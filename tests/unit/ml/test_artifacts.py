from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from parcel_sorter.data import DamageClass, DatasetSplit, ParcelClass, sha256_file
from parcel_sorter.ml.artifacts import (
    CandidateValidationResult,
    ExportBundleManifest,
    ParityReport,
    load_prediction_table,
    save_export_manifest,
    save_prediction_table,
    select_model_candidate,
    verify_onnx_parity,
)
from parcel_sorter.ml.contracts import DecisionThresholds, ModelCandidate, PreprocessContract
from parcel_sorter.ml.evaluation import ClassMetric, EvaluationReport, PredictionRecord

CLASSES = tuple(ParcelClass)


def _thresholds(value: float = 0.6) -> DecisionThresholds:
    return DecisionThresholds(
        tau_ok_pass=value,
        tau_damage_by_class={item: value for item in DamageClass},
    )


def _report(
    split: DatasetSplit,
    *,
    damage_recall: float = 0.9,
    false_reject_rate: float = 0.1,
) -> EvaluationReport:
    per_class = {
        item.value: ClassMetric(precision=0.8, recall=0.8, f1=0.8, support=1) for item in CLASSES
    }
    confusion = {
        actual.value: {predicted.value: int(actual is predicted) for predicted in CLASSES}
        for actual in CLASSES
    }
    return EvaluationReport(
        split=split,
        record_count=5,
        accuracy=0.8,
        damage_recall=damage_recall,
        confirmed_damage_recall=0.8,
        false_reject_rate=false_reject_rate,
        coverage=0.8,
        uncertain_rate=0.2,
        error_rate=0.0,
        confusion_matrix=confusion,
        per_class=per_class,
    )


def _record(index: int, split: DatasetSplit = DatasetSplit.VAL) -> PredictionRecord:
    return PredictionRecord(
        image_id=f"image-{index}",
        split=split,
        true_class=ParcelClass.OK,
        classes=CLASSES,
        probabilities=(0.8, 0.05, 0.05, 0.05, 0.05),
    )


def test_model_selection_uses_validation_safety_metrics() -> None:
    safer = CandidateValidationResult(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        checkpoint_sha256="a" * 64,
        report=_report(DatasetSplit.VAL, damage_recall=0.95, false_reject_rate=0.2),
    )
    lower_recall = CandidateValidationResult(
        candidate=ModelCandidate.YOLOV8N_CLS,
        checkpoint_sha256="b" * 64,
        report=_report(DatasetSplit.VAL, damage_recall=0.94, false_reject_rate=0.01),
    )

    assert select_model_candidate((lower_recall, safer)) is safer
    with pytest.raises(ValueError, match="at least one"):
        select_model_candidate(())
    with pytest.raises(ValueError, match="unique"):
        select_model_candidate((safer, safer))


def test_candidate_result_rejects_test_metrics() -> None:
    with pytest.raises(ValidationError, match="validation reports only"):
        CandidateValidationResult(
            candidate=ModelCandidate.MOBILENET_V3_SMALL,
            checkpoint_sha256="a" * 64,
            report=_report(DatasetSplit.TEST),
        )


def test_parity_requires_twenty_close_rows_and_identical_decisions() -> None:
    torch_rows = [(0.8, 0.05, 0.05, 0.05, 0.05)] * 20
    onnx_rows = [(0.80001, 0.04999, 0.05, 0.05, 0.05)] * 20

    report = verify_onnx_parity(
        torch_rows,
        onnx_rows,
        classes=CLASSES,
        thresholds=_thresholds(),
    )

    assert report.sample_count == 20
    assert report.numerical_parity
    assert report.decision_parity
    assert report.decision_matches == 20


@pytest.mark.parametrize(
    ("torch_rows", "onnx_rows", "message"),
    [
        ([(1.0, 0, 0, 0, 0)] * 19, [(1.0, 0, 0, 0, 0)] * 19, "at least 20"),
        ([(1.0, 0, 0, 0, 0)] * 20, [(1.0, 0, 0, 0, 0)] * 21, "lengths"),
        ([(1.0, 0, 0, 0)] * 20, [(1.0, 0, 0, 0)] * 20, "align"),
    ],
)
def test_parity_rejects_invalid_tables(
    torch_rows: list[tuple[float, ...]],
    onnx_rows: list[tuple[float, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_onnx_parity(
            torch_rows,
            onnx_rows,
            classes=CLASSES,
            thresholds=_thresholds(),
        )


def test_parity_reports_numerical_and_decision_mismatch() -> None:
    torch_rows = [(0.8, 0.05, 0.05, 0.05, 0.05)] * 20
    onnx_rows = [(0.5, 0.2, 0.1, 0.1, 0.1)] * 20

    report = verify_onnx_parity(
        torch_rows,
        onnx_rows,
        classes=CLASSES,
        thresholds=_thresholds(0.6),
    )

    assert not report.numerical_parity
    assert not report.decision_parity


def test_parity_report_rejects_inconsistent_claims() -> None:
    with pytest.raises(ValidationError, match="decision_parity"):
        ParityReport(
            sample_count=20,
            absolute_tolerance=1e-4,
            max_absolute_difference=1e-5,
            numerical_parity=True,
            decision_matches=19,
            decision_parity=True,
        )
    with pytest.raises(ValidationError, match="numerical_parity"):
        ParityReport(
            sample_count=20,
            absolute_tolerance=1e-4,
            max_absolute_difference=1e-3,
            numerical_parity=True,
            decision_matches=20,
            decision_parity=True,
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        ParityReport(
            sample_count=20,
            absolute_tolerance=1e-4,
            max_absolute_difference=1e-5,
            numerical_parity=True,
            decision_matches=21,
            decision_parity=False,
        )


def test_prediction_table_round_trip_is_hash_verified(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "val.jsonl"
    records = tuple(_record(index) for index in range(2))

    digest = save_prediction_table(path, records)

    assert digest == sha256_file(path)
    assert load_prediction_table(path, expected_sha256=digest) == records
    assert path.with_suffix(".jsonl.sha256").is_file()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_prediction_table(path, expected_sha256="0" * 64)


def test_empty_prediction_tables_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        save_prediction_table(tmp_path / "empty.jsonl", ())
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_prediction_table(empty, expected_sha256=sha256_file(empty))


def test_export_manifest_requires_frozen_splits_and_parity(tmp_path: Path) -> None:
    parity = ParityReport(
        sample_count=20,
        absolute_tolerance=1e-4,
        max_absolute_difference=1e-5,
        numerical_parity=True,
        decision_matches=20,
        decision_parity=True,
    )
    manifest = ExportBundleManifest(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        run_id="run-12345",
        git_commit="a" * 40,
        dataset_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        source_checkpoint_sha256="d" * 64,
        onnx_sha256="e" * 64,
        onnx_opset=18,
        input_name="images",
        output_name="probabilities",
        classes=CLASSES,
        preprocess=PreprocessContract(
            preprocess_id="rgb320-v1",
            resize="resize_square",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        thresholds=_thresholds(),
        validation_report=_report(DatasetSplit.VAL),
        test_report=_report(DatasetSplit.TEST),
        parity=parity,
    )

    path = tmp_path / "bundle" / "model_card.json"
    assert save_export_manifest(path, manifest) == sha256_file(path)

    bad_payload = manifest.model_dump()
    bad_payload["parity"] = parity.model_copy(
        update={"decision_matches": 19, "decision_parity": False}
    )
    with pytest.raises(ValidationError, match="requires numerical and decision parity"):
        ExportBundleManifest.model_validate(bad_payload)
