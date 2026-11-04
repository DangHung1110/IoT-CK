"""Tests for the temporary binary v0 evaluation protocol."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from parcel_sorter.data import BinaryParcelClass, DatasetSplit, sha256_file
from parcel_sorter.ml.binary_evaluation import (
    BinaryCandidateValidationResult,
    BinaryDecisionCode,
    BinaryPredictionRecord,
    BinaryThresholds,
    apply_binary_policy,
    evaluate_binary_predictions,
    load_binary_prediction_table,
    save_binary_prediction_table,
    select_binary_model_candidate,
    select_binary_validation_thresholds,
)
from parcel_sorter.ml.contracts import ModelCandidate

CLASSES = (BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL)


def _record(
    image_id: str,
    true_class: BinaryParcelClass,
    probabilities: tuple[float, float],
    *,
    split: DatasetSplit = DatasetSplit.VAL,
) -> BinaryPredictionRecord:
    return BinaryPredictionRecord(
        image_id=image_id,
        split=split,
        true_class=true_class,
        classes=CLASSES,
        probabilities=probabilities,
    )


def _records(split: DatasetSplit = DatasetSplit.VAL) -> tuple[BinaryPredictionRecord, ...]:
    return (
        _record("damage-hit", BinaryParcelClass.DAMAGE, (0.9, 0.1), split=split),
        _record("damage-miss", BinaryParcelClass.DAMAGE, (0.4, 0.6), split=split),
        _record("normal-pass", BinaryParcelClass.NORMAL, (0.1, 0.9), split=split),
        _record("normal-reject", BinaryParcelClass.NORMAL, (0.6, 0.4), split=split),
    )


def test_binary_evaluation_reports_argmax_and_policy_metrics() -> None:
    report = evaluate_binary_predictions(
        _records(),
        BinaryThresholds(tau_normal_pass=0.7, tau_damage_confirm=0.7),
        required_split=DatasetSplit.VAL,
    )

    assert report.record_count == 4
    assert report.accuracy == 0.5
    assert report.damage_class_precision == 0.5
    assert report.damage_class_recall == 0.5
    assert report.damage_class_f1 == 0.5
    assert report.damage_safety_recall == 1.0
    assert report.confirmed_damage_recall == 0.5
    assert report.false_reject_rate == 0.5
    assert report.uncertain_rate == 0.5
    assert report.confusion_matrix["damage"]["normal"] == 1


def test_binary_policy_fails_closed_and_threshold_selection_is_validation_only() -> None:
    thresholds = BinaryThresholds(tau_normal_pass=0.8, tau_damage_confirm=0.8)
    assert apply_binary_policy((0.9, 0.1), CLASSES, thresholds) is BinaryDecisionCode.DAMAGE
    assert apply_binary_policy((0.45, 0.55), CLASSES, thresholds) is BinaryDecisionCode.AI_UNCERTAIN
    assert (
        apply_binary_policy((float("nan"), 0.2), CLASSES, thresholds) is BinaryDecisionCode.AI_ERROR
    )

    with pytest.raises(ValueError, match="validation predictions only"):
        select_binary_validation_thresholds(_records(DatasetSplit.TEST), (0.6, 0.8))

    selection = select_binary_validation_thresholds(
        _records(),
        (0.6, 0.8),
        min_damage_safety_recall=0.5,
        max_false_reject_rate=0.5,
    )
    assert len(selection.evaluated_points) == 4
    assert selection.best.constraint_met


def test_binary_candidate_selection_never_accepts_test_evidence() -> None:
    thresholds = BinaryThresholds(tau_normal_pass=0.7, tau_damage_confirm=0.7)
    mobile_report = evaluate_binary_predictions(_records(), thresholds)
    mobile = BinaryCandidateValidationResult(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        checkpoint_sha256="a" * 64,
        thresholds=thresholds,
        constraint_met=True,
        report=mobile_report,
    )
    better_records = tuple(
        record.model_copy(
            update={
                "probabilities": (
                    (0.95, 0.05) if record.true_class is BinaryParcelClass.DAMAGE else (0.05, 0.95)
                )
            }
        )
        for record in _records()
    )
    yolo = BinaryCandidateValidationResult(
        candidate=ModelCandidate.YOLOV8N_CLS,
        checkpoint_sha256="b" * 64,
        thresholds=thresholds,
        constraint_met=True,
        report=evaluate_binary_predictions(better_records, thresholds),
    )

    assert select_binary_model_candidate((mobile, yolo)) is yolo
    assert (
        select_binary_model_candidate((mobile, yolo.model_copy(update={"constraint_met": False})))
        is mobile
    )
    with pytest.raises(ValidationError, match="validation reports only"):
        BinaryCandidateValidationResult(
            candidate=ModelCandidate.YOLOV8N_CLS,
            checkpoint_sha256="b" * 64,
            thresholds=thresholds,
            constraint_met=True,
            report=evaluate_binary_predictions(_records(DatasetSplit.TEST), thresholds),
        )


def test_binary_prediction_table_round_trip_is_hash_verified(tmp_path: Path) -> None:
    path = tmp_path / "predictions" / "val.jsonl"
    digest = save_binary_prediction_table(path, _records())

    assert digest == sha256_file(path)
    assert load_binary_prediction_table(path, expected_sha256=digest) == _records()
    assert path.with_suffix(".jsonl.sha256").is_file()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_binary_prediction_table(path, expected_sha256="0" * 64)


def test_binary_prediction_contract_rejects_bad_probabilities() -> None:
    with pytest.raises(ValidationError, match="sum to one"):
        _record("bad", BinaryParcelClass.NORMAL, (0.2, 0.2))
    with pytest.raises(ValidationError, match="finite"):
        _record("bad", BinaryParcelClass.NORMAL, (float("nan"), 0.0))


def test_binary_evaluation_and_selection_fail_closed_on_ambiguous_inputs(
    tmp_path: Path,
) -> None:
    thresholds = BinaryThresholds(tau_normal_pass=0.7, tau_damage_confirm=0.7)
    mixed = list(_records())
    mixed[-1] = mixed[-1].model_copy(update={"split": DatasetSplit.TEST})
    with pytest.raises(ValueError, match="exactly one split"):
        evaluate_binary_predictions(mixed, thresholds)

    changed_order = list(_records())
    changed_order[-1] = changed_order[-1].model_copy(update={"classes": tuple(reversed(CLASSES))})
    with pytest.raises(ValueError, match="class order changed"):
        evaluate_binary_predictions(changed_order, thresholds)

    report = evaluate_binary_predictions(_records(), thresholds)
    result = BinaryCandidateValidationResult(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        checkpoint_sha256="a" * 64,
        thresholds=thresholds,
        constraint_met=True,
        report=report,
    )
    with pytest.raises(ValueError, match="at least one"):
        select_binary_model_candidate(())
    with pytest.raises(ValueError, match="unique"):
        select_binary_model_candidate((result, result))

    with pytest.raises(ValueError, match="must not be empty"):
        save_binary_prediction_table(tmp_path / "empty.jsonl", ())
    empty = tmp_path / "existing-empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_binary_prediction_table(empty, expected_sha256=sha256_file(empty))
