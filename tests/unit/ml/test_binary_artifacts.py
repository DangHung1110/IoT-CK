from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from parcel_sorter.data import BinaryParcelClass, sha256_file
from parcel_sorter.ml.binary_artifacts import (
    BinaryOnnxBundleManifest,
    BinaryParityReport,
    load_binary_bundle,
    save_binary_bundle_manifest,
    verify_binary_onnx_parity,
)
from parcel_sorter.ml.binary_evaluation import BinaryThresholds
from parcel_sorter.ml.contracts import ModelCandidate, PreprocessContract

CLASSES = (BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL)
THRESHOLDS = BinaryThresholds(tau_normal_pass=0.95, tau_damage_confirm=0.5)


def _preprocess() -> PreprocessContract:
    return PreprocessContract(
        preprocess_id="binary-v0",
        input_size=64,
        resize="resize_square",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def _manifest(onnx_sha256: str) -> BinaryOnnxBundleManifest:
    return BinaryOnnxBundleManifest(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        run_id="mobilenet-run-v0",
        evaluation_id="evaluation-v0",
        training_git_commit="a" * 40,
        dataset_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        source_checkpoint_sha256="d" * 64,
        frozen_selection_sha256="e" * 64,
        onnx_sha256=onnx_sha256,
        onnx_opset=18,
        input_name="images",
        output_name="logits",
        classes=CLASSES,
        preprocess=_preprocess(),
        thresholds=THRESHOLDS,
        parity=BinaryParityReport(
            sample_count=20,
            absolute_tolerance=1e-4,
            max_absolute_difference=1e-6,
            numerical_parity=True,
            decision_matches=20,
            decision_parity=True,
        ),
    )


def test_binary_parity_requires_close_probabilities_and_matching_policy() -> None:
    torch_rows = [(0.8, 0.2)] * 20
    onnx_rows = [(0.800001, 0.199999)] * 20

    report = verify_binary_onnx_parity(
        torch_rows,
        onnx_rows,
        classes=CLASSES,
        thresholds=THRESHOLDS,
    )

    assert report.numerical_parity
    assert report.decision_parity
    assert report.decision_matches == 20


def test_binary_parity_reports_mismatch() -> None:
    report = verify_binary_onnx_parity(
        [(0.8, 0.2)] * 20,
        [(0.2, 0.8)] * 20,
        classes=CLASSES,
        thresholds=THRESHOLDS,
    )

    assert not report.numerical_parity
    assert not report.decision_parity


@pytest.mark.parametrize(
    ("left", "right", "kwargs", "message"),
    [
        ([(0.8, 0.2)] * 19, [(0.8, 0.2)] * 19, {}, "at least 20"),
        ([(0.8, 0.2)] * 20, [(0.8, 0.2)] * 19, {}, "different lengths"),
        ([(1.0,)] * 20, [(1.0,)] * 20, {}, "align"),
        ([(0.8, 0.2)] * 20, [(0.8, 0.2)] * 20, {"min_samples": 19}, "at least 20"),
        (
            [(0.8, 0.2)] * 20,
            [(0.8, 0.2)] * 20,
            {"absolute_tolerance": 0},
            "positive finite",
        ),
    ],
)
def test_binary_parity_rejects_invalid_inputs(
    left: list[tuple[float, ...]],
    right: list[tuple[float, ...]],
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_binary_onnx_parity(
            left,
            right,
            classes=CLASSES,
            thresholds=THRESHOLDS,
            **kwargs,
        )


def test_bundle_round_trip_and_hash_enforcement(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    model = bundle / "model.onnx"
    model.write_bytes(b"verified-model")
    manifest = _manifest(sha256_file(model))

    digest = save_binary_bundle_manifest(bundle / "manifest.json", manifest)
    loaded_model, loaded_manifest = load_binary_bundle(bundle)

    assert digest == sha256_file(bundle / "manifest.json")
    assert loaded_model == model
    assert loaded_manifest == manifest

    model.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_binary_bundle(bundle)


def test_bundle_rejects_missing_files_and_invalid_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest is missing"):
        load_binary_bundle(tmp_path)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="ONNX model is missing"):
        load_binary_bundle(tmp_path)

    payload = _manifest("f" * 64).model_dump(mode="json")
    payload["candidate"] = ModelCandidate.YOLOV8N_CLS.value
    with pytest.raises(ValidationError, match="frozen MobileNet"):
        BinaryOnnxBundleManifest.model_validate(payload)

    parity = payload["parity"]
    parity["decision_matches"] = 19
    with pytest.raises(ValidationError, match="decision_parity"):
        BinaryParityReport.model_validate(parity)


def test_bundle_manifest_is_plain_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest("f" * 64)
    save_binary_bundle_manifest(path, manifest)
    assert json.loads(path.read_text(encoding="utf-8"))["classes"] == ["damage", "normal"]
