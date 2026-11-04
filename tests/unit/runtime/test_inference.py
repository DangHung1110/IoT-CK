from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from parcel_sorter.data import BinaryParcelClass, sha256_file
from parcel_sorter.ml.binary_artifacts import (
    BinaryOnnxBundleManifest,
    BinaryParityReport,
    save_binary_bundle_manifest,
)
from parcel_sorter.ml.binary_evaluation import BinaryDecisionCode, BinaryThresholds
from parcel_sorter.ml.contracts import ModelCandidate, PreprocessContract
from parcel_sorter.runtime.inference import OnnxBinaryClassifier, annotate_frame


class _FakeSession:
    def __init__(
        self,
        logits: list[list[float]],
        *,
        input_name: str = "images",
        output_name: str = "logits",
        input_shape: list[int] | None = None,
    ) -> None:
        self.logits = np.asarray(logits, dtype=np.float32)
        self.input_name = input_name
        self.output_name = output_name
        self.input_shape = input_shape or [1, 3, 64, 64]
        self.feed: dict[str, Any] | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.input_name, shape=self.input_shape)]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.output_name)]

    def run(self, outputs: list[str], feed: dict[str, Any]) -> list[np.ndarray]:
        assert outputs == ["logits"]
        self.feed = feed
        return [self.logits]


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    model_path = bundle / "model.onnx"
    model_path.write_bytes(b"fake-onnx")
    manifest = BinaryOnnxBundleManifest(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        run_id="mobilenet-run-v0",
        evaluation_id="evaluation-v0",
        training_git_commit="a" * 40,
        dataset_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        source_checkpoint_sha256="d" * 64,
        frozen_selection_sha256="e" * 64,
        onnx_sha256=sha256_file(model_path),
        onnx_opset=18,
        input_name="images",
        output_name="logits",
        classes=(BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL),
        preprocess=PreprocessContract(
            preprocess_id="binary-v0",
            input_size=64,
            resize="resize_square",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        thresholds=BinaryThresholds(tau_normal_pass=0.95, tau_damage_confirm=0.5),
        parity=BinaryParityReport(
            sample_count=20,
            absolute_tolerance=1e-4,
            max_absolute_difference=1e-6,
            numerical_parity=True,
            decision_matches=20,
            decision_parity=True,
        ),
    )
    save_binary_bundle_manifest(bundle / "manifest.json", manifest)
    return bundle


def _clock() -> Iterator[float]:
    yield 10.0
    yield 10.012


def test_classifier_preprocesses_and_applies_damage_policy(tmp_path: Path) -> None:
    session = _FakeSession([[4.0, 1.0]])
    ticks = _clock()
    classifier = OnnxBinaryClassifier(
        _bundle(tmp_path),
        session_factory=lambda path, providers: session,
        clock=lambda: next(ticks),
    )

    result = classifier.predict_rgb(np.full((80, 120, 3), 255, dtype=np.uint8))

    assert result.decision is BinaryDecisionCode.DAMAGE
    assert result.probabilities[BinaryParcelClass.DAMAGE] == pytest.approx(0.95257, rel=1e-4)
    assert result.latency_ms == pytest.approx(12.0)
    assert session.feed is not None
    assert session.feed["images"].shape == (1, 3, 64, 64)
    assert session.feed["images"].dtype == np.float32
    assert np.all(session.feed["images"] == 1.0)


def test_classifier_can_return_uncertain_and_accept_bgr(tmp_path: Path) -> None:
    session = _FakeSession([[0.0, 0.4]])
    ticks = _clock()
    classifier = OnnxBinaryClassifier(
        _bundle(tmp_path),
        session_factory=lambda path, providers: session,
        clock=lambda: next(ticks),
    )

    result = classifier.predict_bgr(np.zeros((64, 64, 3), dtype=np.uint8))

    assert result.decision is BinaryDecisionCode.AI_UNCERTAIN


@pytest.mark.parametrize(
    ("session", "message"),
    [
        (_FakeSession([[1.0, 0.0]], input_name="wrong"), "input"),
        (_FakeSession([[1.0, 0.0]], output_name="wrong"), "output"),
        (_FakeSession([[1.0, 0.0]], input_shape=[1, 3, 32, 32]), "shape"),
    ],
)
def test_classifier_rejects_incompatible_onnx_contract(
    tmp_path: Path, session: _FakeSession, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OnnxBinaryClassifier(_bundle(tmp_path), session_factory=lambda path, providers: session)


def test_classifier_rejects_bad_images_and_outputs(tmp_path: Path) -> None:
    classifier = OnnxBinaryClassifier(
        _bundle(tmp_path),
        session_factory=lambda path, providers: _FakeSession([[1.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="three RGB channels"):
        classifier.predict_rgb(np.zeros((64, 64), dtype=np.uint8))

    bad_output = OnnxBinaryClassifier(
        _bundle(tmp_path / "bad-output"),
        session_factory=lambda path, providers: _FakeSession([[1.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="output shape"):
        bad_output.predict_rgb(np.zeros((64, 64, 3), dtype=np.uint8))


def test_annotation_preserves_shape_and_changes_pixels(tmp_path: Path) -> None:
    session = _FakeSession([[4.0, 1.0]])
    ticks = _clock()
    classifier = OnnxBinaryClassifier(
        _bundle(tmp_path),
        session_factory=lambda path, providers: session,
        clock=lambda: next(ticks),
    )
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    result = classifier.predict_rgb(frame)

    annotated = annotate_frame(frame, result)

    assert annotated.shape == frame.shape
    assert np.count_nonzero(annotated) > 0
    assert np.count_nonzero(frame) == 0
