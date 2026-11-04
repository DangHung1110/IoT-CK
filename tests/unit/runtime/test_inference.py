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
from parcel_sorter.runtime.inference import (
    BinaryInferenceResult,
    OnnxBinaryClassifier,
    TemporalDecisionSmoother,
    annotate_frame,
    annotate_roi_frame,
    centered_square_roi,
)


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


def _inference_result(
    damage: float, normal: float, decision: BinaryDecisionCode
) -> BinaryInferenceResult:
    return BinaryInferenceResult(
        probabilities={
            BinaryParcelClass.DAMAGE: damage,
            BinaryParcelClass.NORMAL: normal,
        },
        decision=decision,
        latency_ms=2.0,
    )


def test_temporal_smoother_warms_up_then_requires_repeated_decisions() -> None:
    smoother = TemporalDecisionSmoother(
        classes=(BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL),
        thresholds=BinaryThresholds(tau_normal_pass=0.95, tau_damage_confirm=0.5),
        window_size=5,
        stable_frames=3,
    )
    raw = _inference_result(0.99, 0.01, BinaryDecisionCode.DAMAGE)

    results = [smoother.update(raw) for _ in range(7)]

    assert all(item.decision is BinaryDecisionCode.AI_UNCERTAIN for item in results[:6])
    assert not results[5].stable
    assert results[6].stable
    assert results[6].decision is BinaryDecisionCode.DAMAGE
    assert results[6].frames_in_window == 5
    assert results[6].probabilities[BinaryParcelClass.DAMAGE] == pytest.approx(0.99)

    smoother.reset()
    reset_result = smoother.update(raw)
    assert reset_result.frames_in_window == 1
    assert not reset_result.stable


def test_temporal_smoother_averages_probabilities_and_rejects_invalid_config() -> None:
    thresholds = BinaryThresholds(tau_normal_pass=0.95, tau_damage_confirm=0.5)
    classes = (BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL)
    with pytest.raises(ValueError, match="window_size"):
        TemporalDecisionSmoother(classes=classes, thresholds=thresholds, window_size=0)
    with pytest.raises(ValueError, match="stable_frames"):
        TemporalDecisionSmoother(
            classes=classes, thresholds=thresholds, window_size=1, stable_frames=0
        )

    smoother = TemporalDecisionSmoother(
        classes=classes, thresholds=thresholds, window_size=2, stable_frames=1
    )
    smoother.update(_inference_result(0.2, 0.8, BinaryDecisionCode.AI_UNCERTAIN))
    result = smoother.update(_inference_result(0.8, 0.2, BinaryDecisionCode.DAMAGE))
    assert result.probabilities[BinaryParcelClass.DAMAGE] == pytest.approx(0.5)
    assert result.decision is BinaryDecisionCode.DAMAGE


def test_centered_square_roi_crops_expected_pixels_and_validates_input() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi, crop = centered_square_roi(frame, 0.6)

    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (70, 20, 130, 80)
    assert crop.shape == (60, 60, 3)
    with pytest.raises(ValueError, match="ROI scale"):
        centered_square_roi(frame, 0)
    with pytest.raises(ValueError, match="three color channels"):
        centered_square_roi(np.zeros((10, 10), dtype=np.uint8), 0.5)


def test_roi_annotation_draws_boundary_and_stabilizing_status() -> None:
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    roi, _ = centered_square_roi(frame, 0.5)
    result = BinaryInferenceResult(
        probabilities={
            BinaryParcelClass.DAMAGE: 0.6,
            BinaryParcelClass.NORMAL: 0.4,
        },
        decision=BinaryDecisionCode.AI_UNCERTAIN,
        latency_ms=2,
        stable=False,
        frames_in_window=2,
        window_size=5,
    )

    annotated = annotate_roi_frame(frame, roi, result)

    assert annotated.shape == frame.shape
    assert np.count_nonzero(annotated) > 0


def test_inference_result_rejects_impossible_window_progress() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        BinaryInferenceResult(
            probabilities={
                BinaryParcelClass.DAMAGE: 0.6,
                BinaryParcelClass.NORMAL: 0.4,
            },
            decision=BinaryDecisionCode.DAMAGE,
            latency_ms=2,
            frames_in_window=3,
            window_size=2,
        )
