"""CPU-only ONNX inference for images and webcam frames."""

from __future__ import annotations

import importlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from parcel_sorter.data import BinaryParcelClass
from parcel_sorter.ml.binary_artifacts import load_binary_bundle
from parcel_sorter.ml.binary_evaluation import (
    BinaryDecisionCode,
    BinaryThresholds,
    apply_binary_policy,
)


class BinaryInferenceResult(BaseModel):
    """One runtime result with probabilities, policy decision, and latency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probabilities: dict[BinaryParcelClass, float]
    decision: BinaryDecisionCode
    latency_ms: float = Field(ge=0)
    stable: bool = True
    frames_in_window: int = Field(default=1, ge=1)
    window_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_window_progress(self) -> BinaryInferenceResult:
        if self.frames_in_window > self.window_size:
            raise ValueError("frames_in_window cannot exceed window_size")
        return self


@dataclass(frozen=True)
class SquareRoi:
    """Pixel coordinates for a centered square camera region."""

    x1: int
    y1: int
    x2: int
    y2: int


class TemporalDecisionSmoother:
    """Average a rolling probability window and require stable repeated decisions."""

    def __init__(
        self,
        *,
        classes: tuple[BinaryParcelClass, ...],
        thresholds: BinaryThresholds,
        window_size: int = 5,
        stable_frames: int = 3,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        self.classes = classes
        self.thresholds = thresholds
        self.window_size = window_size
        self.stable_frames = stable_frames
        self._history: deque[BinaryInferenceResult] = deque(maxlen=window_size)
        self._candidate: BinaryDecisionCode | None = None
        self._candidate_streak = 0

    def reset(self) -> None:
        """Clear evidence when the parcel in the ROI changes."""
        self._history.clear()
        self._candidate = None
        self._candidate_streak = 0

    def update(self, result: BinaryInferenceResult) -> BinaryInferenceResult:
        """Return a fail-safe uncertain result until the rolling decision is stable."""
        self._history.append(result)
        count = len(self._history)
        averaged = {
            parcel_class: sum(item.probabilities[parcel_class] for item in self._history) / count
            for parcel_class in self.classes
        }
        probability_values = tuple(averaged[item] for item in self.classes)
        candidate = apply_binary_policy(probability_values, self.classes, self.thresholds)
        if count < self.window_size:
            self._candidate = None
            self._candidate_streak = 0
        elif candidate is self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = candidate
            self._candidate_streak = 1
        stable = (
            count == self.window_size
            and candidate not in {BinaryDecisionCode.AI_UNCERTAIN, BinaryDecisionCode.AI_ERROR}
            and self._candidate_streak >= self.stable_frames
        )
        return BinaryInferenceResult(
            probabilities=averaged,
            decision=candidate if stable else BinaryDecisionCode.AI_UNCERTAIN,
            latency_ms=sum(item.latency_ms for item in self._history) / count,
            stable=stable,
            frames_in_window=count,
            window_size=self.window_size,
        )


SessionFactory = Callable[[str, list[str]], Any]


def _default_session_factory(model_path: str, providers: list[str]) -> Any:
    ort = importlib.import_module("onnxruntime")
    return ort.InferenceSession(model_path, providers=providers)


class OnnxBinaryClassifier:
    """Verified fixed-batch MobileNet ONNX classifier for local CPU inference."""

    def __init__(
        self,
        bundle_dir: Path,
        *,
        session_factory: SessionFactory = _default_session_factory,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        model_path, manifest = load_binary_bundle(bundle_dir)
        self.manifest = manifest
        self._clock = clock
        self._session = session_factory(str(model_path), ["CPUExecutionProvider"])
        self._validate_session_contract()

    def _validate_session_contract(self) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != self.manifest.input_name:
            raise ValueError("ONNX input does not match bundle manifest")
        if len(outputs) != 1 or outputs[0].name != self.manifest.output_name:
            raise ValueError("ONNX output does not match bundle manifest")
        expected_input = [
            self.manifest.fixed_batch_size,
            3,
            self.manifest.preprocess.input_size,
            self.manifest.preprocess.input_size,
        ]
        if list(inputs[0].shape) != expected_input:
            raise ValueError(f"ONNX input shape must be {expected_input}")

    def _preprocess_rgb(self, rgb_image: Any) -> Any:
        numpy = importlib.import_module("numpy")
        pillow = importlib.import_module("PIL.Image")
        array = numpy.asarray(rgb_image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("input image must have three RGB channels")
        image = pillow.fromarray(array.astype("uint8"), mode="RGB")
        size = self.manifest.preprocess.input_size
        image = image.resize((size, size), resample=pillow.Resampling.BILINEAR)
        tensor = numpy.asarray(image, dtype=numpy.float32) / self.manifest.preprocess.scale
        mean = numpy.asarray(self.manifest.preprocess.mean, dtype=numpy.float32)
        std = numpy.asarray(self.manifest.preprocess.std, dtype=numpy.float32)
        tensor = (tensor - mean) / std
        return numpy.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])

    def predict_rgb(self, rgb_image: Any) -> BinaryInferenceResult:
        """Classify one RGB array and apply the frozen safety thresholds."""
        numpy = importlib.import_module("numpy")
        tensor = self._preprocess_rgb(rgb_image)
        started = self._clock()
        outputs = self._session.run([self.manifest.output_name], {self.manifest.input_name: tensor})
        latency_ms = (self._clock() - started) * 1000
        if len(outputs) != 1:
            raise ValueError("ONNX session returned an unexpected output count")
        logits = numpy.asarray(outputs[0], dtype=numpy.float64)
        if logits.shape != (1, len(self.manifest.classes)):
            raise ValueError("ONNX output shape does not match the class contract")
        logits = logits[0] - numpy.max(logits[0])
        exponentials = numpy.exp(logits)
        probabilities_array = exponentials / numpy.sum(exponentials)
        probability_values = tuple(float(value) for value in probabilities_array.tolist())
        decision = apply_binary_policy(
            probability_values,
            self.manifest.classes,
            self.manifest.thresholds,
        )
        probabilities = dict(zip(self.manifest.classes, probability_values, strict=True))
        return BinaryInferenceResult(
            probabilities=probabilities,
            decision=decision,
            latency_ms=latency_ms,
        )

    def predict_bgr(self, bgr_image: Any) -> BinaryInferenceResult:
        """Classify one OpenCV BGR frame."""
        cv2 = importlib.import_module("cv2")
        return self.predict_rgb(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))


def annotate_frame(frame: Any, result: BinaryInferenceResult) -> Any:
    """Draw a compact decision overlay onto an OpenCV frame."""
    cv2 = importlib.import_module("cv2")
    colors = {
        BinaryDecisionCode.PASS: (0, 180, 0),
        BinaryDecisionCode.DAMAGE: (0, 0, 230),
        BinaryDecisionCode.AI_UNCERTAIN: (0, 180, 255),
        BinaryDecisionCode.AI_ERROR: (120, 120, 120),
    }
    damage = result.probabilities[BinaryParcelClass.DAMAGE]
    normal = result.probabilities[BinaryParcelClass.NORMAL]
    status = (
        result.decision.value
        if result.stable
        else f"STABILIZING {result.frames_in_window}/{result.window_size}"
    )
    lines = (
        f"{status}  {result.latency_ms:.1f} ms",
        f"damage={damage:.3f} normal={normal:.3f}",
    )
    output = frame.copy()
    color = colors[result.decision]
    for index, line in enumerate(lines):
        y = 35 + index * 32
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return output


def centered_square_roi(frame: Any, scale: float) -> tuple[SquareRoi, Any]:
    """Return a centered square crop without pretending to detect an object."""
    if not 0 < scale <= 1:
        raise ValueError("ROI scale must be in (0, 1]")
    if getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        raise ValueError("webcam frame must have three color channels")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    side = max(1, round(min(height, width) * scale))
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    roi = SquareRoi(x1=x1, y1=y1, x2=x1 + side, y2=y1 + side)
    return roi, frame[roi.y1 : roi.y2, roi.x1 : roi.x2]


def annotate_roi_frame(frame: Any, roi: SquareRoi, result: BinaryInferenceResult) -> Any:
    """Draw the fixed crop boundary and keyboard guidance on a webcam frame."""
    cv2 = importlib.import_module("cv2")
    output = annotate_frame(frame, result)
    color = (0, 180, 0) if result.stable else (0, 180, 255)
    cv2.rectangle(output, (roi.x1, roi.y1), (roi.x2, roi.y2), color, 2)
    label_y = max(92, roi.y1 - 10)
    cv2.putText(
        output,
        "FIXED ROI - place one parcel inside | R reset | Q quit",
        (roi.x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
    )
    cv2.putText(
        output,
        "FIXED ROI - place one parcel inside | R reset | Q quit",
        (roi.x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    return output
