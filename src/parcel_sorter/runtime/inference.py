"""CPU-only ONNX inference for images and webcam frames."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from parcel_sorter.data import BinaryParcelClass
from parcel_sorter.ml.binary_artifacts import load_binary_bundle
from parcel_sorter.ml.binary_evaluation import BinaryDecisionCode, apply_binary_policy


class BinaryInferenceResult(BaseModel):
    """One runtime result with probabilities, policy decision, and latency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probabilities: dict[BinaryParcelClass, float]
    decision: BinaryDecisionCode
    latency_ms: float = Field(ge=0)


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
    lines = (
        f"{result.decision.value}  {result.latency_ms:.1f} ms",
        f"damage={damage:.3f} normal={normal:.3f}",
    )
    output = frame.copy()
    color = colors[result.decision]
    for index, line in enumerate(lines):
        y = 35 + index * 32
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(output, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return output
