"""Laptop and edge inference runtime."""

from parcel_sorter.runtime.inference import (
    BinaryInferenceResult,
    OnnxBinaryClassifier,
    annotate_frame,
)

__all__ = ["BinaryInferenceResult", "OnnxBinaryClassifier", "annotate_frame"]
