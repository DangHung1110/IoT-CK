"""Auditable ONNX bundle contracts for the temporary binary v0 model."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from parcel_sorter.data import BinaryParcelClass, sha256_file
from parcel_sorter.ml.binary_evaluation import (
    BinaryDecisionCode,
    BinaryThresholds,
    apply_binary_policy,
)
from parcel_sorter.ml.contracts import ModelCandidate, PreprocessContract


class BinaryParityReport(BaseModel):
    """Numerical and policy parity evidence for a fixed-batch ONNX export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=20)
    absolute_tolerance: float = Field(gt=0)
    max_absolute_difference: float = Field(ge=0)
    numerical_parity: bool
    decision_matches: int = Field(ge=0)
    decision_parity: bool

    @model_validator(mode="after")
    def validate_claims(self) -> BinaryParityReport:
        if self.decision_matches > self.sample_count:
            raise ValueError("decision_matches cannot exceed sample_count")
        if self.numerical_parity is not (self.max_absolute_difference <= self.absolute_tolerance):
            raise ValueError("numerical_parity does not match the recorded difference")
        if self.decision_parity is not (self.decision_matches == self.sample_count):
            raise ValueError("decision_parity does not match decision_matches")
        return self


class BinaryOnnxBundleManifest(BaseModel):
    """Machine-readable contract shipped beside the binary ONNX model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    candidate: ModelCandidate
    run_id: str = Field(min_length=8)
    evaluation_id: str = Field(min_length=8)
    training_git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    onnx_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    onnx_opset: int = Field(ge=17, le=21)
    fixed_batch_size: int = Field(default=1, ge=1, le=1)
    input_name: str = Field(min_length=1)
    output_name: str = Field(min_length=1)
    classes: tuple[BinaryParcelClass, ...]
    preprocess: PreprocessContract
    thresholds: BinaryThresholds
    parity: BinaryParityReport

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> BinaryOnnxBundleManifest:
        if self.candidate is not ModelCandidate.MOBILENET_V3_SMALL:
            raise ValueError("binary v0 runtime bundle must contain the frozen MobileNet winner")
        if len(self.classes) != len(BinaryParcelClass) or set(self.classes) != set(
            BinaryParcelClass
        ):
            raise ValueError("classes must contain damage and normal exactly once")
        if not self.parity.numerical_parity or not self.parity.decision_parity:
            raise ValueError("runtime bundle requires numerical and decision parity")
        return self


def verify_binary_onnx_parity(
    pytorch_probabilities: Sequence[Sequence[float]],
    onnx_probabilities: Sequence[Sequence[float]],
    *,
    classes: tuple[BinaryParcelClass, ...],
    thresholds: BinaryThresholds,
    absolute_tolerance: float = 1e-4,
    min_samples: int = 20,
) -> BinaryParityReport:
    """Require close probabilities and identical runtime decisions."""
    if min_samples < 20:
        raise ValueError("parity must require at least 20 samples")
    if absolute_tolerance <= 0 or not math.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be a positive finite value")
    if len(classes) != len(BinaryParcelClass) or set(classes) != set(BinaryParcelClass):
        raise ValueError("classes must contain damage and normal exactly once")
    if len(pytorch_probabilities) != len(onnx_probabilities):
        raise ValueError("PyTorch and ONNX parity tables have different lengths")
    if len(pytorch_probabilities) < min_samples:
        raise ValueError(f"parity requires at least {min_samples} samples")

    max_difference = 0.0
    decision_matches = 0
    for torch_row, onnx_row in zip(pytorch_probabilities, onnx_probabilities, strict=True):
        if len(torch_row) != len(classes) or len(onnx_row) != len(classes):
            raise ValueError("parity probability rows must align with classes")
        difference = max(
            abs(float(left) - float(right)) for left, right in zip(torch_row, onnx_row, strict=True)
        )
        if not math.isfinite(difference):
            raise ValueError("parity probabilities must be finite")
        max_difference = max(max_difference, difference)
        torch_decision = apply_binary_policy(torch_row, classes, thresholds)
        onnx_decision = apply_binary_policy(onnx_row, classes, thresholds)
        decision_matches += int(
            torch_decision is not BinaryDecisionCode.AI_ERROR and torch_decision is onnx_decision
        )

    sample_count = len(pytorch_probabilities)
    return BinaryParityReport(
        sample_count=sample_count,
        absolute_tolerance=absolute_tolerance,
        max_absolute_difference=max_difference,
        numerical_parity=max_difference <= absolute_tolerance,
        decision_matches=decision_matches,
        decision_parity=decision_matches == sample_count,
    )


def save_binary_bundle_manifest(path: Path, manifest: BinaryOnnxBundleManifest) -> str:
    """Atomically save a binary bundle manifest and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def load_binary_bundle(bundle_dir: Path) -> tuple[Path, BinaryOnnxBundleManifest]:
    """Load a bundle and reject a missing or modified ONNX payload."""
    manifest_path = bundle_dir / "manifest.json"
    model_path = bundle_dir / "model.onnx"
    if not manifest_path.is_file():
        raise ValueError(f"bundle manifest is missing: {manifest_path}")
    if not model_path.is_file():
        raise ValueError(f"ONNX model is missing: {model_path}")
    manifest = BinaryOnnxBundleManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    actual = sha256_file(model_path)
    if actual != manifest.onnx_sha256:
        raise ValueError(f"ONNX SHA-256 mismatch: expected {manifest.onnx_sha256}, got {actual}")
    return model_path, manifest
