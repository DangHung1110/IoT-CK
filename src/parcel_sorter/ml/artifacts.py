"""Versioned prediction, model-selection, parity, and ONNX bundle contracts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from parcel_sorter.data import DatasetSplit, ParcelClass, sha256_file
from parcel_sorter.ml.contracts import DecisionThresholds, ModelCandidate, PreprocessContract
from parcel_sorter.ml.evaluation import EvaluationReport, PredictionRecord
from parcel_sorter.ml.policy import apply_decision_policy


class CandidateValidationResult(BaseModel):
    """One candidate's validation evidence used before touching held-out test data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ModelCandidate
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report: EvaluationReport

    @model_validator(mode="after")
    def require_validation_split(self) -> CandidateValidationResult:
        if self.report.split is not DatasetSplit.VAL:
            raise ValueError("candidate selection accepts validation reports only")
        return self


class ParityReport(BaseModel):
    """Numerical and decision parity evidence for a fixed-batch ONNX export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=20)
    absolute_tolerance: float = Field(gt=0)
    max_absolute_difference: float = Field(ge=0)
    numerical_parity: bool
    decision_matches: int = Field(ge=0)
    decision_parity: bool

    @model_validator(mode="after")
    def validate_claims(self) -> ParityReport:
        expected_numerical = self.max_absolute_difference <= self.absolute_tolerance
        expected_decision = self.decision_matches == self.sample_count
        if self.decision_matches > self.sample_count:
            raise ValueError("decision_matches cannot exceed sample_count")
        if self.numerical_parity is not expected_numerical:
            raise ValueError("numerical_parity does not match the recorded difference")
        if self.decision_parity is not expected_decision:
            raise ValueError("decision_parity does not match decision_matches")
        return self


class ExportBundleManifest(BaseModel):
    """Machine-readable model card shipped beside an ONNX artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    candidate: ModelCandidate
    run_id: str = Field(min_length=8)
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    onnx_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    onnx_opset: int = Field(ge=17, le=21)
    fixed_batch_size: int = Field(default=1, ge=1, le=1)
    input_name: str = Field(min_length=1)
    output_name: str = Field(min_length=1)
    classes: tuple[ParcelClass, ...]
    preprocess: PreprocessContract
    thresholds: DecisionThresholds
    validation_report: EvaluationReport
    test_report: EvaluationReport
    parity: ParityReport

    @model_validator(mode="after")
    def validate_splits_and_class_map(self) -> ExportBundleManifest:
        if self.validation_report.split is not DatasetSplit.VAL:
            raise ValueError("validation_report must use the validation split")
        if self.test_report.split is not DatasetSplit.TEST:
            raise ValueError("test_report must use the held-out test split")
        if len(self.classes) != len(ParcelClass) or set(self.classes) != set(ParcelClass):
            raise ValueError("classes must contain every parcel class exactly once")
        if not self.parity.numerical_parity or not self.parity.decision_parity:
            raise ValueError("an export bundle requires numerical and decision parity")
        return self


def select_model_candidate(
    results: Sequence[CandidateValidationResult],
) -> CandidateValidationResult:
    """Select on validation only, prioritizing damage recall and false rejects."""
    if not results:
        raise ValueError("at least one candidate validation result is required")
    if len({result.candidate for result in results}) != len(results):
        raise ValueError("candidate validation results must be unique")

    def rank(result: CandidateValidationResult) -> tuple[float, float, float, float, float]:
        report = result.report
        return (
            report.damage_recall,
            -report.false_reject_rate,
            report.confirmed_damage_recall,
            report.coverage,
            report.accuracy,
        )

    return max(results, key=rank)


def verify_onnx_parity(
    pytorch_probabilities: Sequence[Sequence[float]],
    onnx_probabilities: Sequence[Sequence[float]],
    *,
    classes: tuple[ParcelClass, ...],
    thresholds: DecisionThresholds,
    absolute_tolerance: float = 1e-4,
    min_samples: int = 20,
) -> ParityReport:
    """Require close probabilities and identical shared-policy decisions."""
    if min_samples < 20:
        raise ValueError("parity must require at least 20 samples")
    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    if len(classes) != len(ParcelClass) or set(classes) != set(ParcelClass):
        raise ValueError("classes must contain every parcel class exactly once")
    if len(pytorch_probabilities) != len(onnx_probabilities):
        raise ValueError("PyTorch and ONNX parity tables have different lengths")
    if len(pytorch_probabilities) < min_samples:
        raise ValueError(f"parity requires at least {min_samples} samples")

    max_difference = 0.0
    decision_matches = 0
    for torch_row, onnx_row in zip(pytorch_probabilities, onnx_probabilities, strict=True):
        if len(torch_row) != len(classes) or len(onnx_row) != len(classes):
            raise ValueError("parity probability rows must align with classes")
        max_difference = max(
            max_difference,
            max(
                abs(float(left) - float(right))
                for left, right in zip(torch_row, onnx_row, strict=True)
            ),
        )
        torch_decision = apply_decision_policy(torch_row, classes, thresholds)
        onnx_decision = apply_decision_policy(onnx_row, classes, thresholds)
        decision_matches += int(torch_decision.code is onnx_decision.code)

    sample_count = len(pytorch_probabilities)
    return ParityReport(
        sample_count=sample_count,
        absolute_tolerance=absolute_tolerance,
        max_absolute_difference=max_difference,
        numerical_parity=max_difference <= absolute_tolerance,
        decision_matches=decision_matches,
        decision_parity=decision_matches == sample_count,
    )


def save_prediction_table(path: Path, records: Sequence[PredictionRecord]) -> str:
    """Atomically persist immutable JSONL predictions and a sibling SHA-256 file."""
    if not records:
        raise ValueError("prediction table must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
    temporary.replace(path)
    digest = sha256_file(path)
    digest_path = path.with_suffix(f"{path.suffix}.sha256")
    digest_path.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def load_prediction_table(path: Path, *, expected_sha256: str) -> tuple[PredictionRecord, ...]:
    """Verify and load a prediction table without permitting silent mutation."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"prediction table SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    records = tuple(
        PredictionRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError("prediction table must not be empty")
    return records


def save_export_manifest(path: Path, manifest: ExportBundleManifest) -> str:
    """Atomically save the JSON model card and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)
