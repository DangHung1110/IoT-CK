"""Leakage-safe evaluation and threshold selection for the temporary binary v0 model."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import product
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from parcel_sorter.data import BinaryParcelClass, DatasetSplit, sha256_file
from parcel_sorter.ml.contracts import ModelCandidate
from parcel_sorter.ml.evaluation import ClassMetric


class BinaryDecisionCode(StrEnum):
    """Stable policy outcomes for the temporary damage/normal classifier."""

    PASS = "PASS"
    DAMAGE = "DAMAGE"
    AI_UNCERTAIN = "AI_UNCERTAIN"
    AI_ERROR = "AI_ERROR"


class BinaryThresholds(BaseModel):
    """Two explicit confidence thresholds selected using validation predictions only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tau_normal_pass: float = Field(ge=0, le=1)
    tau_damage_confirm: float = Field(ge=0, le=1)


class BinaryPredictionRecord(BaseModel):
    """One immutable probability row from a binary candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str = Field(min_length=1)
    split: DatasetSplit
    true_class: BinaryParcelClass
    classes: tuple[BinaryParcelClass, ...]
    probabilities: tuple[float, ...]

    @model_validator(mode="after")
    def validate_probability_contract(self) -> Self:
        expected = set(BinaryParcelClass)
        if len(self.classes) != len(expected) or set(self.classes) != expected:
            raise ValueError("classes must contain damage and normal exactly once")
        if len(self.probabilities) != len(self.classes):
            raise ValueError("probabilities must align with classes")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in self.probabilities):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=0, abs_tol=1e-3):
            raise ValueError("probabilities must sum to one")
        return self


class BinaryEvaluationReport(BaseModel):
    """Argmax and fail-safe policy metrics for one immutable split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: DatasetSplit
    record_count: int = Field(gt=0)
    accuracy: float = Field(ge=0, le=1)
    macro_precision: float = Field(ge=0, le=1)
    macro_recall: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    damage_class_precision: float = Field(ge=0, le=1)
    damage_class_recall: float = Field(ge=0, le=1)
    damage_class_f1: float = Field(ge=0, le=1)
    damage_safety_recall: float = Field(ge=0, le=1)
    confirmed_damage_recall: float = Field(ge=0, le=1)
    false_reject_rate: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    uncertain_rate: float = Field(ge=0, le=1)
    confusion_matrix: dict[str, dict[str, int]]
    per_class: dict[str, ClassMetric]


class BinaryThresholdSweepPoint(BaseModel):
    """One audited validation-only threshold pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    thresholds: BinaryThresholds
    constraint_met: bool
    damage_safety_recall: float = Field(ge=0, le=1)
    confirmed_damage_recall: float = Field(ge=0, le=1)
    false_reject_rate: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    uncertain_rate: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)


class BinaryThresholdSelection(BaseModel):
    """Selected validation point plus the full frontier for audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    best: BinaryThresholdSweepPoint
    evaluated_points: tuple[BinaryThresholdSweepPoint, ...]


class BinaryCandidateValidationResult(BaseModel):
    """Frozen validation evidence used to choose a candidate before test is opened."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ModelCandidate
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    thresholds: BinaryThresholds
    constraint_met: bool
    report: BinaryEvaluationReport

    @model_validator(mode="after")
    def require_validation_split(self) -> Self:
        if self.report.split is not DatasetSplit.VAL:
            raise ValueError("candidate selection accepts validation reports only")
        return self


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _argmax(record: BinaryPredictionRecord) -> BinaryParcelClass:
    index = max(range(len(record.probabilities)), key=record.probabilities.__getitem__)
    return record.classes[index]


def apply_binary_policy(
    probabilities: Sequence[float],
    classes: Sequence[BinaryParcelClass],
    thresholds: BinaryThresholds,
) -> BinaryDecisionCode:
    """Apply the fail-safe binary policy shared by offline evaluation and v0 runtime tests."""
    try:
        record = BinaryPredictionRecord(
            image_id="policy-input",
            split=DatasetSplit.VAL,
            true_class=BinaryParcelClass.NORMAL,
            classes=tuple(classes),
            probabilities=tuple(float(value) for value in probabilities),
        )
    except (TypeError, ValueError):
        return BinaryDecisionCode.AI_ERROR
    top_class = _argmax(record)
    confidence = max(record.probabilities)
    if top_class is BinaryParcelClass.NORMAL and confidence >= thresholds.tau_normal_pass:
        return BinaryDecisionCode.PASS
    if top_class is BinaryParcelClass.DAMAGE and confidence >= thresholds.tau_damage_confirm:
        return BinaryDecisionCode.DAMAGE
    return BinaryDecisionCode.AI_UNCERTAIN


def evaluate_binary_predictions(
    records: Sequence[BinaryPredictionRecord],
    thresholds: BinaryThresholds,
    *,
    required_split: DatasetSplit | None = None,
) -> BinaryEvaluationReport:
    """Evaluate exactly one split without mutating or re-running model predictions."""
    if not records:
        raise ValueError("at least one prediction record is required")
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise ValueError("evaluation records must come from exactly one split")
    split = next(iter(splits))
    if required_split is not None and split is not required_split:
        raise ValueError(f"expected {required_split.value} predictions, got {split.value}")
    class_order = records[0].classes
    if any(record.classes != class_order for record in records):
        raise ValueError("class order changed inside the prediction table")

    confusion = {
        actual.value: {predicted.value: 0 for predicted in class_order} for actual in class_order
    }
    decisions: list[BinaryDecisionCode] = []
    correct = 0
    for record in records:
        predicted = _argmax(record)
        confusion[record.true_class.value][predicted.value] += 1
        correct += int(predicted is record.true_class)
        decisions.append(apply_binary_policy(record.probabilities, record.classes, thresholds))

    per_class: dict[str, ClassMetric] = {}
    for parcel_class in class_order:
        true_positive = confusion[parcel_class.value][parcel_class.value]
        predicted_total = sum(confusion[actual.value][parcel_class.value] for actual in class_order)
        support = sum(confusion[parcel_class.value].values())
        precision = _ratio(true_positive, predicted_total)
        recall = _ratio(true_positive, support)
        per_class[parcel_class.value] = ClassMetric(
            precision=precision,
            recall=recall,
            f1=_ratio(2 * precision * recall, precision + recall),
            support=support,
        )

    damage_indices = [
        index
        for index, record in enumerate(records)
        if record.true_class is BinaryParcelClass.DAMAGE
    ]
    normal_indices = [
        index
        for index, record in enumerate(records)
        if record.true_class is BinaryParcelClass.NORMAL
    ]
    safely_rejected = sum(
        decisions[index] is not BinaryDecisionCode.PASS for index in damage_indices
    )
    confirmed_damage = sum(
        decisions[index] is BinaryDecisionCode.DAMAGE for index in damage_indices
    )
    false_rejects = sum(decisions[index] is not BinaryDecisionCode.PASS for index in normal_indices)
    covered = sum(
        decision in {BinaryDecisionCode.PASS, BinaryDecisionCode.DAMAGE} for decision in decisions
    )
    uncertain = sum(decision is BinaryDecisionCode.AI_UNCERTAIN for decision in decisions)
    metrics = tuple(per_class[item.value] for item in class_order)
    damage_metric = per_class[BinaryParcelClass.DAMAGE.value]
    total = len(records)
    return BinaryEvaluationReport(
        split=split,
        record_count=total,
        accuracy=_ratio(correct, total),
        macro_precision=sum(item.precision for item in metrics) / len(metrics),
        macro_recall=sum(item.recall for item in metrics) / len(metrics),
        macro_f1=sum(item.f1 for item in metrics) / len(metrics),
        damage_class_precision=damage_metric.precision,
        damage_class_recall=damage_metric.recall,
        damage_class_f1=damage_metric.f1,
        damage_safety_recall=_ratio(safely_rejected, len(damage_indices)),
        confirmed_damage_recall=_ratio(confirmed_damage, len(damage_indices)),
        false_reject_rate=_ratio(false_rejects, len(normal_indices)),
        coverage=_ratio(covered, total),
        uncertain_rate=_ratio(uncertain, total),
        confusion_matrix=confusion,
        per_class=per_class,
    )


def select_binary_validation_thresholds(
    records: Sequence[BinaryPredictionRecord],
    threshold_grid: Iterable[float],
    *,
    min_damage_safety_recall: float = 0.95,
    max_false_reject_rate: float = 0.10,
    max_combinations: int = 10_000,
) -> BinaryThresholdSelection:
    """Choose thresholds on validation only and retain every evaluated point."""
    if not records or any(record.split is not DatasetSplit.VAL for record in records):
        raise ValueError("threshold selection accepts validation predictions only")
    if not 0 <= min_damage_safety_recall <= 1 or not 0 <= max_false_reject_rate <= 1:
        raise ValueError("selection constraints must be in [0, 1]")
    grid = tuple(sorted(set(float(value) for value in threshold_grid)))
    if not grid or any(value < 0 or value > 1 for value in grid):
        raise ValueError("threshold_grid must contain values in [0, 1]")
    if len(grid) ** 2 > max_combinations:
        raise ValueError(
            f"threshold grid produces {len(grid) ** 2} combinations; limit is {max_combinations}"
        )

    points: list[BinaryThresholdSweepPoint] = []
    for normal_threshold, damage_threshold in product(grid, repeat=2):
        thresholds = BinaryThresholds(
            tau_normal_pass=normal_threshold,
            tau_damage_confirm=damage_threshold,
        )
        report = evaluate_binary_predictions(
            records,
            thresholds,
            required_split=DatasetSplit.VAL,
        )
        points.append(
            BinaryThresholdSweepPoint(
                thresholds=thresholds,
                constraint_met=(
                    report.damage_safety_recall >= min_damage_safety_recall
                    and report.false_reject_rate <= max_false_reject_rate
                ),
                damage_safety_recall=report.damage_safety_recall,
                confirmed_damage_recall=report.confirmed_damage_recall,
                false_reject_rate=report.false_reject_rate,
                coverage=report.coverage,
                uncertain_rate=report.uncertain_rate,
                macro_f1=report.macro_f1,
                accuracy=report.accuracy,
            )
        )

    def rank(
        point: BinaryThresholdSweepPoint,
    ) -> tuple[bool, float, float, float, float, float, float]:
        return (
            point.constraint_met,
            point.damage_safety_recall,
            -point.false_reject_rate,
            point.confirmed_damage_recall,
            point.coverage,
            point.macro_f1,
            point.accuracy,
        )

    return BinaryThresholdSelection(best=max(points, key=rank), evaluated_points=tuple(points))


def select_binary_model_candidate(
    results: Sequence[BinaryCandidateValidationResult],
) -> BinaryCandidateValidationResult:
    """Freeze the winning candidate using validation evidence only."""
    if not results:
        raise ValueError("at least one candidate validation result is required")
    if len({result.candidate for result in results}) != len(results):
        raise ValueError("candidate validation results must be unique")

    def rank(
        result: BinaryCandidateValidationResult,
    ) -> tuple[bool, float, float, float, float, float]:
        report = result.report
        return (
            result.constraint_met,
            report.damage_safety_recall,
            -report.false_reject_rate,
            report.confirmed_damage_recall,
            report.macro_f1,
            report.accuracy,
        )

    return max(results, key=rank)


def save_binary_prediction_table(path: Path, records: Sequence[BinaryPredictionRecord]) -> str:
    """Atomically persist immutable binary predictions with a sibling checksum."""
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
    path.with_suffix(f"{path.suffix}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def load_binary_prediction_table(
    path: Path, *, expected_sha256: str
) -> tuple[BinaryPredictionRecord, ...]:
    """Verify and reload a binary prediction table for reproducible metric computation."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"prediction table SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    records = tuple(
        BinaryPredictionRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError("prediction table must not be empty")
    return records
