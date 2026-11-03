"""Auditable metrics and validation-only threshold selection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from parcel_sorter.data import DamageClass, DatasetSplit, ParcelClass
from parcel_sorter.ml.contracts import DecisionThresholds
from parcel_sorter.ml.policy import (
    DecisionCode,
    apply_decision_policy,
    is_confirmed_damage,
)


class PredictionRecord(BaseModel):
    """Immutable model output table row used for every offline metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str = Field(min_length=1)
    split: DatasetSplit
    true_class: ParcelClass
    classes: tuple[ParcelClass, ...]
    probabilities: tuple[float, ...]

    @model_validator(mode="after")
    def validate_shapes(self) -> Self:
        """Keep class order explicit and probability vectors aligned."""
        if len(self.classes) != len(ParcelClass) or set(self.classes) != set(ParcelClass):
            raise ValueError("classes must contain every parcel class exactly once")
        if len(self.probabilities) != len(self.classes):
            raise ValueError("probabilities must align with classes")
        return self


class ClassMetric(BaseModel):
    """Per-class metric derived from immutable argmax predictions."""

    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    f1: float
    support: int


class EvaluationReport(BaseModel):
    """Model and policy metrics for one dataset split."""

    model_config = ConfigDict(frozen=True)

    split: DatasetSplit
    record_count: int
    accuracy: float
    damage_recall: float
    confirmed_damage_recall: float
    false_reject_rate: float
    coverage: float
    uncertain_rate: float
    error_rate: float
    confusion_matrix: dict[str, dict[str, int]]
    per_class: dict[str, ClassMetric]


class ThresholdSweepPoint(BaseModel):
    """One auditable point produced by validation threshold search."""

    model_config = ConfigDict(frozen=True)

    thresholds: DecisionThresholds
    constraint_met: bool
    damage_recall: float
    confirmed_damage_recall: float
    false_reject_rate: float
    coverage: float
    accuracy: float


class ThresholdSelection(BaseModel):
    """Best threshold point and all evaluated points for audit."""

    model_config = ConfigDict(frozen=True)

    best: ThresholdSweepPoint
    evaluated_points: tuple[ThresholdSweepPoint, ...]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _argmax_class(record: PredictionRecord) -> ParcelClass:
    index = max(range(len(record.probabilities)), key=record.probabilities.__getitem__)
    return record.classes[index]


def evaluate_predictions(
    records: Sequence[PredictionRecord],
    thresholds: DecisionThresholds,
    *,
    required_split: DatasetSplit | None = None,
) -> EvaluationReport:
    """Evaluate immutable predictions with the exact runtime decision policy."""
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
        true_class.value: {predicted.value: 0 for predicted in class_order}
        for true_class in class_order
    }
    decisions = []
    correct = 0
    for record in records:
        predicted = _argmax_class(record)
        confusion[record.true_class.value][predicted.value] += 1
        correct += int(predicted is record.true_class)
        decisions.append(apply_decision_policy(record.probabilities, record.classes, thresholds))

    damage_indices = [
        index for index, record in enumerate(records) if record.true_class is not ParcelClass.OK
    ]
    ok_indices = [
        index for index, record in enumerate(records) if record.true_class is ParcelClass.OK
    ]
    safely_rejected_damage = sum(
        decisions[index].code not in {DecisionCode.PASS, DecisionCode.AI_ERROR}
        for index in damage_indices
    )
    confirmed_damage = sum(is_confirmed_damage(decisions[index].code) for index in damage_indices)
    false_rejects = sum(decisions[index].code is not DecisionCode.PASS for index in ok_indices)
    covered = sum(
        decision.code is DecisionCode.PASS or is_confirmed_damage(decision.code)
        for decision in decisions
    )
    uncertain = sum(decision.code is DecisionCode.AI_UNCERTAIN for decision in decisions)
    errors = sum(decision.code is DecisionCode.AI_ERROR for decision in decisions)

    per_class: dict[str, ClassMetric] = {}
    for parcel_class in class_order:
        true_positive = confusion[parcel_class.value][parcel_class.value]
        predicted_total = sum(
            confusion[true_class.value][parcel_class.value] for true_class in class_order
        )
        support = sum(confusion[parcel_class.value].values())
        precision = _ratio(true_positive, predicted_total)
        recall = _ratio(true_positive, support)
        per_class[parcel_class.value] = ClassMetric(
            precision=precision,
            recall=recall,
            f1=_ratio(2 * precision * recall, precision + recall),
            support=support,
        )

    total = len(records)
    return EvaluationReport(
        split=split,
        record_count=total,
        accuracy=_ratio(correct, total),
        damage_recall=_ratio(safely_rejected_damage, len(damage_indices)),
        confirmed_damage_recall=_ratio(confirmed_damage, len(damage_indices)),
        false_reject_rate=_ratio(false_rejects, len(ok_indices)),
        coverage=_ratio(covered, total),
        uncertain_rate=_ratio(uncertain, total),
        error_rate=_ratio(errors, total),
        confusion_matrix=confusion,
        per_class=per_class,
    )


def select_validation_thresholds(
    records: Sequence[PredictionRecord],
    threshold_grid: Iterable[float],
    *,
    min_damage_recall: float = 0.95,
    max_false_reject_rate: float = 0.10,
    max_combinations: int = 10_000,
) -> ThresholdSelection:
    """Sweep thresholds on validation only and retain every audited point."""
    if not records or any(record.split is not DatasetSplit.VAL for record in records):
        raise ValueError("threshold selection accepts validation predictions only")
    if not 0 <= min_damage_recall <= 1 or not 0 <= max_false_reject_rate <= 1:
        raise ValueError("selection constraints must be in [0, 1]")

    grid = tuple(sorted(set(float(value) for value in threshold_grid)))
    if not grid or any(value < 0 or value > 1 for value in grid):
        raise ValueError("threshold_grid must contain values in [0, 1]")
    combinations = len(grid) ** (1 + len(DamageClass))
    if combinations > max_combinations:
        raise ValueError(
            f"threshold grid produces {combinations} combinations; limit is {max_combinations}"
        )

    points: list[ThresholdSweepPoint] = []
    for values in product(grid, repeat=1 + len(DamageClass)):
        thresholds = DecisionThresholds(
            tau_ok_pass=values[0],
            tau_damage_by_class={
                damage_class: values[index]
                for index, damage_class in enumerate(DamageClass, start=1)
            },
        )
        report = evaluate_predictions(
            records,
            thresholds,
            required_split=DatasetSplit.VAL,
        )
        points.append(
            ThresholdSweepPoint(
                thresholds=thresholds,
                constraint_met=(
                    report.damage_recall >= min_damage_recall
                    and report.false_reject_rate <= max_false_reject_rate
                ),
                damage_recall=report.damage_recall,
                confirmed_damage_recall=report.confirmed_damage_recall,
                false_reject_rate=report.false_reject_rate,
                coverage=report.coverage,
                accuracy=report.accuracy,
            )
        )

    def rank(point: ThresholdSweepPoint) -> tuple[bool, float, float, float, float, float]:
        return (
            point.constraint_met,
            point.damage_recall,
            -point.false_reject_rate,
            point.confirmed_damage_recall,
            point.coverage,
            point.accuracy,
        )

    return ThresholdSelection(
        best=max(points, key=rank),
        evaluated_points=tuple(points),
    )
