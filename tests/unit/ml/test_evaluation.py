"""Tests for leakage-safe metrics and threshold selection."""

import pytest

from parcel_sorter.data import DamageClass, DatasetSplit, ParcelClass
from parcel_sorter.ml import (
    DecisionThresholds,
    PredictionRecord,
    evaluate_predictions,
    select_validation_thresholds,
)

CLASSES = tuple(ParcelClass)


def prediction(
    image_id: str,
    true_class: ParcelClass,
    probabilities: tuple[float, ...],
    *,
    split: DatasetSplit = DatasetSplit.VAL,
) -> PredictionRecord:
    return PredictionRecord(
        image_id=image_id,
        split=split,
        true_class=true_class,
        classes=CLASSES,
        probabilities=probabilities,
    )


def thresholds(ok: float = 0.7, damage: float = 0.7) -> DecisionThresholds:
    return DecisionThresholds(
        tau_ok_pass=ok,
        tau_damage_by_class={damage_class: damage for damage_class in DamageClass},
    )


def validation_records() -> tuple[PredictionRecord, ...]:
    return (
        prediction("ok-pass", ParcelClass.OK, (0.8, 0.05, 0.05, 0.05, 0.05)),
        prediction("ok-reject", ParcelClass.OK, (0.6, 0.1, 0.1, 0.1, 0.1)),
        prediction("rach", ParcelClass.RACH, (0.05, 0.8, 0.05, 0.05, 0.05)),
        prediction("mop", ParcelClass.MOP, (0.05, 0.1, 0.7, 0.1, 0.05)),
        prediction("vo-safe", ParcelClass.VO, (0.55, 0.1, 0.1, 0.15, 0.1)),
        prediction("uot", ParcelClass.UOT, (0.05, 0.05, 0.05, 0.05, 0.8)),
    )


def test_evaluation_uses_policy_for_safety_metrics() -> None:
    report = evaluate_predictions(
        validation_records(),
        thresholds(),
        required_split=DatasetSplit.VAL,
    )

    assert report.record_count == 6
    assert report.damage_recall == 1.0
    assert report.confirmed_damage_recall == 0.75
    assert report.false_reject_rate == 0.5
    assert report.coverage == pytest.approx(4 / 6)
    assert report.uncertain_rate == pytest.approx(2 / 6)
    assert report.confusion_matrix["vo"]["ok"] == 1
    assert report.per_class["rach"].recall == 1.0


def test_evaluation_rejects_mixed_splits_and_class_order_changes() -> None:
    records = list(validation_records())
    records[-1] = records[-1].model_copy(update={"split": DatasetSplit.TEST})
    with pytest.raises(ValueError, match="exactly one split"):
        evaluate_predictions(records, thresholds())

    records = list(validation_records())
    records[-1] = records[-1].model_copy(update={"classes": tuple(reversed(CLASSES))})
    with pytest.raises(ValueError, match="class order changed"):
        evaluate_predictions(records, thresholds())


def test_threshold_selection_refuses_test_predictions() -> None:
    test_record = prediction(
        "held-out",
        ParcelClass.OK,
        (0.9, 0.025, 0.025, 0.025, 0.025),
        split=DatasetSplit.TEST,
    )

    with pytest.raises(ValueError, match="validation predictions only"):
        select_validation_thresholds((test_record,), (0.6, 0.8))


def test_threshold_selection_is_auditable_and_bounded() -> None:
    selection = select_validation_thresholds(
        validation_records(),
        (0.6, 0.8),
        min_damage_recall=1.0,
        max_false_reject_rate=0.5,
    )

    assert len(selection.evaluated_points) == 32
    assert selection.best.constraint_met
    assert selection.best.damage_recall == 1.0

    with pytest.raises(ValueError, match="limit is 10"):
        select_validation_thresholds(validation_records(), (0.5, 0.6), max_combinations=10)
