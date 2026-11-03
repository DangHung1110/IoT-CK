"""Tests for the single fail-safe model decision policy."""

import math

from parcel_sorter.data import DamageClass, ParcelClass
from parcel_sorter.ml import (
    DecisionCode,
    DecisionThresholds,
    apply_decision_policy,
    is_confirmed_damage,
)

CLASSES = tuple(ParcelClass)
THRESHOLDS = DecisionThresholds(
    tau_ok_pass=0.8,
    tau_damage_by_class={damage_class: 0.7 for damage_class in DamageClass},
)


def test_policy_passes_confident_ok() -> None:
    result = apply_decision_policy((0.82, 0.05, 0.04, 0.04, 0.05), CLASSES, THRESHOLDS)

    assert result.code is DecisionCode.PASS
    assert result.top_class is ParcelClass.OK


def test_policy_confirms_damage_using_per_class_threshold() -> None:
    result = apply_decision_policy((0.05, 0.75, 0.05, 0.10, 0.05), CLASSES, THRESHOLDS)

    assert result.code is DecisionCode.DAMAGE_RACH
    assert is_confirmed_damage(result.code)


def test_policy_rejects_low_confidence_as_uncertain() -> None:
    result = apply_decision_policy((0.3, 0.25, 0.2, 0.15, 0.1), CLASSES, THRESHOLDS)

    assert result.code is DecisionCode.AI_UNCERTAIN
    assert result.top_class is ParcelClass.OK


def test_policy_returns_error_for_invalid_outputs() -> None:
    cases = (
        ((0.5, 0.5), (ParcelClass.OK, ParcelClass.RACH)),
        ((math.nan, 0.25, 0.25, 0.25, 0.25), CLASSES),
        ((0.4, 0.1, 0.1, 0.1, 0.1), CLASSES),
        ((0.2, 0.2, 0.2, 0.2, 0.2), ("ok", "rach", "mop", "vo", "unknown")),
    )

    for probabilities, classes in cases:
        assert (
            apply_decision_policy(probabilities, classes, THRESHOLDS).code is DecisionCode.AI_ERROR
        )
