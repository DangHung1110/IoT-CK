"""Single fail-safe classification policy used by runtime and evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from parcel_sorter.data import DamageClass, ParcelClass
from parcel_sorter.ml.contracts import DecisionThresholds


class DecisionCode(StrEnum):
    """Stable machine-readable outcomes for parcel routing."""

    PASS = "PASS"
    AI_UNCERTAIN = "AI_UNCERTAIN"
    AI_ERROR = "AI_ERROR"
    DAMAGE_RACH = "DAMAGE_RACH"
    DAMAGE_MOP = "DAMAGE_MOP"
    DAMAGE_VO = "DAMAGE_VO"
    DAMAGE_UOT = "DAMAGE_UOT"


_DAMAGE_DECISIONS = {
    DamageClass.RACH: DecisionCode.DAMAGE_RACH,
    DamageClass.MOP: DecisionCode.DAMAGE_MOP,
    DamageClass.VO: DecisionCode.DAMAGE_VO,
    DamageClass.UOT: DecisionCode.DAMAGE_UOT,
}


class DecisionResult(BaseModel):
    """Result returned without raising for malformed model output."""

    model_config = ConfigDict(frozen=True)

    code: DecisionCode
    top_class: ParcelClass | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None


def _error(reason: str) -> DecisionResult:
    return DecisionResult(code=DecisionCode.AI_ERROR, reason=reason)


def apply_decision_policy(
    probabilities: Sequence[float],
    classes: Sequence[str | ParcelClass],
    thresholds: DecisionThresholds,
) -> DecisionResult:
    """Convert one probability vector into a fail-safe routing decision."""
    if len(probabilities) != len(classes) or not probabilities:
        return _error("probability and class shapes do not match")

    try:
        parsed_classes = tuple(ParcelClass(item) for item in classes)
    except ValueError:
        return _error("model output contains an unknown class")

    if len(parsed_classes) != len(ParcelClass) or set(parsed_classes) != set(ParcelClass):
        return _error("model output must contain every parcel class exactly once")

    values = tuple(float(value) for value in probabilities)
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        return _error("probabilities must be finite values in [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-3):
        return _error("probabilities must sum to one")

    top_index = max(range(len(values)), key=values.__getitem__)
    top_class = parsed_classes[top_index]
    confidence = values[top_index]

    if top_class is ParcelClass.OK:
        if confidence >= thresholds.tau_ok_pass:
            return DecisionResult(
                code=DecisionCode.PASS,
                top_class=top_class,
                confidence=confidence,
            )
    else:
        damage_class = DamageClass(top_class.value)
        if confidence >= thresholds.tau_damage_by_class[damage_class]:
            return DecisionResult(
                code=_DAMAGE_DECISIONS[damage_class],
                top_class=top_class,
                confidence=confidence,
            )

    return DecisionResult(
        code=DecisionCode.AI_UNCERTAIN,
        top_class=top_class,
        confidence=confidence,
        reason="top-class confidence is below its decision threshold",
    )


def is_confirmed_damage(code: DecisionCode) -> bool:
    """Return whether a decision confirms one of the four damage classes."""
    return code in set(_DAMAGE_DECISIONS.values())
