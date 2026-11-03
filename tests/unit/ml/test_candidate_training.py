from __future__ import annotations

from pathlib import Path

import pytest

from parcel_sorter.data import ParcelClass
from parcel_sorter.ml.colab_training import checkpoint_evidence
from parcel_sorter.ml.contracts import ModelCandidate
from parcel_sorter.ml.mobilenet_training import _classification_rates
from parcel_sorter.ml.training_runtime import TrainingRuntimeError
from parcel_sorter.ml.yolo_training import (
    ULTRALYTICS_VERSION,
    YOLO_CHECKPOINT,
    _ordered_model_names,
)


def test_classification_rates_measure_damage_detection_and_false_reject() -> None:
    accuracy, damage_recall, false_reject = _classification_rates(
        targets=[0, 0, 1, 2],
        predictions=[0, 1, 4, 0],
        ok_index=0,
    )

    assert accuracy == pytest.approx(0.25)
    assert damage_recall == pytest.approx(0.5)
    assert false_reject == pytest.approx(0.5)


def test_classification_rates_reject_empty_validation() -> None:
    with pytest.raises(TrainingRuntimeError, match="no samples"):
        _classification_rates([], [], ok_index=0)


def test_yolo_model_names_must_be_contiguous() -> None:
    assert _ordered_model_names({0: "mop", 1: "ok"}) == ("mop", "ok")
    assert _ordered_model_names(["mop", "ok"]) == ("mop", "ok")
    with pytest.raises(TrainingRuntimeError, match="contiguous"):
        _ordered_model_names({0: "mop", 2: "ok"})
    with pytest.raises(TrainingRuntimeError, match="supported"):
        _ordered_model_names("mop")


def test_checkpoint_evidence_is_explicit_for_both_candidates() -> None:
    mobile = checkpoint_evidence(ModelCandidate.MOBILENET_V3_SMALL)
    yolo = checkpoint_evidence(ModelCandidate.YOLOV8N_CLS)

    assert mobile.name == "MobileNet_V3_Small_Weights.IMAGENET1K_V1"
    assert mobile.license_name == "BSD-3-Clause"
    assert yolo.name == YOLO_CHECKPOINT
    assert yolo.license_name == "AGPL-3.0"
    assert ULTRALYTICS_VERSION == "8.4.191"
    assert next(iter(sorted(ParcelClass, key=lambda item: item.value))) is ParcelClass.MOP


def test_training_modules_do_not_import_ai_packages_at_module_load() -> None:
    import sys

    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "ultralytics" not in sys.modules
    assert not list(Path("models").glob("*.pt"))
