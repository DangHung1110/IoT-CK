"""Tests for reproducible model configuration contracts."""

import pytest
from pydantic import ValidationError

from parcel_sorter.data import BinaryParcelClass, DamageClass, ParcelClass
from parcel_sorter.ml import (
    DecisionThresholds,
    ExperimentConfig,
    ModelCandidate,
    PreprocessContract,
)


def thresholds(value: float = 0.7) -> DecisionThresholds:
    return DecisionThresholds(
        tau_ok_pass=value,
        tau_damage_by_class={damage_class: value for damage_class in DamageClass},
    )


def preprocess() -> PreprocessContract:
    return PreprocessContract(
        preprocess_id="roi-square-rgb320-imagenet-v1",
        resize="resize_square",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )


def test_thresholds_require_every_damage_class() -> None:
    with pytest.raises(ValidationError, match="damage thresholds mismatch"):
        DecisionThresholds(
            tau_ok_pass=0.7,
            tau_damage_by_class={DamageClass.RACH: 0.7},
        )


def test_preprocess_rejects_zero_standard_deviation() -> None:
    with pytest.raises(ValidationError, match="std values must be positive"):
        PreprocessContract(
            preprocess_id="invalid-preprocess",
            resize="resize_square",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 0.0, 1.0),
        )


def test_experiment_accepts_explicit_complete_class_order() -> None:
    config = ExperimentConfig(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        dataset_version="v0",
        dataset_manifest_sha256="a" * 64,
        dataset_archive_sha256="b" * 64,
        split_seed=42,
        seed=42,
        classes=tuple(reversed(tuple(ParcelClass))),
        preprocess=preprocess(),
        pretrained_checkpoint="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    )

    assert config.classes[0] is ParcelClass.UOT
    assert config.augmentation.rotation_degrees == 8.0
    assert config.head_epochs == 5
    assert config.finetune_epochs == 45
    assert config.patience == 7


def test_experiment_rejects_incomplete_class_order() -> None:
    with pytest.raises(ValidationError, match="five-class or every binary label"):
        ExperimentConfig(
            candidate=ModelCandidate.YOLOV8N_CLS,
            dataset_version="v0",
            dataset_manifest_sha256="a" * 64,
            dataset_archive_sha256="b" * 64,
            split_seed=42,
            seed=42,
            classes=(ParcelClass.OK,),
            preprocess=preprocess(),
            pretrained_checkpoint="yolov8n-cls.pt",
        )


def test_experiment_accepts_binary_v0_contract_and_identifies_pass_class() -> None:
    config = ExperimentConfig(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        dataset_version="v0",
        dataset_manifest_sha256="a" * 64,
        dataset_archive_sha256="b" * 64,
        split_seed=42,
        seed=42,
        classes=(BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL),
        preprocess=preprocess(),
        pretrained_checkpoint="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    )

    assert config.pass_class is BinaryParcelClass.NORMAL
