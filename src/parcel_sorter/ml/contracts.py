"""Versioned configuration contracts shared by training, evaluation, and runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parcel_sorter.data import DamageClass, ParcelClass

SHA256_PATTERN = r"^[0-9a-f]{64}$"
DATASET_VERSION_PATTERN = r"^v\d+(?:\.\d+){0,2}$"


class ModelCandidate(StrEnum):
    """Candidate architectures that must be compared before model selection."""

    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    YOLOV8N_CLS = "yolov8n_cls"


class PreprocessContract(BaseModel):
    """Exact image preprocessing contract recorded in every model artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    preprocess_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,63}$")
    input_size: int = Field(default=320, ge=64, le=1024)
    roi: Literal["square"] = "square"
    resize: Literal["resize_square", "letterbox"]
    interpolation: Literal["bilinear"] = "bilinear"
    color: Literal["RGB"] = "RGB"
    scale: float = Field(default=255.0, gt=0)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    layout: Literal["NCHW"] = "NCHW"

    @field_validator("std")
    @classmethod
    def validate_nonzero_std(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        """Prevent invalid normalization that would divide by zero."""
        if any(component <= 0 for component in value):
            raise ValueError("preprocess std values must be positive")
        return value


class AugmentationContract(BaseModel):
    """Conservative baseline augmentation shared semantically by both candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    vertical_flip_probability: float = Field(default=0.0, ge=0, le=1)
    rotation_degrees: float = Field(default=8.0, ge=0, le=15)
    brightness: float = Field(default=0.1, ge=0, le=0.25)
    contrast: float = Field(default=0.1, ge=0, le=0.25)
    scale_min: float = Field(default=0.9, ge=0.8, le=1)
    scale_max: float = Field(default=1.1, ge=1, le=1.2)

    @model_validator(mode="after")
    def validate_scale_range(self) -> Self:
        """Keep the lower scale bound below the upper bound."""
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must not exceed scale_max")
        return self


class DecisionThresholds(BaseModel):
    """Thresholds used by both offline evaluation and runtime decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tau_ok_pass: float = Field(ge=0, le=1)
    tau_damage_by_class: dict[DamageClass, float]

    @model_validator(mode="after")
    def validate_all_damage_classes(self) -> Self:
        """Require one explicit threshold for every damage class."""
        expected = set(DamageClass)
        actual = set(self.tau_damage_by_class)
        if actual != expected:
            missing = sorted(item.value for item in expected - actual)
            extra = sorted(str(item) for item in actual - expected)
            raise ValueError(f"damage thresholds mismatch; missing={missing}, extra={extra}")
        return self


class ExperimentConfig(BaseModel):
    """Reproducible configuration passed to a Colab training runner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: ModelCandidate
    dataset_version: str = Field(pattern=DATASET_VERSION_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_archive_sha256: str = Field(pattern=SHA256_PATTERN)
    split_seed: int = Field(ge=0)
    seed: int = Field(ge=0)
    classes: tuple[ParcelClass, ...] = tuple(ParcelClass)
    preprocess: PreprocessContract
    augmentation: AugmentationContract = AugmentationContract()
    pretrained_checkpoint: str = Field(min_length=1)
    batch_size: int = Field(default=32, ge=1)
    head_epochs: int = Field(default=3, ge=1)
    finetune_epochs: int = Field(default=20, ge=1)
    patience: int = Field(default=5, ge=1)
    checkpoint_interval: int = Field(default=1, ge=1)
    learning_rate_head: float = Field(default=1e-3, gt=0)
    learning_rate_finetune: float = Field(default=1e-4, gt=0)
    use_amp: bool = True

    @field_validator("classes")
    @classmethod
    def validate_class_mapping(cls, value: tuple[ParcelClass, ...]) -> tuple[ParcelClass, ...]:
        """Accept an explicit order but require every parcel class exactly once."""
        if len(value) != len(ParcelClass) or set(value) != set(ParcelClass):
            raise ValueError("classes must contain every parcel class exactly once")
        return value
