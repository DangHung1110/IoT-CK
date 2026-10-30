"""Versioned dataset manifest contract and leakage checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonScalar = str | int | float | bool | None
SHA256_PATTERN = r"^[0-9a-f]{64}$"
DATASET_VERSION_PATTERN = r"^v\d+(?:\.\d+){0,2}$"


class ParcelClass(StrEnum):
    """Single-label parcel condition used by the classifier."""

    OK = "ok"
    RACH = "rach"
    MOP = "mop"
    VO = "vo"
    UOT = "uot"


class DamageClass(StrEnum):
    """Damage types allowed in the multi-value observation field."""

    RACH = "rach"
    MOP = "mop"
    VO = "vo"
    UOT = "uot"


class DatasetSplit(StrEnum):
    """Dataset partitions with a held-out test set."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DataSource(StrEnum):
    """Provenance category for an image."""

    CAPTURED = "captured"
    EXTERNAL = "external"
    FEEDBACK = "feedback"


class ManifestRecord(BaseModel):
    """One immutable image entry in a versioned dataset manifest."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    image_id: str = Field(min_length=1)
    raw_path: str = Field(min_length=1)
    derived_path: str = Field(min_length=1)
    sha256_raw: str = Field(pattern=SHA256_PATTERN)
    sha256_derived: str = Field(pattern=SHA256_PATTERN)
    box_id: str = Field(min_length=1)
    primary_class: ParcelClass
    observed_defects: tuple[DamageClass, ...]
    source: DataSource
    source_detail: str = Field(min_length=1)
    license: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    captured_at_utc: datetime
    capture_params: dict[str, JsonScalar]
    labeler: str = Field(min_length=1)
    reviewed_by: str | None = None
    split: DatasetSplit
    dataset_version: str = Field(pattern=DATASET_VERSION_PATTERN)

    @field_validator("raw_path", "derived_path")
    @classmethod
    def validate_drive_relative_path(cls, value: str) -> str:
        """Require portable paths relative to the dataset root on Drive."""
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or any(":" in part for part in path.parts)
            or path.as_posix() != value
        ):
            raise ValueError("path must be a canonical POSIX path relative to the dataset root")
        return value

    @field_validator("captured_at_utc")
    @classmethod
    def validate_utc_timestamp(cls, value: datetime) -> datetime:
        """Reject ambiguous or non-UTC capture timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at_utc must be timezone-aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("captured_at_utc must use UTC offset +00:00")
        return value

    @field_validator("observed_defects")
    @classmethod
    def validate_unique_defects(cls, value: tuple[DamageClass, ...]) -> tuple[DamageClass, ...]:
        """Prevent duplicate damage tags from skewing reports."""
        if len(value) != len(set(value)):
            raise ValueError("observed_defects must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_label_semantics(self) -> Self:
        """Keep the primary label consistent with all observed damage tags."""
        if self.primary_class is ParcelClass.OK and self.observed_defects:
            raise ValueError("an ok image cannot contain observed defects")

        if self.primary_class is not ParcelClass.OK:
            primary_damage = DamageClass(self.primary_class.value)
            if primary_damage not in self.observed_defects:
                raise ValueError("primary damage class must appear in observed_defects")

        return self


class ValidationIssue(BaseModel):
    """Machine-readable manifest validation finding."""

    code: str
    message: str
    image_ids: tuple[str, ...] = ()


class ManifestValidationReport(BaseModel):
    """Aggregate validation result suitable for JSON reports."""

    dataset_versions: tuple[str, ...]
    record_count: int
    box_count: int
    class_counts_by_split: dict[str, dict[str, int]]
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether the manifest passed every hard rule."""
        return not self.errors


class ManifestValidationError(ValueError):
    """Raised when a caller requires a valid manifest but receives errors."""

    def __init__(self, report: ManifestValidationReport) -> None:
        self.report = report
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.errors)
        super().__init__(f"manifest validation failed: {details}")


def _class_counts(records: list[ManifestRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in DatasetSplit:
        split_counts = Counter(record.primary_class for record in records if record.split is split)
        counts[split.value] = {
            parcel_class.value: split_counts[parcel_class] for parcel_class in ParcelClass
        }
    return counts


def _cross_split_issues(
    records: list[ManifestRecord],
    *,
    values_for_record: Callable[[ManifestRecord], tuple[str, ...]],
    code: str,
    label: str,
) -> list[ValidationIssue]:
    splits_by_value: dict[str, set[DatasetSplit]] = defaultdict(set)
    images_by_value: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for value in values_for_record(record):
            splits_by_value[value].add(record.split)
            images_by_value[value].add(record.image_id)

    issues: list[ValidationIssue] = []
    for value, splits in sorted(splits_by_value.items()):
        if len(splits) > 1:
            split_names = ", ".join(sorted(split.value for split in splits))
            issues.append(
                ValidationIssue(
                    code=code,
                    message=f"{label} {value!r} appears in multiple splits: {split_names}",
                    image_ids=tuple(sorted(images_by_value[value])),
                )
            )
    return issues


def _duplicate_path_issues(records: list[ManifestRecord]) -> list[ValidationIssue]:
    images_by_path: dict[str, set[str]] = defaultdict(set)
    for record in records:
        images_by_path[record.raw_path].add(record.image_id)
        images_by_path[record.derived_path].add(record.image_id)

    return [
        ValidationIssue(
            code="duplicate_path",
            message=f"path {path!r} is assigned to multiple image IDs",
            image_ids=tuple(sorted(image_ids)),
        )
        for path, image_ids in sorted(images_by_path.items())
        if len(image_ids) > 1
    ]


def validate_manifest(
    records: list[ManifestRecord], *, require_all_classes: bool = True
) -> ManifestValidationReport:
    """Validate provenance, group isolation, and held-out test invariants."""
    errors: list[ValidationIssue] = []

    image_id_counts = Counter(record.image_id for record in records)
    duplicate_ids = tuple(
        sorted(image_id for image_id, count in image_id_counts.items() if count > 1)
    )
    if duplicate_ids:
        errors.append(
            ValidationIssue(
                code="duplicate_image_id",
                message="image_id values must be unique",
                image_ids=duplicate_ids,
            )
        )

    versions = tuple(sorted({record.dataset_version for record in records}))
    if len(versions) != 1:
        errors.append(
            ValidationIssue(
                code="mixed_dataset_versions",
                message="a manifest must contain exactly one dataset version",
            )
        )

    errors.extend(
        _cross_split_issues(
            records,
            values_for_record=lambda record: (record.box_id,),
            code="box_split_leakage",
            label="box_id",
        )
    )
    errors.extend(
        _cross_split_issues(
            records,
            values_for_record=lambda record: (record.sha256_raw, record.sha256_derived),
            code="hash_split_leakage",
            label="SHA-256",
        )
    )
    errors.extend(
        _cross_split_issues(
            records,
            values_for_record=lambda record: (record.session_id,),
            code="session_split_leakage",
            label="session_id",
        )
    )
    errors.extend(_duplicate_path_issues(records))

    for record in records:
        if record.source is not DataSource.CAPTURED and record.split is not DatasetSplit.TRAIN:
            errors.append(
                ValidationIssue(
                    code="non_captured_outside_train",
                    message=(
                        f"{record.source.value} data is only allowed in train, "
                        f"not {record.split.value}"
                    ),
                    image_ids=(record.image_id,),
                )
            )

    if require_all_classes:
        expected = set(ParcelClass)
        for split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
            present = {record.primary_class for record in records if record.split is split}
            missing = sorted(parcel_class.value for parcel_class in expected - present)
            if missing:
                errors.append(
                    ValidationIssue(
                        code="missing_class",
                        message=f"{split.value} is missing classes: {', '.join(missing)}",
                    )
                )

    return ManifestValidationReport(
        dataset_versions=versions,
        record_count=len(records),
        box_count=len({record.box_id for record in records}),
        class_counts_by_split=_class_counts(records),
        errors=tuple(
            sorted(errors, key=lambda issue: (issue.code, issue.message, issue.image_ids))
        ),
    )


def assert_manifest_valid(
    records: list[ManifestRecord], *, require_all_classes: bool = True
) -> ManifestValidationReport:
    """Return a valid report or raise with the complete structured report."""
    report = validate_manifest(records, require_all_classes=require_all_classes)
    if not report.valid:
        raise ManifestValidationError(report)
    return report


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file checksum without loading the full image into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
