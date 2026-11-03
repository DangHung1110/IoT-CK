"""Prepare an auditable Kaggle-only dataset for Colab smoke training."""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, Self
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parcel_sorter.data.manifest import DatasetSplit, ParcelClass, sha256_file

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
KAGGLE_HANDLE_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*/versions/[1-9][0-9]*$"
)
MANIFEST_FIELDS = (
    "image_id",
    "derived_path",
    "sha256_derived",
    "group_id",
    "primary_class",
    "split",
    "source",
    "source_detail",
    "license",
)


class KaggleImportError(ValueError):
    """Raised when Kaggle data cannot be prepared without ambiguity."""


class SplitRatios(BaseModel):
    """Requested group-aware split ratios."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: float = Field(default=0.70, gt=0, lt=1)
    val: float = Field(default=0.15, gt=0, lt=1)
    test: float = Field(default=0.15, gt=0, lt=1)

    @model_validator(mode="after")
    def require_unit_sum(self) -> Self:
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("train, val, and test ratios must sum to 1")
        return self


class KaggleImportConfig(BaseModel):
    """Pinned source, explicit labels, split rules, and dataset identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_handle: str
    source_url: str = Field(pattern=r"^https://www\.kaggle\.com/datasets/")
    license_name: str = Field(min_length=1)
    dataset_version: str = Field(default="v1", pattern=r"^v\d+(?:\.\d+){0,2}$")
    classes: tuple[str, ...] = tuple(parcel_class.value for parcel_class in ParcelClass)
    class_folder_map: dict[str, str]
    scope: str = "v0_kaggle_smoke_only_not_final_kpi"
    split_seed: int = 42
    ratios: SplitRatios = SplitRatios()
    grouping_strategy: Literal["perceptual_hash", "relative_path"] = "perceptual_hash"
    group_regex: str | None = None

    @field_validator("classes")
    @classmethod
    def normalize_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip().casefold() for item in value if str(item).strip())
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("classes must be non-empty and unique")
        return normalized

    @field_validator("dataset_handle")
    @classmethod
    def require_versioned_handle(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not KAGGLE_HANDLE_PATTERN.fullmatch(normalized):
            raise ValueError("dataset_handle must be owner/dataset/versions/N for reproducibility")
        return normalized

    @field_validator("class_folder_map")
    @classmethod
    def normalize_folder_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        normalized = {
            key.strip().casefold(): str(item).strip().casefold()
            for key, item in value.items()
            if key.strip()
        }
        if not normalized or len(normalized) != len(value):
            raise ValueError("class folder names must be non-empty and unique")
        if any(not item for item in normalized.values()):
            raise ValueError("class folder targets must be non-empty")
        return normalized

    @model_validator(mode="after")
    def require_complete_mapping(self) -> Self:
        if set(self.class_folder_map.values()) != set(self.classes):
            raise ValueError("class folder mapping must cover all configured classes")
        return self

    @field_validator("group_regex")
    @classmethod
    def validate_group_regex(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        try:
            pattern = re.compile(value)
        except re.error as error:
            raise ValueError(f"invalid group_regex: {error}") from error
        if pattern.groups < 1:
            raise ValueError("group_regex must contain at least one capture group")
        return value


class KaggleImageInventoryRecord(BaseModel):
    """One source image and its accept/reject decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_relative_path: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_class: str | None = None
    group_id: str | None = None
    split: DatasetSplit | None = None
    status: str
    detail: str | None = None


class KaggleImportReport(BaseModel):
    """Machine-readable evidence written beside the Drive artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    scope: str = "v0_kaggle_smoke_only_not_final_kpi"
    generated_at_utc: datetime
    config: KaggleImportConfig
    grouping_strategy: str
    image_file_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    unmapped_count: int = Field(ge=0)
    class_counts_by_split: dict[str, dict[str, int]]
    raw_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory: tuple[KaggleImageInventoryRecord, ...]


class PreparedKaggleDataset(BaseModel):
    """Local artifacts ready to be copied to Drive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: KaggleImportReport
    raw_archive: Path
    dataset_archive: Path
    manifest: Path


class PersistedKaggleDataset(BaseModel):
    """Verified immutable Drive destinations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_evidence_dir: Path
    dataset_archive: Path
    manifest: Path
    report: Path
    hashes: Path


class _AcceptedImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: Path
    source_relative_path: str
    digest: str
    target_class: str
    group_id: str
    split: DatasetSplit | None = None


def _canonical_relative_path(path: Path, *, root: Path) -> str:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    if relative.is_absolute() or ".." in relative.parts:
        raise KaggleImportError(f"unsafe source path: {path}")
    return relative.as_posix()


def _mapped_class(relative_path: str, mapping: dict[str, str]) -> str | None:
    matches = {
        mapping[part.casefold()]
        for part in PurePosixPath(relative_path).parts[:-1]
        if part.casefold() in mapping
    }
    if len(matches) > 1:
        raise KaggleImportError(f"ambiguous class folders in {relative_path!r}")
    return next(iter(matches)) if matches else None


def _average_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8)).getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if value >= average else "0" for value in pixels)
    return f"{int(bits, 2):016x}"


def _group_id(
    relative_path: str,
    path: Path,
    group_regex: str | None,
    grouping_strategy: Literal["perceptual_hash", "relative_path"],
) -> str:
    if group_regex is None:
        if grouping_strategy == "relative_path":
            return f"path:{relative_path}"
        return f"ahash:{_average_hash(path)}"
    match = re.search(group_regex, relative_path)
    if match is None:
        raise KaggleImportError(f"group_regex did not match source path {relative_path!r}")
    captured = "|".join(part or "" for part in match.groups())
    if not captured:
        raise KaggleImportError(f"group_regex captured an empty group for {relative_path!r}")
    return f"regex:{captured}"


def create_deterministic_zip(source_root: Path, *, destination: Path) -> str:
    """Create a reproducible ZIP with sorted names and fixed timestamps."""
    if destination.exists():
        raise KaggleImportError(f"archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in source_root.rglob("*") if path.is_file())
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = _canonical_relative_path(path, root=source_root)
            info = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    return sha256_file(destination)


def _seed_for_class(seed: int, parcel_class: str) -> int:
    digest = sha256(f"{seed}:{parcel_class}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _assign_group_splits(
    images: list[_AcceptedImage], *, config: KaggleImportConfig
) -> dict[str, DatasetSplit]:
    images_by_group: dict[str, list[_AcceptedImage]] = defaultdict(list)
    for image in images:
        images_by_group[image.group_id].append(image)

    classes_by_group = {
        group_id: {image.target_class for image in grouped}
        for group_id, grouped in images_by_group.items()
    }
    conflicts = sorted(
        group_id for group_id, classes in classes_by_group.items() if len(classes) > 1
    )
    if conflicts:
        raise KaggleImportError(
            "visual groups contain conflicting labels: " + ", ".join(conflicts[:10])
        )

    assignment: dict[str, DatasetSplit] = {}
    ratios = {
        DatasetSplit.TRAIN: config.ratios.train,
        DatasetSplit.VAL: config.ratios.val,
        DatasetSplit.TEST: config.ratios.test,
    }
    for parcel_class in config.classes:
        groups = [
            group_id for group_id, classes in classes_by_group.items() if classes == {parcel_class}
        ]
        if len(groups) < 3:
            raise KaggleImportError(
                f"class {parcel_class!r} needs at least 3 independent visual groups "
                "to populate train, val, and test"
            )
        rng = random.Random(_seed_for_class(config.split_seed, parcel_class))
        rng.shuffle(groups)
        split_order = [DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST]
        counts = {split: 0 for split in split_order}
        total = sum(len(images_by_group[group_id]) for group_id in groups)

        for split, group_id in zip(split_order, groups[:3], strict=True):
            assignment[group_id] = split
            counts[split] += len(images_by_group[group_id])
        for group_id in groups[3:]:
            split = max(
                split_order,
                key=lambda item: (
                    ratios[item] * total - counts[item],
                    -split_order.index(item),
                ),
            )
            assignment[group_id] = split
            counts[split] += len(images_by_group[group_id])
    return assignment


def _write_manifest(path: Path, images: list[_AcceptedImage], config: KaggleImportConfig) -> str:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for image in sorted(images, key=lambda item: item.digest):
            if image.split is None:
                raise KaggleImportError("internal error: image has no assigned split")
            suffix = image.source_path.suffix.lower()
            writer.writerow(
                {
                    "image_id": image.digest,
                    "derived_path": (
                        f"{image.split.value}/{image.target_class}/{image.digest[:20]}{suffix}"
                    ),
                    "sha256_derived": image.digest,
                    "group_id": image.group_id,
                    "primary_class": image.target_class,
                    "split": image.split.value,
                    "source": "external",
                    "source_detail": config.dataset_handle,
                    "license": config.license_name,
                }
            )
    return sha256_file(path)


def prepare_kaggle_smoke_dataset(
    *,
    source_root: Path,
    workspace: Path,
    config: KaggleImportConfig,
    generated_at: datetime | None = None,
) -> PreparedKaggleDataset:
    """Validate, deduplicate, group-split, and package one Kaggle download."""
    if not source_root.is_dir():
        raise KaggleImportError(f"downloaded dataset directory is missing: {source_root}")
    if workspace.exists():
        raise KaggleImportError(f"import workspace already exists: {workspace}")
    workspace.mkdir(parents=True)

    raw_archive = workspace / "kaggle_raw_source.zip"
    raw_hash = create_deterministic_zip(source_root, destination=raw_archive)
    inventory: list[KaggleImageInventoryRecord] = []
    accepted: list[_AcceptedImage] = []
    seen_hashes: dict[str, str] = {}
    image_paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in image_paths:
        relative = _canonical_relative_path(image_path, root=source_root)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            inventory.append(
                KaggleImageInventoryRecord(
                    source_relative_path=relative,
                    status="invalid",
                    detail=type(error).__name__,
                )
            )
            continue

        digest = sha256_file(image_path)
        if digest in seen_hashes:
            inventory.append(
                KaggleImageInventoryRecord(
                    source_relative_path=relative,
                    sha256=digest,
                    status="duplicate",
                    detail=seen_hashes[digest],
                )
            )
            continue
        seen_hashes[digest] = relative
        target_class = _mapped_class(relative, config.class_folder_map)
        if target_class is None:
            inventory.append(
                KaggleImageInventoryRecord(
                    source_relative_path=relative,
                    sha256=digest,
                    status="unmapped",
                )
            )
            continue
        accepted.append(
            _AcceptedImage(
                source_path=image_path,
                source_relative_path=relative,
                digest=digest,
                target_class=target_class,
                group_id=_group_id(
                    relative,
                    image_path,
                    config.group_regex,
                    config.grouping_strategy,
                ),
            )
        )

    assignment = _assign_group_splits(accepted, config=config)
    assigned = [
        accepted_image.model_copy(update={"split": assignment[accepted_image.group_id]})
        for accepted_image in accepted
    ]
    dataset_root = workspace / f"dataset_{config.dataset_version}"
    for accepted_image in assigned:
        if accepted_image.split is None:
            raise KaggleImportError("internal error: image has no assigned split")
        destination = (
            dataset_root
            / accepted_image.split.value
            / accepted_image.target_class
            / f"{accepted_image.digest[:20]}{accepted_image.source_path.suffix.lower()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(accepted_image.source_path, destination)
        inventory.append(
            KaggleImageInventoryRecord(
                source_relative_path=accepted_image.source_relative_path,
                sha256=accepted_image.digest,
                target_class=accepted_image.target_class,
                group_id=accepted_image.group_id,
                split=accepted_image.split,
                status="accepted",
            )
        )

    dataset_archive = workspace / f"dataset_{config.dataset_version}_splits.zip"
    dataset_hash = create_deterministic_zip(dataset_root, destination=dataset_archive)
    manifest = workspace / f"dataset_{config.dataset_version}.csv"
    manifest_hash = _write_manifest(manifest, assigned, config)
    counts: dict[str, dict[str, int]] = {}
    for split in DatasetSplit:
        split_counts = Counter(image.target_class for image in assigned if image.split is split)
        counts[split.value] = {label: split_counts[label] for label in config.classes}

    report = KaggleImportReport(
        scope=config.scope,
        generated_at_utc=generated_at or datetime.now(UTC),
        config=config,
        grouping_strategy=("regex" if config.group_regex else config.grouping_strategy),
        image_file_count=len(image_paths),
        accepted_count=len(assigned),
        duplicate_count=sum(item.status == "duplicate" for item in inventory),
        invalid_count=sum(item.status == "invalid" for item in inventory),
        unmapped_count=sum(item.status == "unmapped" for item in inventory),
        class_counts_by_split=counts,
        raw_archive_sha256=raw_hash,
        dataset_archive_sha256=dataset_hash,
        manifest_sha256=manifest_hash,
        inventory=tuple(sorted(inventory, key=lambda item: item.source_relative_path)),
    )
    return PreparedKaggleDataset(
        report=report,
        raw_archive=raw_archive,
        dataset_archive=dataset_archive,
        manifest=manifest,
    )


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise KaggleImportError(f"Drive copy failed checksum verification: {destination}")


def persist_kaggle_smoke_dataset(
    *, prepared: PreparedKaggleDataset, drive_root: Path
) -> PersistedKaggleDataset:
    """Copy immutable evidence and training inputs to their stable Drive paths."""
    config = prepared.report.config
    source_slug = config.dataset_handle.replace("/", "-")
    evidence = drive_root / "data" / "external" / "kaggle" / source_slug
    dataset_archive = (
        drive_root / "data" / "archives" / f"dataset_{config.dataset_version}_splits.zip"
    )
    manifest = drive_root / "data" / "manifests" / f"dataset_{config.dataset_version}.csv"
    targets = (evidence, dataset_archive, manifest)
    if any(path.exists() for path in targets):
        raise KaggleImportError(
            "Drive destination already exists; bump DATASET_VERSION or Kaggle version"
        )
    evidence.mkdir(parents=True)
    raw_destination = evidence / prepared.raw_archive.name
    _copy_verified(prepared.raw_archive, raw_destination, prepared.report.raw_archive_sha256)
    _copy_verified(
        prepared.dataset_archive,
        dataset_archive,
        prepared.report.dataset_archive_sha256,
    )
    _copy_verified(prepared.manifest, manifest, prepared.report.manifest_sha256)

    report_path = evidence / "import_report.json"
    report_path.write_text(prepared.report.model_dump_json(indent=2), encoding="utf-8")
    hashes_path = evidence / "final_artifact_hashes.json"
    hashes_path.write_text(
        json.dumps(
            {
                "dataset_archive": {
                    "path": dataset_archive.relative_to(drive_root).as_posix(),
                    "sha256": prepared.report.dataset_archive_sha256,
                },
                "manifest": {
                    "path": manifest.relative_to(drive_root).as_posix(),
                    "sha256": prepared.report.manifest_sha256,
                },
                "raw_source": {
                    "path": raw_destination.relative_to(drive_root).as_posix(),
                    "sha256": prepared.report.raw_archive_sha256,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PersistedKaggleDataset(
        source_evidence_dir=evidence,
        dataset_archive=dataset_archive,
        manifest=manifest,
        report=report_path,
        hashes=hashes_path,
    )
