"""Verified dataset staging from Google Drive into a Colab-local workspace."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, Field

from parcel_sorter.data import (
    BinaryParcelClass,
    DatasetSplit,
    ModelClass,
    ParcelClass,
    sha256_file,
)

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
MAX_ARCHIVE_MEMBERS = 100_000
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3


class DatasetIntegrityError(ValueError):
    """Raised when an archive or extracted ImageFolder tree is unsafe or invalid."""


class DatasetSplitLayout(BaseModel):
    """Auditable image counts for one extracted dataset split."""

    model_config = ConfigDict(frozen=True)

    path: Path
    image_counts: dict[ModelClass, int]


class PreparedDataset(BaseModel):
    """Identity and local layout of a dataset verified before training."""

    model_config = ConfigDict(frozen=True)

    archive_source: Path
    manifest_source: Path
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_root: Path
    class_order: tuple[ModelClass, ...]
    splits: dict[DatasetSplit, DatasetSplitLayout]


def verify_file_sha256(path: Path, *, expected: str, label: str) -> str:
    """Verify one regular file against its expected lowercase SHA-256 digest."""
    if not path.is_file():
        raise DatasetIntegrityError(f"{label} is not a regular file: {path}")
    if len(expected) != 64 or expected.lower() != expected:
        raise DatasetIntegrityError(f"expected {label} SHA-256 must be 64 lowercase hex characters")
    try:
        int(expected, 16)
    except ValueError as error:
        raise DatasetIntegrityError(f"expected {label} SHA-256 is not hexadecimal") from error

    actual = sha256_file(path)
    if actual != expected:
        raise DatasetIntegrityError(
            f"{label} SHA-256 mismatch: expected {expected}, calculated {actual}"
        )
    return actual


def _validated_member_path(member: ZipInfo) -> PurePosixPath:
    name = member.filename
    path = PurePosixPath(name)
    unix_mode = member.external_attr >> 16
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
        or stat.S_ISLNK(unix_mode)
        or member.flag_bits & 0x1
    ):
        raise DatasetIntegrityError(f"unsafe ZIP member: {name!r}")
    return path


def safe_extract_zip(
    archive_path: Path,
    *,
    destination: Path,
    max_members: int = MAX_ARCHIVE_MEMBERS,
    max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
) -> None:
    """Extract a bounded ZIP after rejecting traversal, links, and duplicate targets."""
    if max_members <= 0 or max_uncompressed_bytes <= 0:
        raise ValueError("ZIP extraction limits must be positive")
    if destination.exists() and any(destination.iterdir()):
        raise DatasetIntegrityError(f"extraction destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > max_members:
                raise DatasetIntegrityError(
                    f"archive contains {len(members)} members; limit is {max_members}"
                )
            total_size = sum(member.file_size for member in members)
            if total_size > max_uncompressed_bytes:
                raise DatasetIntegrityError(
                    f"archive expands to {total_size} bytes; limit is {max_uncompressed_bytes}"
                )

            seen: set[str] = set()
            for member in members:
                relative = _validated_member_path(member)
                normalized = relative.as_posix().rstrip("/")
                if normalized in seen:
                    raise DatasetIntegrityError(f"duplicate ZIP target: {normalized!r}")
                seen.add(normalized)

                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except BadZipFile as error:
        raise DatasetIntegrityError(f"invalid ZIP archive: {archive_path}") from error


def _find_dataset_root(extraction_root: Path) -> Path:
    candidates = {
        path.parent
        for path in extraction_root.rglob(DatasetSplit.TRAIN.value)
        if path.is_dir() and (path.parent / DatasetSplit.VAL.value).is_dir()
    }
    if len(candidates) != 1:
        formatted = ", ".join(sorted(path.as_posix() for path in candidates)) or "none"
        raise DatasetIntegrityError(
            f"archive must contain exactly one train/val dataset root; found: {formatted}"
        )
    return candidates.pop()


def _inspect_split(path: Path, *, class_order: tuple[ModelClass, ...]) -> DatasetSplitLayout:
    expected_names = {item.value for item in class_order}
    actual_dirs = {item.name for item in path.iterdir() if item.is_dir()}
    unexpected_files = sorted(item.name for item in path.iterdir() if item.is_file())
    if actual_dirs != expected_names or unexpected_files:
        missing = sorted(expected_names - actual_dirs)
        extra = sorted(actual_dirs - expected_names)
        raise DatasetIntegrityError(
            f"invalid split tree at {path}: missing={missing}, extra={extra}, "
            f"files_at_split_root={unexpected_files}"
        )

    counts: dict[ModelClass, int] = {}
    for parcel_class in class_order:
        class_dir = path / parcel_class.value
        unsupported = sorted(
            item.relative_to(class_dir).as_posix()
            for item in class_dir.rglob("*")
            if item.is_file() and item.suffix.lower() not in IMAGE_SUFFIXES
        )
        if unsupported:
            raise DatasetIntegrityError(
                f"unsupported files in {class_dir}: {', '.join(unsupported[:10])}"
            )
        count = sum(
            1
            for item in class_dir.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if count == 0:
            raise DatasetIntegrityError(f"class directory has no images: {class_dir}")
        counts[parcel_class] = count
    return DatasetSplitLayout(path=path, image_counts=counts)


def inspect_imagefolder_tree(
    extraction_root: Path,
    *,
    class_order: tuple[ModelClass, ...],
) -> tuple[Path, dict[DatasetSplit, DatasetSplitLayout]]:
    """Validate a complete binary or five-class train/val/test tree."""
    actual = set(class_order)
    if actual not in (set(ParcelClass), set(BinaryParcelClass)) or len(class_order) != len(actual):
        raise DatasetIntegrityError(
            "class_order must contain every five-class or binary label exactly once"
        )
    dataset_root = _find_dataset_root(extraction_root)
    split_layouts: dict[DatasetSplit, DatasetSplitLayout] = {}
    for split in DatasetSplit:
        split_path = dataset_root / split.value
        if not split_path.is_dir():
            raise DatasetIntegrityError(f"required split directory is missing: {split_path}")
        split_layouts[split] = _inspect_split(split_path, class_order=class_order)
    return dataset_root, split_layouts


def validate_drive_source(path: Path, *, drive_root: Path) -> None:
    """Require a dataset input to be a real file below the configured Drive root."""
    if not path.is_file():
        raise DatasetIntegrityError(f"Drive source is not a regular file: {path}")
    try:
        path.resolve().relative_to(drive_root.resolve())
    except ValueError as error:
        raise DatasetIntegrityError(
            f"source must be below Drive root {drive_root}: {path}"
        ) from error


def stage_verified_dataset(
    *,
    archive_source: Path,
    manifest_source: Path,
    expected_archive_sha256: str,
    expected_manifest_sha256: str,
    drive_root: Path,
    local_workspace: Path,
    class_order: tuple[ModelClass, ...],
) -> PreparedDataset:
    """Verify Drive inputs, copy the archive locally, and validate extracted data."""
    validate_drive_source(archive_source, drive_root=drive_root)
    validate_drive_source(manifest_source, drive_root=drive_root)
    archive_hash = verify_file_sha256(
        archive_source, expected=expected_archive_sha256, label="dataset archive"
    )
    manifest_hash = verify_file_sha256(
        manifest_source, expected=expected_manifest_sha256, label="dataset manifest"
    )
    if local_workspace.exists():
        raise DatasetIntegrityError(f"local workspace already exists: {local_workspace}")
    local_workspace.mkdir(parents=True)
    local_archive = local_workspace / "dataset.zip"
    shutil.copy2(archive_source, local_archive)
    if sha256_file(local_archive) != archive_hash:
        raise DatasetIntegrityError("local archive copy failed SHA-256 verification")

    extraction_root = local_workspace / "extracted"
    safe_extract_zip(local_archive, destination=extraction_root)
    dataset_root, splits = inspect_imagefolder_tree(extraction_root, class_order=class_order)
    return PreparedDataset(
        archive_source=archive_source,
        manifest_source=manifest_source,
        archive_sha256=archive_hash,
        manifest_sha256=manifest_hash,
        dataset_root=dataset_root,
        class_order=class_order,
        splits=splits,
    )
