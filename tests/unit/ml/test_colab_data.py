from __future__ import annotations

import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from parcel_sorter.data import DatasetSplit, ParcelClass, sha256_file
from parcel_sorter.ml.colab_data import (
    DatasetIntegrityError,
    inspect_imagefolder_tree,
    safe_extract_zip,
    stage_verified_dataset,
    validate_drive_source,
    verify_file_sha256,
)

CLASS_ORDER = (
    ParcelClass.OK,
    ParcelClass.RACH,
    ParcelClass.MOP,
    ParcelClass.VO,
    ParcelClass.UOT,
)


def _write_dataset_zip(path: Path, *, root: str = "dataset_v1") -> None:
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for split in DatasetSplit:
            for parcel_class in CLASS_ORDER:
                archive.writestr(
                    f"{root}/{split.value}/{parcel_class.value}/sample.jpg",
                    f"{split.value}-{parcel_class.value}".encode(),
                )


def _write_dataset_tree(root: Path) -> None:
    for split in DatasetSplit:
        for parcel_class in CLASS_ORDER:
            class_dir = root / "dataset_v1" / split.value / parcel_class.value
            class_dir.mkdir(parents=True, exist_ok=True)
            (class_dir / "sample.JPG").write_bytes(b"image")


def test_verify_file_sha256_accepts_matching_regular_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")

    assert verify_file_sha256(
        artifact, expected=sha256_file(artifact), label="artifact"
    ) == sha256_file(artifact)


@pytest.mark.parametrize("expected", ["0" * 64, "ABC", "z" * 64])
def test_verify_file_sha256_rejects_bad_expectation(tmp_path: Path, expected: str) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"actual")

    with pytest.raises(DatasetIntegrityError):
        verify_file_sha256(artifact, expected=expected, label="artifact")


def test_verify_file_sha256_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetIntegrityError, match="not a regular file"):
        verify_file_sha256(tmp_path / "missing", expected="0" * 64, label="artifact")


def test_safe_extract_zip_extracts_regular_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    _write_dataset_zip(archive_path)

    destination = tmp_path / "output"
    safe_extract_zip(archive_path, destination=destination)

    assert (destination / "dataset_v1" / "train" / "ok" / "sample.jpg").is_file()


@pytest.mark.parametrize("member_name", ["../escape.txt", "/absolute.txt", "C:/drive.txt"])
def test_safe_extract_zip_rejects_unsafe_paths(tmp_path: Path, member_name: str) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, b"bad")

    with pytest.raises(DatasetIntegrityError, match="unsafe ZIP member"):
        safe_extract_zip(archive_path, destination=tmp_path / "output")


def test_safe_extract_zip_rejects_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.zip"
    member = ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "target")

    with pytest.raises(DatasetIntegrityError, match="unsafe ZIP member"):
        safe_extract_zip(archive_path, destination=tmp_path / "output")


def test_safe_extract_zip_enforces_limits_and_empty_destination(tmp_path: Path) -> None:
    archive_path = tmp_path / "small.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("one.txt", b"12345")
        archive.writestr("two.txt", b"1")

    with pytest.raises(DatasetIntegrityError, match="members"):
        safe_extract_zip(archive_path, destination=tmp_path / "members", max_members=1)
    with pytest.raises(DatasetIntegrityError, match="expands"):
        safe_extract_zip(archive_path, destination=tmp_path / "bytes", max_uncompressed_bytes=1)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="not empty"):
        safe_extract_zip(archive_path, destination=occupied)


def test_safe_extract_zip_rejects_invalid_archive_and_limits(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not a zip")

    with pytest.raises(DatasetIntegrityError, match="invalid ZIP"):
        safe_extract_zip(invalid, destination=tmp_path / "invalid-output")
    with pytest.raises(ValueError, match="positive"):
        safe_extract_zip(invalid, destination=tmp_path / "limits", max_members=0)


def test_inspect_imagefolder_tree_preserves_explicit_class_order(tmp_path: Path) -> None:
    _write_dataset_tree(tmp_path)

    dataset_root, splits = inspect_imagefolder_tree(tmp_path, class_order=CLASS_ORDER)

    assert dataset_root == tmp_path / "dataset_v1"
    assert tuple(splits[DatasetSplit.TRAIN].image_counts) == CLASS_ORDER
    assert all(count == 1 for count in splits[DatasetSplit.TEST].image_counts.values())


def test_inspect_imagefolder_tree_rejects_missing_split_and_bad_order(tmp_path: Path) -> None:
    _write_dataset_tree(tmp_path)
    test_path = tmp_path / "dataset_v1" / "test"
    for child in test_path.rglob("*"):
        if child.is_file():
            child.unlink()
    for child in sorted(test_path.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
    test_path.rmdir()

    with pytest.raises(DatasetIntegrityError, match="required split"):
        inspect_imagefolder_tree(tmp_path, class_order=CLASS_ORDER)
    with pytest.raises(DatasetIntegrityError, match="class_order"):
        inspect_imagefolder_tree(tmp_path, class_order=CLASS_ORDER[:-1])


def test_inspect_imagefolder_tree_rejects_ambiguous_and_invalid_trees(tmp_path: Path) -> None:
    _write_dataset_tree(tmp_path / "first")
    _write_dataset_tree(tmp_path / "second")
    with pytest.raises(DatasetIntegrityError, match="exactly one"):
        inspect_imagefolder_tree(tmp_path, class_order=CLASS_ORDER)

    isolated = tmp_path / "isolated"
    _write_dataset_tree(isolated)
    train = isolated / "dataset_v1" / "train"
    (train / "unexpected").mkdir()
    (train / "loose.txt").write_text("bad", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="invalid split tree"):
        inspect_imagefolder_tree(isolated, class_order=CLASS_ORDER)


def test_inspect_imagefolder_tree_rejects_empty_or_unsupported_class(tmp_path: Path) -> None:
    _write_dataset_tree(tmp_path)
    image = tmp_path / "dataset_v1" / "train" / "ok" / "sample.JPG"
    image.unlink()
    with pytest.raises(DatasetIntegrityError, match="no images"):
        inspect_imagefolder_tree(tmp_path, class_order=CLASS_ORDER)

    image.write_bytes(b"image")
    (image.parent / "notes.txt").write_text("unsupported", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="unsupported files"):
        inspect_imagefolder_tree(tmp_path, class_order=CLASS_ORDER)


def test_stage_verified_dataset_copies_and_validates_drive_artifacts(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    archive = drive_root / "data" / "archives" / "dataset_v1_splits.zip"
    manifest = drive_root / "data" / "manifests" / "dataset_v1.csv"
    archive.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    _write_dataset_zip(archive)
    manifest.write_text("manifest", encoding="utf-8")

    prepared = stage_verified_dataset(
        archive_source=archive,
        manifest_source=manifest,
        expected_archive_sha256=sha256_file(archive),
        expected_manifest_sha256=sha256_file(manifest),
        drive_root=drive_root,
        local_workspace=tmp_path / "content" / "dataset-workspace",
        class_order=CLASS_ORDER,
    )

    assert prepared.dataset_root.name == "dataset_v1"
    assert prepared.class_order == CLASS_ORDER
    assert set(prepared.splits) == set(DatasetSplit)


def test_stage_verified_dataset_rejects_existing_workspace_and_outside_source(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside")
    with pytest.raises(DatasetIntegrityError, match="below Drive root"):
        validate_drive_source(outside, drive_root=drive_root)

    archive = drive_root / "dataset.zip"
    manifest = drive_root / "manifest.csv"
    _write_dataset_zip(archive)
    manifest.write_text("manifest", encoding="utf-8")
    workspace = tmp_path / "existing"
    workspace.mkdir()
    with pytest.raises(DatasetIntegrityError, match="already exists"):
        stage_verified_dataset(
            archive_source=archive,
            manifest_source=manifest,
            expected_archive_sha256=sha256_file(archive),
            expected_manifest_sha256=sha256_file(manifest),
            drive_root=drive_root,
            local_workspace=workspace,
            class_order=CLASS_ORDER,
        )
