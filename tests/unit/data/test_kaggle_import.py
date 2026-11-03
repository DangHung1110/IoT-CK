from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image
from pydantic import ValidationError

from parcel_sorter.data.kaggle_import import (
    KaggleImportConfig,
    KaggleImportError,
    SplitRatios,
    create_deterministic_zip,
    persist_kaggle_smoke_dataset,
    prepare_kaggle_smoke_dataset,
)
from parcel_sorter.data.manifest import DatasetSplit, ParcelClass, sha256_file


def _mapping() -> dict[str, ParcelClass]:
    return {item.value: item for item in ParcelClass}


def _config(**updates: object) -> KaggleImportConfig:
    values: dict[str, object] = {
        "dataset_handle": "owner/parcel-data/versions/2",
        "source_url": "https://www.kaggle.com/datasets/owner/parcel-data",
        "license_name": "CC-BY-4.0",
        "class_folder_map": _mapping(),
        "group_regex": r"/([^/]+_g\d+)_",
    }
    values.update(updates)
    return KaggleImportConfig.model_validate(values)


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (7, 5), color=color).save(path)


def _source_tree(root: Path) -> None:
    for class_index, parcel_class in enumerate(ParcelClass):
        for group_index in range(4):
            _write_image(
                root
                / "images"
                / parcel_class.value
                / f"{parcel_class.value}_g{group_index}_view.png",
                color=(class_index * 30, group_index * 40, class_index + group_index),
            )
    duplicate = root / "images" / "ok" / "ok_g9_duplicate.png"
    duplicate.write_bytes((root / "images" / "ok" / "ok_g0_view.png").read_bytes())
    invalid = root / "images" / "rach" / "rach_g9_invalid.png"
    invalid.write_bytes(b"not an image")
    _write_image(root / "images" / "unknown" / "unknown_g0_view.png", color=(1, 2, 3))


def test_config_requires_pinned_source_complete_mapping_and_valid_ratios() -> None:
    with pytest.raises(ValidationError, match="owner/dataset/versions/N"):
        _config(dataset_handle="owner/parcel-data")
    with pytest.raises(ValidationError, match="configured classes"):
        _config(class_folder_map={"ok": ParcelClass.OK})
    with pytest.raises(ValidationError, match="sum to 1"):
        SplitRatios(train=0.6, val=0.2, test=0.3)
    with pytest.raises(ValidationError, match="capture group"):
        _config(group_regex=r"no-capture")


def _binary_config() -> KaggleImportConfig:
    return KaggleImportConfig.model_validate(
        {
            "dataset_handle": "ducnguyentien69/parcel-classification/versions/1",
            "source_url": ("https://www.kaggle.com/datasets/ducnguyentien69/parcel-classification"),
            "license_name": "MIT",
            "dataset_version": "v0",
            "classes": ("damage", "normal"),
            "class_folder_map": {"normal": "normal", "damage": "damage"},
            "scope": "v0_kaggle_binary_smoke_only_not_final_kpi",
            "group_regex": r"/([^/]+_g\d+)_",
        }
    )


def test_prepare_supports_binary_kaggle_smoke_dataset(tmp_path: Path) -> None:
    source = tmp_path / "binary-download"
    for class_index, folder_name in enumerate(("normal", "damage")):
        for group_index in range(4):
            _write_image(
                source
                / "cls_dataset"
                / "train"
                / folder_name
                / f"{folder_name}_g{group_index}_view.png",
                color=(class_index * 100, group_index * 20, class_index),
            )

    prepared = prepare_kaggle_smoke_dataset(
        source_root=source,
        workspace=tmp_path / "binary-work",
        config=_binary_config(),
    )

    report = prepared.report
    assert report.scope == "v0_kaggle_binary_smoke_only_not_final_kpi"
    assert report.accepted_count == 8
    assert set(report.class_counts_by_split["train"]) == {"normal", "damage"}
    assert all(
        report.class_counts_by_split[split.value][label] > 0
        for split in DatasetSplit
        for label in ("normal", "damage")
    )
    with prepared.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["primary_class"] for row in rows} == {"normal", "damage"}
    with ZipFile(prepared.dataset_archive) as archive:
        names = archive.namelist()
    assert any(name.startswith("train/normal/") for name in names)
    assert any(name.startswith("test/damage/") for name in names)


def test_prepare_builds_deduplicated_group_isolated_three_way_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "download"
    _source_tree(source)
    prepared = prepare_kaggle_smoke_dataset(
        source_root=source,
        workspace=tmp_path / "work",
        config=_config(),
        generated_at=datetime(2026, 11, 3, tzinfo=UTC),
    )

    report = prepared.report
    assert report.scope == "v0_kaggle_smoke_only_not_final_kpi"
    assert report.accepted_count == 20
    assert report.duplicate_count == report.invalid_count == report.unmapped_count == 1
    assert all(
        report.class_counts_by_split[split.value][parcel_class.value] > 0
        for split in DatasetSplit
        for parcel_class in ParcelClass
    )
    with prepared.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert sha256_file(prepared.manifest) == report.manifest_sha256
    with ZipFile(prepared.dataset_archive) as archive:
        names = archive.namelist()
    assert all(any(name.startswith(f"{split.value}/") for name in names) for split in DatasetSplit)


def test_prepare_is_deterministic_and_rejects_bad_grouping(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_tree(source)
    first = prepare_kaggle_smoke_dataset(
        source_root=source, workspace=tmp_path / "one", config=_config()
    )
    second = prepare_kaggle_smoke_dataset(
        source_root=source, workspace=tmp_path / "two", config=_config()
    )
    assert first.report.dataset_archive_sha256 == second.report.dataset_archive_sha256
    assert first.report.manifest_sha256 == second.report.manifest_sha256

    with pytest.raises(KaggleImportError, match="did not match"):
        prepare_kaggle_smoke_dataset(
            source_root=source,
            workspace=tmp_path / "bad-regex",
            config=_config(group_regex=r"never/([^/]+)"),
        )
    with pytest.raises(KaggleImportError, match="already exists"):
        prepare_kaggle_smoke_dataset(
            source_root=source, workspace=tmp_path / "one", config=_config()
        )


def test_prepare_requires_three_groups_per_class_and_unambiguous_labels(
    tmp_path: Path,
) -> None:
    too_small = tmp_path / "small"
    for class_index, parcel_class in enumerate(ParcelClass):
        for group_index in range(2):
            _write_image(
                too_small / parcel_class.value / f"{parcel_class.value}_g{group_index}_view.png",
                color=(class_index, group_index, 0),
            )
    with pytest.raises(KaggleImportError, match="at least 3"):
        prepare_kaggle_smoke_dataset(
            source_root=too_small,
            workspace=tmp_path / "small-work",
            config=_config(),
        )

    ambiguous = tmp_path / "ambiguous"
    _write_image(ambiguous / "ok" / "rach" / "ok_g0_view.png", color=(1, 1, 1))
    with pytest.raises(KaggleImportError, match="ambiguous"):
        prepare_kaggle_smoke_dataset(
            source_root=ambiguous,
            workspace=tmp_path / "ambiguous-work",
            config=_config(),
        )


def test_persist_writes_verified_drive_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_tree(source)
    prepared = prepare_kaggle_smoke_dataset(
        source_root=source, workspace=tmp_path / "work", config=_config()
    )
    drive = tmp_path / "drive"

    persisted = persist_kaggle_smoke_dataset(prepared=prepared, drive_root=drive)

    assert persisted.dataset_archive == drive / "data/archives/dataset_v1_splits.zip"
    assert persisted.manifest == drive / "data/manifests/dataset_v1.csv"
    hashes = json.loads(persisted.hashes.read_text(encoding="utf-8"))
    assert hashes["dataset_archive"]["sha256"] == prepared.report.dataset_archive_sha256
    assert hashes["manifest"]["sha256"] == prepared.report.manifest_sha256
    with pytest.raises(KaggleImportError, match="already exists"):
        persist_kaggle_smoke_dataset(prepared=prepared, drive_root=drive)


def test_deterministic_zip_rejects_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.txt").write_text("one", encoding="utf-8")
    destination = tmp_path / "archive.zip"
    first_hash = create_deterministic_zip(source, destination=destination)
    assert first_hash == sha256_file(destination)
    with pytest.raises(KaggleImportError, match="already exists"):
        create_deterministic_zip(source, destination=destination)
