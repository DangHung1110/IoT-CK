"""Tests for the versioned dataset manifest contract."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from parcel_sorter.data.manifest import (
    DamageClass,
    DatasetSplit,
    DataSource,
    ManifestRecord,
    ManifestValidationError,
    ParcelClass,
    assert_manifest_valid,
    sha256_file,
    validate_manifest,
)


def make_record(
    index: int,
    *,
    primary_class: ParcelClass,
    split: DatasetSplit,
    box_id: str | None = None,
    session_id: str | None = None,
    source: DataSource = DataSource.CAPTURED,
    license_name: str = "project-owned",
) -> ManifestRecord:
    """Build a deterministic valid record for a test."""
    defects = () if primary_class is ParcelClass.OK else (DamageClass(primary_class.value),)
    return ManifestRecord(
        image_id=f"image-{index}",
        raw_path=f"data/raw/session-{index}/image.jpg",
        derived_path=f"data/derived/v0/image-{index}.jpg",
        sha256_raw=f"{index * 2 + 1:064x}",
        sha256_derived=f"{index * 2 + 2:064x}",
        box_id=box_id or f"box-{index}",
        primary_class=primary_class,
        observed_defects=defects,
        source=source,
        source_detail="test fixture",
        license=license_name,
        session_id=session_id or f"session-{split.value}",
        camera_id="camera-1",
        captured_at_utc=datetime(2026, 10, 30, 3, 0, tzinfo=UTC),
        capture_params={"distance_cm": 25, "flash": False},
        labeler="tester",
        reviewed_by=None,
        split=split,
        dataset_version="v0",
    )


@pytest.fixture
def valid_records() -> list[ManifestRecord]:
    """Create all classes in train/val plus an isolated test record."""
    records: list[ManifestRecord] = []
    index = 1
    for split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
        for parcel_class in ParcelClass:
            records.append(make_record(index, primary_class=parcel_class, split=split))
            index += 1
    records.append(
        make_record(
            index,
            primary_class=ParcelClass.OK,
            split=DatasetSplit.TEST,
            session_id="session-test",
        )
    )
    return records


def issue_codes(records: list[ManifestRecord]) -> set[str]:
    """Return error codes for concise assertions."""
    return {issue.code for issue in validate_manifest(records).errors}


def test_valid_manifest_passes(valid_records: list[ManifestRecord]) -> None:
    report = assert_manifest_valid(valid_records)

    assert report.valid
    assert report.record_count == 11
    assert report.box_count == 11
    assert report.class_counts_by_split["train"]["uot"] == 1


def test_box_cannot_cross_splits(valid_records: list[ManifestRecord]) -> None:
    valid_records[5] = valid_records[5].model_copy(update={"box_id": valid_records[0].box_id})

    assert "box_split_leakage" in issue_codes(valid_records)


def test_hash_cannot_cross_splits(valid_records: list[ManifestRecord]) -> None:
    valid_records[5] = valid_records[5].model_copy(
        update={"sha256_raw": valid_records[0].sha256_raw}
    )

    assert "hash_split_leakage" in issue_codes(valid_records)


def test_image_ids_must_be_unique(valid_records: list[ManifestRecord]) -> None:
    valid_records[1] = valid_records[1].model_copy(update={"image_id": valid_records[0].image_id})

    assert "duplicate_image_id" in issue_codes(valid_records)


def test_external_data_is_train_only(valid_records: list[ManifestRecord]) -> None:
    valid_records[5] = valid_records[5].model_copy(update={"source": DataSource.EXTERNAL})

    assert "non_captured_outside_train" in issue_codes(valid_records)


def test_feedback_data_is_train_only(valid_records: list[ManifestRecord]) -> None:
    valid_records[5] = valid_records[5].model_copy(update={"source": DataSource.FEEDBACK})

    assert "non_captured_outside_train" in issue_codes(valid_records)


def test_session_cannot_cross_train_and_val(valid_records: list[ManifestRecord]) -> None:
    valid_records[5] = valid_records[5].model_copy(update={"session_id": "session-train"})

    assert "session_split_leakage" in issue_codes(valid_records)


def test_test_session_is_held_out(valid_records: list[ManifestRecord]) -> None:
    valid_records[-1] = valid_records[-1].model_copy(update={"session_id": "session-train"})

    assert "session_split_leakage" in issue_codes(valid_records)


def test_train_and_val_require_every_class(valid_records: list[ManifestRecord]) -> None:
    without_val_uot = [
        record
        for record in valid_records
        if not (record.split is DatasetSplit.VAL and record.primary_class is ParcelClass.UOT)
    ]

    assert "missing_class" in issue_codes(without_val_uot)


def test_manifest_must_use_one_version(valid_records: list[ManifestRecord]) -> None:
    valid_records[0] = valid_records[0].model_copy(update={"dataset_version": "v1"})

    assert "mixed_dataset_versions" in issue_codes(valid_records)


def test_assertion_exposes_structured_report(valid_records: list[ManifestRecord]) -> None:
    valid_records[5] = valid_records[5].model_copy(update={"box_id": valid_records[0].box_id})

    with pytest.raises(ManifestValidationError) as exc_info:
        assert_manifest_valid(valid_records)

    assert not exc_info.value.report.valid
    assert exc_info.value.report.errors[0].code == "box_split_leakage"


def test_every_record_requires_nonempty_license() -> None:
    record = make_record(
        1,
        primary_class=ParcelClass.OK,
        split=DatasetSplit.TRAIN,
        source=DataSource.EXTERNAL,
    )

    with pytest.raises(ValidationError, match="at least 1 character"):
        ManifestRecord.model_validate(
            {
                **record.model_dump(),
                "license": "",
            }
        )


def test_primary_damage_must_be_observed() -> None:
    record = make_record(1, primary_class=ParcelClass.RACH, split=DatasetSplit.TRAIN)

    with pytest.raises(ValidationError, match="primary damage class"):
        ManifestRecord.model_validate({**record.model_dump(), "observed_defects": []})


def test_ok_record_cannot_have_damage_tag() -> None:
    record = make_record(1, primary_class=ParcelClass.OK, split=DatasetSplit.TRAIN)

    with pytest.raises(ValidationError, match="ok image"):
        ManifestRecord.model_validate(
            {**record.model_dump(), "observed_defects": [DamageClass.MOP]}
        )


def test_damage_tags_must_be_unique() -> None:
    record = make_record(1, primary_class=ParcelClass.MOP, split=DatasetSplit.TRAIN)

    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ManifestRecord.model_validate(
            {
                **record.model_dump(),
                "observed_defects": [DamageClass.MOP, DamageClass.MOP],
            }
        )


def test_paths_are_relative_and_portable() -> None:
    record = make_record(1, primary_class=ParcelClass.OK, split=DatasetSplit.TRAIN)

    with pytest.raises(ValidationError, match="POSIX path"):
        ManifestRecord.model_validate({**record.model_dump(), "raw_path": "../secret.jpg"})

    with pytest.raises(ValidationError, match="canonical POSIX path"):
        ManifestRecord.model_validate(
            {**record.model_dump(), "raw_path": "C:/dataset/raw/image.jpg"}
        )

    with pytest.raises(ValidationError, match="canonical POSIX path"):
        ManifestRecord.model_validate({**record.model_dump(), "raw_path": "data//raw/image.jpg"})


def test_paths_must_be_unique(valid_records: list[ManifestRecord]) -> None:
    valid_records[1] = valid_records[1].model_copy(update={"raw_path": valid_records[0].raw_path})

    assert "duplicate_path" in issue_codes(valid_records)


def test_report_order_is_independent_of_input_order(
    valid_records: list[ManifestRecord],
) -> None:
    valid_records[5] = valid_records[5].model_copy(update={"source": DataSource.EXTERNAL})
    valid_records[6] = valid_records[6].model_copy(update={"box_id": valid_records[1].box_id})

    forward = validate_manifest(valid_records)
    reverse = validate_manifest(list(reversed(valid_records)))

    assert forward.errors == reverse.errors


def test_capture_timestamp_must_be_utc() -> None:
    record = make_record(1, primary_class=ParcelClass.OK, split=DatasetSplit.TRAIN)

    with pytest.raises(ValidationError, match="UTC offset"):
        ManifestRecord.model_validate(
            {
                **record.model_dump(),
                "captured_at_utc": datetime(
                    2026, 10, 30, 10, 0, tzinfo=timezone(timedelta(hours=7))
                ),
            }
        )


def test_capture_timestamp_must_be_timezone_aware() -> None:
    record = make_record(1, primary_class=ParcelClass.OK, split=DatasetSplit.TRAIN)

    with pytest.raises(ValidationError, match="timezone-aware"):
        ManifestRecord.model_validate(
            {
                **record.model_dump(),
                "captured_at_utc": datetime(2026, 10, 30, 3, 0),
            }
        )


def test_sha256_file_streams_content(tmp_path: Path) -> None:
    fixture = tmp_path / "image.bin"
    fixture.write_bytes(b"parcel-image")

    expected = "8dfaa0c4ce5717bbe66369cdfd798eb2c72feda951c3d8e350c2e04ab89cd712"  # pragma: allowlist secret  # noqa: E501

    assert sha256_file(fixture, chunk_size=3) == expected


def test_sha256_file_rejects_invalid_chunk_size(tmp_path: Path) -> None:
    fixture = tmp_path / "image.bin"
    fixture.write_bytes(b"parcel-image")

    with pytest.raises(ValueError, match="positive"):
        sha256_file(fixture, chunk_size=0)
