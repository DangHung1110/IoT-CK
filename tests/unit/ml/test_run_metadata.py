from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError

from parcel_sorter.colab import AcceleratorInfo
from parcel_sorter.data import DatasetSplit, ParcelClass
from parcel_sorter.ml.colab_data import DatasetSplitLayout, PreparedDataset
from parcel_sorter.ml.contracts import (
    ExperimentConfig,
    ModelCandidate,
    PreprocessContract,
)
from parcel_sorter.ml.run_metadata import (
    PretrainedCheckpointEvidence,
    ResumeIdentityError,
    TrainingRunManifest,
    assert_resume_identity,
    build_run_id,
    dataset_run_identity,
    initialize_training_run,
    load_training_run,
)

GIT_COMMIT = "a" * 40
ARCHIVE_HASH = "b" * 64
MANIFEST_HASH = "c" * 64
CLASS_ORDER = tuple(ParcelClass)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        candidate=ModelCandidate.MOBILENET_V3_SMALL,
        dataset_version="v1",
        dataset_manifest_sha256=MANIFEST_HASH,
        dataset_archive_sha256=ARCHIVE_HASH,
        split_seed=42,
        seed=42,
        classes=CLASS_ORDER,
        preprocess=PreprocessContract(
            preprocess_id="mobilenet-rgb320-v1",
            resize="resize_square",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        pretrained_checkpoint="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
    )


def _prepared(tmp_path: Path) -> PreparedDataset:
    counts = {parcel_class: 1 for parcel_class in CLASS_ORDER}
    return PreparedDataset(
        archive_source=tmp_path / "drive" / "dataset.zip",
        manifest_source=tmp_path / "drive" / "manifest.csv",
        archive_sha256=ARCHIVE_HASH,
        manifest_sha256=MANIFEST_HASH,
        dataset_root=tmp_path / "content" / "dataset",
        class_order=CLASS_ORDER,
        splits={
            split: DatasetSplitLayout(
                path=tmp_path / "content" / "dataset" / split.value,
                image_counts=counts,
            )
            for split in DatasetSplit
        },
    )


def _checkpoint() -> PretrainedCheckpointEvidence:
    return PretrainedCheckpointEvidence(
        name="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        source_url=HttpUrl(
            "https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html"
        ),
        license_name="BSD-3-Clause",
        license_url=HttpUrl("https://github.com/pytorch/vision/blob/main/LICENSE"),
    )


def _manifest(tmp_path: Path) -> TrainingRunManifest:
    config = _config()
    created_at = datetime(2026, 11, 3, 4, 5, 6, tzinfo=UTC)
    return TrainingRunManifest(
        run_id=build_run_id(config=config, git_commit=GIT_COMMIT, created_at=created_at),
        created_at_utc=created_at,
        git_commit=GIT_COMMIT,
        git_dirty=False,
        config=config,
        dataset=dataset_run_identity(_prepared(tmp_path)),
        checkpoint=_checkpoint(),
        packages={"torch": "2.9.0"},
        accelerator=AcceleratorInfo(
            torch_version="2.9.0",
            cuda_available=True,
            cuda_version="12.8",
            device_name="T4",
        ),
    )


def test_build_run_id_contains_reproducible_human_readable_fields() -> None:
    run_id = build_run_id(
        config=_config(),
        git_commit=GIT_COMMIT,
        created_at=datetime(2026, 11, 3, 4, 5, 6, tzinfo=UTC),
    )

    assert run_id == "20261103t040506z-mobilenet_v3_small-v1-s42-aaaaaaaa"


def test_manifest_requires_utc_and_cross_contract_consistency(tmp_path: Path) -> None:
    valid = _manifest(tmp_path)
    payload = valid.model_dump()
    payload["created_at_utc"] = datetime(2026, 11, 3)
    with pytest.raises(ValidationError, match="timezone-aware"):
        TrainingRunManifest.model_validate(payload)

    payload["created_at_utc"] = datetime(2026, 11, 3, tzinfo=timezone(timedelta(hours=7)))
    with pytest.raises(ValidationError, match="UTC offset"):
        TrainingRunManifest.model_validate(payload)

    payload = valid.model_dump()
    payload["dataset"]["archive_sha256"] = "d" * 64
    with pytest.raises(ValidationError, match="archive hash"):
        TrainingRunManifest.model_validate(payload)


def test_initialize_and_load_training_run_round_trip(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run_dir = initialize_training_run(runs_root=tmp_path / "runs", manifest=manifest)

    assert load_training_run(run_dir) == manifest
    assert {path.name for path in run_dir.iterdir()} == {
        "checkpoints",
        "exports",
        "metrics",
        "run_manifest.json",
    }
    with pytest.raises(FileExistsError):
        initialize_training_run(runs_root=tmp_path / "runs", manifest=manifest)


def test_load_training_run_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ResumeIdentityError, match="missing"):
        load_training_run(tmp_path)


def test_resume_requires_exact_immutable_identity(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert_resume_identity(
        manifest,
        config=manifest.config,
        dataset=manifest.dataset,
        git_commit=manifest.git_commit,
        checkpoint=manifest.checkpoint,
    )

    changed_config = manifest.config.model_copy(update={"seed": 99})
    changed_dataset = manifest.dataset.model_copy(update={"archive_sha256": "d" * 64})
    changed_checkpoint = manifest.checkpoint.model_copy(update={"license_name": "changed"})
    with pytest.raises(
        ResumeIdentityError,
        match="config, dataset, git_commit, checkpoint",
    ):
        assert_resume_identity(
            manifest,
            config=changed_config,
            dataset=changed_dataset,
            git_commit="e" * 40,
            checkpoint=changed_checkpoint,
        )
