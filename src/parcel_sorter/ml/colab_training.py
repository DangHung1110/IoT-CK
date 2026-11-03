"""Colab orchestration for verified data, immutable runs, and candidate training."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import HttpUrl

from parcel_sorter.colab import (
    collect_accelerator_info,
    collect_git_state,
    collect_package_versions,
    validate_colab_drive_root,
)
from parcel_sorter.ml.colab_data import PreparedDataset, stage_verified_dataset
from parcel_sorter.ml.contracts import ExperimentConfig, ModelCandidate
from parcel_sorter.ml.mobilenet_training import (
    MOBILENET_CHECKPOINT,
    train_mobilenet_candidate,
)
from parcel_sorter.ml.run_metadata import (
    PretrainedCheckpointEvidence,
    TrainingRunManifest,
    assert_resume_identity,
    build_run_id,
    dataset_run_identity,
    initialize_training_run,
    load_training_run,
)
from parcel_sorter.ml.training_runtime import TrainingRuntimeError, assert_colab_gpu_runtime
from parcel_sorter.ml.yolo_training import YOLO_CHECKPOINT, train_yolov8n_candidate


def checkpoint_evidence(candidate: ModelCandidate) -> PretrainedCheckpointEvidence:
    """Return the pinned upstream checkpoint provenance for one comparison candidate."""
    if candidate is ModelCandidate.MOBILENET_V3_SMALL:
        return PretrainedCheckpointEvidence(
            name=MOBILENET_CHECKPOINT,
            source_url=HttpUrl(
                "https://pytorch.org/vision/stable/models/generated/"
                "torchvision.models.mobilenet_v3_small.html"
            ),
            license_name="BSD-3-Clause",
            license_url=HttpUrl("https://github.com/pytorch/vision/blob/main/LICENSE"),
        )
    return PretrainedCheckpointEvidence(
        name=YOLO_CHECKPOINT,
        source_url=HttpUrl(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt"
        ),
        license_name="AGPL-3.0",
        license_url=HttpUrl("https://github.com/ultralytics/ultralytics/blob/main/LICENSE"),
    )


def _validate_local_workspace(path: Path) -> None:
    try:
        path.resolve().relative_to(Path("/content").resolve())
    except ValueError as error:
        raise TrainingRuntimeError("training workspace must be below Colab /content") from error


def _prepare_dataset(
    config: ExperimentConfig,
    *,
    drive_root: Path,
    archive_relative_path: str,
    manifest_relative_path: str,
    local_workspace: Path,
) -> PreparedDataset:
    _validate_local_workspace(local_workspace)
    return stage_verified_dataset(
        archive_source=drive_root / archive_relative_path,
        manifest_source=drive_root / manifest_relative_path,
        expected_archive_sha256=config.dataset_archive_sha256,
        expected_manifest_sha256=config.dataset_manifest_sha256,
        drive_root=drive_root,
        local_workspace=local_workspace,
        class_order=config.classes,
    )


def train_candidate_on_colab(
    *,
    config: ExperimentConfig,
    repo_dir: Path,
    drive_root: str,
    archive_relative_path: str,
    manifest_relative_path: str,
    local_workspace: Path,
    resume_run_id: str | None = None,
    resume_checkpoint_relative_path: str | None = None,
) -> tuple[Path, Path]:
    """Run one exact candidate on Colab and return its Drive run and best checkpoint."""
    assert_colab_gpu_runtime()
    drive_path = validate_colab_drive_root(drive_root)
    git_commit, git_dirty = collect_git_state(repo_dir)
    if git_dirty:
        raise TrainingRuntimeError("refusing to train from a dirty Git checkout")
    prepared = _prepare_dataset(
        config,
        drive_root=drive_path,
        archive_relative_path=archive_relative_path,
        manifest_relative_path=manifest_relative_path,
        local_workspace=local_workspace,
    )
    dataset_identity = dataset_run_identity(prepared)
    evidence = checkpoint_evidence(config.candidate)
    runs_root = drive_path / "models" / "runs"
    resume_checkpoint: Path | None = None
    if resume_run_id is None:
        if resume_checkpoint_relative_path is not None:
            raise TrainingRuntimeError("a resume checkpoint requires resume_run_id")
        created_at = datetime.now(UTC)
        manifest = TrainingRunManifest(
            run_id=build_run_id(config=config, git_commit=git_commit, created_at=created_at),
            created_at_utc=created_at,
            git_commit=git_commit,
            git_dirty=False,
            config=config,
            dataset=dataset_identity,
            checkpoint=evidence,
            packages=collect_package_versions(),
            accelerator=collect_accelerator_info(),
        )
        run_dir = initialize_training_run(runs_root=runs_root, manifest=manifest)
    else:
        run_dir = runs_root / resume_run_id
        existing = load_training_run(run_dir)
        assert_resume_identity(
            existing,
            config=config,
            dataset=dataset_identity,
            git_commit=git_commit,
            checkpoint=evidence,
        )
        if resume_checkpoint_relative_path is None:
            raise TrainingRuntimeError("resume requires an explicit checkpoint path")
        candidate_checkpoint = run_dir / resume_checkpoint_relative_path
        try:
            candidate_checkpoint.resolve().relative_to(run_dir.resolve())
        except ValueError as error:
            raise TrainingRuntimeError(
                "resume checkpoint must be inside its run directory"
            ) from error
        if not candidate_checkpoint.is_file():
            raise TrainingRuntimeError(f"resume checkpoint does not exist: {candidate_checkpoint}")
        resume_checkpoint = candidate_checkpoint

    if config.candidate is ModelCandidate.MOBILENET_V3_SMALL:
        best_checkpoint = train_mobilenet_candidate(
            config=config,
            prepared=prepared,
            run_dir=run_dir,
            resume_checkpoint=resume_checkpoint,
        )
    else:
        best_checkpoint = train_yolov8n_candidate(
            config=config,
            prepared=prepared,
            run_dir=run_dir,
            resume_checkpoint=resume_checkpoint,
        )
    return run_dir, best_checkpoint
