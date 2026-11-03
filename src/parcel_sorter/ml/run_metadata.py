"""Immutable experiment identity and resumable run directories on Google Drive."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from parcel_sorter.colab import AcceleratorInfo
from parcel_sorter.data import DatasetSplit, ParcelClass
from parcel_sorter.ml.colab_data import PreparedDataset
from parcel_sorter.ml.contracts import ExperimentConfig

RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{7,127}$"


class PretrainedCheckpointEvidence(BaseModel):
    """Auditable origin and license of an external pretrained checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    source_url: HttpUrl
    license_name: str = Field(min_length=1)
    license_url: HttpUrl


class DatasetRunIdentity(BaseModel):
    """Stable dataset evidence used to reject an incompatible resume attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    archive_source: Path
    manifest_source: Path
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    class_order: tuple[ParcelClass, ...]
    image_counts: dict[DatasetSplit, dict[ParcelClass, int]]


class TrainingRunManifest(BaseModel):
    """Complete immutable identity written before a candidate begins training."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    created_at_utc: datetime
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    config: ExperimentConfig
    dataset: DatasetRunIdentity
    checkpoint: PretrainedCheckpointEvidence
    packages: dict[str, str | None]
    accelerator: AcceleratorInfo

    @field_validator("created_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Reject timestamps that cannot identify the experiment globally."""
        offset = value.utcoffset()
        if value.tzinfo is None or offset is None:
            raise ValueError("created_at_utc must be timezone-aware")
        if offset.total_seconds() != 0:
            raise ValueError("created_at_utc must use UTC offset +00:00")
        return value

    @model_validator(mode="after")
    def validate_cross_contract_identity(self) -> Self:
        """Keep the manifest, experiment config, and checkpoint mutually consistent."""
        if self.dataset.archive_sha256 != self.config.dataset_archive_sha256:
            raise ValueError("dataset archive hash does not match experiment config")
        if self.dataset.manifest_sha256 != self.config.dataset_manifest_sha256:
            raise ValueError("dataset manifest hash does not match experiment config")
        if self.dataset.class_order != self.config.classes:
            raise ValueError("dataset class order does not match experiment config")
        if self.checkpoint.name != self.config.pretrained_checkpoint:
            raise ValueError("checkpoint evidence does not match experiment config")
        return self


class ResumeIdentityError(ValueError):
    """Raised when a resume attempt does not match the original experiment."""


def build_run_id(
    *, config: ExperimentConfig, git_commit: str, created_at: datetime | None = None
) -> str:
    """Build a readable unique ID while keeping its identity inputs in the manifest."""
    timestamp = (created_at or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp.lower()}-{config.candidate.value}-{config.dataset_version}-"
        f"s{config.seed}-{git_commit[:8]}"
    )


def dataset_run_identity(prepared: PreparedDataset) -> DatasetRunIdentity:
    """Convert ephemeral staged paths into the stable dataset evidence for a run."""
    return DatasetRunIdentity(
        archive_source=prepared.archive_source,
        manifest_source=prepared.manifest_source,
        archive_sha256=prepared.archive_sha256,
        manifest_sha256=prepared.manifest_sha256,
        class_order=prepared.class_order,
        image_counts={split: layout.image_counts for split, layout in prepared.splits.items()},
    )


def initialize_training_run(*, runs_root: Path, manifest: TrainingRunManifest) -> Path:
    """Create a new Drive run atomically and refuse accidental overwrites."""
    run_dir = runs_root / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "run_manifest.json"
    temporary_path = run_dir / ".run_manifest.json.tmp"
    temporary_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    for directory_name in ("checkpoints", "metrics", "exports"):
        (run_dir / directory_name).mkdir()
    return run_dir


def load_training_run(run_dir: Path) -> TrainingRunManifest:
    """Load the original manifest instead of silently generating new run identity."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ResumeIdentityError(f"run manifest is missing: {manifest_path}")
    return TrainingRunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def assert_resume_identity(
    existing: TrainingRunManifest,
    *,
    config: ExperimentConfig,
    dataset: DatasetRunIdentity,
    git_commit: str,
    checkpoint: PretrainedCheckpointEvidence,
) -> None:
    """Allow resume only for the exact code, data, class map, and checkpoint origin."""
    mismatches: list[str] = []
    if existing.config != config:
        mismatches.append("config")
    if existing.dataset != dataset:
        mismatches.append("dataset")
    if existing.git_commit != git_commit:
        mismatches.append("git_commit")
    if existing.checkpoint != checkpoint:
        mismatches.append("checkpoint")
    if mismatches:
        raise ResumeIdentityError(f"resume identity mismatch: {', '.join(mismatches)}")
