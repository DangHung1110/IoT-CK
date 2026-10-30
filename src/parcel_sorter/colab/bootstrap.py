"""Safe Colab and Google Drive bootstrap without running model training."""

from __future__ import annotations

import importlib
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

REQUIRED_DRIVE_DIRS = (
    "data/raw",
    "data/derived",
    "data/external",
    "data/feedback",
    "data/manifests",
    "data/archives",
    "models/runs",
    "models/exports",
    "reports/bootstrap",
)

REPORT_PACKAGES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "pydantic",
    "torch",
    "torchvision",
    "ultralytics",
)


class AcceleratorInfo(BaseModel):
    """Torch and CUDA state reported by the Colab runtime."""

    model_config = ConfigDict(frozen=True)

    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None


class ColabBootstrapReport(BaseModel):
    """Reproducibility evidence written to Google Drive before training."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(default=1, ge=1)
    generated_at_utc: datetime
    is_colab: bool
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    python_version: str
    platform: str
    drive_root: str
    packages: dict[str, str | None]
    accelerator: AcceleratorInfo


class _CudaApi(Protocol):
    def is_available(self) -> bool: ...

    def get_device_name(self, device: int) -> str: ...


class _TorchVersion(Protocol):
    cuda: str | None


class _TorchModule(Protocol):
    __version__: str
    cuda: _CudaApi
    version: _TorchVersion


def is_colab_runtime() -> bool:
    """Return whether the Google Colab runtime module is importable."""
    try:
        importlib.import_module("google.colab")
    except ModuleNotFoundError:
        return False
    return True


def assert_colab_runtime() -> None:
    """Prevent bootstrap and future training notebooks from running locally."""
    if not is_colab_runtime():
        raise RuntimeError("this notebook must run in Google Colab, never on the laptop")


def validate_colab_drive_root(root: str) -> Path:
    """Allow writes only below the mounted user's MyDrive directory."""
    path = PurePosixPath(root)
    expected_prefix = ("/", "content", "drive", "MyDrive")
    if (
        path.parts[:4] != expected_prefix
        or len(path.parts) <= 4
        or ".." in path.parts
        or path.as_posix() != root
    ):
        raise ValueError("drive_root must be a canonical child of /content/drive/MyDrive")
    return Path(path.as_posix())


def ensure_drive_layout(root: Path) -> tuple[Path, ...]:
    """Create and return the versioned project directories on Google Drive."""
    created: list[Path] = []
    for relative_path in REQUIRED_DRIVE_DIRS:
        directory = root / relative_path
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return tuple(created)


def collect_git_state(repo_dir: Path) -> tuple[str, bool]:
    """Return the exact checked-out commit and whether tracked files are dirty."""
    commit = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def collect_package_versions() -> dict[str, str | None]:
    """Read installed versions without importing the training libraries."""
    versions: dict[str, str | None] = {}
    for package_name in REPORT_PACKAGES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def collect_accelerator_info() -> AcceleratorInfo:
    """Inspect the Colab accelerator; absence of torch is reported, not installed."""
    try:
        torch_module = cast(_TorchModule, importlib.import_module("torch"))
    except ModuleNotFoundError:
        return AcceleratorInfo(
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            device_name=None,
        )

    cuda_available = torch_module.cuda.is_available()
    return AcceleratorInfo(
        torch_version=torch_module.__version__,
        cuda_available=cuda_available,
        cuda_version=torch_module.version.cuda,
        device_name=torch_module.cuda.get_device_name(0) if cuda_available else None,
    )


def build_bootstrap_report(
    *,
    repo_dir: Path,
    drive_root: Path,
    generated_at: datetime | None = None,
) -> ColabBootstrapReport:
    """Collect the environment identity needed before any experiment starts."""
    commit, dirty = collect_git_state(repo_dir)
    timestamp = generated_at or datetime.now(UTC)
    return ColabBootstrapReport(
        generated_at_utc=timestamp,
        is_colab=is_colab_runtime(),
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        platform=platform.platform(),
        drive_root=drive_root.as_posix(),
        packages=collect_package_versions(),
        accelerator=collect_accelerator_info(),
    )


def save_bootstrap_report(report: ColabBootstrapReport, *, drive_root: Path) -> Path:
    """Atomically write the bootstrap report into its own Drive run directory."""
    run_id = report.generated_at_utc.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_dir = drive_root / "reports" / "bootstrap" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "colab_environment.json"
    temporary_path = report_dir / ".colab_environment.json.tmp"
    temporary_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(report_path)
    return report_path


def bootstrap_colab(*, repo_dir: Path, drive_root: str) -> tuple[Path, ColabBootstrapReport]:
    """Validate Colab, initialize Drive, and persist evidence without training."""
    assert_colab_runtime()
    drive_path = validate_colab_drive_root(drive_root)
    ensure_drive_layout(drive_path)
    report = build_bootstrap_report(repo_dir=repo_dir, drive_root=drive_path)
    report_path = save_bootstrap_report(report, drive_root=drive_path)
    return report_path, report
