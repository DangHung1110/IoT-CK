"""Tests for the no-training Colab and Drive bootstrap."""

import importlib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from parcel_sorter.colab import bootstrap
from parcel_sorter.colab.bootstrap import (
    AcceleratorInfo,
    ColabBootstrapReport,
    assert_colab_runtime,
    ensure_drive_layout,
    save_bootstrap_report,
    validate_colab_drive_root,
)


def test_drive_root_must_be_under_colab_mydrive() -> None:
    valid = validate_colab_drive_root("/content/drive/MyDrive/parcel_sorter")

    assert valid.as_posix() == "/content/drive/MyDrive/parcel_sorter"

    for invalid in (
        "/content/parcel_sorter",
        "/content/drive/MyDrive",
        "/content/drive/MyDrive/../secrets",
        "/content/drive/MyDrive//parcel_sorter",
    ):
        with pytest.raises(ValueError, match="MyDrive"):
            validate_colab_drive_root(invalid)


def test_drive_layout_is_created_under_selected_root(tmp_path: Path) -> None:
    created = ensure_drive_layout(tmp_path)

    assert len(created) == len(bootstrap.REQUIRED_DRIVE_DIRS)
    assert all(path.is_dir() for path in created)
    assert {path.relative_to(tmp_path).as_posix() for path in created} == set(
        bootstrap.REQUIRED_DRIVE_DIRS
    )


def test_local_runtime_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "is_colab_runtime", lambda: False)

    with pytest.raises(RuntimeError, match="never on the laptop"):
        assert_colab_runtime()


def test_colab_runtime_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as context:
        context.setattr(importlib, "import_module", lambda _name: object())
        assert bootstrap.is_colab_runtime()

    def missing_colab(_name: str) -> None:
        raise ModuleNotFoundError

    with monkeypatch.context() as context:
        context.setattr(importlib, "import_module", missing_colab)
        assert not bootstrap.is_colab_runtime()


def test_git_state_records_commit_and_dirty_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    responses = iter(
        (
            SimpleNamespace(stdout=f"{'a' * 40}\n"),
            SimpleNamespace(stdout=" M tracked.py\n"),
        )
    )

    def fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return next(responses)

    monkeypatch.setattr("parcel_sorter.colab.bootstrap.subprocess.run", fake_run)

    assert bootstrap.collect_git_state(tmp_path) == ("a" * 40, True)


def test_package_inventory_marks_missing_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_version(name: str) -> str:
        if name == "torch":
            return "2.test"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr("parcel_sorter.colab.bootstrap.metadata.version", fake_version)

    versions = bootstrap.collect_package_versions()

    assert versions["torch"] == "2.test"
    assert versions["onnx"] is None


def test_accelerator_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_torch(name: str) -> None:
        assert name == "torch"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("parcel_sorter.colab.bootstrap.importlib.import_module", missing_torch)

    assert bootstrap.collect_accelerator_info() == AcceleratorInfo(
        torch_version=None,
        cuda_available=False,
        cuda_version=None,
        device_name=None,
    )


def test_accelerator_reports_colab_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.test",
        version=SimpleNamespace(cuda="12.test"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _device: "Test GPU",
        ),
    )
    monkeypatch.setattr(
        "parcel_sorter.colab.bootstrap.importlib.import_module", lambda _name: fake_torch
    )

    assert bootstrap.collect_accelerator_info() == AcceleratorInfo(
        torch_version="2.test",
        cuda_available=True,
        cuda_version="12.test",
        device_name="Test GPU",
    )


def test_report_is_written_to_unique_drive_run(tmp_path: Path) -> None:
    report = ColabBootstrapReport(
        generated_at_utc=datetime(2026, 10, 30, 4, 30, tzinfo=UTC),
        is_colab=True,
        git_commit="a" * 40,
        git_dirty=False,
        python_version="3.11.test",
        platform="colab-test",
        drive_root="/content/drive/MyDrive/parcel_sorter",
        packages={"torch": "2.test"},
        accelerator=AcceleratorInfo(
            torch_version="2.test",
            cuda_available=True,
            cuda_version="12.test",
            device_name="Test GPU",
        ),
    )

    report_path = save_bootstrap_report(report, drive_root=tmp_path)

    assert report_path.relative_to(tmp_path).as_posix() == (
        "reports/bootstrap/20261030T043000Z/colab_environment.json"
    )
    loaded = ColabBootstrapReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert loaded == report
    assert not (report_path.parent / ".colab_environment.json.tmp").exists()

    with pytest.raises(FileExistsError):
        save_bootstrap_report(report, drive_root=tmp_path)


def test_report_builder_collects_reproducibility_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated_at = datetime(2026, 10, 30, 5, 0, tzinfo=UTC)
    accelerator = AcceleratorInfo(
        torch_version="2.test",
        cuda_available=True,
        cuda_version="12.test",
        device_name="Test GPU",
    )
    monkeypatch.setattr(bootstrap, "collect_git_state", lambda _repo_dir: ("b" * 40, False))
    monkeypatch.setattr(bootstrap, "is_colab_runtime", lambda: True)
    monkeypatch.setattr(bootstrap, "collect_package_versions", lambda: {"torch": "2.test"})
    monkeypatch.setattr(bootstrap, "collect_accelerator_info", lambda: accelerator)
    monkeypatch.setattr(
        "parcel_sorter.colab.bootstrap.platform.python_version", lambda: "3.11.test"
    )
    monkeypatch.setattr("parcel_sorter.colab.bootstrap.platform.platform", lambda: "colab-test")

    report = bootstrap.build_bootstrap_report(
        repo_dir=tmp_path,
        drive_root=tmp_path / "drive",
        generated_at=generated_at,
    )

    assert report.generated_at_utc == generated_at
    assert report.git_commit == "b" * 40
    assert report.python_version == "3.11.test"
    assert report.accelerator == accelerator


def test_bootstrap_orchestrates_drive_and_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated_at = datetime(2026, 10, 30, 5, 30, tzinfo=UTC)
    report = ColabBootstrapReport(
        generated_at_utc=generated_at,
        is_colab=True,
        git_commit="c" * 40,
        git_dirty=False,
        python_version="3.11.test",
        platform="colab-test",
        drive_root="/content/drive/MyDrive/parcel_sorter",
        packages={},
        accelerator=AcceleratorInfo(
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            device_name=None,
        ),
    )
    monkeypatch.setattr(bootstrap, "assert_colab_runtime", lambda: None)
    monkeypatch.setattr(bootstrap, "validate_colab_drive_root", lambda _root: tmp_path)
    monkeypatch.setattr(bootstrap, "build_bootstrap_report", lambda **_kwargs: report)

    report_path, returned_report = bootstrap.bootstrap_colab(
        repo_dir=tmp_path / "repo",
        drive_root="/content/drive/MyDrive/parcel_sorter",
    )

    assert returned_report == report
    assert report_path.is_file()
    assert all(
        (tmp_path / relative_path).is_dir() for relative_path in bootstrap.REQUIRED_DRIVE_DIRS
    )
