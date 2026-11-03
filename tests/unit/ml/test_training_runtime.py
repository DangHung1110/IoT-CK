from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from parcel_sorter.data import BinaryParcelClass, ParcelClass
from parcel_sorter.ml import training_runtime
from parcel_sorter.ml.training_runtime import (
    EpochMetrics,
    TrainingRuntimeError,
    append_epoch_metrics,
    assert_colab_gpu_runtime,
    class_index_remap,
    seed_training_runtime,
    write_json_atomic,
)


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.seeds: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)


class _FakeTorch:
    def __init__(self, *, available: bool = True) -> None:
        self.cuda = _FakeCuda(available=available)
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))
        self.seeds: list[int] = []
        self.deterministic: tuple[bool, bool] | None = None

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool) -> None:
        self.deterministic = (mode, warn_only)


def _metrics(epoch: int = 1) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        phase="head",
        train_loss=0.5,
        val_loss=0.4,
        val_accuracy=0.8,
        val_damage_recall=0.9,
        val_false_reject_rate=0.1,
        duration_seconds=2.0,
    )


def test_epoch_selection_prioritizes_damage_recall_then_false_rejects() -> None:
    baseline = _metrics()
    better_damage = baseline.model_copy(update={"val_damage_recall": 0.91})
    fewer_false_rejects = baseline.model_copy(update={"val_false_reject_rate": 0.05})

    assert better_damage.selection_key > fewer_false_rejects.selection_key
    assert fewer_false_rejects.selection_key > baseline.selection_key


def test_class_index_remap_uses_explicit_contract_order() -> None:
    framework = {"mop": 0, "ok": 1, "rach": 2, "uot": 3, "vo": 4}
    order = tuple(ParcelClass)

    assert class_index_remap(framework, class_order=order) == {
        0: 2,
        1: 0,
        2: 1,
        3: 4,
        4: 3,
    }

    assert class_index_remap({"damage": 0, "normal": 1}, class_order=tuple(BinaryParcelClass)) == {
        0: 1,
        1: 0,
    }


@pytest.mark.parametrize(
    "mapping",
    [
        {"ok": 0},
        {"mop": 0, "ok": 1, "rach": 2, "uot": 3, "vo": 9},
    ],
)
def test_class_index_remap_rejects_wrong_names_or_indices(mapping: dict[str, int]) -> None:
    with pytest.raises(TrainingRuntimeError):
        class_index_remap(mapping, class_order=tuple(ParcelClass))


def test_assert_colab_gpu_runtime_is_lazy_and_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch()
    monkeypatch.setattr(training_runtime, "assert_colab_runtime", lambda: None)
    monkeypatch.setattr(
        "parcel_sorter.ml.training_runtime.importlib.import_module", lambda _: fake_torch
    )
    assert assert_colab_gpu_runtime() is fake_torch

    fake_torch.cuda.available = False
    with pytest.raises(TrainingRuntimeError, match="CUDA"):
        assert_colab_gpu_runtime()


def test_assert_colab_gpu_runtime_reports_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_runtime, "assert_colab_runtime", lambda: None)

    def missing(_: str) -> Any:
        raise ModuleNotFoundError

    monkeypatch.setattr("parcel_sorter.ml.training_runtime.importlib.import_module", missing)
    with pytest.raises(TrainingRuntimeError, match="missing PyTorch"):
        assert_colab_gpu_runtime()


def test_seed_runtime_sets_deterministic_torch_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch()
    fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda _: None))
    monkeypatch.setattr(
        "parcel_sorter.ml.training_runtime.importlib.import_module", lambda _: fake_numpy
    )

    seed_training_runtime(42, torch_module=fake_torch)

    assert fake_torch.seeds == [42]
    assert fake_torch.cuda.seeds == [42]
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False
    assert fake_torch.deterministic == (True, True)
    with pytest.raises(ValueError, match="non-negative"):
        seed_training_runtime(-1, torch_module=fake_torch)


def test_metric_and_json_writes_are_machine_readable(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics" / "epochs.jsonl"
    append_epoch_metrics(metrics_path, _metrics(1))
    append_epoch_metrics(metrics_path, _metrics(2))
    records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [record["epoch"] for record in records] == [1, 2]

    summary_path = tmp_path / "metrics" / "summary.json"
    write_json_atomic(summary_path, {"ok": True})
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"ok": True}
    assert not summary_path.with_name(".summary.json.tmp").exists()
