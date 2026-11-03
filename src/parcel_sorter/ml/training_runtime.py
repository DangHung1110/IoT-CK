"""Colab-only runtime gates and durable training metric/checkpoint helpers."""

from __future__ import annotations

import importlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from parcel_sorter.colab import assert_colab_runtime
from parcel_sorter.data import ParcelClass


class TrainingRuntimeError(RuntimeError):
    """Raised before training when the Colab GPU contract is not satisfied."""


class _CudaApi(Protocol):
    def is_available(self) -> bool: ...

    def manual_seed_all(self, seed: int) -> None: ...


class _CudnnApi(Protocol):
    deterministic: bool
    benchmark: bool


class _BackendsApi(Protocol):
    cudnn: _CudnnApi


class _TorchApi(Protocol):
    cuda: _CudaApi
    backends: _BackendsApi

    def manual_seed(self, seed: int) -> Any: ...

    def use_deterministic_algorithms(self, mode: bool, *, warn_only: bool) -> None: ...


class EpochMetrics(BaseModel):
    """One append-only epoch record used for audit and early stopping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epoch: int = Field(ge=1)
    phase: str = Field(min_length=1)
    train_loss: float = Field(ge=0)
    val_loss: float = Field(ge=0)
    val_accuracy: float = Field(ge=0, le=1)
    val_damage_recall: float = Field(ge=0, le=1)
    val_false_reject_rate: float = Field(ge=0, le=1)
    duration_seconds: float = Field(ge=0)

    @property
    def selection_key(self) -> tuple[float, float, float, float]:
        """Prioritize damage recall, then false rejects, accuracy, and loss."""
        return (
            self.val_damage_recall,
            -self.val_false_reject_rate,
            self.val_accuracy,
            -self.val_loss,
        )


def assert_colab_gpu_runtime() -> Any:
    """Fail closed unless this process is a Google Colab runtime with CUDA."""
    assert_colab_runtime()
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise TrainingRuntimeError("Colab runtime is missing PyTorch") from error
    typed_torch = cast(_TorchApi, torch)
    if not typed_torch.cuda.is_available():
        raise TrainingRuntimeError(
            "CUDA is unavailable; select a Colab GPU runtime before training"
        )
    return torch


def seed_training_runtime(seed: int, *, torch_module: Any) -> None:
    """Seed Python, NumPy when present, and the supplied Colab Torch module."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
    except ModuleNotFoundError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(seed)

    torch = cast(_TorchApi, torch_module)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def class_index_remap(
    framework_mapping: Mapping[str, int],
    *,
    class_order: tuple[ParcelClass, ...],
) -> dict[int, int]:
    """Map framework-discovered indices onto the single experiment class order."""
    expected_names = {parcel_class.value for parcel_class in class_order}
    if set(framework_mapping) != expected_names:
        raise TrainingRuntimeError(
            "framework class names do not match the experiment class contract"
        )
    indices = list(framework_mapping.values())
    if sorted(indices) != list(range(len(class_order))):
        raise TrainingRuntimeError("framework class indices must be contiguous from zero")
    target_by_name = {parcel_class.value: index for index, parcel_class in enumerate(class_order)}
    return {
        framework_index: target_by_name[name] for name, framework_index in framework_mapping.items()
    }


def append_epoch_metrics(path: Path, metrics: EpochMetrics) -> None:
    """Durably append one JSONL record and flush it to the mounted Drive filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(metrics.model_dump_json())
        handle.write("\n")
        handle.flush()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON metadata through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
