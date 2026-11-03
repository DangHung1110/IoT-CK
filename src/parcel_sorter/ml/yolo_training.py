"""Google Colab trainer for the explicitly pinned YOLOv8n classification candidate."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Any

from parcel_sorter.ml.colab_data import PreparedDataset
from parcel_sorter.ml.contracts import ExperimentConfig, ModelCandidate
from parcel_sorter.ml.training_runtime import (
    TrainingRuntimeError,
    assert_colab_gpu_runtime,
    seed_training_runtime,
    write_json_atomic,
)

YOLO_CHECKPOINT = "yolov8n-cls.pt"
ULTRALYTICS_VERSION = "8.4.191"


def _ordered_model_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, dict):
        try:
            return tuple(str(names[index]) for index in range(len(names)))
        except KeyError as error:
            raise TrainingRuntimeError("YOLO class indices are not contiguous from zero") from error
    if isinstance(names, list | tuple):
        return tuple(str(name) for name in names)
    raise TrainingRuntimeError("YOLO did not expose a supported class mapping")


def train_yolov8n_candidate(
    *,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    run_dir: Path,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Train the pinned YOLOv8n classifier on the same verified Colab dataset."""
    if config.candidate is not ModelCandidate.YOLOV8N_CLS:
        raise TrainingRuntimeError("YOLO trainer received the wrong candidate config")
    if config.pretrained_checkpoint != YOLO_CHECKPOINT:
        raise TrainingRuntimeError(f"YOLO checkpoint must be pinned to {YOLO_CHECKPOINT}")
    try:
        installed_version = metadata.version("ultralytics")
    except metadata.PackageNotFoundError as error:
        raise TrainingRuntimeError("Ultralytics is not installed in the Colab runtime") from error
    if installed_version != ULTRALYTICS_VERSION:
        raise TrainingRuntimeError(
            f"Ultralytics must be exactly {ULTRALYTICS_VERSION}, found {installed_version}"
        )

    torch = assert_colab_gpu_runtime()
    seed_training_runtime(config.seed, torch_module=torch)
    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLO(
        str(resume_checkpoint) if resume_checkpoint is not None else YOLO_CHECKPOINT
    )
    output_dir = run_dir / "yolo"
    common_options = {
        "data": str(prepared.dataset_root),
        "project": str(run_dir),
        "name": "yolo",
        "exist_ok": resume_checkpoint is not None,
        "imgsz": config.preprocess.input_size,
        "batch": config.batch_size,
        "epochs": config.head_epochs + config.finetune_epochs,
        "seed": config.seed,
        "deterministic": True,
        "device": 0,
        "workers": 2,
        "amp": config.use_amp,
        "save": True,
        "save_period": config.checkpoint_interval,
        "patience": config.patience,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": config.learning_rate_finetune,
        "degrees": config.augmentation.rotation_degrees,
        "scale": max(
            1.0 - config.augmentation.scale_min,
            config.augmentation.scale_max - 1.0,
        ),
        "fliplr": config.augmentation.horizontal_flip_probability,
        "flipud": config.augmentation.vertical_flip_probability,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": config.augmentation.brightness,
        "erasing": 0.0,
        "auto_augment": None,
        "verbose": True,
    }
    if resume_checkpoint is not None:
        common_options["resume"] = True
    model.train(**common_options)

    class_order = _ordered_model_names(model.names)
    expected_order = tuple(parcel_class.value for parcel_class in config.classes)
    if class_order != expected_order:
        raise TrainingRuntimeError(
            f"YOLO artifact class order {class_order} differs from contract {expected_order}"
        )
    best_checkpoint = output_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise TrainingRuntimeError(f"YOLO best checkpoint is missing: {best_checkpoint}")
    write_json_atomic(
        run_dir / "metrics" / "training_summary.json",
        {
            "best_checkpoint": best_checkpoint.as_posix(),
            "class_order": class_order,
            "max_epochs": config.head_epochs + config.finetune_epochs,
            "early_stopping": {
                "enabled": True,
                "patience": config.patience,
                "monitor": "Ultralytics validation fitness",
            },
            "ultralytics_version": installed_version,
        },
    )
    return best_checkpoint
