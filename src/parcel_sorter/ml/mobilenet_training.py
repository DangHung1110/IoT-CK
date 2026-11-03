"""Google Colab trainer for the MobileNetV3-Small comparison candidate."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parcel_sorter.data import DatasetSplit
from parcel_sorter.ml.colab_data import PreparedDataset
from parcel_sorter.ml.contracts import ExperimentConfig, ModelCandidate
from parcel_sorter.ml.training_runtime import (
    EpochMetrics,
    TrainingRuntimeError,
    append_epoch_metrics,
    assert_colab_gpu_runtime,
    class_index_remap,
    seed_training_runtime,
    write_json_atomic,
)

MOBILENET_CHECKPOINT = "MobileNet_V3_Small_Weights.IMAGENET1K_V1"


@dataclass(frozen=True)
class _IndexRemap:
    mapping: dict[int, int]

    def __call__(self, source_index: int) -> int:
        return self.mapping[source_index]


def _build_transforms(config: ExperimentConfig, torchvision: Any) -> tuple[Any, Any]:
    interpolation = torchvision.transforms.InterpolationMode.BILINEAR
    common = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=config.preprocess.mean,
            std=config.preprocess.std,
        ),
    ]
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (config.preprocess.input_size, config.preprocess.input_size),
                interpolation=interpolation,
                antialias=True,
            ),
            torchvision.transforms.RandomAffine(
                degrees=config.augmentation.rotation_degrees,
                scale=(config.augmentation.scale_min, config.augmentation.scale_max),
                interpolation=interpolation,
            ),
            torchvision.transforms.ColorJitter(
                brightness=config.augmentation.brightness,
                contrast=config.augmentation.contrast,
            ),
            torchvision.transforms.RandomHorizontalFlip(
                p=config.augmentation.horizontal_flip_probability
            ),
            torchvision.transforms.RandomVerticalFlip(
                p=config.augmentation.vertical_flip_probability
            ),
            *common,
        ]
    )
    validation_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (config.preprocess.input_size, config.preprocess.input_size),
                interpolation=interpolation,
                antialias=True,
            ),
            *common,
        ]
    )
    return train_transform, validation_transform


def _build_loaders(
    config: ExperimentConfig,
    prepared: PreparedDataset,
    *,
    torch: Any,
    torchvision: Any,
) -> tuple[Any, Any]:
    train_transform, validation_transform = _build_transforms(config, torchvision)
    train_dataset = torchvision.datasets.ImageFolder(
        str(prepared.splits[DatasetSplit.TRAIN].path),
        transform=train_transform,
    )
    validation_dataset = torchvision.datasets.ImageFolder(
        str(prepared.splits[DatasetSplit.VAL].path),
        transform=validation_transform,
    )
    if train_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise TrainingRuntimeError("train and validation class mappings differ")
    remap = _IndexRemap(class_index_remap(train_dataset.class_to_idx, class_order=config.classes))
    train_dataset.target_transform = remap
    validation_dataset.target_transform = remap
    generator = torch.Generator().manual_seed(config.seed)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    return train_loader, validation_loader


def _build_model(config: ExperimentConfig, *, torch: Any, torchvision: Any) -> Any:
    if config.pretrained_checkpoint != MOBILENET_CHECKPOINT:
        raise TrainingRuntimeError(f"MobileNet checkpoint must be pinned to {MOBILENET_CHECKPOINT}")
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = torchvision.models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(config.classes))
    return model.cuda()


def _set_backbone_trainable(model: Any, *, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


def _classification_rates(
    targets: list[int], predictions: list[int], *, ok_index: int
) -> tuple[float, float, float]:
    if not targets:
        raise TrainingRuntimeError("validation loader produced no samples")
    correct = sum(
        actual == predicted for actual, predicted in zip(targets, predictions, strict=True)
    )
    damage_total = sum(actual != ok_index for actual in targets)
    detected_damage = sum(
        actual != ok_index and predicted != ok_index
        for actual, predicted in zip(targets, predictions, strict=True)
    )
    ok_total = sum(actual == ok_index for actual in targets)
    false_rejects = sum(
        actual == ok_index and predicted != ok_index
        for actual, predicted in zip(targets, predictions, strict=True)
    )
    damage_recall = detected_damage / damage_total if damage_total else 0.0
    false_reject_rate = false_rejects / ok_total if ok_total else 0.0
    return correct / len(targets), damage_recall, false_reject_rate


def _run_loader(
    loader: Any,
    *,
    model: Any,
    criterion: Any,
    torch: Any,
    training: bool,
    optimizer: Any | None,
    scaler: Any | None,
) -> tuple[float, list[int], list[int]]:
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    predictions: list[int] = []
    grad_context: Callable[[], Any] = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for images, labels in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if training:
                if optimizer is None or scaler is None:
                    raise TrainingRuntimeError("training requires optimizer and scaler")
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler is not None and scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                assert scaler is not None
                assert optimizer is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            if not training:
                targets.extend(int(item) for item in labels.detach().cpu().tolist())
                predictions.extend(
                    int(item) for item in logits.detach().argmax(dim=1).cpu().tolist()
                )
    if total_samples == 0:
        raise TrainingRuntimeError("data loader produced no samples")
    return total_loss / total_samples, targets, predictions


def _save_checkpoint_atomic(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_mobilenet_candidate(
    *,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    run_dir: Path,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Train MobileNet in frozen-head and fine-tune phases only on a Colab GPU."""
    if config.candidate is not ModelCandidate.MOBILENET_V3_SMALL:
        raise TrainingRuntimeError("MobileNet trainer received the wrong candidate config")
    torch = assert_colab_gpu_runtime()
    torchvision = importlib.import_module("torchvision")
    seed_training_runtime(config.seed, torch_module=torch)
    train_loader, validation_loader = _build_loaders(
        config, prepared, torch=torch, torchvision=torchvision
    )
    model = _build_model(config, torch=torch, torchvision=torchvision)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)
    ok_index = config.classes.index(config.pass_class)
    phases = (
        ("head", config.head_epochs, config.learning_rate_head, False),
        ("finetune", config.finetune_epochs, config.learning_rate_finetune, True),
    )
    start_epoch = 1
    resume_state: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    best_path = run_dir / "checkpoints" / "best.pt"
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location="cuda", weights_only=False)
        if resume_state["config"] != config.model_dump(mode="json"):
            raise TrainingRuntimeError("resume checkpoint config does not match this run")
        if tuple(resume_state["class_order"]) != tuple(item.value for item in config.classes):
            raise TrainingRuntimeError("resume checkpoint class order does not match this run")
        model.load_state_dict(resume_state["model_state"])
        start_epoch = int(resume_state["epoch"]) + 1
        saved_key = resume_state.get("best_key")
        if saved_key:
            if len(saved_key) != 4:
                raise TrainingRuntimeError("resume checkpoint has an invalid best key")
            best_key = (
                float(saved_key[0]),
                float(saved_key[1]),
                float(saved_key[2]),
                float(saved_key[3]),
            )

    global_epoch = 0
    for phase_name, phase_epochs, learning_rate, backbone_trainable in phases:
        phase_start = global_epoch + 1
        phase_end = global_epoch + phase_epochs
        global_epoch = phase_end
        if start_epoch > phase_end:
            continue
        _set_backbone_trainable(model, trainable=backbone_trainable)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        if (
            resume_state is not None
            and resume_state.get("phase") == phase_name
            and start_epoch >= phase_start
        ):
            optimizer.load_state_dict(resume_state["optimizer_state"])
        stale_epochs = 0
        for epoch in range(max(start_epoch, phase_start), phase_end + 1):
            started = time.monotonic()
            train_loss, _, _ = _run_loader(
                train_loader,
                model=model,
                criterion=criterion,
                torch=torch,
                training=True,
                optimizer=optimizer,
                scaler=scaler,
            )
            val_loss, targets, predictions = _run_loader(
                validation_loader,
                model=model,
                criterion=criterion,
                torch=torch,
                training=False,
                optimizer=None,
                scaler=None,
            )
            accuracy, damage_recall, false_reject_rate = _classification_rates(
                targets, predictions, ok_index=ok_index
            )
            metrics = EpochMetrics(
                epoch=epoch,
                phase=phase_name,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=accuracy,
                val_damage_recall=damage_recall,
                val_false_reject_rate=false_reject_rate,
                duration_seconds=time.monotonic() - started,
            )
            append_epoch_metrics(run_dir / "metrics" / "epochs.jsonl", metrics)
            payload = {
                "epoch": epoch,
                "phase": phase_name,
                "config": config.model_dump(mode="json"),
                "class_order": [item.value for item in config.classes],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_key": list(best_key) if best_key is not None else None,
            }
            if epoch % config.checkpoint_interval == 0:
                _save_checkpoint_atomic(
                    torch,
                    run_dir / "checkpoints" / f"epoch-{epoch:04d}.pt",
                    payload,
                )
            if best_key is None or metrics.selection_key > best_key:
                best_key = metrics.selection_key
                payload["best_key"] = list(best_key)
                _save_checkpoint_atomic(torch, best_path, payload)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= config.patience:
                break
        start_epoch = phase_end + 1
        resume_state = None
    if not best_path.is_file():
        raise TrainingRuntimeError("training completed without a best checkpoint")
    write_json_atomic(
        run_dir / "metrics" / "training_summary.json",
        {"best_checkpoint": best_path.as_posix(), "best_selection_key": best_key},
    )
    return best_path
