"""Colab-only prediction runners for binary MobileNet and YOLO checkpoints."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Any

from parcel_sorter.data import BinaryParcelClass, DatasetSplit
from parcel_sorter.ml.binary_evaluation import BinaryPredictionRecord
from parcel_sorter.ml.colab_data import IMAGE_SUFFIXES, PreparedDataset
from parcel_sorter.ml.contracts import ExperimentConfig, ModelCandidate
from parcel_sorter.ml.run_metadata import TrainingRunManifest
from parcel_sorter.ml.training_runtime import (
    TrainingRuntimeError,
    assert_colab_gpu_runtime,
    class_index_remap,
)
from parcel_sorter.ml.yolo_training import ULTRALYTICS_VERSION


def validate_binary_run_pair(
    manifests: tuple[TrainingRunManifest, ...],
) -> dict[ModelCandidate, TrainingRunManifest]:
    """Require a fair, same-data pair before any validation or test inference."""
    if len(manifests) != 2:
        raise TrainingRuntimeError("evaluation requires exactly two candidate runs")
    by_candidate = {manifest.config.candidate: manifest for manifest in manifests}
    expected = {ModelCandidate.MOBILENET_V3_SMALL, ModelCandidate.YOLOV8N_CLS}
    if set(by_candidate) != expected or len(by_candidate) != len(manifests):
        raise TrainingRuntimeError("evaluation requires one MobileNet and one YOLO run")
    reference = manifests[0]
    if reference.git_dirty or any(manifest.git_dirty for manifest in manifests):
        raise TrainingRuntimeError("evaluation refuses runs created from a dirty checkout")
    if set(reference.config.classes) != set(BinaryParcelClass):
        raise TrainingRuntimeError("this notebook evaluates the binary v0 contract only")

    for manifest in manifests[1:]:
        if manifest.git_commit != reference.git_commit:
            raise TrainingRuntimeError("candidate runs were trained from different Git commits")
        if manifest.dataset != reference.dataset:
            raise TrainingRuntimeError("candidate runs do not share the exact dataset identity")
        comparable_fields = (
            "dataset_version",
            "dataset_manifest_sha256",
            "dataset_archive_sha256",
            "split_seed",
            "seed",
            "classes",
            "augmentation",
            "batch_size",
            "head_epochs",
            "finetune_epochs",
            "patience",
            "use_amp",
        )
        mismatches = [
            field
            for field in comparable_fields
            if getattr(manifest.config, field) != getattr(reference.config, field)
        ]
        if mismatches:
            raise TrainingRuntimeError(
                "candidate run configs are not comparable: " + ", ".join(mismatches)
            )
    return by_candidate


def resolve_best_checkpoint(run_dir: Path, candidate: ModelCandidate) -> Path:
    """Resolve and validate the trainer-specific best checkpoint path."""
    relative = (
        Path("checkpoints/best.pt")
        if candidate is ModelCandidate.MOBILENET_V3_SMALL
        else Path("yolo/weights/best.pt")
    )
    checkpoint = run_dir / relative
    if not checkpoint.is_file():
        raise TrainingRuntimeError(f"best checkpoint is missing: {checkpoint}")
    try:
        checkpoint.resolve().relative_to(run_dir.resolve())
    except ValueError as error:
        raise TrainingRuntimeError("checkpoint escaped its run directory") from error
    return checkpoint


def _mobilenet_model(
    config: ExperimentConfig, checkpoint: Path, *, torch: Any, torchvision: Any
) -> Any:
    model = torchvision.models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(config.classes))
    payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if payload.get("config") != config.model_dump(mode="json"):
        raise TrainingRuntimeError("MobileNet checkpoint config does not match run manifest")
    expected_order = tuple(item.value for item in config.classes)
    if tuple(payload.get("class_order", ())) != expected_order:
        raise TrainingRuntimeError("MobileNet checkpoint class order does not match run manifest")
    model.load_state_dict(payload["model_state"])
    return model.cuda().eval()


def predict_mobilenet_binary(
    *,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    checkpoint: Path,
    split: DatasetSplit,
) -> tuple[BinaryPredictionRecord, ...]:
    """Run deterministic MobileNet inference for exactly one immutable split."""
    torch = assert_colab_gpu_runtime()
    torchvision = importlib.import_module("torchvision")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (config.preprocess.input_size, config.preprocess.input_size),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=config.preprocess.mean,
                std=config.preprocess.std,
            ),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(
        str(prepared.splits[split].path), transform=transform
    )
    remap = class_index_remap(dataset.class_to_idx, class_order=config.classes)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    model = _mobilenet_model(config, checkpoint, torch=torch, torchvision=torchvision)
    records: list[BinaryPredictionRecord] = []
    cursor = 0
    with torch.no_grad():
        for images, source_labels in loader:
            probabilities = torch.softmax(model(images.cuda(non_blocking=True)), dim=1)
            rows = probabilities.detach().cpu().tolist()
            labels = source_labels.tolist()
            for row, source_label in zip(rows, labels, strict=True):
                source_path = Path(dataset.samples[cursor][0])
                target_index = remap[int(source_label)]
                records.append(
                    BinaryPredictionRecord(
                        image_id=source_path.relative_to(prepared.dataset_root).as_posix(),
                        split=split,
                        true_class=BinaryParcelClass(config.classes[target_index].value),
                        classes=tuple(BinaryParcelClass(item.value) for item in config.classes),
                        probabilities=tuple(float(value) for value in row),
                    )
                )
                cursor += 1
    if cursor != len(dataset):
        raise TrainingRuntimeError("MobileNet prediction count does not match split size")
    return tuple(records)


def _ordered_yolo_names(names: Any) -> tuple[str, ...]:
    if isinstance(names, dict):
        try:
            return tuple(str(names[index]) for index in range(len(names)))
        except KeyError as error:
            raise TrainingRuntimeError("YOLO class indices are not contiguous") from error
    if isinstance(names, list | tuple):
        return tuple(str(item) for item in names)
    raise TrainingRuntimeError("YOLO did not expose a supported class mapping")


def predict_yolo_binary(
    *,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    checkpoint: Path,
    split: DatasetSplit,
) -> tuple[BinaryPredictionRecord, ...]:
    """Run pinned YOLO classification inference for one immutable split."""
    assert_colab_gpu_runtime()
    installed = metadata.version("ultralytics")
    if installed != ULTRALYTICS_VERSION:
        raise TrainingRuntimeError(
            f"Ultralytics must be exactly {ULTRALYTICS_VERSION}, found {installed}"
        )
    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLO(str(checkpoint))
    expected_order = tuple(item.value for item in config.classes)
    if _ordered_yolo_names(model.names) != expected_order:
        raise TrainingRuntimeError("YOLO checkpoint class order does not match run manifest")
    split_path = prepared.splits[split].path
    image_paths = sorted(
        path
        for path in split_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=config.preprocess.input_size,
        batch=config.batch_size,
        device=0,
        stream=True,
        verbose=False,
    )
    records: list[BinaryPredictionRecord] = []
    classes = tuple(BinaryParcelClass(item.value) for item in config.classes)
    for result in results:
        if result.probs is None:
            raise TrainingRuntimeError("YOLO classification result has no probabilities")
        source_path = Path(result.path)
        records.append(
            BinaryPredictionRecord(
                image_id=source_path.relative_to(prepared.dataset_root).as_posix(),
                split=split,
                true_class=BinaryParcelClass(source_path.parent.name),
                classes=classes,
                probabilities=tuple(
                    float(value) for value in result.probs.data.detach().cpu().tolist()
                ),
            )
        )
    if len(records) != len(image_paths):
        raise TrainingRuntimeError("YOLO prediction count does not match split size")
    return tuple(records)


def predict_binary_candidate(
    *,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    checkpoint: Path,
    split: DatasetSplit,
) -> tuple[BinaryPredictionRecord, ...]:
    """Dispatch one binary candidate without allowing any training operation."""
    if config.candidate is ModelCandidate.MOBILENET_V3_SMALL:
        return predict_mobilenet_binary(
            config=config,
            prepared=prepared,
            checkpoint=checkpoint,
            split=split,
        )
    return predict_yolo_binary(
        config=config,
        prepared=prepared,
        checkpoint=checkpoint,
        split=split,
    )
