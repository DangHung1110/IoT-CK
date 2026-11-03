from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from parcel_sorter.colab import AcceleratorInfo
from parcel_sorter.data import DatasetSplit, ParcelClass
from parcel_sorter.ml import colab_training, mobilenet_training, yolo_training
from parcel_sorter.ml.colab_data import DatasetSplitLayout, PreparedDataset
from parcel_sorter.ml.contracts import (
    ExperimentConfig,
    ModelCandidate,
    PreprocessContract,
)
from parcel_sorter.ml.training_runtime import TrainingRuntimeError

ARCHIVE_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
GIT_COMMIT = "c" * 40


def _config(candidate: ModelCandidate) -> ExperimentConfig:
    class_order = tuple(sorted(ParcelClass, key=lambda item: item.value))
    is_mobile = candidate is ModelCandidate.MOBILENET_V3_SMALL
    return ExperimentConfig(
        candidate=candidate,
        dataset_version="v1",
        dataset_manifest_sha256=MANIFEST_HASH,
        dataset_archive_sha256=ARCHIVE_HASH,
        split_seed=42,
        seed=42,
        classes=class_order,
        preprocess=PreprocessContract(
            preprocess_id="candidate-rgb320-v1",
            resize="resize_square",
            mean=(0.485, 0.456, 0.406) if is_mobile else (0.0, 0.0, 0.0),
            std=(0.229, 0.224, 0.225) if is_mobile else (1.0, 1.0, 1.0),
        ),
        pretrained_checkpoint=(
            mobilenet_training.MOBILENET_CHECKPOINT if is_mobile else yolo_training.YOLO_CHECKPOINT
        ),
        batch_size=2,
        head_epochs=1,
        finetune_epochs=1,
        patience=1,
    )


def _prepared(tmp_path: Path, config: ExperimentConfig) -> PreparedDataset:
    counts = {parcel_class: 1 for parcel_class in config.classes}
    return PreparedDataset(
        archive_source=tmp_path / "drive" / "archive.zip",
        manifest_source=tmp_path / "drive" / "manifest.csv",
        archive_sha256=ARCHIVE_HASH,
        manifest_sha256=MANIFEST_HASH,
        dataset_root=tmp_path / "content" / "dataset",
        class_order=config.classes,
        splits={
            split: DatasetSplitLayout(
                path=tmp_path / "content" / "dataset" / split.value,
                image_counts=counts,
            )
            for split in DatasetSplit
        },
    )


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeModel:
    def __init__(self) -> None:
        self._parameters = [_Parameter(), _Parameter()]
        self.features = SimpleNamespace(parameters=lambda: self._parameters[:1])
        self.loaded: dict[str, Any] | None = None

    def parameters(self) -> list[_Parameter]:
        return self._parameters

    def state_dict(self) -> dict[str, str]:
        return {"weight": "fake"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = state


class _FakeOptimizer:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.loaded: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, str]:
        return {"optimizer": "fake"}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loaded = state


def test_mobilenet_builds_conservative_train_and_deterministic_val_transforms() -> None:
    config = _config(ModelCandidate.MOBILENET_V3_SMALL)

    def operation(name: str) -> Any:
        return lambda *args, **kwargs: (name, args, kwargs)

    transforms = SimpleNamespace(
        InterpolationMode=SimpleNamespace(BILINEAR="bilinear"),
        ToTensor=operation("tensor"),
        Normalize=operation("normalize"),
        Compose=lambda items: tuple(items),
        Resize=operation("resize"),
        RandomAffine=operation("affine"),
        ColorJitter=operation("jitter"),
        RandomHorizontalFlip=operation("horizontal"),
        RandomVerticalFlip=operation("vertical"),
    )
    train, validation = mobilenet_training._build_transforms(
        config, SimpleNamespace(transforms=transforms)
    )

    assert [item[0] for item in train] == [
        "resize",
        "affine",
        "jitter",
        "horizontal",
        "vertical",
        "tensor",
        "normalize",
    ]
    assert [item[0] for item in validation] == ["resize", "tensor", "normalize"]


def test_mobilenet_loaders_remap_framework_classes_to_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.MOBILENET_V3_SMALL)
    prepared = _prepared(tmp_path, config)
    discovered_mapping = {"mop": 0, "ok": 1, "rach": 2, "uot": 3, "vo": 4}
    datasets: list[Any] = []

    class FakeDataset:
        def __init__(self, path: str, *, transform: Any) -> None:
            self.path = path
            self.transform = transform
            self.class_to_idx = discovered_mapping
            self.target_transform: Any | None = None
            datasets.append(self)

    class FakeGenerator:
        def manual_seed(self, seed: int) -> FakeGenerator:
            assert seed == 42
            return self

    loaders: list[tuple[Any, dict[str, Any]]] = []

    def fake_loader(dataset: Any, **options: Any) -> tuple[Any, dict[str, Any]]:
        result = (dataset, options)
        loaders.append(result)
        return result

    fake_torch = SimpleNamespace(
        Generator=FakeGenerator,
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=fake_loader)),
    )
    fake_torchvision = SimpleNamespace(datasets=SimpleNamespace(ImageFolder=FakeDataset))
    monkeypatch.setattr(
        mobilenet_training,
        "_build_transforms",
        lambda *_: ("train-transform", "val-transform"),
    )

    train_loader, validation_loader = mobilenet_training._build_loaders(
        config,
        prepared,
        torch=fake_torch,
        torchvision=fake_torchvision,
    )

    assert train_loader == loaders[0]
    assert validation_loader == loaders[1]
    assert loaders[0][1]["shuffle"] is True
    assert loaders[1][1]["shuffle"] is False
    target_transform = datasets[0].target_transform
    assert target_transform is not None
    assert target_transform(0) == 0
    assert target_transform(1) == 1


def test_mobilenet_builds_exact_pinned_classifier_head() -> None:
    config = _config(ModelCandidate.MOBILENET_V3_SMALL)

    class FakeNetwork:
        def __init__(self) -> None:
            self.classifier = [SimpleNamespace(in_features=128)]

        def cuda(self) -> FakeNetwork:
            return self

    network = FakeNetwork()
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Linear=lambda inputs, outputs: (inputs, outputs))
    )
    fake_torchvision = SimpleNamespace(
        models=SimpleNamespace(
            MobileNet_V3_Small_Weights=SimpleNamespace(IMAGENET1K_V1="weights"),
            mobilenet_v3_small=lambda *, weights: network,
        )
    )

    built = mobilenet_training._build_model(config, torch=fake_torch, torchvision=fake_torchvision)

    assert built is network
    assert network.classifier[-1] == (128, 5)

    wrong = config.model_copy(update={"pretrained_checkpoint": "unreviewed"})
    with pytest.raises(TrainingRuntimeError, match="pinned"):
        mobilenet_training._build_model(wrong, torch=fake_torch, torchvision=fake_torchvision)


def test_mobilenet_two_phase_runner_writes_periodic_and_best_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.MOBILENET_V3_SMALL)
    prepared = _prepared(tmp_path, config)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(CrossEntropyLoss=lambda: object()),
        amp=SimpleNamespace(GradScaler=lambda *_args, **_kwargs: object()),
        optim=SimpleNamespace(AdamW=_FakeOptimizer),
    )
    model = _FakeModel()
    monkeypatch.setattr(mobilenet_training, "assert_colab_gpu_runtime", lambda: fake_torch)
    monkeypatch.setattr(
        "parcel_sorter.ml.mobilenet_training.importlib.import_module", lambda _: object()
    )
    monkeypatch.setattr(mobilenet_training, "seed_training_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(mobilenet_training, "_build_loaders", lambda *_a, **_k: (1, 2))
    monkeypatch.setattr(mobilenet_training, "_build_model", lambda *_a, **_k: model)

    def fake_loader(*_args: Any, training: bool, **_kwargs: Any) -> Any:
        if training:
            return 0.5, [], []
        return 0.4, [1, 0], [1, 0]

    monkeypatch.setattr(mobilenet_training, "_run_loader", fake_loader)
    monkeypatch.setattr(mobilenet_training, "append_epoch_metrics", lambda *_: None)

    def fake_save(_: Any, path: Path, __: dict[str, Any]) -> None:
        path.write_bytes(b"fake checkpoint")

    monkeypatch.setattr(mobilenet_training, "_save_checkpoint_atomic", fake_save)

    best = mobilenet_training.train_mobilenet_candidate(
        config=config,
        prepared=prepared,
        run_dir=run_dir,
    )

    assert best == run_dir / "checkpoints" / "best.pt"
    assert best.is_file()
    assert all(parameter.requires_grad for parameter in model._parameters)
    assert (run_dir / "metrics" / "training_summary.json").is_file()


def test_mobilenet_runner_rejects_wrong_candidate(tmp_path: Path) -> None:
    config = _config(ModelCandidate.YOLOV8N_CLS)
    with pytest.raises(TrainingRuntimeError, match="wrong candidate"):
        mobilenet_training.train_mobilenet_candidate(
            config=config,
            prepared=_prepared(tmp_path, config),
            run_dir=tmp_path,
        )


class _FakeYolo:
    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        self.names = {0: "mop", 1: "ok", 2: "rach", 3: "uot", 4: "vo"}

    def train(self, **options: Any) -> None:
        output = Path(options["project"]) / str(options["name"]) / "weights"
        output.mkdir(parents=True, exist_ok=True)
        (output / "best.pt").write_bytes(b"fake yolo")


def test_yolo_runner_uses_pinned_package_and_artifact_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.YOLOV8N_CLS)
    monkeypatch.setattr("parcel_sorter.ml.yolo_training.metadata.version", lambda _: "8.4.191")
    monkeypatch.setattr(yolo_training, "assert_colab_gpu_runtime", lambda: object())
    monkeypatch.setattr(yolo_training, "seed_training_runtime", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "parcel_sorter.ml.yolo_training.importlib.import_module",
        lambda _: SimpleNamespace(YOLO=_FakeYolo),
    )

    best = yolo_training.train_yolov8n_candidate(
        config=config,
        prepared=_prepared(tmp_path, config),
        run_dir=tmp_path / "run",
    )

    assert best.is_file()
    assert (tmp_path / "run" / "metrics" / "training_summary.json").is_file()


def test_yolo_runner_rejects_wrong_candidate_checkpoint_and_version(tmp_path: Path) -> None:
    mobile = _config(ModelCandidate.MOBILENET_V3_SMALL)
    with pytest.raises(TrainingRuntimeError, match="wrong candidate"):
        yolo_training.train_yolov8n_candidate(
            config=mobile,
            prepared=_prepared(tmp_path, mobile),
            run_dir=tmp_path,
        )

    yolo = _config(ModelCandidate.YOLOV8N_CLS)
    wrong_checkpoint = yolo.model_copy(update={"pretrained_checkpoint": "newer.pt"})
    with pytest.raises(TrainingRuntimeError, match="pinned"):
        yolo_training.train_yolov8n_candidate(
            config=wrong_checkpoint,
            prepared=_prepared(tmp_path, yolo),
            run_dir=tmp_path,
        )


def _patch_colab_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drive_root: Path,
    prepared: PreparedDataset,
) -> None:
    monkeypatch.setattr(colab_training, "assert_colab_gpu_runtime", lambda: object())
    monkeypatch.setattr(colab_training, "validate_colab_drive_root", lambda _: drive_root)
    monkeypatch.setattr(colab_training, "collect_git_state", lambda _: (GIT_COMMIT, False))
    monkeypatch.setattr(colab_training, "_prepare_dataset", lambda *_a, **_k: prepared)
    monkeypatch.setattr(colab_training, "collect_package_versions", lambda: {"torch": "fake"})
    monkeypatch.setattr(
        colab_training,
        "collect_accelerator_info",
        lambda: AcceleratorInfo(
            torch_version="fake",
            cuda_available=True,
            cuda_version="fake",
            device_name="fake",
        ),
    )


def test_colab_orchestrator_initializes_and_resumes_exact_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.MOBILENET_V3_SMALL)
    drive_root = tmp_path / "drive"
    prepared = _prepared(tmp_path, config)
    _patch_colab_orchestrator(monkeypatch, drive_root=drive_root, prepared=prepared)

    def fake_mobile(**options: Any) -> Path:
        path = Path(options["run_dir"]) / "checkpoints" / "best.pt"
        path.write_bytes(b"best")
        return path

    monkeypatch.setattr(colab_training, "train_mobilenet_candidate", fake_mobile)
    run_dir, best = colab_training.train_candidate_on_colab(
        config=config,
        repo_dir=tmp_path / "repo",
        drive_root="ignored",
        archive_relative_path="archive.zip",
        manifest_relative_path="manifest.csv",
        local_workspace=Path("/content/mobile"),
    )
    assert best.is_file()

    resume_checkpoint = run_dir / "checkpoints" / "epoch-0001.pt"
    resume_checkpoint.write_bytes(b"resume")
    resumed_dir, _ = colab_training.train_candidate_on_colab(
        config=config,
        repo_dir=tmp_path / "repo",
        drive_root="ignored",
        archive_relative_path="archive.zip",
        manifest_relative_path="manifest.csv",
        local_workspace=Path("/content/mobile-resume"),
        resume_run_id=run_dir.name,
        resume_checkpoint_relative_path="checkpoints/epoch-0001.pt",
    )
    assert resumed_dir == run_dir


def test_colab_orchestrator_dispatches_yolo_and_rejects_dirty_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.YOLOV8N_CLS)
    drive_root = tmp_path / "drive"
    prepared = _prepared(tmp_path, config)
    _patch_colab_orchestrator(monkeypatch, drive_root=drive_root, prepared=prepared)

    def fake_yolo(**options: Any) -> Path:
        path = Path(options["run_dir"]) / "checkpoints" / "best.pt"
        path.write_bytes(b"best")
        return path

    monkeypatch.setattr(colab_training, "train_yolov8n_candidate", fake_yolo)
    _, best = colab_training.train_candidate_on_colab(
        config=config,
        repo_dir=tmp_path / "repo",
        drive_root="ignored",
        archive_relative_path="archive.zip",
        manifest_relative_path="manifest.csv",
        local_workspace=Path("/content/yolo"),
    )
    assert best.is_file()

    monkeypatch.setattr(colab_training, "collect_git_state", lambda _: (GIT_COMMIT, True))
    with pytest.raises(TrainingRuntimeError, match="dirty"):
        colab_training.train_candidate_on_colab(
            config=config,
            repo_dir=tmp_path / "repo",
            drive_root="ignored",
            archive_relative_path="archive.zip",
            manifest_relative_path="manifest.csv",
            local_workspace=Path("/content/dirty"),
        )


def test_colab_workspace_guard_rejects_paths_outside_content(tmp_path: Path) -> None:
    colab_training._validate_local_workspace(Path("/content/model-work"))
    with pytest.raises(TrainingRuntimeError, match="below Colab"):
        colab_training._validate_local_workspace(tmp_path / "outside")
