from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import HttpUrl

from parcel_sorter.colab import AcceleratorInfo
from parcel_sorter.data import BinaryParcelClass, DatasetSplit, ModelClass
from parcel_sorter.ml import colab_evaluation
from parcel_sorter.ml.colab_data import DatasetSplitLayout, PreparedDataset
from parcel_sorter.ml.colab_evaluation import (
    _mobilenet_model,
    predict_binary_candidate,
    predict_mobilenet_binary,
    predict_yolo_binary,
    resolve_best_checkpoint,
    validate_binary_run_pair,
)
from parcel_sorter.ml.contracts import ExperimentConfig, ModelCandidate, PreprocessContract
from parcel_sorter.ml.run_metadata import (
    DatasetRunIdentity,
    PretrainedCheckpointEvidence,
    TrainingRunManifest,
)
from parcel_sorter.ml.training_runtime import TrainingRuntimeError

ARCHIVE_HASH = "a" * 64
MANIFEST_HASH = "b" * 64
GIT_COMMIT = "c" * 40
CLASSES = (BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL)


def _config(candidate: ModelCandidate) -> ExperimentConfig:
    mobile = candidate is ModelCandidate.MOBILENET_V3_SMALL
    return ExperimentConfig(
        candidate=candidate,
        dataset_version="v0",
        dataset_manifest_sha256=MANIFEST_HASH,
        dataset_archive_sha256=ARCHIVE_HASH,
        split_seed=42,
        seed=42,
        classes=CLASSES,
        preprocess=PreprocessContract(
            preprocess_id="mobile-rgb320-v1" if mobile else "yolo-rgb320-v1",
            resize="resize_square",
            mean=(0.485, 0.456, 0.406) if mobile else (0.0, 0.0, 0.0),
            std=(0.229, 0.224, 0.225) if mobile else (1.0, 1.0, 1.0),
        ),
        pretrained_checkpoint=(
            "MobileNet_V3_Small_Weights.IMAGENET1K_V1" if mobile else "yolov8n-cls.pt"
        ),
    )


def _dataset() -> DatasetRunIdentity:
    counts: dict[ModelClass, int] = {
        BinaryParcelClass.DAMAGE: 10,
        BinaryParcelClass.NORMAL: 10,
    }
    return DatasetRunIdentity(
        archive_source=Path("/content/drive/MyDrive/data/archive.zip"),
        manifest_source=Path("/content/drive/MyDrive/data/manifest.csv"),
        archive_sha256=ARCHIVE_HASH,
        manifest_sha256=MANIFEST_HASH,
        class_order=CLASSES,
        image_counts={split: counts for split in DatasetSplit},
    )


def _manifest(candidate: ModelCandidate) -> TrainingRunManifest:
    config = _config(candidate)
    return TrainingRunManifest(
        run_id=f"run-{candidate.value}",
        created_at_utc=datetime(2026, 11, 3, tzinfo=UTC),
        git_commit=GIT_COMMIT,
        git_dirty=False,
        config=config,
        dataset=_dataset(),
        checkpoint=PretrainedCheckpointEvidence(
            name=config.pretrained_checkpoint,
            source_url=HttpUrl("https://example.com/checkpoint"),
            license_name="test-license",
            license_url=HttpUrl("https://example.com/license"),
        ),
        packages={},
        accelerator=AcceleratorInfo(
            torch_version="test",
            cuda_available=True,
            cuda_version="test",
            device_name="test",
        ),
    )


def test_binary_run_pair_requires_same_clean_identity() -> None:
    mobile = _manifest(ModelCandidate.MOBILENET_V3_SMALL)
    yolo = _manifest(ModelCandidate.YOLOV8N_CLS)

    pair = validate_binary_run_pair((mobile, yolo))
    assert pair[ModelCandidate.MOBILENET_V3_SMALL] is mobile
    assert pair[ModelCandidate.YOLOV8N_CLS] is yolo

    with pytest.raises(TrainingRuntimeError, match="exactly two"):
        validate_binary_run_pair((mobile,))
    with pytest.raises(TrainingRuntimeError, match="dirty checkout"):
        validate_binary_run_pair((mobile.model_copy(update={"git_dirty": True}), yolo))
    with pytest.raises(TrainingRuntimeError, match="different Git commits"):
        validate_binary_run_pair((mobile, yolo.model_copy(update={"git_commit": "d" * 40})))
    changed_dataset = yolo.dataset.model_copy(update={"archive_sha256": "e" * 64})
    with pytest.raises(TrainingRuntimeError, match="dataset identity"):
        validate_binary_run_pair((mobile, yolo.model_copy(update={"dataset": changed_dataset})))


def test_best_checkpoint_path_is_candidate_specific(tmp_path: Path) -> None:
    mobile_run = tmp_path / "mobile"
    yolo_run = tmp_path / "yolo"
    mobile_checkpoint = mobile_run / "checkpoints" / "best.pt"
    yolo_checkpoint = yolo_run / "yolo" / "weights" / "best.pt"
    mobile_checkpoint.parent.mkdir(parents=True)
    yolo_checkpoint.parent.mkdir(parents=True)
    mobile_checkpoint.write_bytes(b"mobile")
    yolo_checkpoint.write_bytes(b"yolo")

    assert (
        resolve_best_checkpoint(mobile_run, ModelCandidate.MOBILENET_V3_SMALL) == mobile_checkpoint
    )
    assert resolve_best_checkpoint(yolo_run, ModelCandidate.YOLOV8N_CLS) == yolo_checkpoint
    with pytest.raises(TrainingRuntimeError, match="missing"):
        resolve_best_checkpoint(tmp_path / "missing", ModelCandidate.MOBILENET_V3_SMALL)


def _prepared(tmp_path: Path) -> PreparedDataset:
    dataset_root = tmp_path / "dataset"
    splits: dict[DatasetSplit, DatasetSplitLayout] = {}
    counts: dict[ModelClass, int] = {
        BinaryParcelClass.DAMAGE: 1,
        BinaryParcelClass.NORMAL: 1,
    }
    for split in DatasetSplit:
        split_path = dataset_root / split.value
        for parcel_class in CLASSES:
            class_path = split_path / parcel_class.value
            class_path.mkdir(parents=True)
            (class_path / f"{split.value}-{parcel_class.value}.png").write_bytes(b"fake")
        splits[split] = DatasetSplitLayout(path=split_path, image_counts=counts)
    return PreparedDataset(
        archive_source=tmp_path / "archive.zip",
        manifest_source=tmp_path / "manifest.csv",
        archive_sha256=ARCHIVE_HASH,
        manifest_sha256=MANIFEST_HASH,
        dataset_root=dataset_root,
        class_order=CLASSES,
        splits=splits,
    )


class _FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = values

    def cuda(self, *, non_blocking: bool) -> _FakeTensor:
        assert non_blocking
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> Any:
        return self.values


class _NoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None


def test_mobilenet_checkpoint_and_prediction_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.MOBILENET_V3_SMALL)

    class FakeModel:
        def __init__(self) -> None:
            self.classifier: list[Any] = [SimpleNamespace(in_features=8)]
            self.loaded: dict[str, str] | None = None

        def load_state_dict(self, state: dict[str, str]) -> None:
            self.loaded = state

        def cuda(self) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    model = FakeModel()
    payload = {
        "config": config.model_dump(mode="json"),
        "class_order": [item.value for item in config.classes],
        "model_state": {"weight": "fake"},
    }
    fake_torch_for_model = SimpleNamespace(
        nn=SimpleNamespace(Linear=lambda inputs, outputs: (inputs, outputs)),
        load=lambda *_args, **_kwargs: payload,
    )
    fake_torchvision_for_model = SimpleNamespace(
        models=SimpleNamespace(mobilenet_v3_small=lambda *, weights: model)
    )
    assert (
        _mobilenet_model(
            config,
            tmp_path / "best.pt",
            torch=fake_torch_for_model,
            torchvision=fake_torchvision_for_model,
        )
        is model
    )
    assert model.classifier[-1] == (8, 2)
    assert model.loaded == {"weight": "fake"}

    prepared = _prepared(tmp_path)

    class FakeDataset:
        def __init__(self, path: str, *, transform: Any) -> None:
            self.transform = transform
            split_path = Path(path)
            self.class_to_idx = {"damage": 0, "normal": 1}
            self.samples = [
                (str(split_path / "damage" / "val-damage.png"), 0),
                (str(split_path / "normal" / "val-normal.png"), 1),
            ]

        def __len__(self) -> int:
            return len(self.samples)

    def operation(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args, kwargs

    fake_torchvision = SimpleNamespace(
        transforms=SimpleNamespace(
            Compose=lambda items: items,
            Resize=operation,
            ToTensor=operation,
            Normalize=operation,
            InterpolationMode=SimpleNamespace(BILINEAR="bilinear"),
        ),
        datasets=SimpleNamespace(ImageFolder=FakeDataset),
    )
    fake_torch = SimpleNamespace(
        utils=SimpleNamespace(
            data=SimpleNamespace(
                DataLoader=lambda *_args, **_kwargs: [(_FakeTensor("images"), _FakeTensor([0, 1]))]
            )
        ),
        no_grad=_NoGrad,
        softmax=lambda *_args, **_kwargs: _FakeTensor([[0.9, 0.1], [0.2, 0.8]]),
    )
    monkeypatch.setattr(colab_evaluation, "assert_colab_gpu_runtime", lambda: fake_torch)
    monkeypatch.setattr(
        "parcel_sorter.ml.colab_evaluation.importlib.import_module",
        lambda _: fake_torchvision,
    )
    monkeypatch.setattr(
        colab_evaluation,
        "_mobilenet_model",
        lambda *_args, **_kwargs: lambda _images: "logits",
    )

    records = predict_mobilenet_binary(
        config=config,
        prepared=prepared,
        checkpoint=tmp_path / "best.pt",
        split=DatasetSplit.VAL,
    )
    assert [record.true_class for record in records] == list(CLASSES)
    assert records[0].probabilities == (0.9, 0.1)
    assert records[1].image_id == "val/normal/val-normal.png"


def test_yolo_prediction_and_candidate_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(ModelCandidate.YOLOV8N_CLS)
    prepared = _prepared(tmp_path)

    class FakeData:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def detach(self) -> FakeData:
            return self

        def cpu(self) -> FakeData:
            return self

        def tolist(self) -> list[float]:
            return self.values

    class FakeYolo:
        names: ClassVar[dict[int, str]] = {0: "damage", 1: "normal"}

        def __init__(self, checkpoint: str) -> None:
            self.checkpoint = checkpoint

        def predict(self, **options: Any) -> list[Any]:
            paths = [Path(item) for item in options["source"]]
            return [
                SimpleNamespace(
                    path=str(path),
                    probs=SimpleNamespace(
                        data=FakeData([0.9, 0.1] if path.parent.name == "damage" else [0.1, 0.9])
                    ),
                )
                for path in paths
            ]

    monkeypatch.setattr(colab_evaluation, "assert_colab_gpu_runtime", lambda: object())
    monkeypatch.setattr("parcel_sorter.ml.colab_evaluation.metadata.version", lambda _: "8.4.191")
    monkeypatch.setattr(
        "parcel_sorter.ml.colab_evaluation.importlib.import_module",
        lambda _: SimpleNamespace(YOLO=FakeYolo),
    )
    records = predict_yolo_binary(
        config=config,
        prepared=prepared,
        checkpoint=tmp_path / "best.pt",
        split=DatasetSplit.TEST,
    )
    assert len(records) == 2
    assert {record.true_class for record in records} == set(CLASSES)
    assert all(record.split is DatasetSplit.TEST for record in records)

    monkeypatch.setattr(colab_evaluation, "predict_yolo_binary", lambda **_: records)
    assert predict_binary_candidate(
        config=config,
        prepared=prepared,
        checkpoint=tmp_path / "best.pt",
        split=DatasetSplit.TEST,
    ) == tuple(records)
