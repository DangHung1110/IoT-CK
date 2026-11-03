from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def _load_training_notebook() -> dict[str, Any]:
    path = Path(__file__).parents[3] / "notebooks" / "01_colab_train_candidates.ipynb"
    content = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], content)


def test_training_notebook_is_clean_and_delegates_to_source() -> None:
    notebook = _load_training_notebook()
    cells_by_id = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert "drive.mount" in code
    assert 'GIT_REF = "feat/model-training-pipeline"' in code
    assert "train_candidate_on_colab" in code
    assert "mobilenet_v3_small,yolov8n_cls" in code
    assert "DATASET_ARCHIVE_SHA256" in code
    assert "DATASET_MANIFEST_SHA256" in code
    assert "BinaryParcelClass" in code
    assert 'DATASET_VERSION = "v0"' in code
    assert "dataset_v0_splits.zip" in code
    assert 'src_dir = str(repo_dir / "src")' in code
    assert "sys.path.insert(0, src_dir)" in code
    assert 'module_name == "parcel_sorter"' in code
    assert 'module_name.startswith("parcel_sorter.")' in code
    assert "del sys.modules[module_name]" in code
    assert "importlib.invalidate_caches()" in code
    assert 'importlib.import_module("parcel_sorter")' in code
    assert "package_path.is_relative_to(repo_dir.resolve())" in code
    assert 'importlib.import_module("parcel_sorter.ml.contracts")' in code
    assert "HEAD_EPOCHS = 5" in code
    assert "FINETUNE_EPOCHS = 45" in code
    assert "PATIENCE = 7" in code
    assert "head_epochs=HEAD_EPOCHS" in code
    assert "finetune_epochs=FINETUNE_EPOCHS" in code
    assert "patience=PATIENCE" in code
    assert "--no-deps" in code
    assert '"requirements" / "colab-training-overlay.txt"' in code
    assert "torch.save" not in code
    assert ".train(" not in code

    validation_code = cells_by_id["validate-inputs"]
    assert 'validation_src_dir = repo_dir / "src"' in validation_code
    assert "validation_src_dir.is_dir()" in validation_code
    assert "sys.path.insert(0, validation_src)" in validation_code
    assert "del sys.modules[module_name]" in validation_code
    assert "importlib.invalidate_caches()" in validation_code
    assert 'importlib.import_module("parcel_sorter.ml.contracts")' in validation_code
    assert 'importlib.import_module("parcel_sorter.ml.yolo_training")' in validation_code


def test_colab_overlay_is_pinned_and_does_not_replace_torch() -> None:
    path = Path(__file__).parents[3] / "requirements" / "colab-training-overlay.txt"
    content = path.read_text(encoding="utf-8")

    assert "ultralytics==8.4.191" in content
    assert "ultralytics-thop==2.2.3" in content
    assert "polars==" not in content
    assert "torch==" not in content
    assert "torchvision==" not in content
