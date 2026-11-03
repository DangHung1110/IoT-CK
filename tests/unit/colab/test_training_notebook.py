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
    assert "--no-deps" in code
    assert '"requirements" / "colab-training-overlay.txt"' in code
    assert "torch.save" not in code
    assert ".train(" not in code


def test_colab_overlay_is_pinned_and_does_not_replace_torch() -> None:
    path = Path(__file__).parents[3] / "requirements" / "colab-training-overlay.txt"
    content = path.read_text(encoding="utf-8")

    assert "ultralytics==8.4.191" in content
    assert "ultralytics-thop==2.2.3" in content
    assert "polars==" not in content
    assert "torch==" not in content
    assert "torchvision==" not in content
