from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def _load_notebook() -> dict[str, Any]:
    path = Path(__file__).parents[3] / "notebooks" / "02_colab_evaluate_candidates.ipynb"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_evaluation_notebook_is_clean_gpu_only_and_leakage_safe() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    cells_by_id = {cell["id"]: "".join(cell["source"]) for cell in cells}
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    ids = [cell["id"] for cell in cells]

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert all(cell.get("execution_count") is None for cell in cells)
    assert all(not cell.get("outputs") for cell in cells)
    assert 'GIT_REF = "feat/model-training-pipeline"' in code
    assert "20261103t153037z-mobilenet_v3_small-v0-s42-7c300430" in code
    assert "20261103t153335z-yolov8n_cls-v0-s42-7c300430" in code
    assert 'src_dir = str(repo_dir / "src")' in code
    assert "sys.path.insert(0, src_dir)" in code
    assert "validate_binary_run_pair" in code
    assert "stage_verified_dataset" in code
    assert "select_binary_validation_thresholds" in code
    assert "select_binary_model_candidate" in code
    assert "save_binary_prediction_table" in code
    assert '"quality_status": "smoke_only"' in code
    assert "test_confusion_matrices.png" in code
    assert ".train(" not in code
    assert "fit(" not in code

    validation_index = ids.index("validation-predict-and-freeze-selection")
    test_index = ids.index("test-evaluate")
    assert validation_index < test_index
    assert "DatasetSplit.TEST" not in cells_by_id["validation-predict-and-freeze-selection"]
    assert "DatasetSplit.TEST" in cells_by_id["test-evaluate"]
    assert "selection_frozen.json" in cells_by_id["validation-predict-and-freeze-selection"]
    assert "select_binary_model_candidate" not in cells_by_id["test-evaluate"]
