from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def _load_notebook() -> dict[str, Any]:
    path = Path(__file__).parents[3] / "notebooks" / "03_colab_export_mobilenet_onnx.ipynb"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_export_notebook_is_clean_frozen_and_parity_gated() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    code = "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert all(cell.get("execution_count") is None for cell in cells)
    assert all(not cell.get("outputs") for cell in cells)
    assert 'GIT_REF = "feat/model-training-pipeline"' in code
    assert 'EVALUATION_ID = "20261104t015526z-binary-v0-7c300430"' in code
    assert "selection_frozen.json" in code
    assert 'selection["selection_basis"] != "validation_only"' in code
    assert 'selection["winner"] != ModelCandidate.MOBILENET_V3_SMALL.value' in code
    assert "winner_checkpoint_sha256" in code
    assert "torch.onnx.export" in code
    assert "onnx.checker.check_model" in code
    assert "verify_binary_onnx_parity" in code
    assert "PARITY_SAMPLES_PER_CLASS = 10" in code
    assert "BinaryOnnxBundleManifest" in code
    assert "model.onnx" in code
    assert "manifest.json" in code
    assert "make_archive" in code
    assert ".train(" not in code
    assert "fit(" not in code
