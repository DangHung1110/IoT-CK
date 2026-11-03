from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def _load_notebook() -> dict[str, Any]:
    path = Path(__file__).parents[3] / "notebooks" / "data_01_kaggle_to_drive.ipynb"
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_kaggle_notebook_is_clean_secure_and_uses_tested_pipeline() -> None:
    notebook = _load_notebook()
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert 'GIT_REF = "feat/model-training-pipeline"' in code
    assert 'KAGGLE_DATASET_HANDLE = "ducnguyentien69/parcel-classification/versions/1"' in code
    assert 'CLASSES = ("damage", "normal")' in code
    assert 'GROUPING_STRATEGY = "relative_path"' in code
    assert 'GROUP_REGEX = ""' in code
    assert "grouping_strategy=GROUPING_STRATEGY" in code  # pragma: allowlist secret
    assert 'DATASET_VERSION = "v0"' in code
    assert 'userdata.get("KAGGLE_API_TOKEN")' in code
    assert "kagglehub==1.0.3" in code
    assert "pillow>=13" not in code
    assert "sys.executable" in code
    assert 'src_dir = str(repo_dir / "src")' in code
    assert "sys.path.insert(0, src_dir)" in code
    assert 'module_name == "parcel_sorter"' in code
    assert 'module_name.startswith("parcel_sorter.")' in code
    assert "del sys.modules[module_name]" in code
    assert "importlib.invalidate_caches()" in code
    assert 'importlib.import_module("parcel_sorter.data.kaggle_import")' in code
    assert "force_download=True" in code
    assert 'Path("/content/kaggle_download")' in code
    assert "prepare_kaggle_smoke_dataset" in code
    assert "persist_kaggle_smoke_dataset" in code
    assert "v0_kaggle_binary_smoke_only_not_final_kpi" in code
    assert "TRAIN_RATIO = 0.70" in code
    assert "VAL_RATIO = 0.15" in code
    assert "TEST_RATIO = 0.15" in code
    assert "print(kaggle_token)" not in code
    assert "kaggle.json" not in code
