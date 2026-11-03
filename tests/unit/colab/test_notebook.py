"""Static safety checks for the user-run Colab bootstrap notebook."""

import json
import tomllib
from pathlib import Path
from typing import Any, cast


def load_notebook() -> dict[str, Any]:
    notebook_path = Path(__file__).parents[3] / "notebooks" / "00_colab_drive_bootstrap.ipynb"
    content = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError("notebook root must be a JSON object")
    return cast(dict[str, Any], content)


def test_notebook_is_clean_and_has_no_training_commands() -> None:
    notebook = load_notebook()
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert "drive.mount" in code
    assert "requirements/colab.lock.txt" not in code
    assert "sys.version_info" in code
    assert "pydantic_version" in code
    assert "--no-deps" in code
    assert "bootstrap_colab" in code
    assert "REPORT_PATH=" in code

    lowered = code.lower()
    for forbidden in (".train(", ".fit(", "yolo train", "model.export("):
        assert forbidden not in lowered


def test_project_metadata_supports_current_colab_python() -> None:
    pyproject_path = Path(__file__).parents[3] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["requires-python"] == ">=3.11,<3.14"
