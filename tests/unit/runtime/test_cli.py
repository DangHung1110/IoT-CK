from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from parcel_sorter.data import BinaryParcelClass
from parcel_sorter.ml.binary_evaluation import BinaryDecisionCode, BinaryThresholds
from parcel_sorter.runtime import cli
from parcel_sorter.runtime.inference import BinaryInferenceResult


def _result() -> BinaryInferenceResult:
    return BinaryInferenceResult(
        probabilities={
            BinaryParcelClass.DAMAGE: 0.9,
            BinaryParcelClass.NORMAL: 0.1,
        },
        decision=BinaryDecisionCode.DAMAGE,
        latency_ms=12.5,
    )


class _FakeClassifier:
    def __init__(self, bundle: Path) -> None:
        self.bundle = bundle
        self.manifest = SimpleNamespace(
            classes=(BinaryParcelClass.DAMAGE, BinaryParcelClass.NORMAL),
            thresholds=BinaryThresholds(tau_normal_pass=0.95, tau_damage_confirm=0.5),
        )

    def predict_bgr(self, frame: Any) -> BinaryInferenceResult:
        return _result()


class _FakeCapture:
    def __init__(
        self, *, opened: bool = True, frames: Iterator[tuple[bool, Any]] | None = None
    ) -> None:
        self.opened = opened
        self.frames = frames or iter([(True, np.zeros((32, 32, 3), dtype=np.uint8))])
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, Any]:
        return next(self.frames)

    def release(self) -> None:
        self.released = True


class _FakeCv2:
    def __init__(
        self,
        *,
        decoded: bool = True,
        write_ok: bool = True,
        capture: _FakeCapture | None = None,
    ) -> None:
        self.decoded = decoded
        self.write_ok = write_ok
        self.capture = capture or _FakeCapture()
        self.shown = 0
        self.destroyed = False
        self.written: Path | None = None

    def imread(self, path: str) -> Any:
        if not self.decoded:
            return None
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def imwrite(self, path: str, frame: Any) -> bool:
        self.written = Path(path)
        return self.write_ok

    def imshow(self, title: str, frame: Any) -> None:
        self.shown += 1

    def waitKey(self, delay: int) -> int:
        return ord("q")

    def destroyAllWindows(self) -> None:
        self.destroyed = True

    def VideoCapture(self, camera: int) -> _FakeCapture:
        return self.capture

    def flip(self, frame: Any, axis: int) -> Any:
        return frame.copy()


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = tmp_path / "parcel.jpg"
    source.write_bytes(b"image-placeholder")
    return bundle, source


def test_image_command_prints_result_writes_and_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _paths(tmp_path)
    output = tmp_path / "nested" / "annotated.jpg"
    fake_cv2 = _FakeCv2()
    monkeypatch.setattr(cli, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(cli, "OnnxBinaryClassifier", _FakeClassifier)
    monkeypatch.setattr(cli, "annotate_frame", lambda frame, result: frame.copy())

    response = CliRunner().invoke(
        cli.app,
        [
            "image",
            "--bundle",
            str(bundle),
            str(source),
            "--output",
            str(output),
            "--show",
        ],
    )

    assert response.exit_code == 0, response.output
    assert '"decision": "DAMAGE"' in response.output
    assert "Annotated image:" in response.output
    assert fake_cv2.written == output
    assert fake_cv2.shown == 1
    assert fake_cv2.destroyed


@pytest.mark.parametrize(
    ("decoded", "write_ok", "message"),
    [
        (False, True, "cannot decode image"),
        (True, False, "cannot write output image"),
    ],
)
def test_image_command_reports_io_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decoded: bool,
    write_ok: bool,
    message: str,
) -> None:
    bundle, source = _paths(tmp_path)
    fake_cv2 = _FakeCv2(decoded=decoded, write_ok=write_ok)
    monkeypatch.setattr(cli, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(cli, "OnnxBinaryClassifier", _FakeClassifier)
    monkeypatch.setattr(cli, "annotate_frame", lambda frame, result: frame)

    response = CliRunner().invoke(
        cli.app,
        ["image", "--bundle", str(bundle), str(source), "--output", str(tmp_path / "out.jpg")],
    )

    assert response.exit_code != 0
    assert message in response.output


def test_webcam_command_releases_camera_on_quit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _paths(tmp_path)
    capture = _FakeCapture()
    fake_cv2 = _FakeCv2(capture=capture)
    monkeypatch.setattr(cli, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(cli, "OnnxBinaryClassifier", _FakeClassifier)
    monkeypatch.setattr(cli, "annotate_roi_frame", lambda frame, roi, result: frame)

    response = CliRunner().invoke(cli.app, ["webcam", "--bundle", str(bundle)])

    assert response.exit_code == 0, response.output
    assert fake_cv2.shown == 1
    assert capture.released
    assert fake_cv2.destroyed


def test_webcam_command_rejects_unavailable_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _paths(tmp_path)
    capture = _FakeCapture(opened=False)
    fake_cv2 = _FakeCv2(capture=capture)
    monkeypatch.setattr(cli, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(cli, "OnnxBinaryClassifier", _FakeClassifier)

    response = CliRunner().invoke(cli.app, ["webcam", "--bundle", str(bundle), "--camera", "0"])

    assert response.exit_code != 0
    assert "cannot open camera" in response.output
    assert capture.released


def test_webcam_command_releases_camera_when_frame_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _ = _paths(tmp_path)
    capture = _FakeCapture(frames=iter([(False, None)]))
    fake_cv2 = _FakeCv2(capture=capture)
    monkeypatch.setattr(cli, "_load_cv2", lambda: fake_cv2)
    monkeypatch.setattr(cli, "OnnxBinaryClassifier", _FakeClassifier)

    response = CliRunner().invoke(cli.app, ["webcam", "--bundle", str(bundle)])

    assert response.exit_code != 0
    assert isinstance(response.exception, RuntimeError)
    assert capture.released
    assert fake_cv2.destroyed


def test_main_invokes_typer_app(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_app() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "app", fake_app)
    cli.main()
    assert called
