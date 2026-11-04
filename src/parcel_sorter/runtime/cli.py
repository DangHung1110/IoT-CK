"""Command-line interface for local image and webcam inference."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated, Any

import typer

from parcel_sorter.runtime.inference import OnnxBinaryClassifier, annotate_frame

app = typer.Typer(no_args_is_help=True, help="Run the frozen binary parcel model locally.")


def _load_cv2() -> Any:
    return importlib.import_module("cv2")


@app.command()
def image(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    source: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(help="Optional annotated image path.")] = None,
    show: Annotated[bool, typer.Option(help="Open a preview window.")] = False,
) -> None:
    """Classify one image."""
    cv2 = _load_cv2()
    frame = cv2.imread(str(source))
    if frame is None:
        raise typer.BadParameter(f"cannot decode image: {source}", param_hint="source")
    classifier = OnnxBinaryClassifier(bundle)
    result = classifier.predict_bgr(frame)
    annotated = annotate_frame(frame, result)
    typer.echo(result.model_dump_json(indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), annotated):
            raise typer.BadParameter(f"cannot write output image: {output}", param_hint="output")
        typer.echo(f"Annotated image: {output.resolve()}")
    if show:
        cv2.imshow("Parcel sorter inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


@app.command()
def webcam(
    bundle: Annotated[Path, typer.Option(exists=True, file_okay=False, dir_okay=True)],
    camera: Annotated[int, typer.Option(min=0, help="OpenCV camera index.")] = 0,
) -> None:
    """Run live inference until Q or Escape is pressed."""
    cv2 = _load_cv2()
    classifier = OnnxBinaryClassifier(bundle)
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        raise typer.BadParameter(f"cannot open camera index {camera}", param_hint="camera")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("webcam stopped returning frames")
            result = classifier.predict_bgr(frame)
            cv2.imshow("Parcel sorter - Q/Esc to quit", annotate_frame(frame, result))
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Run the Typer application."""
    app()


if __name__ == "__main__":
    main()
