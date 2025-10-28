import os

from pdebug.otn import manager as otn_manager
from pdebug.utils.env import SECUREMR_INSTALLED

import typer

if SECUREMR_INSTALLED:
    import securemr


@otn_manager.NODE.register(name="onnx2qnn")
def main(
    input_path: str = None,
    output: str = "tmp_output",
    benchmark: bool = False,
    runs: int = 100,
):
    """Convert onnx to qnn, support benchmark."""
    assert SECUREMR_INSTALLED, "securemr is required."
    assert input_path.endswith(".onnx"), "Only support onnx file as input."

    input_path = os.path.abspath(input_path)
    qnn_model = securemr.onnx_to_qnn(input_path, output=output)
    if benchmark:
        qnn_model.set_target("android")
        qnn_model.benchmark(runs=runs, output_dir=output)
    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
