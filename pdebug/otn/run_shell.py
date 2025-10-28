import os
import random
import string
import subprocess
import sys
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.utils.fileio import run_with_print

import typer


@otn_manager.NODE.register(name="run_shell")
def main(
    path: str,
    shell: str = None,
    output: str = None,
    debug: bool = False,
    **kwargs,
):
    """Run shell."""
    assert shell, "Please set shell."
    if not output:
        output = path
    cmd = shell.format(path=path, output=output, **kwargs)
    typer.echo(typer.style(f"RunShell:\n {cmd}", fg=typer.colors.YELLOW))

    file_id = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=8)
    )
    tmp_shell = f"/tmp/.tmp_run_{file_id}.sh"
    with open(tmp_shell, "w") as fid:
        fid.write(cmd)

    run_with_print(["bash", tmp_shell], get_return_from_cmd=False)

    if not debug:
        os.system(f"rm {tmp_shell}")
    return output


if __name__ == "__main__":
    typer.run(main)
