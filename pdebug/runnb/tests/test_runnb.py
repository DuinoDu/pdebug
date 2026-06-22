import os
import subprocess
import sys

notebook = os.path.join(os.path.dirname(__file__), "hello.ipynb")


def test_run_func(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pdebug.runnb.runnb",
            notebook,
            "hello",
            "--cache",
            str(tmp_path),
        ],
        check=True,
    )


def test_run_test(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pdebug.runnb.runnb",
            notebook,
            "runtest",
            "--cache",
            str(tmp_path),
        ],
        check=True,
    )
