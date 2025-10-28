import os
import subprocess

notebook = os.path.join(os.path.dirname(__file__), "hello.ipynb")


def test_run_func():
    cmd = f"runnb {notebook} hello"
    subprocess.run(cmd.split(" "))


def test_run_test():
    cmd = f"runnb {notebook} runtest"
    subprocess.run(cmd.split(" "))
