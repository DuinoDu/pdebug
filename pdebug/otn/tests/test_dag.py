import os

from pdebug.otn import manager as otn_manager

import pytest


def add(x, a=1):
    return x + a


def sub(x, b=2):
    return x - b


@pytest.mark.skipif(True, reason="TODO: fix bug")
def test_dag_simple(tmpdir):
    config_str = """Input = {}
Input["value"] = 1

Add = {}
Add["a"] = 100
Add["input"] = "Input.value"
Add["output"] = "add_output"

Sub = {}
Sub["b"] = 200
Sub["input"] = "add_output"
Sub["output"] = "sub_output"

Main = "dag"
"""
    otn_manager.NODE.register(name="add", obj=add)
    otn_manager.NODE.register(name="sub", obj=sub)

    config_file = os.path.join(tmpdir, "dag_config.py")
    with open(config_file, "w") as fid:
        fid.write(config_str)

    # os.system(f"piata-cli --config-file {config_file}")
    res = otn_manager.create("dag")(config_file)
    assert res["sub_output"] == 1 + 100 - 200
