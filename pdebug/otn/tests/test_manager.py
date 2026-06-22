import subprocess
import sys

from pdebug.otn import cli as otn_cli
from pdebug.otn import manager as otn_manager

import pytest


def test_manager_import_registers_core_nodes_without_extensions():
    script = (
        "from pdebug.otn import manager as m\n"
        "assert 'dag' in m.NODE\n"
        "assert 'single_node' in m.NODE\n"
        "assert 'run_shell' in m.NODE\n"
        "assert not m._EXTENSIONS_LOADED\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_create_missing_node_loads_extensions_once(monkeypatch):
    calls = []

    def fake_find_node_from_folder(node_dir, skip=(), strict=True):
        calls.append((node_dir, strict))
        return []

    monkeypatch.setattr(
        otn_manager.NODE, "find_node_from_folder", fake_find_node_from_folder
    )
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)

    try:
        otn_manager.create("__missing_test_node__")
    except ValueError as exc:
        assert "Unknown node name" in str(exc)
    else:
        raise AssertionError("create() should fail for a missing node")

    assert len(calls) == len(otn_manager._EXTENSION_DIRS)
    assert all(strict is False for _, strict in calls)
    assert otn_manager._EXTENSIONS_LOADED

    calls.clear()
    try:
        otn_manager.create("__still_missing_test_node__")
    except ValueError:
        pass
    assert calls == []


def test_create_reports_skipped_extension_import_errors(monkeypatch):
    def fake_find_node_from_folder(node_dir, skip=(), strict=True):
        return [(f"{node_dir}/bad_node.py", RuntimeError("broken import"))]

    monkeypatch.setattr(
        otn_manager.NODE, "find_node_from_folder", fake_find_node_from_folder
    )
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(otn_manager, "_EXTENSION_LOAD_ERRORS", [])

    try:
        otn_manager.create("__missing_test_node__")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("create() should fail for a missing node")

    assert "Skipped extension modules" in message
    assert "RuntimeError: broken import" in message


def test_strict_extension_load_runs_after_non_strict_load(monkeypatch):
    calls = []

    def fake_find_node_from_folder(node_dir, skip=(), strict=True):
        calls.append(strict)
        return []

    monkeypatch.setattr(
        otn_manager.NODE, "find_node_from_folder", fake_find_node_from_folder
    )
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)

    otn_manager.load_extension_nodes(strict=False)
    otn_manager.load_extension_nodes(strict=True)

    assert calls.count(False) == len(otn_manager._EXTENSION_DIRS)
    assert calls.count(True) == len(otn_manager._EXTENSION_DIRS)


def test_find_node_from_folder_strict_modes(tmp_path):
    good_node = tmp_path / "good_node.py"
    good_node.write_text(
        "from pdebug.otn import manager as otn_manager\n"
        "@otn_manager.NODE.register(name='tmp_test_node')\n"
        "def tmp_test_node():\n"
        "    return 'ok'\n"
    )
    bad_node = tmp_path / "bad_node.py"
    bad_node.write_text("raise RuntimeError('bad optional dependency')\n")

    otn_manager.NODE.find_node_from_folder(str(tmp_path), strict=False)
    assert otn_manager.create("tmp_test_node")() == "ok"

    sys.modules.pop("bad_node", None)
    try:
        otn_manager.NODE.find_node_from_folder(str(tmp_path), strict=True)
    except RuntimeError as exc:
        assert "bad optional dependency" in str(exc)
    else:
        raise AssertionError("strict=True should raise import errors")


def test_create_loads_node_from_extension_dir(tmp_path, monkeypatch):
    node_file = tmp_path / "lazy_node.py"
    node_file.write_text(
        "from pdebug.otn import manager as otn_manager\n"
        "@otn_manager.NODE.register(name='tmp_lazy_node')\n"
        "def tmp_lazy_node():\n"
        "    return 'lazy'\n"
    )
    monkeypatch.setattr(otn_manager, "_EXTENSION_DIRS", (str(tmp_path),))
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)

    assert otn_manager.create("tmp_lazy_node")() == "lazy"


def test_otn_cli_preserves_create_error_details(monkeypatch):
    def fail_create(name):
        raise ValueError("Unknown node name\nSkipped extension modules")

    monkeypatch.setattr(otn_cli.otn_manager, "create", fail_create)

    with pytest.raises(RuntimeError) as exc_info:
        otn_cli.main(
            ctx=type("Ctx", (), {"args": []})(),
            node="missing",
            list_node=False,
            help_node=False,
            print_node_file=False,
            force_single_process=True,
        )

    assert "Skipped extension modules" in str(exc_info.value)
