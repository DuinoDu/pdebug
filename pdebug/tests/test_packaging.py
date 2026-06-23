"""Packaging configuration tests."""

import re
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _section(text, name):
    pattern = rf"(?ms)^\[{re.escape(name)}\]\s*(.*?)(?=^\[|\Z)"
    match = re.search(pattern, text)
    assert match is not None, f"Missing [{name}] section"
    return match.group(1)


def test_setuptools_discovers_pdebug_packages():
    text = PYPROJECT.read_text()
    find_section = _section(text, "tool.setuptools.packages.find")

    assert re.search(
        r'^\s*where\s*=\s*\[\s*["\']\.\s*["\']\s*\]',
        find_section,
        re.M,
    )
    assert re.search(
        r'^\s*include\s*=\s*\[\s*["\']pdebug\*["\']\s*\]',
        find_section,
        re.M,
    )


def test_setuptools_does_not_package_only_top_level_pdebug():
    text = PYPROJECT.read_text()

    assert not re.search(
        r'^\s*packages\s*=\s*\[\s*["\']pdebug["\']\s*\]\s*$',
        text,
        re.M,
    )


def test_wheel_contains_subpackages_and_console_scripts(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(PROJECT_ROOT),
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(tmp_path),
        ],
        check=True,
    )
    wheel = next(tmp_path.glob("pdebug-*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        entry_points_file = next(
            name
            for name in names
            if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = archive.read(entry_points_file).decode()

    assert "pdebug/otn/manager.py" in names
    assert "pdebug/otn/infer/manifests/moondream.otn.json" in names
    assert "pdebug/otn/infer/manifests/sam6d_docker.otn.json" in names
    assert "pdebug/piata/input.py" in names
    assert "pdebug/pdag/runner.py" in names
    assert "runnb = pdebug.runnb.runnb:main" in entry_points
    assert "otn-cli = pdebug.otn.cli:cli" in entry_points
    assert "piata-cli = pdebug.piata.cli:cli" in entry_points
