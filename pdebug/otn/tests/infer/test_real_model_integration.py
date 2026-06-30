"""Opt-in real inference smoke tests for manifest-backed model nodes.

These tests intentionally do not run during normal ``make test``. They clone
official model repositories, download weights, and run tiny real inference
cases only when ``PDEBUG_RUN_MODEL_INTEGRATION=1`` is set.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.base import InferenceNode
from pdebug.otn.tests.infer.model_integration_support import (
    FixtureFactory,
    CASE_ENV,
    RUN_ENV,
    assert_expected_outputs,
    build_common_args,
    case_required_args,
    model_cache_root,
    prepare_case_environment,
    pytest_skip_if_disabled,
    require_cuda,
    require_docker,
    require_python_modules_or_fail,
)

try:
    from pdebug.otn.tests.infer.model_integration_cases import (
        MODEL_INTEGRATION_CASES,
    )
    from pdebug.otn.tests.infer.model_integration_matrix import (
        MODEL_INTEGRATION_MATRIX,
    )
except ImportError:  # pragma: no cover - protects partial local edits
    MODEL_INTEGRATION_CASES = ()
    MODEL_INTEGRATION_MATRIX = ()


pytestmark = pytest.mark.model_integration

SUBPROCESS_ENV = "PDEBUG_MODEL_INTEGRATION_SUBPROCESS"


@pytest.mark.parametrize(
    "case",
    MODEL_INTEGRATION_CASES,
    ids=lambda case: case.case_id,
)
def test_real_model_inference_case(case, tmp_path):
    """Run one real model smoke case through the new manifest entrypoint."""
    pytest_skip_if_disabled(case.case_id)
    if os.getenv(SUBPROCESS_ENV) != "1":
        _run_real_model_case_in_subprocess(case)
        return

    _run_real_model_case(case, tmp_path)


def _run_real_model_case_in_subprocess(case) -> None:
    test_id = (
        f"{Path(__file__).resolve()}::test_real_model_inference_case"
        f"[{case.case_id}]"
    )
    env = os.environ.copy()
    env[RUN_ENV] = "1"
    env[CASE_ENV] = case.case_id
    env[SUBPROCESS_ENV] = "1"
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", test_id],
        cwd=Path(__file__).resolve().parents[4],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output = _bounded_subprocess_output(result.stdout or "")
        pytest.fail(
            f"{case.case_id} real inference subprocess failed with "
            f"exit code {result.returncode}.\n{output}"
        )


def _bounded_subprocess_output(output: str) -> str:
    if len(output) <= 30000:
        return output
    return (
        output[:8000]
        + "\n\n... subprocess output truncated ...\n\n"
        + output[-22000:]
    )


def _run_real_model_case(case, tmp_path):
    """Run one real model smoke case in the current process."""
    if case.requires_cuda:
        require_cuda()
    if case.requires_docker:
        require_docker()
    prepare_case_environment(case)
    if case.required_modules:
        require_python_modules_or_fail(case, case.required_modules)

    cache_root = model_cache_root()
    fixtures = FixtureFactory(cache_root / "fixtures" / case.case_id)
    output_root = tmp_path / "model_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    node = otn_manager.create(case.node)
    assert isinstance(node, InferenceNode)

    args = build_common_args(case, fixtures, output_root)
    result = node(**args)

    if isinstance(result, (str, Path)):
        assert Path(result).exists() or Path(result).parent.exists()
    assert_expected_outputs(case, output_root)


def test_model_integration_matrix_covers_requested_files():
    """Guard against silently dropping one of the requested model files."""
    expected_files = {
        "co-tracker.py",
        "depth_anything.py",
        "foundpose.py",
        "genpose2.py",
        "groundingdino_node.py",
        "hunyuan3d_paint.py",
        "hunyuan3d_rembg.py",
        "hunyuan3d_shape.py",
        "internimage_semseg.py",
        "langsam.py",
        "ml_depth_pro_node.py",
        "moondream_node.py",
        "oneposeviagen_3dgen.py",
        "oneposeviagen_pose.py",
        "oneposeviagen_scale.py",
        "orient_anything.py",
        "qwen2_5_vl.py",
        "sam2_node.py",
        "sam6d.py",
        "segment_anything_node.py",
        "spatracker.py",
        "vggt_node.py",
    }
    covered = {Path(case.file).name for case in MODEL_INTEGRATION_CASES}
    assert expected_files <= covered
    metadata_covered = {
        Path(str(item["file"])).name for item in MODEL_INTEGRATION_MATRIX
    }
    assert expected_files <= metadata_covered


def test_model_integration_case_ids_are_unique():
    ids = [case.case_id for case in MODEL_INTEGRATION_CASES]
    assert len(ids) == len(set(ids))


def test_model_integration_cases_have_source_metadata():
    metadata_files = {str(item["file"]) for item in MODEL_INTEGRATION_MATRIX}
    for case in MODEL_INTEGRATION_CASES:
        assert case.file in metadata_files
        assert case.official_sources
        assert case.heavy_reason


@pytest.mark.parametrize(
    "case",
    MODEL_INTEGRATION_CASES,
    ids=lambda case: case.case_id,
)
def test_model_integration_case_args_match_node_signature(case, tmp_path):
    """Validate case wiring without downloading or running real models."""
    fixtures = FixtureFactory(tmp_path / "fixtures")
    output_root = tmp_path / "outputs"
    args = build_common_args(
        case, fixtures, output_root, prepare_repos=False
    )

    node = otn_manager.create(case.node)
    assert isinstance(node, InferenceNode)
    missing = case_required_args(case) - set(args)
    assert not missing
    manifest_args = set(node.spec.parameters)
    unexpected = set(args) - manifest_args
    assert not unexpected
