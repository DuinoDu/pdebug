"""Integration tests for Cortexia-style inference nodes."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

LANCE_DATASET_PATH = Path("/home/duino/all_in_one_annotated.lance")
LENA_URL = (
    "https://github.com/mikolalysenko/lena/blob/master/lena.png?raw=true"
)

pytestmark = pytest.mark.skipif(
    not LANCE_DATASET_PATH.exists(),
    reason="Test dataset /home/duino/all_in_one_annotated.lance is required.",
)


def _load_column(dataset_path: Path, column_name: str):
    import lance

    ds = lance.dataset(str(dataset_path))
    table = ds.to_table(columns=[column_name])
    cell = table[column_name][0]
    if hasattr(cell, "as_py"):
        return cell.as_py()
    return cell


def test_moondream_node_single_and_lance(tmp_path):
    from pdebug.otn.infer.moondream_node import moondream_node

    single_output = tmp_path / "moondream.json"
    result_path = moondream_node(
        LENA_URL, output=str(single_output), unittest=True
    )
    assert Path(result_path) == single_output
    assert single_output.exists()
    with single_output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["model_name"] == "moondream"
    assert "caption" in payload

    lance_output = tmp_path / "moondream_output.lance"
    dataset_path = moondream_node(
        str(LANCE_DATASET_PATH), output=str(lance_output), unittest=True
    )
    assert Path(dataset_path) == lance_output
    column = _load_column(lance_output, "cortexia_caption")
    assert column["model_name"] == "moondream"
    assert isinstance(column["caption"], str) and len(column["caption"]) > 0


def test_groundingdino_node_single_and_lance(tmp_path):
    from pdebug.otn.infer.groundingdino_node import groundingdino_node

    single_output = tmp_path / "groundingdino.json"
    result_path = groundingdino_node(
        LENA_URL, output=str(single_output), unittest=True
    )
    assert Path(result_path) == single_output
    with single_output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["model_name"] == "GroundingDINO"
    assert isinstance(payload["detections"], list)
    assert "bbox" in payload["detections"][0]

    lance_output = tmp_path / "groundingdino_output.lance"
    dataset_path = groundingdino_node(
        str(LANCE_DATASET_PATH), output=str(lance_output), unittest=True
    )
    assert Path(dataset_path) == lance_output
    column = _load_column(lance_output, "cortexia_detection")
    assert column["model_name"] == "GroundingDINO"
    assert isinstance(column["detections"], list)
    assert "bbox" in column["detections"][0]


def test_segment_anything_node_single_and_lance(tmp_path):
    from pdebug.otn.infer.segment_anything_node import segment_anything_node

    single_output = tmp_path / "segment_anything.json"
    result_path = segment_anything_node(
        LENA_URL, output=str(single_output), unittest=True
    )
    assert Path(result_path) == single_output
    with single_output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["model_name"] == "SegmentAnything"
    assert isinstance(payload["segmentations"], list)

    lance_output = tmp_path / "segment_anything_output.lance"
    dataset_path = segment_anything_node(
        str(LANCE_DATASET_PATH), output=str(lance_output), unittest=True
    )
    assert Path(dataset_path) == lance_output
    column = _load_column(lance_output, "cortexia_segmentation")
    assert column["model_name"] == "SegmentAnything"
    assert isinstance(column["segmentations"], list)


def test_ml_depth_pro_node_single_and_lance(tmp_path):
    from pdebug.otn.infer.ml_depth_pro_node import ml_depth_pro_node

    single_output = tmp_path / "ml_depth_pro.json"
    result_path = ml_depth_pro_node(
        LENA_URL, output=str(single_output), unittest=True
    )
    assert Path(result_path) == single_output
    with single_output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["model_name"] == "ml-depth-pro"
    assert "mean_depth" in payload

    lance_output = tmp_path / "ml_depth_pro_output.lance"
    dataset_path = ml_depth_pro_node(
        str(LANCE_DATASET_PATH), output=str(lance_output), unittest=True
    )
    assert Path(dataset_path) == lance_output
    column = _load_column(lance_output, "cortexia_depth")
    assert column["model_name"] == "ml-depth-pro"
    assert "mean_depth" in column


def test_qwen_node_lance_dataset(tmp_path):
    from pdebug.otn.infer import qwen2_5_vl

    lance_output = tmp_path / "qwen_output.lance"
    dataset_path = qwen2_5_vl.main(
        input_path=str(LANCE_DATASET_PATH),
        output=str(lance_output),
        unittest=True,
    )
    assert Path(dataset_path) == lance_output
    column = _load_column(lance_output, "cortexia_tags")
    assert column["model_name"] == "Qwen2.5-VL"
    assert isinstance(column["tags"], list)
    assert len(column["tags"]) > 0
