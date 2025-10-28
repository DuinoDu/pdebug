from pathlib import Path

from pdebug.piata.input import Input

import numpy as np
import pytest

DATASET_PATH = Path("/home/duino/all_in_one_annotated.lance")


pytestmark = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="Test dataset /home/duino/all_in_one_annotated.lance is required.",
)


def test_lance_vita_reader_iteration_and_reset():
    reader = Input(str(DATASET_PATH), name="lance_vita").get_reader()

    assert len(reader) == 8

    first = next(reader)
    assert "image" in first
    assert first["image"].ndim == 3
    assert first["image"].dtype == np.uint8
    assert first["image_name"].startswith("lance_")
    assert first["frame_index"] == 0
    assert isinstance(first["frame_number_resolved"], int)
    assert isinstance(first["source_video_id_resolved"], str)
    assert first["timestamp_seconds"] is not None

    reader.reset()
    reset_first = next(reader)
    np.testing.assert_array_equal(first["image"], reset_first["image"])

    fetched = reader.imread(reset_first["image_name"])
    np.testing.assert_array_equal(reset_first["image"], fetched)
