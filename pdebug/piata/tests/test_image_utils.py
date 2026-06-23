"""Practical tests for Piata image helpers."""
from __future__ import annotations

from pdebug.piata import (
    bbox_from_mask,
    compute_image_stats,
    decode_bitmask,
    decode_depth_map,
    decode_depth_map_uint16,
    decode_png_u8,
    depth_stub,
    deterministic_caption,
    encode_bitmask,
    encode_depth_map,
    encode_depth_map_uint16,
    encode_png_u8,
    load_image,
    scaled_bbox,
    segmentation_stub,
)

import numpy as np
from PIL import Image


def test_load_image_reads_local_frame_as_rgb(tmp_path):
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[:, :, 0] = 240
    image[1:3, 2:4] = [10, 120, 200]
    image_path = tmp_path / "camera_frame.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    loaded = load_image(image_path)

    np.testing.assert_array_equal(loaded, image)


def test_image_stats_drive_caption_and_scaled_detection_box():
    image = np.full((60, 100, 3), 32, dtype=np.uint8)
    image[:, 50:] = 96

    stats = compute_image_stats(image)
    caption = deterministic_caption(stats, prefix="inspection frame")
    bbox = scaled_bbox(stats, scale=0.4)

    assert stats["height"] == 60.0
    assert stats["width"] == 100.0
    assert caption == "inspection frame 100x60 mean=64.00 std=32.00"
    assert bbox == [30.0, 20.0, 70.0, 40.0]


def test_segmentation_mask_payloads_roundtrip_for_object_region():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    bitmask_payload = encode_bitmask(mask)
    decoded_bitmask = decode_bitmask(bitmask_payload)
    decoded_png = decode_png_u8(encode_png_u8(mask * 255))

    np.testing.assert_array_equal(decoded_bitmask, mask.astype(bool))
    np.testing.assert_array_equal(decoded_png, mask * 255)
    assert bbox_from_mask(decoded_bitmask) == [3.0, 2.0, 6.0, 4.0]


def test_segmentation_stub_produces_decodable_regions_from_image_stats():
    stats = {"height": 12.0, "width": 16.0, "mean": 80.0}

    segments = segmentation_stub(stats, segments=2)

    assert [segment["label"] for segment in segments] == [
        "segment_0",
        "segment_1",
    ]
    for segment in segments:
        mask = decode_bitmask(segment["mask"])
        assert mask.shape == (12, 16)
        assert mask.any()
        assert segment["bbox"] == bbox_from_mask(mask)
        assert segment["area"] == float(mask.sum())


def test_depth_payloads_roundtrip_for_metric_depth_map():
    depth_map = np.array(
        [
            [0.25, 0.5, 0.75],
            [1.0, 1.25, 1.5],
        ],
        dtype=np.float32,
    )

    decoded_float16 = decode_depth_map(encode_depth_map(depth_map))
    decoded_uint16 = decode_depth_map_uint16(
        encode_depth_map_uint16(depth_map)
    )

    np.testing.assert_allclose(decoded_float16, depth_map, atol=1e-3)
    np.testing.assert_allclose(decoded_uint16, depth_map, atol=1e-3)


def test_depth_stub_generates_usable_depth_summary_and_map():
    stats = {"height": 4.0, "width": 5.0, "mean": 127.5}

    result = depth_stub(stats)
    depth_map = decode_depth_map(result["map"])

    assert depth_map.shape == (4, 5)
    assert result["min_depth"] == 0.0
    assert result["max_depth"] == 0.6
    assert result["mean_depth"] == 0.3
    np.testing.assert_allclose(depth_map[0, 0], 0.0, atol=1e-3)
    np.testing.assert_allclose(depth_map[-1, -1], 0.6, atol=1e-3)
