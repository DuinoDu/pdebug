"""Lightweight contract tests for LangSAM OTN nodes.

Real LangSAM/SAM2 model execution is covered by the opt-in model integration
matrix. These tests keep the default suite focused on pdebug's input/output
contract and avoid loading external checkpoints or Hydra configs.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pdebug.otn.infer import langsam as langsam_module
from pdebug.otn.infer.langsam import langsam_predict, langsam_sam


class FakeSAM:
    """Small SAM stand-in that matches the methods pdebug uses."""

    def build_model(self, sam_type, ckpt_path, device="cuda"):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.device = device

    def generate(self, image_np):
        mask = np.zeros(image_np.shape[:2], dtype=bool)
        mask[50:150, 50:150] = True
        return [
            {
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox": [50, 50, 100, 100],
                "predicted_iou": 0.99,
                "stability_score": 0.98,
            }
        ]


class FakeLangSAM:
    """Small LangSAM stand-in for language-guided segmentation output."""

    def __init__(self, sam_type, ckpt_path=None, device="cuda"):
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.device = device

    def predict(self, images, prompts, box_threshold, text_threshold):
        image = images[0]
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[50:150, 50:150] = True
        return [
            {
                "masks": np.array([mask]),
                "boxes": np.array([[50, 50, 150, 150]], dtype=np.float32),
                "scores": np.array([0.95], dtype=np.float32),
            }
        ]


class TestLangSAMNodes:
    """Test pdebug's LangSAM node IO contracts without real model loading."""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_images(self, temp_dir):
        img_dir = temp_dir / "images"
        img_dir.mkdir()

        img = Image.new("RGB", (224, 224), color="red")
        pixels = np.array(img)
        pixels[50:150, 50:150] = [0, 255, 0]
        Image.fromarray(pixels).save(img_dir / "test_000.jpg")
        return str(img_dir)

    @pytest.fixture
    def fake_sam_module(self, monkeypatch):
        module = types.ModuleType("lang_sam.models.sam")
        module.SAM = FakeSAM
        monkeypatch.setitem(sys.modules, "lang_sam.models.sam", module)
        return module

    def test_langs_sam_auto_mask(
        self, fake_sam_module, temp_dir, sample_images
    ):
        output_dir = temp_dir / "sam_output"

        langsam_sam(
            input_path=sample_images,
            output=str(output_dir),
            sam_type="sam2.1_hiera_small",
            use_auto_mask=True,
            device="cpu",
        )

        assert output_dir.exists()
        mask_files = list((output_dir / "masks").glob("*.png"))
        meta_files = list((output_dir / "metadata").glob("*.json"))
        assert len(mask_files) == 1
        assert len(meta_files) == 1

        metadata = json.loads(meta_files[0].read_text())
        assert metadata == [
            {
                "file": "test_000_mask_000.png",
                "area": 10000,
                "bbox": [50, 50, 100, 100],
                "predicted_iou": 0.99,
                "stability_score": 0.98,
            }
        ]

    def test_langs_predict_language_guided(
        self, monkeypatch, temp_dir, sample_images
    ):
        output_dir = temp_dir / "langsam_output"
        monkeypatch.setattr(langsam_module, "_load_langsam", lambda: FakeLangSAM)

        langsam_predict(
            input_path=sample_images,
            texts="object,red square",
            output=str(output_dir),
            sam_type="sam2.1_hiera_small",
            box_threshold=0.3,
            text_threshold=0.25,
            device="cpu",
            cache=False,
        )

        assert output_dir.exists()
        mask_files = list((output_dir / "masks").glob("*.png"))
        vis_files = list((output_dir / "vis").glob("*.jpg"))
        meta_files = list((output_dir / "metadata").glob("*.json"))
        assert len(mask_files) == 1
        assert len(vis_files) == 1
        assert len(meta_files) == 1

        metadata = json.loads(meta_files[0].read_text())
        assert metadata["prompt"] == "object"
        assert metadata["detections"] == [
            {
                "mask_file": "test_000_mask_000.png",
                "bbox": [50.0, 50.0, 150.0, 150.0],
                "score": pytest.approx(0.95),
                "area": 10000.0,
            }
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
