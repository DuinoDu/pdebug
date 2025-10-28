import json
import os
import shutil
import tempfile
from pathlib import Path

from pdebug.otn.infer.langsam import installed, langsam_predict, langsam_sam

import numpy as np
import pytest
from PIL import Image

if not installed():
    pytest.skip("langsam is required.", allow_module_level=True)


class TestLangSAMNodes:
    """Test cases for LangSAM OTN nodes."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample test images."""
        img_dir = temp_dir / "images"
        img_dir.mkdir()

        # Create a few test images
        for i in range(1):
            img = Image.new("RGB", (224, 224), color="red")
            # Add some variation
            pixels = np.array(img)
            pixels[50:150, 50:150] = [0, 255, 0]  # Green square
            img = Image.fromarray(pixels)
            img.save(img_dir / f"test_{i:03d}.jpg")
        return str(img_dir)

    def test_langs_sam_auto_mask(self, temp_dir, sample_images):
        """Test SAM automatic mask generation."""
        output_dir = temp_dir / "sam_output"

        # Test automatic mask generation
        langsam_sam(
            input_path=sample_images,
            output=str(output_dir),
            sam_type="sam2.1_hiera_small",
            use_auto_mask=True,
        )

        # Check output structure
        assert output_dir.exists()
        assert (output_dir / "masks").exists()
        assert len(list((output_dir / "masks").glob("*.png"))) > 0
        assert (output_dir / "metadata").exists()

        # Check metadata files
        meta_files = list((output_dir / "metadata").glob("*.json"))
        assert len(meta_files) > 0

        # Check metadata content
        with open(meta_files[0]) as f:
            metadata = json.load(f)
        assert isinstance(metadata, list)
        assert len(metadata) > 0
        assert "area" in metadata[0]
        assert "bbox" in metadata[0]

    def test_langs_predict_language_guided(self, temp_dir, sample_images):
        """Test language-guided segmentation."""
        output_dir = temp_dir / "langsam_output"

        # Test language-guided segmentation
        langsam_predict(
            input_path=sample_images,
            texts="object,red square",
            output=str(output_dir),
            sam_type="sam2.1_hiera_small",
            box_threshold=0.3,
            text_threshold=0.25,
        )

        # Check output structure
        assert output_dir.exists()
        assert (output_dir / "masks").exists()
        assert (output_dir / "vis").exists()
        assert (output_dir / "metadata").exists()

        # Check files exist
        mask_files = list((output_dir / "masks").glob("*.png"))
        vis_files = list((output_dir / "vis").glob("*.jpg"))
        meta_files = list((output_dir / "metadata").glob("*.json"))

        assert len(mask_files) > 0
        assert len(vis_files) > 0
        assert len(meta_files) > 0

        # Check metadata content
        with open(meta_files[0]) as f:
            metadata = json.load(f)
        assert "prompt" in metadata
        assert "detections" in metadata
        assert isinstance(metadata["detections"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
