import os
import shutil
from pathlib import Path

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# CACHE_DIR = Path("/tmp") / "debug_wonder3d"
CACHE_DIR = Path(
    "/home/duino/code/github/ws/001_object_tracking/output/3DVideo_2025-08-13-09-24-15-730/wonder_input"
)


def dump_data(images, normals, input_image):
    CACHE_DIR.mkdir(exist_ok=True)
    for idx, (image, normal) in enumerate(zip(images, normals)):
        image.save(CACHE_DIR / f"rgb_{idx}.png")
        normal.save(CACHE_DIR / f"normal_{idx}.png")
    input_image.save(CACHE_DIR / f"input_image.png")


def load_data(num_images=6):
    """
    Returns:
        images: list of 6 RGB image, shape is 256x256
        normals: list of 6 RGBA image, shape is 256x256
        input_image: RGBA image, shape is 2048x2048, front view
    """
    images, normals = [], []
    for idx in range(num_images):
        images.append(Image.open(CACHE_DIR / f"rgb_{idx}.png"))
        normals.append(Image.open(CACHE_DIR / f"normal_{idx}.png"))
    input_image = Image.open(CACHE_DIR / f"input_image.png")
    return images, normals, input_image


@otn_manager.NODE.register(name="create_mv4wonder")
def create_mv4wonder(
    rgb_path: str,
    normal_path: str,
    view_idx: str,
    output: str = "tmp_output",
):
    """Create 6 view rgb and normal images for wonder3d."""
    assert "," in view_idx
    view_idx = [int(i) for i in view_idx.split(",")]
    rgb_files = Input(rgb_path, name="imgdir").get_reader().imgfiles
    nor_files = Input(normal_path, name="imgdir").get_reader().imgfiles

    target_shape = (256, 256)
    target_shape_big = (2048, 2048)

    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(exist_ok=True)

    for idx in view_idx:
        img_file = rgb_files[idx]
        nor_file = nor_files[idx]
        image = Image.open(img_file)  #  output / "rgb_{idx}.png")
        normal = Image.open(nor_file)  #  output / "normal_{idx}.png")

        if idx == view_idx[0]:
            image.resize((target_shape_big)).save(output / f"input_image.png")

        # convert image(RGBA) to RGB, and set rgb pixel where alpha=0 to 0
        if image.mode == "RGBA":
            image_array = np.array(image)
            alpha = image_array[:, :, 3]
            rgb = image_array[:, :, :3]
            rgb[alpha == 0] = 255
            image = Image.fromarray(rgb, "RGB")

        image = image.resize(target_shape)
        normal = normal.resize(target_shape)
        view_id = view_idx.index(idx)
        image.save(output / f"rgb_{view_id}.png")
        normal.save(output / f"normal_{view_id}.png")
    print(f"saved to {output}")


def load_normal_map_rgba(image_path):
    assert os.path.exists(image_path), f"{image_path} not exists"

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:  # RGBA
        normals = img[:, :, :3].astype(np.float32) / 255.0 * 2.0 - 1.0
        normals = normals[:, :, [2, 1, 0]]  # BGR -> RGB
        mask = img[:, :, 3]
    else:
        normals = img.astype(np.float32) / 255.0 * 2.0 - 1.0
        mask = None
    return normals, mask


def vis_normal_map(normal_map, mask, savename=None):
    normal_data = np.array(normal_map)
    normal_data[:, :, :3] = (normal_data[:, :, :3] + 1.0) / 2.0 * 255
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, :, None]
        normal_data = np.concatenate((normal_data, mask), axis=2)
    normal_data = normal_data.astype(np.uint8)
    if savename:
        cv2.imwrite(savename, normal_data)
    return normal_data


def debug():
    all_res = []
    for i in range(6):
        normal_file1 = f"/tmp/debug_wonder3d/normal_{i}.png"
        normal_file2 = f"/home/duino/code/github/ws/001_object_tracking/output/3DVideo_2025-08-13-09-24-15-730/wonder_input/normal_{i}.png"
        n1, mask1 = load_normal_map_rgba(normal_file1)
        n2, mask2 = load_normal_map_rgba(normal_file2)
        # n2[:, :, 0] *= -1

        res1 = vis_normal_map(n1, mask1)
        res2 = vis_normal_map(n2, mask2)

        res1 = cv2.imread(normal_file1, cv2.IMREAD_UNCHANGED)
        res2 = cv2.imread(normal_file2, cv2.IMREAD_UNCHANGED)

        res = np.concatenate((res1, res2), axis=1)
        all_res.append(res)
    cv2.imwrite("good.png", np.concatenate(all_res, axis=0))


if __name__ == "__main__":
    debug()
