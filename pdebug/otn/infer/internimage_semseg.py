import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils import install_x
from pdebug.utils.env import HUGGINGFACE_HUB_INSTALLED, TORCH_INSTALLED
from pdebug.utils.fileio import do_system, download_file
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer

if HUGGINGFACE_HUB_INSTALLED:
    from huggingface_hub import hf_hub_download


__all__ = [
    "init_model",
    "prepare",
    "COLOR_PALETTE",
    "ADE_CLASSES",
    "_load_internimage_model",
]


def patch_for_mmcv():
    # Patch for mmcv, force weights_only=False in torch.load
    # Required for torch>=2.6
    import torch

    native_load = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return native_load(*args, **kwargs)

    torch.load = _load


def prepare():
    """Prepare deps for internimage"""

    repo_path = install_x.get_repo("DCNv4")
    if not repo_path:
        print("Try to install repo by `bash $INSTALL/DCNv4.sh` ...")
        install_x.install("DCNv4")

    repo = Path(repo_path).resolve()
    sys.path.append(str(repo / "segmentation"))

    try:
        import mmcv_custom
        import mmseg_custom
        from mmcv.runner import load_checkpoint
        from mmseg.apis import inference_segmentor, init_segmentor
        from mmseg.core import get_classes
        from mmseg.core.evaluation import get_palette
    except ModuleNotFoundError as e:
        print(e)
        install_x.install("DCNv4")
        try:
            import mmcv_custom
            import mmseg_custom
            from mmcv.runner import load_checkpoint
            from mmseg.apis import inference_segmentor, init_segmentor
            from mmseg.core import get_classes
            from mmseg.core.evaluation import get_palette
        except ModuleNotFoundError as e:
            print(e)
            install_x.print_and_exit(
                "Try to fix deps error by `bash $INSTALL/internimage_dcnv4.sh`"
            )
    return repo_path


@lru_cache(maxsize=1)
def _cached_repo_path() -> str:
    return prepare()


@lru_cache(maxsize=1)
def _cached_internimage_model(
    model_type: str = "big", device: str = "cuda:0"
) -> Tuple[object, object, object]:
    repo = _cached_repo_path()
    model, inference_segmentor, color_palette = init_model(
        repo, model_type, device
    )
    global COLOR_PALETTE
    COLOR_PALETTE = color_palette
    return model, inference_segmentor, color_palette


def _load_internimage_model(
    model_type: str = "big", device: str = "cuda:0"
) -> Tuple[object, object, object]:
    return _cached_internimage_model(model_type, device)


def init_model(repo: str, model_type: str = "small", device: str = "cuda:0"):

    import mmcv_custom
    import mmseg_custom
    from mmcv.runner import load_checkpoint
    from mmseg.apis import inference_segmentor, init_segmentor
    from mmseg.core import get_classes
    from mmseg.core.evaluation import get_palette

    patch_for_mmcv()

    assert (
        HUGGINGFACE_HUB_INSTALLED
    ), "huggingface_hub is required, but not found. Please install it first."

    MODEL_LIST = [
        "upernet_flash_internimage_s_512_160k_ade20k",
        "upernet_flash_internimage_b_512_160k_ade20k",
        "upernet_flash_internimage_t_512_160k_ade20k",
        "upernet_flash_internimage_l_640_160k_ade20k",
        "mask2former_flash_internimage_s_640_160k_ade20k_ss",
        "mask2former_flash_internimage_b_640_160k_ade20k_ss",
        "mask2former_flash_internimage_t_512_160k_ade20k_ss",
        "mask2former_flash_internimage_l_640_160k_ade20k_ss",
    ]
    if model_type == "small":
        model_name = MODEL_LIST[0]
    elif model_type == "big":
        model_name = MODEL_LIST[-1]
    else:
        assert model_type in MODEL_LIST
        model_name = model_type

    cfg_file = Path(repo) / f"segmentation/configs/ade20k/{model_name}.py"
    checkpoint = hf_hub_download(
        repo_id="OpenGVLab/DCNv4",
        filename=f"{model_name}.pth",
        revision="main",
        force_download=False,
        resume_download=True,
    )
    color_palette = get_palette("ade20k")

    model = init_segmentor(str(cfg_file), checkpoint=None, device=device)
    checkpoint = load_checkpoint(model, str(checkpoint), map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = get_classes(args.palette)
    return model, inference_segmentor, color_palette


_original_cached_model_clear = _cached_internimage_model.cache_clear


def _internimage_cache_clear() -> None:
    if _cached_internimage_model.cache_info().currsize:
        try:
            model, _, _ = _cached_internimage_model()
            if TORCH_INSTALLED:
                import torch
                import gc

                if torch.cuda.is_available():
                    try:
                        if hasattr(model, "to"):
                            model.to("cpu")
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                        if callable(ipc_collect):
                            ipc_collect()
                    except Exception:
                        pass
                del model
                gc.collect()
        except Exception:
            pass
    _original_cached_model_clear()
    if TORCH_INSTALLED:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except Exception:
            pass


_cached_internimage_model.cache_clear = _internimage_cache_clear  # type: ignore[assignment]
_load_internimage_model.cache_clear = _cached_internimage_model.cache_clear  # type: ignore[attr-defined]


@otn_manager.NODE.register(name="internimage_semseg")
def main(
    input_path: str = None,
    output: str = "tmp_semseg",
    vis_output: str = "tmp_semseg_vis",
    repo: str = None,
    unittest: bool = False,
    device: str = "cuda:0",
    cache: bool = True,
    topk: int = -1,
    savevideo: bool = False,
    model_type: str = "small",
):
    """Inference of internimage-DCNv4 semseg.

    GPU Memory: 6.5G

    Args:
        model_type: model type name. Available: small for upernet_s, big for mask2former_l.
            Defualt is "small".

    Unittest:
        >> otn-cli --node internimage_semseg --unittest True
    """
    output = Path(output)
    output.mkdir(exist_ok=True)
    if vis_output:
        vis_output = Path(vis_output)
        vis_output.mkdir(exist_ok=True)

    if unittest:
        if not os.getenv("PDEBUG_TEST_IMAGE", None):
            url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
            print(f"Run unittest on {url}")
            download_file(url, "/tmp/lenna.png", exit_when_failed=True)
            input_files = ["/tmp/lenna.png"]
        else:
            test_image = os.getenv("PDEBUG_TEST_IMAGE")
            print(f"Run unittest on {test_image}")
            input_files = [test_image]
    else:
        input_files = (
            Input(input_path, name="imgdir", topk=topk).get_reader().imgfiles
        )
        if vis_output and savevideo:
            writer = Output(
                vis_output / "visualization.mp4", name="video_ffmpeg"
            ).get_writer()

    repo = prepare()
    model, inference_segmentor, color_palette = init_model(
        repo, model_type, device
    )

    global COLOR_PALETTE
    COLOR_PALETTE = color_palette

    t = tqdm.tqdm(total=len(input_files))
    for imgfile in input_files:
        t.update()

        savename = output / os.path.basename(imgfile)
        if cache and savename.exists():
            continue

        result = inference_segmentor(model, imgfile)
        if result and len(result) == 1:
            result = result[0]
        cv2.imwrite(savename, result.astype(np.uint8))

        if unittest or vis_output:
            vis_seg = draw.semseg(
                result,
                image=imgfile,
                colors=color_palette,
                classes=model.CLASSES,
            )

        if unittest or (vis_output and not savevideo):
            savename = vis_output / os.path.basename(imgfile)
            cv2.imwrite(savename, vis_seg)

        if (not unittest) and vis_output and savevideo:
            writer.write_frame(vis_seg)

    if vis_output and savevideo and not unittest:
        writer.save()
    return output


ADE_CLASSES = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed ",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
]

COLOR_PALETTE = None


@otn_manager.NODE.register(name="remove_dynamic")
def remove_dynamic(
    input_path: str = None,
    mask_path: str = None,
    output: str = "tmp_semseg",
    remove_ade_names: str = "person,car",
    remove_mask_idx: str = None,
    cache: bool = True,
    topk: int = -1,
    downsample: int = None,
):
    """Mask out rgb region based on mask file."""
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(exist_ok=True)

    rgb_reader = Input(
        input_path, name="imgdir", topk=topk, downsample=downsample
    ).get_reader()
    mask_reader = Input(
        mask_path, name="imgdir", topk=topk, downsample=downsample
    ).get_reader()

    if remove_ade_names:
        target_ids = [
            ADE_CLASSES.index(name) for name in remove_ade_names.split(",")
        ]
    elif remove_mask_idx:
        target_ids = [int(s) for s in remove_mask_idx.split(",")]
    else:
        raise ValueError(
            "Please provide mask idx to remove, by --revemo-ade-names or --remove-mask-ids"
        )

    t = tqdm.tqdm(total=len(rgb_reader))
    for rgb, mask in zip(rgb_reader, mask_reader):
        savename = output / os.path.basename(rgb_reader.filename)
        if cache and Path(savename).exists():
            continue
        t.update()
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        for target_id in target_ids:
            valid = mask == target_id
            if valid.sum() > 0:
                rgb[valid] = rgb[valid].mean(0)
        cv2.imwrite(savename, rgb)


if __name__ == "__main__":
    typer.run(main)
