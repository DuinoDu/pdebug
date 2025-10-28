import os
import sys

__all__ = [
    "IS_WINDOWS",
    "IS_MAC",
    "IS_LINUX",
    "IS_PYTHON3",
    "IS_PYTHON3_6",
    "IS_PYTHON3_7",
    "IS_PYTHON3_8",
    "IS_PYTHON3_9",
    "IS_PYTHON3_10",
    "IS_PYTHON3_11",
    "TORCH_INSTALLED",
    "TORCHVISION_INSTALLED",
    "TORCHMETRICS_INSTALLED",
    "TIMM_INSTALLED",
    "JITTOR_INSTALLED",
    "VISDOM_INSTALLED",
    "OPEN3D_INSTALLED",
    "PLOTLY_INSTALLED",
    "RICH_INSTALLED",
    "SCIPY_INSTALLED",
    "PYNVML_INSTALLED",
    "XMLTODICT_INSTALLED",
    "HAS_CUDA",
    "CRUISE_INSTALLED",
    "PYARROW_INSTALLED",
    "IMANTICS_INSTALLED",
    "FFMPEG_INSTALLED",
    "PANDAS_INSTALLED",
    "MMSEG_INSTALLED",
    "PYTORCH3D_INSTALLED",
    "TRIMESH_INSTALLED",
    "PYRENDER_INSTALLED",
    "RERUN_INSTALLED",
    "PLOTEXT_INSTALLED",
    "DORA_INSTALLED",
    "BLENDER_INSTALLED",
    "VISPY_INSTALLED",
    "YOURDFPY_INSTALLED",
    "SPATIALMP4_INSTALLED",
    "VGGT_INSTALLED",
    "VISER_INSTALLED",
    "SECUREMR_INSTALLED",
    "HUGGINGFACE_HUB_INSTALLED",
    "TRANSFORMERS_INSTALLED",
    "MOVIEPY_INSTALLED",
]

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform == "xorg"

IS_PYTHON3 = sys.version_info.major == 3
IS_PYTHON3_6 = sys.version_info.major == 3 and sys.version_info.minor == 6
IS_PYTHON3_7 = sys.version_info.major == 3 and sys.version_info.minor == 7
IS_PYTHON3_8 = sys.version_info.major == 3 and sys.version_info.minor == 8
IS_PYTHON3_9 = sys.version_info.major == 3 and sys.version_info.minor == 9
IS_PYTHON3_10 = sys.version_info.major == 3 and sys.version_info.minor == 10
IS_PYTHON3_11 = sys.version_info.major == 3 and sys.version_info.minor == 11

try:
    import torch

    TORCH_INSTALLED = True
except ModuleNotFoundError:
    TORCH_INSTALLED = False

try:
    import torchvision

    TORCHVISION_INSTALLED = True
except ModuleNotFoundError:
    TORCHVISION_INSTALLED = False

try:
    import torchmetrics

    TORCHMETRICS_INSTALLED = True
except ModuleNotFoundError:
    TORCHMETRICS_INSTALLED = False

try:
    import timm

    TIMM_INSTALLED = True
except ModuleNotFoundError:
    TIMM_INSTALLED = False

try:
    import jittor

    JITTOR_INSTALLED = True
except ModuleNotFoundError:
    JITTOR_INSTALLED = False

try:
    import visdom

    VISDOM_INSTALLED = True
except ModuleNotFoundError:
    VISDOM_INSTALLED = False

try:
    import open3d

    OPEN3D_INSTALLED = True
except ModuleNotFoundError:
    OPEN3D_INSTALLED = False

try:
    import plotly

    PLOTLY_INSTALLED = True
except ModuleNotFoundError:
    PLOTLY_INSTALLED = False

try:
    import rich
    from rich.traceback import install

    install(max_frames=0)

    RICH_INSTALLED = True
except ModuleNotFoundError:
    RICH_INSTALLED = False

try:
    import scipy

    SCIPY_INSTALLED = True
except ModuleNotFoundError:
    SCIPY_INSTALLED = False


try:
    import pynvml

    PYNVML_INSTALLED = True
    if os.environ.get("DISABLE_PYNVML", "0") == "1" or IS_MAC:
        PYNVML_INSTALLED = False
except ModuleNotFoundError:
    PYNVML_INSTALLED = False

try:
    import xmltodict

    XMLTODICT_INSTALLED = True
except ModuleNotFoundError:
    XMLTODICT_INSTALLED = False


def has_cuda():
    import shutil

    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import commands
    except Exception as e:
        import subprocess as commands
    cmd = "nvidia-smi --list-gpus"
    (status, output) = commands.getstatusoutput(cmd)
    if "NVIDIA" not in output:
        return False

    return True


HAS_CUDA = has_cuda()


try:
    import cruise

    CRUISE_INSTALLED = True
except ModuleNotFoundError:
    CRUISE_INSTALLED = False


try:
    import pyarrow

    PYARROW_INSTALLED = True
except ModuleNotFoundError:
    PYARROW_INSTALLED = False


try:
    import imantics

    IMANTICS_INSTALLED = True
except ModuleNotFoundError:
    IMANTICS_INSTALLED = False


try:
    import ffmpeg

    FFMPEG_INSTALLED = True
except ModuleNotFoundError:
    FFMPEG_INSTALLED = False

try:
    import pandas

    PANDAS_INSTALLED = True
except ModuleNotFoundError:
    PANDAS_INSTALLED = False

try:
    import mmseg

    MMSEG_INSTALLED = True
except ModuleNotFoundError:
    MMSEG_INSTALLED = False

try:
    import pytorch3d

    PYTORCH3D_INSTALLED = True
except ModuleNotFoundError:
    PYTORCH3D_INSTALLED = False

try:
    import trimesh

    TRIMESH_INSTALLED = True
except ModuleNotFoundError:
    TRIMESH_INSTALLED = False

try:
    import pyrender

    PYRENDER_INSTALLED = True
except ModuleNotFoundError:
    PYRENDER_INSTALLED = False


try:
    import rerun

    RERUN_INSTALLED = True
except ModuleNotFoundError:
    RERUN_INSTALLED = False

if RERUN_INSTALLED:
    import rerun

    IS_RERUN_VERSION_0_20 = rerun.__version__.split(".")[1] == "20"
    IS_RERUN_VERSION_0_21 = rerun.__version__.split(".")[1] == "21"
    IS_RERUN_VERSION_0_22 = rerun.__version__.split(".")[1] == "22"
    IS_RERUN_VERSION_0_23 = rerun.__version__.split(".")[1] == "23"


try:
    import plotext

    PLOTEXT_INSTALLED = True
except ModuleNotFoundError:
    PLOTEXT_INSTALLED = False

try:
    import dora

    DORA_INSTALLED = True
except ModuleNotFoundError:
    DORA_INSTALLED = False

try:
    import bpy

    BLENDER_INSTALLED = True
except ModuleNotFoundError:
    BLENDER_INSTALLED = False

try:
    import vispy

    VISPY_INSTALLED = True
except ModuleNotFoundError:
    VISPY_INSTALLED = False

try:
    import yourdfpy

    YOURDFPY_INSTALLED = True
except ModuleNotFoundError:
    YOURDFPY_INSTALLED = False

try:
    import spatialmp4

    SPATIALMP4_INSTALLED = True
except ModuleNotFoundError as e:
    SPATIALMP4_INSTALLED = False

try:
    import vggt

    VGGT_INSTALLED = True
except ModuleNotFoundError as e:
    VGGT_INSTALLED = False

try:
    import viser

    VISER_INSTALLED = True
except ModuleNotFoundError as e:
    VISER_INSTALLED = False

try:
    import securemr

    SECUREMR_INSTALLED = True
except ModuleNotFoundError as e:
    SECUREMR_INSTALLED = False

try:
    import huggingface_hub

    HUGGINGFACE_HUB_INSTALLED = True
except ModuleNotFoundError as e:
    HUGGINGFACE_HUB_INSTALLED = False

try:
    import transformers

    TRANSFORMERS_INSTALLED = True
except ModuleNotFoundError as e:
    TRANSFORMERS_INSTALLED = False


try:
    import moviepy

    MOVIEPY_INSTALLED = True
except ModuleNotFoundError as e:
    MOVIEPY_INSTALLED = False
