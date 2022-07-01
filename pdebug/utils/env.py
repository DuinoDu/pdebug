__all__ = ["TORCH_INSTALLED", "JITTOR_INSTALLED", "VISDOM_INSTALLED",
           "OPEN3D_INSTALLED"]

try:
    import torch

    TORCH_INSTALLED = True
except ModuleNotFoundError:
    TORCH_INSTALLED = False

try:
    import jittor

    JITTOR_INSTALLED = True
except ModuleNotFoundError:
    JITTOR_INSTALLED = False

try:
    import visdom
    from visdom.server import download_scripts, start_server

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
