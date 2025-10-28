"""
Depend:

onnx>=1.12.0
onnxsim>=0.4.0
onnxruntime>=1.12.0
tensorboard
torch

"""

import os

############### config ###############
import typer
from easydict import EasyDict as edict

opt = edict()

# ape benchvise cam can cat
# driller duck eggbox glue
# holepuncher iron lamp phone
category = "iron"

opt.data = f"./configs/linemod/{category}.yaml"
opt.static_camera = "./configs/linemod/linemod_camera.json"
opt.source = f"../data/LINEMOD/{category}/JPEGImages"  # infer imgdir
opt.mesh_data = f"../data/LINEMOD/{category}/{category}.ply"

opt.resume = None
opt.weights = "./runs/train/exp013/weights/best.pt"  # test | infer
opt.epochs = 1000
opt.eval_interval = 200
opt.batch_size_train = 8
opt.batch_size_test = 8
opt.num_workers_train = 8
opt.num_workers_test = 0
opt.img_size = 640  # DON'T change img_size, keep it 640
opt.conf_thres_test = 0.01
opt.conf_thres_infer = 0.25
opt.nms_iou_threshold = 0.2
opt.detect_max_det = 2
opt.num_keypoints = 9
opt.single_cls = True
opt.verbose = True
opt.save_txt = False
opt.save_hybrid = False
opt.save_conf = False
opt.save_json = False
opt.symetric = False
opt.test_plotting = False
opt.task = "val"  # val | test

opt.debug_dataset = False
opt.debug_target = False
opt.debug_single_sample = False
opt.train_augment = True

opt.device = "0"  # 0 | 0,1,2,3 | cpu
opt.pretrained = "./yolov5x.pt"
opt.cfg = "./models/yolov5x_6dpose_bifpn.yaml"
opt.hyp = "./configs/hyp.single.yaml"
opt.hyp_cfg = edict()
# opt.hyp_cfg.fliplr = True          # apply left-right flip
opt.hyp_cfg.background = 0.9  # apply random background
opt.hyp_cfg.translate = 0.4  # apply random translate, 0~0.5
opt.hyp_cfg.blur = 0.5  # apply random blur, 0~1.0
# opt.hyp_cfg.degrees = 0.0           # apply rotation translate, > 0
# opt.hyp_cfg.scale = 4.0             # apply random scale, > 0
# opt.random_scale_large_only = True  # apply random enlarge scale only
opt.rect = True
opt.nosave = False  # only save final checkpoint
opt.notest = False  # only test final epoch
opt.noautoanchor = False  # disable autoanchor check
opt.evolve = False  # find best hyperparameters
opt.bucket = ""  # not used
opt.cache_images = False
opt.image_weights = False
opt.optimizer = "Adam"
opt.sync_bn = False
opt.local_rank = -1
opt.linear_lr = False
opt.standard_lr = False

opt.log_imgs = 8  # not used, by W&B
opt.log_artifacts = False  # not used, by W&B
opt.entity = None  # not used, by W&B
opt.view_img = False
opt.name = "exp"
opt.exist_ok = False
opt.project = "runs/detect"  # runs/train | runs/detect | runs/test
opt.save_img = True

if opt.debug_dataset or opt.debug_target or opt.debug_single_sample:
    opt.num_workers_train = 0
    opt.num_workers_test = 0
    opt.debug_dataset_output = "tmp_debug_dataset"
    if os.path.exists(opt.debug_dataset_output):
        os.system(f"rm -rf {opt.debug_dataset_output}")
    os.makedirs(opt.debug_dataset_output, exist_ok=True)
if opt.debug_target:
    opt.batch_size_train = 1
    opt.batch_size_test = 1


def edict2dict(obj):
    if isinstance(obj, edict):
        return {k: edict2dict(v) for k, v in obj.items()}
    return obj


app = typer.Typer()

############### models/common.py ###############

"""
Common modules
"""

import logging
import math
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

logger = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(
        self, c1, c2, k=1, s=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution class
    def __init__(
        self, c1, c2, k=1, s=1, p1=0, p2=0
    ):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(
            *(TransformerLayer(c2, num_heads) for _ in range(num_layers))
        )
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return (
            self.tr(p + self.linear(p))
            .permute(1, 2, 0)
            .reshape(b, self.c2, w, h)
        )


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n))
        )


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[GhostBottleneck(c_, c_) for _ in range(n)])


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        (
            b,
            c,
            h,
            w,
        ) = (
            x.size()
        )  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g
        )  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat(
            [self.aap(y) for y in (x if isinstance(x, list) else [x])], 1
        )  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


############### utils/google_utils.py ###############


import os
import platform
import subprocess
import time
from pathlib import Path

import requests
import torch


def gsutil_getsize(url=""):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode("utf-8")
    return eval(s.split(" ")[0]) if len(s) else 0  # bytes


def attempt_download(file, repo="ultralytics/yolov5"):
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        try:
            response = requests.get(
                f"https://api.github.com/repos/{repo}/releases/latest"
            ).json()  # github api
            assets = [
                x["name"] for x in response["assets"]
            ]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response["tag_name"]  # i.e. 'v1.0'
        except:  # fallback plan
            assets = ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
            tag = (
                subprocess.check_output("git tag", shell=True)
                .decode()
                .split()[-1]
            )

        name = file.name
        if name in assets:
            msg = f"{file} missing, try downloading from https://github.com/{repo}/releases/"
            redundant = False  # second download option
            try:  # GitHub
                url = (
                    f"https://github.com/{repo}/releases/download/{tag}/{name}"
                )
                print(f"Downloading {url} to {file}...")
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1e6  # check
            except Exception as e:  # GCP
                print(f"Download error: {e}")
                assert redundant, "No secondary mirror"
                url = f"https://storage.googleapis.com/{repo}/ckpt/{name}"
                print(f"Downloading {url} to {file}...")
                os.system(
                    f"curl -L {url} -o {file}"
                )  # torch.hub.download_url_to_file(url, weights)
            finally:
                if not file.exists() or file.stat().st_size < 1e6:  # check
                    file.unlink(missing_ok=True)  # remove partial downloads
                    print(f"ERROR: Download failure: {msg}")
                print("")
                return


def gdrive_download(id="16TiPfZj7htmTyhntwcZyEEAejOUxuT6m", file="tmp.zip"):
    # Downloads a file from Google Drive. from yolov5.utils.google_utils import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path("cookie")  # gdrive cookie
    print(
        f"Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ",
        end="",
    )
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(
        f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}'
    )
    if os.path.exists("cookie"):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print("Download error ")  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == ".zip":
        print("unzipping... ", end="")
        os.system(f"unzip -q {file}")  # unzip
        file.unlink()  # remove zip to free space

    print(f"Done ({time.time() - t:.1f}s)")
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


#
#
#
#
#
#
#


############### models/experimental.py ###############


import numpy as np
import torch
import torch.nn as nn

## from models.common import Conv, DWConv
## from utils.google_utils import attempt_download


class CrossConv_1(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv_1, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(
                -torch.arange(1.0, n) / 2, requires_grad=True
            )  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv_1(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv_1, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck_1(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(GhostBottleneck_1, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv_1(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv_1(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1e-6, c2).floor()  # c2 indices
            c_ = [
                (i == g).sum() for g in range(groups)
            ]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [
                nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False)
                for g in range(groups)
            ]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(
            w, map_location=map_location, weights_only=False
        )  # load
        model.append(
            ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
        )  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = (
                set()
            )  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


############### utils/downloads.py ###############

"""
Download utils
"""

import os
import platform
import subprocess
import time
import urllib
from pathlib import Path
from zipfile import ZipFile

import requests
import torch


def gsutil_getsize_1(url=""):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode("utf-8")
    return eval(s.split(" ")[0]) if len(s) else 0  # bytes


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file))
        assert (
            file.exists() and file.stat().st_size > min_bytes
        ), assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        os.system(
            f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -"
        )  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print("")


def attempt_download_1(
    file, repo="ultralytics/yolov5"
):  # from utils.downloads import *; attempt_download_1()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        # URL specified
        name = Path(
            urllib.parse.unquote(str(file))
        ).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            name = name.split("?")[
                0
            ]  # parse authentication https://url.com/file.txt?auth...
            safe_download(file=name, url=url, min_bytes=1e5)
            return name

        # GitHub assets
        file.parent.mkdir(
            parents=True, exist_ok=True
        )  # make parent dir (if required)
        try:
            response = requests.get(
                f"https://api.github.com/repos/{repo}/releases/latest"
            ).json()  # github api
            assets = [
                x["name"] for x in response["assets"]
            ]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
            tag = response["tag_name"]  # i.e. 'v1.0'
        except:  # fallback plan
            assets = [
                "yolov5s.pt",
                "yolov5m.pt",
                "yolov5l.pt",
                "yolov5x.pt",
                "yolov5s6.pt",
                "yolov5m6.pt",
                "yolov5l6.pt",
                "yolov5x6.pt",
            ]
            try:
                tag = (
                    subprocess.check_output(
                        "git tag", shell=True, stderr=subprocess.STDOUT
                    )
                    .decode()
                    .split()[-1]
                )
            except:
                tag = "v5.0"  # current release

        if name in assets:
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/",
            )

    return str(file)


def gdrive_download_1(id="16TiPfZj7htmTyhntwcZyEEAejOUxuT6m", file="tmp.zip"):
    # Downloads a file from Google Drive. from yolov5.utils.downloads import *; gdrive_download_1()
    t = time.time()
    file = Path(file)
    cookie = Path("cookie")  # gdrive cookie
    print(
        f"Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ",
        end="",
    )
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(
        f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}'
    )
    if os.path.exists("cookie"):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token_1()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print("Download error ")  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == ".zip":
        print("unzipping... ", end="")
        ZipFile(file).extractall(path=file.parent)  # unzip
        file.unlink()  # remove zip

    print(f"Done ({time.time() - t:.1f}s)")
    return r


def get_token_1(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


#
#
#
#
#
#
#
#
#


############### utils/metrics.py ###############


import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [
        0.0,
        0.0,
        1.0,
        0.0,
    ]  # weights for [ mean_corner_err_2d, acc, acc3d, acc5cm5deg]
    return (x[:, :4] * w).sum(1)


def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=()
):
    """Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = (
        np.zeros((nc, tp.shape[1])),
        np.zeros((nc, 1000)),
        np.zeros((nc, 1000)),
    )
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(
                -px, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(
                -px, -conf[i], precision[:, 0], left=1
            )  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(
                    recall[:, j], precision[:, j]
                )
                if plot and j == 0:
                    py.append(
                        np.interp(px, mrec, mpre)
                    )  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / "PR_curve.png", names)
        plot_mc_curve(
            px, f1, Path(save_dir) / "F1_curve.png", names, ylabel="F1"
        )
        plot_mc_curve(
            px, p, Path(save_dir) / "P_curve.png", names, ylabel="Precision"
        )
        plot_mc_curve(
            px, r, Path(save_dir) / "R_curve.png", names, ylabel="Recall"
        )

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype("int32")


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[
            0
        ]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[
                    np.unique(matches[:, 1], return_index=True)[1]
                ]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[
                    np.unique(matches[:, 0], return_index=True)[1]
                ]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir="", names=()):
        try:
            import seaborn as sn

            array = self.matrix / (
                self.matrix.sum(0).reshape(1, self.nc + 1) + 1e-6
            )  # normalize
            array[
                array < 0.005
            ] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(
                names
            ) == self.nc  # apply names to ticklabels
            sn.heatmap(
                array,
                annot=self.nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                xticklabels=names + ["background FP"] if labels else "auto",
                yticklabels=names + ["background FN"] if labels else "auto",
            ).set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel("True")
            fig.axes[0].set_ylabel("Predicted")
            fig.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(" ".join(map(str, self.matrix[i])))


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label="all classes %.3f mAP@0.5" % ap[:, 0].mean(),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(
    px,
    py,
    save_dir="mc_curve.png",
    names=(),
    xlabel="Confidence",
    ylabel="Metric",
):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]}"
            )  # plot(confidence, metric)
    else:
        ax.plot(
            px, py.T, linewidth=1, color="grey"
        )  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def bbox_iou(
    box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7
):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if (
            CIoU or DIoU
        ):  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center dist ** 2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)),
                    2,
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


############### utils/general.py ###############

"""
General utils
"""

import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml

## from utils.downloads import gsutil_getsize  # AIO: {'gsutil_getsize': 'gsutil_getsize_1', 'attempt_download': 'attempt_download_1', 'gdrive_download': 'gdrive_download_1', 'get_token': 'get_token_1'}
## from utils.metrics import box_iou, fitness

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(
    linewidth=320, formatter={"float_kind": "{:11.5g}".format}
)  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(
    0
)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(
    min(os.cpu_count(), 8)
)  # NumExpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory


class Profile(contextlib.ContextDecorator):
    # Usage: @Profile() decorator or 'with Profile():' context manager
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print(f"Profile results: {time.time() - self.start:.5f}s")


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(
        self, seconds, *, timeout_msg="", suppress_timeout_errors=True
    ):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(
            signal.SIGALRM, self._timeout_handler
        )  # Set handler for SIGALRM
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def methods(instance):
    # Get class/instance methods
    return [
        f
        for f in dir(instance)
        if callable(getattr(instance, f)) and not f.startswith("__")
    ]


def set_logging(rank=-1, verbose=True):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN,
    )


def print_args(name, opt):
    # Print argparser arguments
    print(
        colorstr(f"{name}: ")
        + ", ".join(f"{k}={v}" for k, v in vars(opt).items())
    )


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (
        (False, True) if seed == 0 else (True, False)
    )


def get_latest_run(search_dir="."):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""


def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {
            "Windows": "AppData/Roaming",
            "Linux": ".config",
            "Darwin": "Library/Application Support",
        }  # 3 OS dirs
        path = Path.home() / cfg.get(
            platform.system(), ""
        )  # OS-specific config dir
        path = (
            path if is_writeable(path) else Path("/tmp")
        ) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if test:  # method 1
        file = Path(dir) / "tmp.txt"
        try:
            with open(file, "w"):  # open file with write permissions
                pass
            file.unlink()  # remove file
            return True
        except IOError:
            return False
    else:  # method 2
        return os.access(dir, os.R_OK)  # possible issues on Windows


def is_docker():
    # Is environment a Docker container?
    return Path("/workspace").exists()  # or Path('/.dockerenv').exists()


def is_colab():
    # Is environment a Google Colab instance?
    try:
        import google.colab

        return True
    except ImportError:
        return False


def is_pip():
    # Is file in a pip package?
    return "site-packages" in Path(__file__).resolve().parts


def is_ascii(s=""):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def is_chinese(s="人工智能"):
    # Is string composed of any Chinese characters?
    return re.search("[\u4e00-\u9fff]", s)


def emojis(str=""):
    # Return platform-dependent emoji-safe version of string
    return (
        str.encode().decode("ascii", "ignore")
        if platform.system() == "Windows"
        else str
    )


def file_size(path):
    # Return file/dir size (MB)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    elif path.is_dir():
        return (
            sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
            / 1e6
        )
    else:
        return 0.0


def check_online():
    # Check internet connectivity
    import socket

    try:
        socket.create_connection(
            ("1.1.1.1", 443), 5
        )  # check host accessibility
        return True
    except OSError:
        return False


@try_except
def check_git_status():
    # Recommend 'git pull' if code is out of date
    msg = ", for updates see https://github.com/ultralytics/yolov5"
    print(colorstr("github: "), end="")
    assert Path(".git").exists(), "skipping check (not a git repository)" + msg
    assert not is_docker(), "skipping check (Docker image)" + msg
    assert check_online(), "skipping check (offline)" + msg

    cmd = "git fetch && git config --get remote.origin.url"
    url = (
        check_output(cmd, shell=True, timeout=5)
        .decode()
        .strip()
        .rstrip(".git")
    )  # git fetch
    branch = (
        check_output("git rev-parse --abbrev-ref HEAD", shell=True)
        .decode()
        .strip()
    )  # checked out
    n = int(
        check_output(
            f"git rev-list {branch}..origin/master --count", shell=True
        )
    )  # commits behind
    if n > 0:
        s = f"⚠️ YOLOv5 is out of date by {n} commit{'s' * (n > 1)}. Use `git pull` or `git clone {url}` to update."
    else:
        s = f"up to date with {url} ✅"
    print(emojis(s))  # emoji-safe


def check_python(minimum="3.6.2"):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name="Python ")


def check_version(
    current="0.0.0", minimum="0.0.0", name="version ", pinned=False
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    assert (
        result
    ), f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"


@try_except
def check_requirements(
    requirements=ROOT / "requirements.txt", exclude=(), install=True
):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr("red", "bold", "requirements:")
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert (
            file.exists()
        ), f"{prefix} {file.resolve()} not found, check failed."
        requirements = [
            f"{x.name}{x.specifier}"
            for x in pkg.parse_requirements(file.open())
            if x.name not in exclude
        ]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install:
                print(f"{s}, attempting auto-update...")
                try:
                    assert (
                        check_online()
                    ), f"'pip install {r}' skipped (offline)"
                    print(
                        check_output(f"pip install '{r}'", shell=True).decode()
                    )
                    n += 1
                except Exception as e:
                    print(f"{prefix} {e}")
            else:
                print(f"{s}. Please install and rerun your command.")

    if n:  # if packages updated
        source = file.resolve() if "file" in locals() else requirements
        s = (
            f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n"
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        )
        print(emojis(s))


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(
            f"WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size


def check_imshow():
    # Check if environment supports image displays
    try:
        assert (
            not is_docker()
        ), "cv2.imshow() is disabled in Docker environments"
        assert (
            not is_colab()
        ), "cv2.imshow() is disabled in Google Colab environments"
        cv2.imshow("test", np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(
            f"WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}"
        )
        return False


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffixes
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            assert (
                Path(f).suffix.lower() in suffix
            ), f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=(".yaml", ".yml")):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)


def check_file(file, suffix=""):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == "":  # exists
        return file
    elif file.startswith(("http:/", "https:/")):  # download
        url = str(Path(file)).replace(":/", "://")  # Pathlib turns :// -> :/
        file = Path(
            urllib.parse.unquote(file).split("?")[0]
        ).name  # '%2F' to '/', split https://url.com/file.txt?auth
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, file)
        assert (
            Path(file).exists() and Path(file).stat().st_size > 0
        ), f"File download failed: {url}"  # check
        return file
    else:  # search
        files = []
        for d in "data", "models", "utils":  # search directories
            files.extend(
                glob.glob(str(ROOT / d / "**" / file), recursive=True)
            )  # find file
        assert len(files), f"File not found: {file}"  # assert file was found
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def check_dataset(data, autodownload=True):
    # Download and/or unzip dataset if not found locally
    # Usage: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip

    # Download (optional)
    extract_dir = ""
    if isinstance(data, (str, Path)) and str(data).endswith(
        ".zip"
    ):  # i.e. gs://bucket/dir/coco128.zip
        download(
            data,
            dir="../datasets",
            unzip=True,
            delete=False,
            curl=False,
            threads=1,
        )
        data = next((Path("../datasets") / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors="ignore") as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = extract_dir or Path(
        data.get("path") or ""
    )  # optional 'path' default to '.'
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            data[k] = (
                str(path / data[k])
                if isinstance(data[k], str)
                else [str(path / x) for x in data[k]]
            )

    assert "nc" in data, "Dataset 'nc' key missing."
    if "names" not in data:
        data["names"] = [
            f"class{i}" for i in range(data["nc"])
        ]  # assign class names if missing
    train, val, test, s = [
        data.get(x) for x in ("train", "val", "test", "download")
    ]
    if val:
        val = [
            Path(x).resolve()
            for x in (val if isinstance(val, list) else [val])
        ]  # val path
        if not all(x.exists() for x in val):
            print(
                "\nWARNING: Dataset not found, nonexistent paths: %s"
                % [str(x) for x in val if not x.exists()]
            )
            if s and autodownload:  # download script
                root = (
                    path.parent if "path" in data else ".."
                )  # unzip directory i.e. '../'
                if s.startswith("http") and s.endswith(".zip"):  # URL
                    f = Path(s).name  # filename
                    print(f"Downloading {s} to {f}...")
                    torch.hub.download_url_to_file(s, f)
                    Path(root).mkdir(
                        parents=True, exist_ok=True
                    )  # create root
                    ZipFile(f).extractall(path=root)  # unzip
                    Path(f).unlink()  # remove zip
                    r = None  # success
                elif s.startswith("bash "):  # bash script
                    print(f"Running {s} ...")
                    r = os.system(s)
                else:  # python script
                    r = exec(s, {"yaml": data})  # return None
                print(
                    f"Dataset autodownload {f'success, saved to {root}' if r in (0, None) else 'failure'}\n"
                )
            else:
                raise Exception("Dataset not found.")

    return data  # dictionary


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(":/", "://")  # Pathlib turns :// -> :/
    file = Path(urllib.parse.unquote(url)).name.split("?")[
        0
    ]  # '%2F' to '/', split https://url.com/file.txt?auth
    return file


def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1):
    # Multi-threaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        # Download 1 file
        f = dir / Path(url).name  # filename
        if Path(url).is_file():  # exists in current path
            Path(url).rename(f)  # move to dir
        elif not f.exists():
            print(f"Downloading {url} to {f}...")
            if curl:
                os.system(
                    f"curl -L '{url}' -o '{f}' --retry 9 -C -"
                )  # curl download, retry and resume on fail
            else:
                torch.hub.download_url_to_file(
                    url, f, progress=True
                )  # torch download
        if unzip and f.suffix in (".zip", ".gz"):
            print(f"Unzipping {f}...")
            if f.suffix == ".zip":
                ZipFile(f).extractall(path=dir)  # unzip
            elif f.suffix == ".gz":
                os.system(f"tar xfz {f} --directory {f.parent}")  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(
            lambda x: download_one(*x), zip(url, repeat(dir))
        )  # multi-threaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array(
        [np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels]
    )
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = (
        x[inside],
        y[inside],
    )
    return (
        np.array([x.min(), y.min(), x.max(), y.max()])
        if any(x)
        else np.zeros((1, 4))
    )  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)])
            .reshape(2, -1)
            .T
        )  # segment xy
    return segments


def retrieve_image(img, img1_shape, img0_shape, ratio_pad=None):
    # retrieve image from img1_shape to img0_shape
    gain_width = ratio_pad[0][0]
    gain_height = ratio_pad[0][0]

    # print(ratio_pad[1][1], img0_shape[0]/gain+ratio_pad[1][1], ratio_pad[1][0], img0_shape[1]/gain+ratio_pad[1][0])
    # print(img.shape)
    # og_img = img[ int(ratio_pad[1][1]): int(img0_shape[0]/gain+ratio_pad[1][1]), int(ratio_pad[1][0]): int(img0_shape[1]/gain+ratio_pad[1][0])] # this worked for x-ray
    og_img = img[
        int(ratio_pad[1][1]) : int(
            img0_shape[0] * gain_height + ratio_pad[1][1]
        ),
        int(ratio_pad[1][0]) : int(
            img0_shape[1] * gain_width + ratio_pad[1][0]
        ),
    ]
    og_img = cv2.resize(og_img, (img0_shape[1], img0_shape[0]))
    return og_img


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8, 10, 12, 14, 16]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9, 11, 13, 15, 17]] -= pad[1]  # y padding
    coords[:, :18] /= gain
    # clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    min_wh, max_wh = (
        2,
        4096,
    )  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [
        torch.zeros((0, 6), device=prediction.device)
    ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = (
            x[:, :4] + c,
            x[:, 4],
        )  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (
            1 < n < 3e3
        ):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def strip_optimizer(
    f="best.pt", s=""
):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with ema
    for k in (
        "optimizer",
        "training_results",
        "wandb_id",
        "ema",
        "updates",
    ):  # keys
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1e6  # filesize
    print(
        f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB"
    )


def print_mutation(results, hyp, save_dir, bucket):
    evolve_csv, results_csv, evolve_yaml = (
        save_dir / "evolve.csv",
        save_dir / "results.csv",
        save_dir / "hyp_evolve.yaml",
    )
    keys = (
        "metrics/precision",
        "metrics/recall",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
        "val/box_loss",
        "val/obj_loss",
        "val/cls_loss",
    ) + tuple(
        hyp.keys()
    )  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)

    # Download (optional)
    if bucket:
        url = f"gs://{bucket}/evolve.csv"
        if gsutil_getsize_1(url) > (
            os.path.getsize(evolve_csv) if os.path.exists(evolve_csv) else 0
        ):
            os.system(
                f"gsutil cp {url} {save_dir}"
            )  # download evolve.csv if larger than local

    # Log to evolve.csv
    s = (
        ""
        if evolve_csv.exists()
        else (("%20s," * n % keys).rstrip(",") + "\n")
    )  # add header
    with open(evolve_csv, "a") as f:
        f.write(s + ("%20.5g," * n % vals).rstrip(",") + "\n")

    # Print to screen
    print(colorstr("evolve: ") + ", ".join(f"{x.strip():>20s}" for x in keys))
    print(
        colorstr("evolve: ") + ", ".join(f"{x:20.5g}" for x in vals),
        end="\n\n\n",
    )

    # Save yaml
    with open(evolve_yaml, "w") as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :7]))  #
        f.write(
            "# YOLOv5 Hyperparameter Evolution Results\n"
            + f"# Best generation: {i}\n"
            + f"# Last generation: {len(data)}\n"
            + "# "
            + ", ".join(f"{x.strip():>20s}" for x in keys[:7])
            + "\n"
            + "# "
            + ", ".join(f"{x:>20.5g}" for x in data.values[i, :7])
            + "\n\n"
        )
        yaml.safe_dump(hyp, f, sort_keys=False)

    if bucket:
        os.system(
            f"gsutil cp {evolve_csv} {evolve_yaml} gs://{bucket}"
        )  # upload


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]) : int(a[3]), int(a[0]) : int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('example%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(
                    2, 0, 1
                )  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(
                    im, dtype=np.float32
                )  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(
                1
            )  # classifier prediction
            x[i] = x[i][
                pred_cls1 == pred_cls2
            ]  # retain matching class detections

    return x


def save_one_box(
    xyxy,
    im,
    file="image.jpg",
    gain=1.02,
    pad=10,
    square=False,
    BGR=False,
    save=True,
):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = (
            b[:, 2:].max(1)[0].unsqueeze(1)
        )  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[
        int(xyxy[0, 1]) : int(xyxy[0, 3]),
        int(xyxy[0, 0]) : int(xyxy[0, 2]),
        :: (1 if BGR else -1),
    ]
    if save:
        cv2.imwrite(
            str(increment_path(file, mkdir=True).with_suffix(".jpg")), crop
        )
    return crop


def increment_path(path, exist_ok=False, sep="", mkdir=True):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n:03d}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory

    if not path.exists() and mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    # create "latest" linked to path
    latest = path.parent / "latest"
    if latest.exists():
        os.system(f"rm  -rf {latest}")
    if path.exists():
        os.symlink(path.name, latest)
    return str(path)


############### utils/torch_utils.py ###############


import logging
import math
import os
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None
# logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def git_describe():
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    if Path(".git").exists():
        return subprocess.check_output(
            "git describe --tags --long --always", shell=True
        ).decode("utf-8")[:-1]
    else:
        return ""


def select_device(device="", batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f"YOLOv5 {git_describe()} torch {torch.__version__} "  # string
    cpu = device.lower() == "cpu"
    if cpu:
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if (
            n > 1 and batch_size
        ):  # check that batch_size is compatible with device_count
            assert (
                batch_size % n == 0
            ), f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * len(s)
        for i, d in enumerate(device.split(",") if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += "CPU\n"

    logger.info(s)  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(x, ops, n=100, device=None):
    # profile a pytorch module or list of modules. Example usage:
    #     x = torch.randn(16, 3, 640, 640)  # input
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(x, [m1, m2], n=100)  # profile speed over 100 iterations

    device = device or torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    x = x.to(device)
    x.requires_grad = True
    print(
        torch.__version__,
        device.type,
        torch.cuda.get_device_properties(0) if device.type == "cuda" else "",
    )
    print(
        f"\n{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}"
    )
    for m in ops if isinstance(ops, list) else [ops]:
        m = m.to(device) if hasattr(m, "to") else m  # device
        m = (
            m.half()
            if hasattr(m, "half")
            and isinstance(x, torch.Tensor)
            and x.dtype is torch.float16
            else m
        )  # type
        dtf, dtb, t = 0.0, 0.0, [0.0, 0.0, 0.0]  # dt forward, backward
        try:
            flops = (
                thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2
            )  # GFLOPS
        except:
            flops = 0

        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:  # no backward method
                t[2] = float("nan")
            dtf += (t[1] - t[0]) * 1000 / n  # ms per op forward
            dtb += (t[2] - t[1]) * 1000 / n  # ms per op backward

        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else "list"
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else "list"
        p = (
            sum(list(x.numel() for x in m.parameters()))
            if isinstance(m, nn.Module)
            else 0
        )  # parameters
        print(
            f"{p:12.4g}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}"
        )


def is_parallel(model):
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {
        k: v
        for k, v in da.items()
        if k in db
        and not any(x in k for x in exclude)
        and v.shape == db[k].shape
    }


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [
        i for i, m in enumerate(model.module_list) if isinstance(m, mclass)
    ]


def sparsity(model):
    # Return global model sparsity
    a, b = 0.0, 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune

    print("Pruning model... ", end="")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    print(" %.3g global sparsity" % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(
        torch.mm(w_bn, w_conv).view(fusedconv.weight.size())
    )

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn
    )

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(
        x.numel() for x in model.parameters() if x.requires_grad
    )  # number gradients
    if verbose:
        print(
            "%5s %40s %9s %12s %20s %10s %10s"
            % (
                "layer",
                "name",
                "gradient",
                "parameters",
                "shape",
                "mu",
                "sigma",
            )
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                )
            )

    try:  # FLOPS
        from thop import profile

        stride = (
            max(int(model.stride.max()), 32)
            if hasattr(model, "stride")
            else 32
        )
        img = torch.zeros(
            (1, model.yaml.get("ch", 3), stride, stride),
            device=next(model.parameters()).device,
        )  # input
        flops = (
            profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1e9 * 2
        )  # stride GFLOPS
        img_size = (
            img_size if isinstance(img_size, list) else [img_size, img_size]
        )  # expand if int/float
        fs = ", %.1f GFLOPS" % (
            flops * img_size[0] / stride * img_size[1] / stride
        )  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ""

    logger.info(
        f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"
    )


def load_classifier(name="resnet101", n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)

    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n
    return model


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(
            img, size=s, mode="bilinear", align_corners=False
        )  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(
            img, [0, w - s[1], 0, h - s[0]], value=0.447
        )  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (
            (len(include) and k not in include)
            or k.startswith("_")
            or k in exclude
        ):
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(
            model.module if is_parallel(model) else model
        ).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / 2000)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = (
                model.module.state_dict()
                if is_parallel(model)
                else model.state_dict()
            )  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(
        self, model, include=(), exclude=("process_group", "reducer")
    ):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


############### utils/pose_utils.py ###############

import math
import os
import time

import cv2
import numpy as np
import torch
from matplotlib.path import Path as mat_Path
from scipy import spatial


def get_all_files(directory):
    files = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files


def calcAngularDistance(gt_rot, pr_rot):

    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff)
    return np.rad2deg(np.arccos((trace - 1.0) / 2.0))


def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])


def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype="float32")
    camera_projection = (internal_calibration.dot(transformation)).dot(
        points_3D
    )
    projections_2d[0, :] = camera_projection[0, :] / camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :] / camera_projection[2, :]
    return projections_2d


def calc_pts_diameter(pts):
    diameter = -1
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


def adi(pts_est, pts_gt):
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    e = nn_dists.mean()
    return e


def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)


from pdebug.utils.ddd import get_3D_corners as get_3D_corners__pdebug
from pdebug.utils.ddd import load_points_3d_from_cad


def get_3D_corners(vertices):

    min_x = np.min(vertices[0, :])
    max_x = np.max(vertices[0, :])
    min_y = np.min(vertices[1, :])
    max_y = np.max(vertices[1, :])
    min_z = np.min(vertices[2, :])
    max_z = np.max(vertices[2, :])
    corners = np.array(
        [
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ]
    )

    corners = np.concatenate((np.transpose(corners), np.ones((1, 8))), axis=0)
    return corners


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype="float32")

    assert (
        points_3D.shape[0] == points_2D.shape[0]
    ), "points 3D and points 2D must have same number of vertices"

    _, R_exp, t = cv2.solvePnP(
        points_3D,
        np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
        cameraMatrix,
        distCoeffs,
    )

    R, _ = cv2.Rodrigues(R_exp)
    return R, t


def epnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype="float32")

    assert (
        points_2D.shape[0] == points_2D.shape[0]
    ), "points 3D and points 2D must have same number of vertices"

    _, R_exp, t = cv2.solvePnP(
        points_3D,
        np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    R, _ = cv2.Rodrigues(R_exp)
    return R, t


def PnPRansac(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype="float32")

    assert (
        points_2D.shape[0] == points_2D.shape[0]
    ), "points 3D and points 2D must have same number of vertices"

    _, R_exp, t, _ = cv2.solvePnPRansac(
        points_3D,
        np.ascontiguousarray(points_2D[:, :2]).reshape((-1, 1, 2)),
        cameraMatrix,
        distCoeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )

    R, _ = cv2.Rodrigues(R_exp)
    return R, t


def fix_corner_order(corners2D_gt):
    corners2D_gt_corrected = np.zeros((9, 2), dtype="float32")
    corners2D_gt_corrected[0, :] = corners2D_gt[0, :]
    corners2D_gt_corrected[1, :] = corners2D_gt[1, :]
    corners2D_gt_corrected[2, :] = corners2D_gt[3, :]
    corners2D_gt_corrected[3, :] = corners2D_gt[5, :]
    corners2D_gt_corrected[4, :] = corners2D_gt[7, :]
    corners2D_gt_corrected[5, :] = corners2D_gt[2, :]
    corners2D_gt_corrected[6, :] = corners2D_gt[4, :]
    corners2D_gt_corrected[7, :] = corners2D_gt[6, :]
    corners2D_gt_corrected[8, :] = corners2D_gt[8, :]
    return corners2D_gt_corrected


def corner_confidence(
    gt_corners,
    pr_corners,
    im_grid_width,
    im_grid_height,
    th=0.25,
    sharpness=2,
    device=None,
):
    """gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
    pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
    th        : distance threshold, type: int
    sharpness : sharpness of the exponential that assigns a confidence value to the distance
    -----------
    return    : a list of shape (9,) with 9 confidence values
    """

    th = th * (np.sqrt(im_grid_width**2 + im_grid_height**2))
    dist = gt_corners - pr_corners
    dist = torch.reshape(dist, (-1, 9, 2))

    eps = 1e-5
    dist = torch.sqrt(torch.sum((dist) ** 2, dim=2))
    mask = (dist < th).type(torch.FloatTensor).to(device)
    conf = (torch.exp(sharpness * (1.0 - dist / th)) - 1).to(device)
    conf0 = torch.exp(torch.FloatTensor([sharpness])) - 1 + eps
    conf0 = conf0.to(device)
    conf = conf / conf0.repeat(1, 9)
    conf = mask * conf
    return torch.mean(conf, dim=1)


def do_nms(boxes, nms_iou_threshold):
    """
    Non-Maximum Suppression using numpy.

    Args:
        boxes: numpy array of shape [N, 5] where each row is [x1, y1, x2, y2, score]
        iou_threshold: IoU threshold for suppression

    Returns:
        indices: numpy array of indices that are kept after NMS
    """
    boxes = boxes.astype(np.float32)
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    # Extract coordinates and scores
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by scores in descending order
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        # Select the box with highest score
        i = order[0]
        keep.append(i)

        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Compute width and height of intersection
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # Compute IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Find boxes with IoU less than threshold
        inds = np.where(iou <= nms_iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def box_filter(
    prediction, conf_thres=0.01, classes=None, multi_label=False, max_det=1
):
    """Performs box filtering on inference results

    Args:
        prediction: (num_predict, N, 18 + num_classes)

    Returns:
         detections with shape: [(M, 20)],  (x0, y0,.., x8, y8, conf, cls)
    """
    num_predict = prediction.shape[0]
    num_classes = prediction.shape[2] - 19  # number of classes
    xc = prediction[:, :, 18] > conf_thres  # candidates

    # Settings
    max_det = max_det  # maximum number of detections per image
    # max_nms = 1
    time_limit = 1.0  # seconds to quit after
    multi_label = num_classes > 1 and multi_label

    t = time.time()
    output = [torch.zeros((0, 20))] * num_predict
    for xi, x in enumerate(prediction):  # image index, image inference

        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 19:] *= x[:, 18:19]  # conf = obj_conf * cls_conf
        box = x[:, :18]

        # do nms
        num_boxes = box.shape[0]
        x1 = (
            box.cpu()
            .numpy()
            .reshape(num_boxes, -1, 2)[:, :, 0]
            .min(1)[:, None]
        )
        x2 = (
            box.cpu()
            .numpy()
            .reshape(num_boxes, -1, 2)[:, :, 0]
            .max(1)[:, None]
        )
        y1 = (
            box.cpu()
            .numpy()
            .reshape(num_boxes, -1, 2)[:, :, 1]
            .min(1)[:, None]
        )
        y2 = (
            box.cpu()
            .numpy()
            .reshape(num_boxes, -1, 2)[:, :, 1]
            .max(1)[:, None]
        )
        score = x[:, 18].cpu().numpy()[:, None]
        boxes = np.concatenate((x1, y1, x2, y2, score), axis=1)
        nms_idx = do_nms(boxes, opt.nms_iou_threshold)
        # from pdebug.visp import draw
        # bg = np.zeros((240, 320, 3), dtype=np.uint8)
        # vis_boxes = draw.boxes(bg, boxes)
        x = x[nms_idx]
        box = box[nms_idx]

        # # elif n > max_nms:  # excess boxes
        if multi_label:
            # TODO: what's meaning?
            i, j = (x[:, 19:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # set class_score (x[:, 19:]) to class_id
            conf, j = x[:, 19:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[
                (x[:, 19:20] == torch.tensor(classes, device=x.device)).any(1)
            ]

        n = x.shape[0]  # number of boxes
        # print(f"number of boxes {n}")
        if not n:  # no boxes
            continue
        elif n > max_det:
            x = x[
                x[:, 18].argsort(descending=True)[:max_det]
            ]  # sort by confidence

        # print(f"final{x=}")
        output[xi] = x
        if (time.time() - t) > time_limit:
            print(f"WARNING: filter time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


class MeshPly:
    def __init__(self, filename, color=[0.0, 0.0, 0.0]):

        f = open(filename, "r")
        self.vertices = []
        self.colors = []
        self.indices = []
        self.normals = []

        vertex_mode = False
        face_mode = False

        nb_vertices = 0
        nb_faces = 0

        idx = 0

        with f as open_file_object:
            for line in open_file_object:
                elements = line.split()
                if vertex_mode:
                    self.vertices.append([float(i) for i in elements[:3]])
                    self.normals.append([float(i) for i in elements[3:6]])

                    if elements[6:9]:
                        self.colors.append(
                            [float(i) / 255.0 for i in elements[6:9]]
                        )
                    else:
                        self.colors.append([float(i) / 255.0 for i in color])

                    idx += 1
                    if idx == nb_vertices:
                        vertex_mode = False
                        face_mode = True
                        idx = 0
                elif face_mode:
                    self.indices.append([float(i) for i in elements[1:4]])
                    idx += 1
                    if idx == nb_faces:
                        face_mode = False
                elif elements[0] == "element":
                    if elements[1] == "vertex":
                        nb_vertices = int(elements[2])
                    elif elements[1] == "face":
                        nb_faces = int(elements[2])
                elif elements[0] == "end_header":
                    vertex_mode = True


############### utils/occlude.py ###############

#!/usr/bin/env python

import functools
import os.path
import random
import sys
import xml.etree.ElementTree

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image


def main():
    """Demo of how to use the code"""
    import skimage.data

    # path = 'something/something/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    path = sys.argv[1]

    print("Loading occluders from Pascal VOC dataset...")
    occluders = load_occluders(pascal_voc_root_path=path)
    print("Found {} suitable objects".format(len(occluders)))

    original_im = cv2.resize(skimage.data.astronaut(), (256, 256))
    fig, axarr = plt.subplots(3, 3, figsize=(7, 7))
    for ax in axarr.ravel():
        occluded_im = occlude_with_objects(original_im, occluders)
        ax.imshow(occluded_im, interpolation="none")
        ax.axis("off")

    fig.tight_layout(h_pad=0)
    # plt.savefig('examples.jpg', dpi=150, bbox_inches='tight')
    plt.show()


def load_occluders(pascal_voc_root_path):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    annotation_paths = list_filepaths(
        os.path.join(pascal_voc_root_path, "Annotations")
    )
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = xml_root.find("segmented").text != "0"

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall("object")):
            is_person = obj.find("name").text == "person"
            is_difficult = obj.find("difficult").text != "0"
            is_truncated = obj.find("truncated").text != "0"
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find("bndbox")
                box = [
                    int(bndbox.find(s).text)
                    for s in ["xmin", "ymin", "xmax", "ymax"]
                ]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find("filename").text
        seg_filename = im_filename.replace("jpg", "png")

        im_path = os.path.join(pascal_voc_root_path, "JPEGImages", im_filename)
        seg_path = os.path.join(
            pascal_voc_root_path, "SegmentationObject", seg_filename
        )
        if not os.path.exists(im_path) or not os.path.exists(seg_path):
            continue

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(
                np.uint8
            ) * 255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate(
                [object_image, object_mask[..., np.newaxis]], axis=-1
            )

            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)
    return occluders


def occlude_with_objects(im, occluders):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_scale_factor = min(width_height) / 256
    count = np.random.randint(1, 8)

    for _ in range(count):
        occluder = random.choice(occluders)

        random_scale_factor = np.random.uniform(0.2, 1.0)
        scale_factor = random_scale_factor * im_scale_factor
        occluder = resize_by_factor(occluder, scale_factor)

        center = np.random.uniform([0, 0], width_height)
        paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1] : end_dst[1], start_dst[0] : end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1] : end_src[1], start_src[0] : end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    im_dst[start_dst[1] : end_dst[1], start_dst[0] : end_dst[0]] = (
        alpha * color_src + (1 - alpha) * region_dst
    )


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(
        np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int)
    )
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


# if __name__=='__main__':
#     main()

############### utils/datasets.py ###############

import glob
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ExifTags, Image, ImageMath, ImageOps
from torch.utils.data import Dataset
from tqdm import tqdm

## from utils.general import clean_str
## from utils.torch_utils import torch_distributed_zero_first
## from utils.pose_utils import get_3D_corners, pnp, get_camera_intrinsic, compute_projection, MeshPly
## from utils.occlude import load_occluders, occlude_with_objects

img_formats = [
    "bmp",
    "jpg",
    "jpeg",
    "png",
    "tif",
    "tiff",
    "dng",
    "webp",
]  # acceptable image suffixes
vid_formats = [
    "mov",
    "avi",
    "mp4",
    "mpg",
    "mpeg",
    "m4v",
    "wmv",
    "mkv",
]  # acceptable video suffixes
# logger = logging.getLogger(__name__)

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    opt,
    hyp=None,
    augment=True,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    world_size=1,
    workers=8,
    image_weights=False,
    prefix="",
    bg_file_names=None,
):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabelsPose(
            path,
            imgsz,
            batch_size,
            augment=augment,  # augment images
            hyp=hyp,  # augmentation hyperparameters
            rect=rect,  # rectangular training
            cache_images=cache,
            single_cls=opt.single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            bg_file_names=bg_file_names,
        )

    batch_size = min(batch_size, len(dataset))
    nw = min(
        [
            os.cpu_count() // world_size,
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if rank != -1
        else None
    )
    loader = (
        torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    )
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        # num_workers=0,
        sampler=sampler,
        pin_memory=True,
        collate_fn=LoadImagesAndLabelsPose.collate_fn,
    )
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(
            self, "batch_sampler", _RepeatSampler(self.batch_sampler)
        )
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in img_formats]
        videos = [x for x in files if x.split(".")[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = "image"
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(
                f"video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ",
                end="",
            )

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, "Image Not Found " + path
            print(f"image {self.count}/{self.nf} {path}: ", end="")

        # Padded resize
        img, _, pad = letterbox(img0, self.img_size, stride=self.stride)

        h0 = img0.shape[0]
        w0 = img0.shape[1]
        h = img.shape[0]
        w = img.shape[1]

        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, None, shapes

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe="0", img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord("q"):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f"Camera Error {self.pipe}"
        img_path = "webcam.jpg"
        print(f"webcam {self.count}: ", end="")

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, None

    def __len__(self):
        return 0


class LoadVideo:
    def __init__(
        self, sources, camera_intrinsics, mesh_path, img_size=640, stride=32
    ) -> None:
        pass
        self.img_size = img_size
        self.stride = stride
        with open(camera_intrinsics, "r") as f:
            camera_data = json.load(f)

        # camera_model = None
        self.dtx = np.array(camera_data["distortion"])
        self.mtx = np.array(camera_data["intrinsic"])
        self.intrinsics = []

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [
                    x.strip()
                    for x in f.read().strip().splitlines()
                    if len(x.strip())
                ]
        else:
            sources = [sources]

        n = len(sources)
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            self.intrinsics[i] = list(
                self.mtx[0, 0],
                self.mtx[1, 1],
                w,
                h,
                self.mtx[0, 2],
                self.mtx[1, 2],
                w,
                h,
            )
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {fps:.2f} FPS).")
            thread.start()
        print("")  # newline

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [
            letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0]
            for x in img0
        ]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(
            0, 3, 1, 2
        )  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, self.intrinics, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources="streams.txt", img_size=640, stride=32):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, "r") as f:
                sources = [
                    x.strip()
                    for x in f.read().strip().splitlines()
                    if len(x.strip())
                ]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [
            clean_str(x) for x in sources
        ]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f"{i + 1}/{n}: {s}... ", end="")
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f"Failed to open {s}"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({w}x{h} at {fps:.2f} FPS).")
            thread.start()
        print("")  # newline

        # check for common shapes
        s = np.stack(
            [
                letterbox(x, self.img_size, stride=self.stride)[0].shape
                for x in self.imgs
            ],
            0,
        )  # shapes
        self.rect = (
            np.unique(s, axis=0).shape[0] == 1
        )  # rect inference if all shapes equal
        if not self.rect:
            print(
                "WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams."
            )

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [
            letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0]
            for x in img0
        ]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(
            0, 3, 1, 2
        )  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    From https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,
        }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = (
        os.sep + "images" + os.sep,
        os.sep + "labels" + os.sep,
    )  # /images/, /labels/ substrings
    return [
        "txt".join(x.replace(sa, sb, 1).rsplit(x.split(".")[-1], 1))
        for x in img_paths
    ]


def Linemodimg2label_paths(img_paths):
    # Define label paths as a function of image paths
    return [
        x.replace("JPEGImages", "labels")
        .replace("images", "labels")
        .replace(".jpg", ".txt")
        .replace(".png", ".txt")
        for x in img_paths
    ]


def Linemodimg2mask_paths(img_paths):
    # Define mask paths as a function of image paths
    def rgb2mask(x, remove_two_zeros=True):
        x = (
            x.replace("JPEGImages", "mask")
            .replace("images", "mask")
            .replace(".jpg", ".png")
        )
        if remove_two_zeros:
            basename = os.path.basename(x)[2:]  # split '00' prefix
            return os.path.join(os.path.dirname(x), basename)
        else:
            return x

    remove_two_zeros = os.path.exists(rgb2mask(img_paths[0]))
    return [rgb2mask(x, remove_two_zeros) for x in img_paths]


def Linemodimg2mask_path(img_paths):
    # Define mask paths as a function of image paths
    return (
        img_paths.replace("JPEGImages", "mask")
        .replace("images", "mask")
        .replace("/00", "/")
        .replace(".jpg", ".png")
    )


class LoadImagesAndLabelsPose(Dataset):  # for training/testing
    def __init__(
        self,
        path,
        img_size=(640, 480),
        batch_size=16,
        augment=True,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        bg_file_names=None,
    ):

        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = rect

        self.stride = stride
        self.path = path
        self.bg_file_names = bg_file_names
        self.occluders = None

        if bg_file_names is not None and self.hyp["occlude"] > 0:

            occlude_path = os.path.join(
                bg_file_names[0].split("VOC2012")[0], "VOC2012"
            )
            print(f"Creating occluders from VOC: {occlude_path}")
            self.occluders = load_occluders(pascal_voc_root_path=occlude_path)

        data_dir = os.path.dirname(path)

        images_folder = None
        for folder in os.listdir(data_dir):
            if "images" in folder.lower():
                images_folder = os.path.join(data_dir, folder)

        if images_folder is None:
            raise Exception(
                f"Error loading data from {data_dir}: Could not find images folder"
            )
        # images_folder = os.path.join(data_dir, 'JPEGImages')
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:

                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, "r") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [
                            x.replace("./", parent)
                            if x.startswith("./")
                            else x
                            for x in t
                        ]  # local to global path

                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f"{prefix}{p} does not exist")
            self.img_files = sorted(
                [
                    os.path.join(images_folder, x.replace("/", os.sep))
                    for x in f
                    if x.split(".")[-1].lower() in img_formats
                ]
            )
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
            assert self.img_files, f"{prefix} No images found"
        except Exception as e:
            raise Exception(f"{prefix} Error loading data from {path}: {e}")

        # Check cache
        self.label_files = Linemodimg2label_paths(self.img_files)  # labels
        cache_path = (
            p if p.is_file() else Path(self.label_files[0]).parent
        ).with_suffix(
            ".cache"
        )  # cached labels
        self.mask_files = Linemodimg2mask_paths(self.img_files)  # mask
        if cache_path.is_file():

            cache, exists = (
                torch.load(cache_path, weights_only=False),
                True,
            )  # load
            if (
                cache["hash"]
                != get_hash(
                    self.label_files + self.mask_files + self.img_files
                )
                or "version" not in cache
            ):  # changed
                cache, exists = (
                    self.cache_data(cache_path, prefix),
                    False,
                )  # re-cache
        else:
            cache, exists = self.cache_data(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, nm_mask, nf_mask, ne_mask, nc_mask, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupted, total
        if exists:
            d = (
                f"Scanning '{cache_path}' for images, labels, masks... {nf} found, {nm} missing, {ne} empty, {nc} corrupted, "
                f"{nf_mask} found, {nm_mask} missing, {ne_mask} empty, {nc_mask} corrupted"
            )
            tqdm(
                None, desc=prefix + d, total=n, initial=n
            )  # display cache results
        assert (
            nf > 0 or not augment
        ), f"{prefix}No labels in {cache_path}. Can not train without labels."

        # Read cache
        cache.pop("hash")  # remove hash
        cache.pop("version")  # remove version

        labels, masks, shapes = zip(*cache.values())

        self.labels = list(labels)
        self.masks = list(masks)
        self.shapes = np.array(shapes, dtype=np.float64)

        self.img_files = list(cache.keys())  # update
        self.label_files = Linemodimg2label_paths(cache.keys())  # update

        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.masks = [self.masks[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = (
                np.ceil(np.array(shapes) * img_size / stride + pad).astype(int)
                * stride
            )

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(
                lambda x: load_image(*x), zip(repeat(self), range(n))
            )  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                (
                    self.imgs[i],
                    self.img_hw0[i],
                    self.img_hw[i],
                ) = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f"{prefix}Caching images ({gb / 1E9:.1f}GB)"

    def cache_data(self, path=Path("./data.cache"), prefix=""):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, nm_mask, nf_mask, ne_mask, nc_mask = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )  # number missing, found, empty, duplicate, number missing, found, empty, duplicate
        pbar = tqdm(
            zip(self.img_files, self.label_files, self.mask_files),
            desc="Scanning images",
            total=len(self.img_files),
        )

        for i, (im_file, lb_file, mk_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (
                    shape[1] > 9
                ), f"image size {shape} <10 pixels"
                assert (
                    im.format.lower() in img_formats
                ), f"invalid image format {im.format}"

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, "r") as f:
                        l = [x.split() for x in f.read().strip().splitlines()]
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert (
                            l.shape[1] >= 27
                        ), "labels at least require 21 columns each + 6 camera intrinsics"
                        assert (
                            np.unique(l, axis=0).shape[0] == l.shape[0]
                        ), "duplicate labels"
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 27), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 27), dtype=np.float32)
                    # raise NameError('label missing')

                # verify masks
                if os.path.isfile(mk_file):
                    nf_mask += 1  # mask found
                    mk = Image.open(mk_file)
                    mk.verify()
                    if exif_size(mk) != shape:
                        mk = np.zeros(shape, dtype=np.float32)
                        assert (
                            exif_size(mk) == shape
                        ), "mask does not fit image"
                else:
                    nm_mask += 1  # mask missing
                    mk = None
                    mk_file = None

                x[im_file] = [l, mk_file, shape]
            except Exception as e:
                nc_mask += 1
                print(
                    f"{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}"
                )

            pbar.desc = (
                f"{prefix}Scanning '{path.parent / path.stem}' for images, labels and masks... "
                f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted, "
                f"{nf_mask} found, {nm_mask} missing, {ne_mask} empty, {nc_mask} corrupted"
            )

        if nf == 0:
            print(f"{prefix}WARNING: No labels found in {path}.")
        if nf_mask == 0:
            print(f"{prefix}WARNING: No masks found in {path}.")

        x["hash"] = get_hash(
            self.label_files + self.mask_files + self.img_files
        )
        x["results"] = (
            nf,
            nm,
            ne,
            nc,
            nm_mask,
            nf_mask,
            ne_mask,
            nc_mask,
            i + 1,
        )
        x["version"] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f"{prefix}New cache created: {path}")
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp

        # Load image
        if opt.debug_single_sample:
            index = 11
            print(f"[debug_single_sample] fix index to {index}")
        img, (h0, w0), (h, w) = load_image(self, index)

        img2 = img

        # Augment background
        if self.augment:
            mask = cv2.imread(self.masks[index])
            if (
                hyp["background"]
                and self.bg_file_names is not None
                and self.masks[index] != None
            ):

                if random.random() < hyp["background"]:
                    # Get background image path
                    random_bg_index = random.randint(
                        0, len(self.bg_file_names) - 1
                    )
                    bgpath = self.bg_file_names[random_bg_index]
                    bg = cv2.imread(bgpath)
                    img = change_background(img, mask, bg)

        # Letterbox
        shape = (
            self.batch_shapes[self.batch[index]]
            if self.rect
            else self.img_size
        )  # final letterboxed shape
        img, ratio, pad = letterbox(
            img, shape, auto=False, scaleup=self.augment
        )
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        labels = self.labels[index].copy()
        nl = len(labels)  # number of labels
        labels_og = labels.copy()

        if labels.size:  # normalized xy to pixel xy format
            labels[:, 1:19] = xy_norm2xy_pix(
                labels[:, 1:19],
                ratio[0] * w,
                ratio[1] * h,
                padw=pad[0],
                padh=pad[1],
            )
            # labels[:, 19:21] = compute_new_width_height(labels[:, 1:19])

        if self.augment:

            # Augment colorspace
            augment_hsv(
                img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"]
            )

            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 1:19][:, ::2] = 1 - labels[:, 1:19][:, ::2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1:19][:, 1::2] = 1 - labels[:, 1:19][:, 1::2]
                # TODO: update intrinics

            img, labels = random_pose_perspective(
                img,
                labels,
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
            )

            if (
                hyp["occlude"]
                and self.bg_file_names is not None
                and self.masks[index] != None
            ):

                if random.random() < hyp["occlude"] and self.occluders:
                    img = occlude_with_objects(img, self.occluders)

            # Augment blur
            if random.random() < hyp["blur"]:
                img = gaussian_blur(img, hyp["blur_kernel"])

        if labels.size:  #  pixel xyxy format to normalized xy
            labels[:, 1:21] = xy_pix2xy_norm(
                labels[:, 1:21], img.shape[1], img.shape[0], padw=0, padh=0
            )
            labels[:, 19:21] = compute_new_width_height(labels[:, 1:19])

        if opt.debug_dataset or opt.debug_target:
            corners2D_gt = np.array(
                np.reshape(labels_og[0, 1 : 9 * 2 + 1], [9, 2]),
                dtype="float32",
            )
            corners2D_gt[:, 0] = corners2D_gt[:, 0] * w
            corners2D_gt[:, 1] = corners2D_gt[:, 1] * h
            corners2D_gt_aug = np.array(
                np.reshape(labels[0, 1 : 9 * 2 + 1], [9, 2]), dtype="float32"
            )
            corners2D_gt_aug[:, 0] = corners2D_gt_aug[:, 0] * img.shape[1]
            corners2D_gt_aug[:, 1] = corners2D_gt_aug[:, 1] * img.shape[0]

            corn2D_gt = corners2D_gt[1:, :]
            corn2D_gt_aug = corners2D_gt_aug[1:, :]

            edges_corners = [
                [0, 1],
                [0, 2],
                [0, 4],
                [1, 3],
                [1, 5],
                [2, 3],
                [2, 6],
                [3, 7],
                [4, 5],
                [4, 6],
                [5, 7],
                [6, 7],
            ]
            # # # Visualize

            # matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(img2[:, :, ::-1])  # base
            ax[0].scatter(corners2D_gt.T[0], corners2D_gt.T[1], c="r", s=20)
            for i, (x, y) in enumerate(corners2D_gt):
                ax[0].annotate(
                    str(i),
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="white",
                )

            ax[1].imshow(img[:, :, ::-1])  # warped
            ax[1].scatter(
                corners2D_gt_aug.T[0], corners2D_gt_aug.T[1], c="r", s=20
            )
            # Projections
            for edge in edges_corners:
                ax[1].plot(
                    corn2D_gt_aug[edge, 0],
                    corn2D_gt_aug[edge, 1],
                    color="g",
                    linewidth=1.0,
                )
                ax[0].plot(
                    corn2D_gt[edge, 0],
                    corn2D_gt[edge, 1],
                    color="g",
                    linewidth=1.0,
                )
            savename = (
                opt.debug_dataset_output
                + "/"
                + str(len(os.listdir(opt.debug_dataset_output)))
                + ".png"
            )
            plt.savefig(savename)
            if not opt.debug_target:
                __import__("ipdb").set_trace()
                pass

        labels_out = torch.zeros((nl, 22))
        intrinics = torch.zeros((nl, 6))

        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels[:, :21])
            intrinics[:, :] = torch.from_numpy(labels[:, 21:27])
        # Convert
        if opt.debug_target:
            save_dir = Path("debug_target")
            save_dir.mkdir(exist_ok=True)
            cv2.imwrite(f"{save_dir}/tmp_debug_target_rgb.png", img)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        return (
            torch.from_numpy(img),
            labels_out,
            intrinics,
            self.img_files[index],
            shapes,
        )

    @staticmethod
    def collate_fn(batch):
        img, label, intrinics, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
            # intrinics[i][:, 0] = i
        return (
            torch.stack(img, 0),
            torch.cat(label, 0),
            torch.cat(intrinics, 0),
            path,
            shapes,
        )


def xy_norm2xy_pix(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, ::2] = w * (x[:, ::2]) + padw
    y[:, 1::2] = h * (x[:, 1::2]) + padh
    return y


def xy_pix2xy_norm(x, w=640, h=640, padw=0, padh=0):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, ::2] = (x[:, ::2] - padw) / w
    y[:, 1::2] = (x[:, 1::2] - padh) / h
    return y


def gaussian_blur(cv_image, kernel_size=5):
    return cv2.GaussianBlur(
        cv_image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT
    )


def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        if opt.debug_single_sample:
            print(f"[debug_single_sample] rgb path: {path}")

        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if (
            r != 1
        ):  # always resize down, only resize up if training with augmentation
            interp = (
                cv2.INTER_AREA
                if r < 1 and not self.augment
                else cv2.INTER_LINEAR
            )
            img = cv2.resize(
                img, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return (
            self.imgs[index],
            self.img_hw0[index],
            self.img_hw[index],
        )  # img, hw_original, hw_resized


def resize_mask(msk, img_size, augment):

    h0, w0 = msk.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if (
        r != 1
    ):  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        msk = cv2.resize(msk, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return msk


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def hist_equalize(img, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'img' with img.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(
            yuv[:, :, 0]
        )  # equalize Y channel histogram
    return cv2.cvtColor(
        yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB
    )  # convert YUV image to RGB


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(
            random.uniform(0, w - bw)
        )  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[
            y1b:y2b, x1b:x2b
        ]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(
            labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0
        )

    return img, labels


def letterbox(
    img,
    new_shape=(640, 480),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (
        new_shape[1] - new_unpad[0],
        new_shape[0] - new_unpad[1],
    )  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = (
            new_shape[1] / shape[1],
            new_shape[0] / shape[0],
        )  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return img, ratio, (dw, dh)


def change_background(img, mask, bg):

    # ow, oh = img.shape[:2]

    bg = cv2.resize(bg, img.shape[:2][::-1])  # [:, :, ::-1]
    mask = cv2.resize(
        mask, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    # bg = np.array(bg)
    # mask = mask
    if mask.ndim == 2:
        mask = np.stack((mask, mask, mask), axis=2)

    new_img = np.where(mask[:, :, :] == True, img, bg)

    return new_img


def random_pose_perspective(
    img,
    targets=(),
    segments=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):

    # targets = [cls, xy*9, label_width, label_height]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(
        -perspective, perspective
    )  # x perspective (about y)
    P[2, 1] = random.uniform(
        -perspective, perspective
    )  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)

    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    if getattr(opt, "random_scale_large_only", False):
        s = random.uniform(1, 1 + scale)
    else:
        s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(
        random.uniform(-shear, shear) * math.pi / 180
    )  # x shear (deg)
    S[1, 0] = math.tan(
        random.uniform(-shear, shear) * math.pi / 180
    )  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (
        (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any()
    ):  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 9, 3))
        xy[:, :2] = targets[:, 1:19].reshape(
            n * 9, 2
        )  # x1y1, x2y2, x3y3, x4y4, ...., x9y9
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
            n, 18
        )  # perspective rescale or affine
        targets[:, 1:19] = xy
        targets[:, 19:21] = compute_new_width_height(xy)

    return img, targets


def augmentation_6DoF(
    img,
    mask,
    rotation_matrix_annos,
    translation_vector_annos,
    angle,
    scale,
    camera_matrix,
    mask_values,
):

    """Computes the 6D augmentation.
    Args:
        img: The image to augment
        mask: The segmentation mask of the image
        rotation_matrix_annos: numpy array with shape (num_annotations, 3, 3) which contains the ground truth rotation matrix for each annotated object in the image
        translation_vector_annos: numpy array with shape (num_annotations, 3) which contains the ground truth translation vectors for each annotated object in the image
        angle: rotate the image with the given angle
        scale: scale the image with the given scale
        camera_matrix: The camera matrix of the example
        mask_values: numpy array of shape (num_annotations,) containing the segmentation mask value of each annotated object
    Returns:
        augmented_img: The augmented image
        augmented_rotation_vector_annos: numpy array with shape (num_annotations, 3) which contains the augmented ground truth rotation vectors for each annotated object in the image
        augmented_translation_vector_annos: numpy array with shape (num_annotations, 3) which contains the augmented ground truth translation vectors for each annotated object in the image
        augmented_bbox_annos: numpy array with shape (num_annotations, 4) which contains the augmented ground truth 2D bounding boxes for each annotated object in the image
        still_valid_annos: numpy boolean array of shape (num_annotations,) indicating if the augmented annotation of each object is still valid or not (object rotated out of the image for example)
        is_valid_augmentation: Boolean indicating wheter there is at least one valid annotated object after the augmentation
    """

    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    height, width, _ = img.shape
    # rotate and scale image

    a = random.uniform(-angle, angle) if angle != 0 else 0.0
    s = random.uniform(1 - scale, 1 + scale) if scale != 0 else 0.0
    # print(a, s)
    rot_2d_mat = cv2.getRotationMatrix2D((cx, cy), -a, s)

    augmented_img = cv2.warpAffine(img, rot_2d_mat, (width, height))
    # append the affine transformation also to the mask to extract the augmented bbox afterwards
    augmented_mask = cv2.warpAffine(
        mask, rot_2d_mat, (width, height), flags=cv2.INTER_NEAREST
    )  # use nearest neighbor interpolation to keep valid mask values
    num_annos = rotation_matrix_annos.shape[0]

    augmented_rotation_matrix_annos = np.zeros(
        (num_annos, 3, 3), dtype=np.float32
    )
    augmented_translation_vector_annos = np.zeros(
        (num_annos, 3), dtype=np.float32
    )
    augmented_bbox_annos = np.zeros((num_annos, 4), dtype=np.float32)

    still_valid_annos = np.zeros(
        (num_annos,), dtype=bool
    )  # flag for the annotations if they are still in the image and usable after augmentation or not
    for i in range(num_annos):
        augmented_bbox, is_valid_augmentation = get_bbox_from_mask(
            augmented_mask, mask_value=mask_values[i]
        )

        if not is_valid_augmentation:
            still_valid_annos[i] = False
            continue
        # create additional rotation vector representing the rotation of the given angle around the z-axis in the camera coordinate system
        tmp_rotation_vector = np.zeros((3,))
        tmp_rotation_vector[2] = a / 180.0 * math.pi
        tmp_rotation_matrix, _ = cv2.Rodrigues(tmp_rotation_vector)
        # get the final augmentation rotation
        augmented_rotation_matrix = np.dot(
            tmp_rotation_matrix, rotation_matrix_annos[i, :, :]
        )
        # augmented_rotation_vector, _ = cv2.Rodrigues(augmented_rotation_matrix)

        # also rotate the gt translation vector first and then adjust Tz with the given augmentation scale
        augmented_translation_vector = np.dot(
            np.copy(translation_vector_annos[i]), tmp_rotation_matrix.T
        )
        augmented_translation_vector[2] /= s

        # fill in augmented annotations
        augmented_rotation_matrix_annos[
            i, :, :
        ] = augmented_rotation_matrix  # np.squeeze(augmented_rotation_vector)
        augmented_translation_vector_annos[i, :] = np.squeeze(
            augmented_translation_vector
        )
        augmented_bbox_annos[i, :] = augmented_bbox
        still_valid_annos[i] = True

    return (
        augmented_img,
        augmented_rotation_matrix_annos,
        augmented_translation_vector_annos,
        augmented_bbox_annos,
        still_valid_annos,
        True,
    )


def get_bbox_from_mask(mask, mask_value=None):
    """Computes the 2D bounding box from the input mask
    Args:
        mask: The segmentation mask of the image
        mask_value: The integer value of the object in the segmentation mask
    Returns:
        numpy array with shape (4,) containing the 2D bounding box
        Boolean indicating if the object is found in the given mask or not
    """
    if mask_value is None:
        seg = np.where(mask != 0)
    else:
        seg = np.where(mask == mask_value)
    # check if mask is empty
    if seg[0].size <= 0 or seg[1].size <= 0:
        return np.zeros((4,), dtype=np.float32), False
    min_x = np.min(seg[1])
    min_y = np.min(seg[0])
    max_x = np.max(seg[1])
    max_y = np.max(seg[0])

    return np.array([min_x, min_y, max_x, max_y], dtype=np.float32), True


def compute_new_width_height(coordinates):
    return np.array(
        [
            [
                np.max(coordinates[:, ::2]) - np.min(coordinates[:, ::2]),
                np.max(coordinates[:, 1::2]) - np.min(coordinates[:, 1::2]),
            ]
        ]
    )


def create_folder(path="./new"):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path="../coco128"):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + "_flat")
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + "/**/*.*", recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


############### utils/plots.py ###############


import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

## from utils.general import xywh2xyxy, xyxy2xywh
## from utils.metrics import fitness

matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    return [
        hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()
    ]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(
        y.min(), y.max(), n
    )
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype="low", analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle(
            [box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]],
            fill=tuple(color),
        )
        draw.text(
            (box[0], box[1] - txt_height + 1),
            label,
            fill=(255, 255, 255),
            font=font,
        )
    return np.asarray(img)


def plot_wh_methods():  # from utils.plots import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, 0.1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(x, ya, ".-", label="YOLOv3")
    plt.plot(x, yb**2, ".-", label="YOLOv5 ^2")
    plt.plot(x, yb**1.6, ".-", label="YOLOv5 ^1.6")
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.grid()
    plt.legend()
    fig.savefig("comparison.png", dpi=200)


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append(
                [i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf]
            )
    return np.array(targets)


def plot_images(
    images,
    targets,
    paths=None,
    fname="images.jpg",
    names=None,
    max_size=640,
    max_subplots=16,
):
    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors
    mosaic = np.full(
        (int(ns * h), int(ns * w), 3), 255, dtype=np.uint8
    )  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y : block_y + h, block_x : block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype("int")
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = (
                None if labels else image_targets[:, 6]
            )  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif (
                    scale_factor < 1
                ):  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = (
                        "%s" % cls if labels else "%s %.1f" % (cls, conf[j])
                    )
                    plot_one_box(
                        box,
                        mosaic,
                        label=label,
                        color=color,
                        line_thickness=tl,
                    )

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[
                0
            ]
            cv2.putText(
                mosaic,
                label,
                (block_x + 5, block_y + t_size[1] + 5),
                0,
                tl / 3,
                [220, 220, 220],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        # Image border
        cv2.rectangle(
            mosaic,
            (block_x, block_y),
            (block_x + w, block_y + h),
            (255, 255, 255),
            thickness=3,
        )

    if fname:
        r = min(1280.0 / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(
            mosaic,
            (int(ns * w * r), int(ns * h * r)),
            interpolation=cv2.INTER_AREA,
        )
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=""):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(
        scheduler
    )  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]["lr"])
    plt.plot(y, ".-", label="LR")
    plt.xlabel("epoch")
    plt.ylabel("LR")
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / "LR.png", dpi=200)
    plt.close()


def plot_test_txt():  # from utils.plots import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt("test.txt", dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect("equal")
    plt.savefig("hist2d.png", dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig("hist1d.png", dpi=200)


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt("targets.txt", dtype=np.float32).T
    s = ["x targets", "y targets", "width targets", "height targets"]
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(
            x[i], bins=100, label="%.3g +/- %.3g" % (x[i].mean(), x[i].std())
        )
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig("targets.jpg", dpi=200)


def plot_study_txt(
    path="", x=None
):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    # ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']]:
    for f in sorted(Path(path).glob("study*.txt")):
        y = np.loadtxt(
            f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2
        ).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = [
            "P",
            "R",
            "mAP@.5",
            "mAP@.5:.95",
            "t_inference (ms/img)",
            "t_NMS (ms/img)",
            "t_total (ms/img)",
        ]
        # for i in range(7):
        #     ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
        #     ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(
            y[6, :j],
            y[3, :j] * 1e2,
            ".-",
            linewidth=2,
            markersize=8,
            label=f.stem.replace("study_coco_", "").replace("yolo", "YOLO"),
        )

    ax2.plot(
        1e3 / np.array([209, 140, 97, 58, 35, 18]),
        [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
        "k.-",
        linewidth=2,
        markersize=8,
        alpha=0.25,
        label="EfficientDet",
    )

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 30)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel("GPU Speed (ms/img)")
    ax2.set_ylabel("COCO AP val")
    ax2.legend(loc="lower right")
    plt.savefig(str(Path(path).name) + ".png", dpi=300)


def plot_labels(labels, save_dir=Path(""), loggers=None):
    import seaborn as sns

    # plot dataset labels
    print("Plotting labels... ")
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    colors = color_list()
    x = pd.DataFrame(b.transpose(), columns=["x", "y", "width", "height"])

    # seaborn correlogram
    sns.pairplot(
        x,
        corner=True,
        diag_kind="auto",
        kind="hist",
        diag_kws=dict(bins=50),
        plot_kws=dict(pmax=0.9),
    )
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use("svg")  # faster
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel("classes")
    sns.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(
            box, width=1, outline=colors[int(cls) % 10]
        )  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / "labels.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()

    # loggers
    for k, v in loggers.items() or {}:
        if k == "wandb" and v:
            v.log(
                {
                    "Labels": [
                        v.Image(str(x), caption=x.name)
                        for x in save_dir.glob("*labels*.jpg")
                    ]
                },
                commit=False,
            )


def plot_evolution(
    yaml_file="data/hyp.finetune.yaml",
):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt("evolve.txt", ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc("font", **{"size": 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(
            y,
            f,
            c=hist2d(y, f, 20),
            cmap="viridis",
            alpha=0.8,
            edgecolors="none",
        )
        plt.plot(mu, f.max(), "k+", markersize=15)
        plt.title(
            "%s = %.3g" % (k, mu), fontdict={"size": 9}
        )  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print("%15s: %.3g" % (k, mu))
    plt.savefig("evolve.png", dpi=200)
    print("\nPlot saved as evolve.png")


def profile_idetection(start=0, stop=0, labels=(), save_dir=""):
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = [
        "Images",
        "Free Storage (GB)",
        "RAM Usage (GB)",
        "Battery",
        "dt_raw (ms)",
        "dt_smooth (ms)",
        "real-world FPS",
    ]
    files = list(Path(save_dir).glob("frames*.txt"))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[
                :, 90:-30
            ]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = results[0] - results[0].min()  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = (
                        labels[fi]
                        if len(labels)
                        else f.stem.replace("frames_", "")
                    )
                    a.plot(
                        t,
                        results[i],
                        marker=".",
                        label=label,
                        linewidth=1,
                        markersize=5,
                    )
                    a.set_title(s[i])
                    a.set_xlabel("time (s)")
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ["top", "right"]:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print("Warning: Plotting error for %s; %s" % (f, e))

    ax[1].legend()
    plt.savefig(Path(save_dir) / "idetection_profile.png", dpi=200)


def plot_results_overlay(
    start=0, stop=0
):  # from utils.plots import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = [
        "train",
        "train",
        "train",
        "Precision",
        "mAP@0.5",
        "val",
        "val",
        "val",
        "Recall",
        "mAP@0.5:0.95",
    ]  # legends
    t = ["Box", "Objectness", "Classification", "P-R", "mAP-F1"]  # titles
    for f in sorted(
        glob.glob("results*.txt") + glob.glob("../../Downloads/results*.txt")
    ):
        results = np.loadtxt(
            f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2
        ).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker=".", label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace(".txt", ".png"), dpi=200)


def plot_results(start=0, stop=0, bucket="", id=(), labels=(), save_dir=""):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = [
        "Box",
        "Objectness",
        "Classification",
        "Precision",
        "Recall",
        "val Box",
        "val Objectness",
        "val Classification",
        "mAP@0.5",
        "mAP@0.5:0.95",
    ]
    if bucket:
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ["results%g.txt" % x for x in id]
        c = ("gsutil cp " + "%s " * len(files) + ".") % tuple(
            "gs://%s/results%g.txt" % (bucket, x) for x in id
        )
        os.system(c)
    else:
        files = list(Path(save_dir).glob("results*.txt"))
    assert len(
        files
    ), "No results.txt files found in %s, nothing to plot." % os.path.abspath(
        save_dir
    )
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(
                f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2
            ).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(
                    x, y, marker=".", label=label, linewidth=2, markersize=8
                )
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print("Warning: Plotting error for %s; %s" % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / "results.png", dpi=200)


############### utils/loss.py ###############


import time

import numpy as np
import torch
import torch.nn as nn

## from utils.metrics import bbox_iou
## from utils.torch_utils import is_parallel
## from utils.pose_utils import corner_confidence


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class PoseLoss:
    # Compute losses
    def __init__(self, model, num_keypoints=9, pretrain_num_epochs=20):
        super(PoseLoss, self).__init__()

        self.device = next(model.parameters()).device  # get model device

        h = model.hyp  # hyperparameters
        self.hyp = h

        self.num_keypoints = num_keypoints
        self.box_loss = nn.L1Loss()

        self.cp, self.cn = smooth_BCE(
            eps=h.get("label_smoothing", 0.0)
        )  # positive, negative BCE targets

        self.obj_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=self.device)
        )
        self.cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=self.device)
        )

        self.pretrain_num_epochs = pretrain_num_epochs
        self.balance = [4.0, 1.0, 0.3, 0.1, 0.03]  # P3-P7

        pose = (
            model.module.model[-1] if is_parallel(model) else model.model[-1]
        )  # Pose() module
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(pose.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7

        for k in "na", "nc", "nl", "anchors":
            setattr(self, k, getattr(pose, k))

    def __call__(self, p, targets, epoch=None):  # predictions, targets, model
        lcls, lbox, lobj = (
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
        )
        tcls, tbox, indices, anchors = self.build_targets(
            p, targets
        )  # targets

        # debug build_targets, visualize selected anchors on image
        self.visualize_targets(tcls, tbox, indices, anchors, p)

        # Losses
        for layer_id, pi in enumerate(p):  # layer index, layer predictions

            b, a, gj, gi = indices[layer_id]  # image, anchor, gridy, gridx
            # print(self.device)
            tobj = (
                torch.zeros_like(pi[..., 0])
                .type(torch.FloatTensor)
                .to(self.device)
            )  # target obj
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[
                    b, a, gj, gi
                ]  # prediction subset corresponding to targets
                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5  # centroid
                pwh = (
                    compute_new_width_height_1(
                        ps[:, 2 : 2 * self.num_keypoints]
                    )
                    * anchors[layer_id]
                )
                pbox = torch.cat((pxy, pwh), 1)
                p3dbox = torch.cat((pxy, ps[:, 2 : 2 * self.num_keypoints]), 1)
                # print(f"{tbox[layer_id][:2*self.num_keypoints].shape}")
                # lbox += self.box_loss(p3dbox, tbox[layer_id][:, :2*self.num_keypoints])

                # Objectness
                confidence = corner_confidence(
                    tbox[layer_id][:, : 2 * self.num_keypoints],
                    p3dbox,
                    im_grid_width=p[layer_id].shape[3],
                    im_grid_height=p[layer_id].shape[2],
                    th=0.25,
                    device=self.device,
                ).clamp(
                    min=0
                )  # .detach() .type(tobj.dtype)
                iou = bbox_iou(
                    pbox, tbox[layer_id][:, np.r_[:2, 18:20]], CIoU=True
                ).squeeze()  # iou(prediction, target)

                tobj[
                    b, a, gj, gi
                ] = confidence  #  (confidence + iou)/2 # target confidence

                p_edge_keypoints = (
                    ps[:, 2 : 2 * self.num_keypoints] * 2.0 - 0.5
                )
                # print(p_edge_keypoints.shape)
                # print(pxy.shape)
                p_keypoints = torch.cat((pxy, p_edge_keypoints), 1)
                t_keypoints = tbox[layer_id][:, : self.num_keypoints * 2]
                # print(p_keypoints.shape)
                # print(t_keypoints.shape)
                l2_dist = self.box_loss(
                    p_keypoints[:, : 2 * self.num_keypoints], t_keypoints
                )
                # l2_dist = (p_keypoints[:, :2*self.num_keypoints][:, 0::2]-t_keypoints[:, 0::2])**2 + (p_keypoints[:, :2*self.num_keypoints][:, 1::2]-t_keypoints[:, 1::2])**2
                # print(f"{l2_dist=}")
                scales = torch.prod(
                    tbox[layer_id][:, -2:], dim=1, keepdim=True
                )
                # print(scales)
                # print(f"loss: {((1 - torch.exp(-l2_dist/(scales+1e-9)))).mean()}")
                lbox += ((1 - torch.exp(-l2_dist / (scales + 1e-9)))).mean()
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, 2 * self.num_keypoints + 1 :],
                        self.cn,
                        device=self.device,
                    )  # targets
                    t[range(n), tcls[layer_id]] = self.cp
                    lcls += self.cls_loss(
                        ps[:, 2 * self.num_keypoints + 1 :], t
                    )

            obji = self.obj_loss(pi[..., 2 * self.num_keypoints], tobj)
            lobj += obji * self.balance[layer_id]

        lobj *= self.hyp["obj"]
        lbox *= self.hyp["box"]
        lcls *= self.hyp["cls"]

        # if epoch is not None and epoch > self.pretrain_num_epochs:
        #     loss = lbox + lobj + lcls
        # else:
        #     loss = lbox + lcls
        loss = lbox + lobj + lcls
        bs = tobj.shape[0]  # batch size

        return loss * bs, torch.cat((lobj, lbox, lcls)).detach()

    def build_targets(self, p, targets):
        """
        Build targets for compute_loss(), input targets(image,class,x,y,w,h)

        Args:
            p: list of tensors, multi-scale predict from model
            targets: tensor of shape (B, 22), where B is batch size, 22 is (1 + 8 + 1)*2 + 2
                1 * 2: center_xy
                8 * 2: corner_xy
                1 * 2: wh
                2: class, score

        Returns:
            tcls: list of tensors, each tensor is a class of the target
            tbox: list of tensors, each tensor is a box of the target
            indices: list of tuples, each tuple is a tuple of image, anchor, grid indices
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

        if opt.debug_target:
            print(f"targets: {targets}")

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(
            23, device=targets.device
        )  # normalized to gridspace gain
        ai = (
            torch.arange(na, device=targets.device)
            .float()
            .view(na, 1)
            .repeat(1, nt)
        )  # same as .repeat_interleave(nt) # anchor indices
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[:, :, None]), 2
        )  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            shape = p[i].shape
            gain[2:22] = torch.tensor(p[i].shape)[
                10 * [3, 2]
            ]  # xyxy gain, 1(center)+8(corner)+1(wh)

            # Match targets to anchors, [3, 8, 23] * [23] = [3, 8, 23]
            t = targets * gain  # xy_norm2xy_feat
            if nt:
                # Matches
                r = t[:, :, 20:22] / anchors[:, None]  # wh ratio
                j = (
                    torch.max(r, 1.0 / r).max(2)[0] < self.hyp["anchor_t"]
                )  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # center xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = (
                    (gxy % 1.0 < g) & (gxy > 1.0)
                ).T  # index of target closer to left top corner
                l, m = (
                    (gxi % 1.0 < g) & (gxi > 1.0)
                ).T  # index of target closer to right bottom corner
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gxy_coords = t[
                :, 4 : 2 * self.num_keypoints + 2
            ]  # other box coords
            gwh = t[
                :, 2 * self.num_keypoints + 2 : 2 * self.num_keypoints + 4
            ]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            if opt.debug_target and i == 0:
                print(f"targets: {targets}")
                print(f"gain: {gain}")
                # print(f"t[:, 2:4]: {t[:, 2:4]}")
                # print(f"offsets: {offsets}")
                print(f"gij: {gij}")
                # print(f"gxy-gij: {gxy - gij}")
                print(f"t[:, 4:20]: {t[:, 4:20]}")

            for key_idx in range(self.num_keypoints - 1):
                gxy_coords[:, key_idx * 2 : key_idx * 2 + 2] = (
                    gxy_coords[:, key_idx * 2 : key_idx * 2 + 2] - gij
                )

            if opt.debug_target and i == 0:
                print(f"gxy_coords: {gxy_coords}")

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1))
            )  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gxy_coords, gwh), 1))  # box

            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def visualize_targets(self, tcls, tbox, indices, anchors, p):
        """
        Visualize targets on image scale.
        """
        if not opt.debug_target:
            return
        # 创建保存目录
        save_dir = Path("debug_target")
        save_dir.mkdir(exist_ok=True)

        # 为每个检测层创建可视化
        strides = [8, 16, 32]
        for layer_id in range(len(tcls)):
            if len(tcls[layer_id]) == 0:
                continue
            # 获取当前层的数据
            layer_tcls = tcls[layer_id]
            layer_tbox = tbox[layer_id]
            layer_indices = indices[layer_id]
            layer_anchors = anchors[layer_id]
            layer_stride = strides[layer_id]

            shape = p[layer_id].shape
            gain = torch.ones(23, device=layer_tbox.device)
            gain[2:22] = torch.tensor(shape)[10 * [3, 2]]  # xyxy gain

            # 创建图像网格
            grid_size = int(np.ceil(np.sqrt(len(layer_tcls))))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            if grid_size == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # 为每个目标创建可视化
            b_list, a_list, gj_list, gi_list = layer_indices
            for i, (cls, box, b, a, gj, gi, anchor) in enumerate(
                zip(
                    layer_tcls,
                    layer_tbox,
                    b_list,
                    a_list,
                    gj_list,
                    gi_list,
                    layer_anchors,
                )
            ):
                if i >= len(axes):
                    break

                ax = axes[i]

                img = cv2.imread(f"{save_dir}/tmp_debug_target_rgb.png")
                img_h, img_w = img.shape[:2]

                # 提取坐标信息
                gxy_offset = box[0:2]
                gxy_coords_offset = box[
                    2 : 2 + 2 * self.num_keypoints
                ].reshape(-1, 2)
                gwh = box[-2:]

                gij = torch.tensor([gi, gj], device=gxy_offset.device).float()
                gxy = gxy_offset + gij
                gxy_coords = gxy_coords_offset + gij

                if layer_id == 0:
                    print(f"  gij: {gij}")
                    # print(f"  gxy_offset: {gxy_offset}")
                    # print(f"  gxy: {gxy}")
                    print(f"  gxy_coords_offset: {gxy_coords_offset}")
                    print(f"  gxy_coords: {gxy_coords}")

                # 转换到图像坐标系
                center_x_pix = int(gxy[0] / gain[2] * img_w)
                center_y_pix = int(gxy[1] / gain[3] * img_h)
                width_pix = int(gwh[0] / gain[2] * img_w)
                height_pix = int(gwh[1] / gain[3] * img_h)

                # 绘制边界框
                x1 = int(center_x_pix - width_pix // 2)
                y1 = int(center_y_pix - height_pix // 2)
                x2 = int(center_x_pix + width_pix // 2)
                y2 = int(center_y_pix + height_pix // 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制关键点
                colors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255),
                    (128, 0, 0),
                    (0, 128, 0),
                    (0, 0, 128),
                ]

                for j, (kp_x, kp_y) in enumerate(gxy_coords):
                    if j < len(colors):
                        kp_x_pix = int(kp_x / gain[2] * img_w)
                        kp_y_pix = int(kp_y / gain[3] * img_h)
                        cv2.circle(img, (kp_x_pix, kp_y_pix), 3, colors[j], -1)
                        cv2.putText(
                            img,
                            str(j),
                            (kp_x_pix + 5, kp_y_pix - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            colors[j],
                            1,
                        )

                # 绘制anchor
                anchor_w, anchor_h = anchor[0], anchor[1]
                anchor_w_pix = int(anchor_w * layer_stride)
                anchor_h_pix = int(anchor_h * layer_stride)
                anchor_x1 = int(center_x_pix - anchor_w_pix // 2)
                anchor_y1 = int(center_y_pix - anchor_h_pix // 2)
                anchor_x2 = int(center_x_pix + anchor_w_pix // 2)
                anchor_y2 = int(center_y_pix + anchor_h_pix // 2)

                cv2.rectangle(
                    img,
                    (anchor_x1, anchor_y1),
                    (anchor_x2, anchor_y2),
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                # 绘制网格位置
                grid_x = int(gi * layer_stride)
                grid_y = int(gj * layer_stride)
                cv2.circle(img, (grid_x, grid_y), 5, (0, 0, 255), -1)

                # 显示图像
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(
                    f"Layer {layer_id}, Class {cls.item()}, Anchor {a.item()}"
                )
                ax.axis("off")

            # 隐藏多余的子图
            for j in range(len(layer_tcls), len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.savefig(
                save_dir / f"targets_layer_{layer_id}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            # 打印统计信息
            print(f"Layer {layer_id}: {len(layer_tcls)} targets visualized")
            if len(layer_tcls) > 0:
                print(f"  - Classes: {torch.unique(layer_tcls).tolist()}")
                print(
                    f"  - Anchors used: {torch.unique(layer_indices[1]).tolist()}"
                )

        __import__("ipdb").set_trace()
        print(f"Target visualizations saved to {save_dir}")


def compute_new_width_height_1(coordinates):
    # print((torch.amax(coordinates[:, ::2], dim=1)-torch.amin(coordinates[:, ::2], dim=1)).shape )
    return torch.stack(
        [
            torch.amax(coordinates[:, ::2], dim=1)
            - torch.amin(coordinates[:, ::2], dim=1),
            torch.amax(coordinates[:, 1::2], dim=1)
            - torch.amin(coordinates[:, 1::2], dim=1),
        ],
        dim=1,
    )


############### utils/autoanchor.py ###############


import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

## from utils.general import colorstr


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    prefix = colorstr("autoanchor: ")
    print(f"\n{prefix}Analyzing anchors... ", end="")
    m = (
        model.module.model[-1] if hasattr(model, "module") else model.model[-1]
    )  # Pose()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(
        0.9, 1.1, size=(shapes.shape[0], 1)
    )  # augment scale
    wh = torch.tensor(
        np.concatenate(
            [l[:, 19:21] * s for s, l in zip(shapes * scale, dataset.labels)]
        )
    ).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1.0 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1.0 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1.0 / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print(
        f"anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}",
        end="",
    )
    if bpr < 0.98:  # threshold to recompute
        print(". Attempting to improve anchors, please wait...")
        na = m.anchor_grid.numel() // 2  # number of anchors
        print(dataset)
        new_anchors = kmean_anchors(
            dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False
        )
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(
                new_anchors, device=m.anchors.device
            ).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(
                m.anchor_grid
            )  # for inference
            m.anchors[:] = new_anchors.clone().view_as(
                m.anchors
            ) / m.stride.to(m.anchors.device).view(
                -1, 1, 1
            )  # loss
            check_anchor_order(m)
            print(
                f"{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future."
            )
        else:
            print(
                f"{prefix}Original anchors better than new anchors. Proceeding with original anchors."
            )
    print("")  # newline


def kmean_anchors(
    path="./data/coco128.yaml",
    n=9,
    img_size=640,
    thr=4.0,
    gen=1000,
    verbose=True,
):
    """Creates kmeans-evolved anchors from training dataset

    Arguments:
        path: path to dataset *.yaml, or a loaded dataset
        n: number of anchors
        img_size: image size used for training
        thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
        gen: generations to evolve anchors using genetic algorithm
        verbose: print all results

    Return:
        k: kmeans evolved anchors

    Usage:
        from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1.0 / thr
    prefix = colorstr("autoanchor: ")

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1.0 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (
            x > thr
        ).float().mean() * n  # best possible recall, anch > thr
        print(
            f"{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr"
        )
        print(
            f"{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, "
            f"past_thr={x[x > thr].mean():.3f}-mean: ",
            end="",
        )
        for i, x in enumerate(k):
            print(
                "%i,%i" % (round(x[0]), round(x[1])),
                end=",  " if i < len(k) - 1 else "\n",
            )  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        print("Doing this")
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(
            data_dict["train"], augment=True, rect=True
        )
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate(
        [l[:, 19:21] * s for s, l in zip(shapes, dataset.labels)]
    )  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(
            f"{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size."
        )
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation
    print(f"{prefix}Running kmeans for {n} anchors on {len(wh)} points...")
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = (
        anchor_fitness(k),
        k.shape,
        0.9,
        0.1,
    )  # fitness, generations, mutation prob, sigma
    pbar = tqdm(
        range(gen), desc=f"{prefix}Evolving anchors with Genetic Algorithm:"
    )  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (
            v == 1
        ).all():  # mutate until a change occurs (prevent duplicates)
            v = (
                (npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1
            ).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f"{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}"
            if verbose:
                print_results(k)

    return print_results(k)


############### models/yolo.py ###############

import argparse
import logging
import sys
from copy import deepcopy

sys.path.append("./")  # to run '$ python *.py' files in subdirectories

## from models.common import *
## from models.experimental import *  # AIO: {'CrossConv': 'CrossConv_1', 'GhostConv': 'GhostConv_1', 'GhostBottleneck': 'GhostBottleneck_1'}
## from utils.autoanchor import check_anchor_order
## from utils.general import check_file, make_divisible, print_args, set_logging
## from utils.torch_utils import copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, \
##     select_device, time_synchronized

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# LOGGER = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(
        self, nc=80, anchors=(), ch=(), inplace=True
    ):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1) for x in ch
        )  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                if (
                    self.grid[i].shape[2:4] != x[i].shape[2:4]
                    or self.onnx_dynamic
                ):
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (
                        y[..., 0:2] * 2.0 - 0.5 + self.grid[i]
                    ) * self.stride[
                        i
                    ]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[
                        i
                    ]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (
                        y[..., 0:2] * 2.0 - 0.5 + self.grid[i]
                    ) * self.stride[
                        i
                    ]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                        1, self.na, 1, 1, 2
                    )  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Pose(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(
        self, nc=1, anchors=(), ch=(), num_keypoints=9
    ):  # detection layer
        super(Pose, self).__init__()
        self.nc = nc  # number of classes
        self.num_keypoints = num_keypoints
        self.no = (
            nc + 1 + self.num_keypoints * 2
        )  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.m = nn.ModuleList(
            nn.Sequential(
                DWConv(x, x, k=3),
                Conv(x, x),
                DWConv(x, x, k=3),
                Conv(x, x),
                DWConv(x, x, k=3),
                Conv(x, x),
                DWConv(x, x, k=3),
                Conv(x, x),
                DWConv(x, x, k=3),
                Conv(x, x),
                DWConv(x, x, k=3),
                nn.Conv2d(x, self.no * self.na, 1),
            )
            for x in ch
        )

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv

            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            if not self.training:  # inference
                x[i] = x[i].view(bs, self.na, nx * ny, self.no).contiguous()
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = (
                        self._make_grid(nx, ny)
                        .to(x[i].device)
                        .view(1, 1, nx * ny, 2)
                    )

                y = x[i].clone()
                # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy

                y[..., 0:2] = (
                    y[..., 0:2].sigmoid() * 2.0
                    - 0.5
                    + self.grid[i].to(x[i].device)
                ) * self.stride[
                    i
                ]  # center coordinate
                for key_idx in range(1, self.num_keypoints):
                    y[..., 2 * key_idx : 2 * key_idx + 2] = (
                        y[..., 2 * key_idx : 2 * key_idx + 2] * 2.0
                        - 0.5
                        + self.grid[i].to(x[i].device)
                    ) * self.stride[i]
                y[..., 2 * self.num_keypoints : self.no] = y[
                    ..., 2 * self.num_keypoints : self.no
                ].sigmoid()  # sigmoid confidence and classes
                z.append(y.view(bs, -1, self.no))
        # from pdebug.debug import yolopose
        # yolopose.scoremap(x, savename="score.png", image="/home/duino/code/github/SAM-6D/SAM-6D/runs/linemod/shoes/JPEGImages/000000.png")
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(
        self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None
    ):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            logger.info(
                f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}"
            )
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            logger.info(
                f"Overriding model.yaml anchors with anchors={anchors}"
            )
            self.yaml["anchors"] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch]
        )  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Pose()
        if isinstance(m, Pose):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor(
                [
                    s / x.shape[-2]
                    for x in self.forward(torch.zeros(1, ch, s, s))
                ]
            )  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(
            x, profile, visualize
        )  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(
                x.flip(fi) if fi else x, si, gs=int(self.stride.max())
            )
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = (
                p[..., 0:1] / scale,
                p[..., 1:2] / scale,
                p[..., 2:4] / scale,
            )  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(
            4 ** (nl - 1 - x) for x in range(e)
        )  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = (
            thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0]
            / 1e9
            * 2
            if thop
            else 0
        )  # FLOPs
        t = time_synchronized()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_synchronized() - t) * 100)
        if m == self.model[0]:
            logger.info(
                f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}"
            )
        logger.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            logger.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(
        self, cf=None
    ):  # initialize biases into Pose(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            for op in mi:
                if isinstance(op, nn.Conv2d):
                    b = op.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                    b.data[:, 4] += math.log(
                        8 / (640 / s) ** 2
                    )  # obj (8 objects per 640 image)
                    b.data[:, 5:] += (
                        math.log(0.6 / (m.nc - 0.99))
                        if cf is None
                        else torch.log(cf / cf.sum())
                    )  # cls
                    op.bias = torch.nn.Parameter(
                        b.view(-1), requires_grad=True
                    )
                else:
                    continue

    # def _initialize_biases(self, cf=None):  # initialize biases into Pose(), cf is class frequency
    #     # https://arxiv.org/abs/1708.02002 section 3.3
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    #     m = self.model[-1]  # Detect() module
    #     for mi, s in zip(m.m, m.stride):  # from
    #         b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
    #         b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
    #         b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
    #         mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ("%6g Conv2d.bias:" + "%10.3g" * 6)
                % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    # def autoshape(self):  # add AutoShape module
    #     logger.info('Adding AutoShape... ')
    #     m = AutoShape(self)  # wrap model
    #     copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
    #     return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info(
        "\n%3s%18s%3s%10s  %-40s%-30s"
        % ("", "from", "n", "params", "module", "arguments")
    )
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (
        (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    )  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(
        d["backbone"] + d["head"]
    ):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv,
            GhostConv_1,
            Bottleneck,
            GhostBottleneck_1,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv_1,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        ]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3Ghost, C3x]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Pose:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = (
            nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        )  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        num_params = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = (
            i,
            f,
            t,
            num_params,
        )  # attach index, 'from' index, type, number params
        logger.info(
            "%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n_, num_params, t, args)
        )  # print
        save.extend(
            x % i for x in ([f] if isinstance(f, int) else f) if x != -1
        )  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


@app.command()
def model_main():
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard


############### test.py ###############

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from threading import Thread

import matplotlib.patches as patches

## from models.experimental import attempt_load  # AIO: {'CrossConv': 'CrossConv_1', 'GhostConv': 'GhostConv_1', 'GhostBottleneck': 'GhostBottleneck_1'}
## from utils.datasets import create_dataloader  # AIO: {'logger': 'logger'}
## from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, scale_coords, set_logging, increment_path, colorstr, retrieve_image
## from utils.plots import plot_images, output_to_target, plot_study_txt
## from utils.torch_utils import select_device, time_synchronized
## from utils.pose_utils import box_filter, get_3D_corners, pnp, epnp, calcAngularDistance, compute_projection, compute_transformation, get_camera_intrinsic, fix_corner_order, calc_pts_diameter, MeshPly
## from utils.loss import PoseLoss  # AIO: {'compute_new_width_height': 'compute_new_width_height_1'}
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm


def test(
    data,
    weights=None,
    batch_size=1,
    imgsz=640,
    conf_thres=0.01,
    num_keypoints=9,
    save_json=False,
    single_cls=True,
    augment=False,
    verbose=False,
    model=None,
    dataloader=None,
    save_dir=Path(""),  # for saving images
    save_txt=False,  # for auto-labelling
    save_hybrid=False,  # for hybrid auto-labelling
    save_conf=False,  # save auto-label confidences
    plots=True,
    nc=1,
    log_imgs=0,  # number of logged images
    compute_loss=False,
    symetric=False,
    test_plotting=False,
):

    # Initialize/load model and set device
    training = model is not None

    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(
            increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
        )  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

    check_dataset(data)  # check
    nc = 1 if single_cls else int(data["nc"])  # number of classes

    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    testing_samples = 0.0
    errs_2d = []
    errs_3d = []
    errs_trans = []
    errs_angle = []
    errs_corner2D = []

    # Variable to save
    testing_errors_trans = []
    testing_errors_angle = []
    testing_errors_pixel = []
    testing_accuracies = []

    edges_corners = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]
    colormap = np.array(
        ["r", "g", "b", "c", "m", "y", "k", "w", "xkcd:sky blue"]
    )

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    wandb = None
    # try:
    #     import wandb  # Weights & Biases
    # except ImportError:
    #     log_imgs = 0

    # Dataloader
    if not training:
        if device.type != "cpu":
            model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(device)
                .type_as(next(model.parameters()))
            )  # run once
        path = (
            data["test"] if opt.task == "test" else data["val"]
        )  # path to val/test images
        dataloader = create_dataloader(
            path,
            imgsz,
            batch_size,
            gs,
            opt,
            rect=True,
            augment=False,
            prefix=colorstr("test: " if opt.task == "test" else "val: "),
            workers=opt.num_workers_test,
        )[0]

    seen = 0
    names = {
        k: v
        for k, v in enumerate(
            model.names if hasattr(model, "names") else model.module.names
        )
    }

    t1, t2, t3, t4, t5, t6 = [], [], [], [], [], []
    if compute_loss:
        pose_loss = PoseLoss(model, num_keypoints, pretrain_num_epochs=0)
    loss_items = torch.zeros(3, device=device, requires_grad=False)
    loss = torch.zeros(1, device=device, requires_grad=False)
    # Get the intrinsic camerea matrix, mesh, vertices and corners of the model
    # mesh_list = []
    # for mesh_id in range(8):
    #     mesh_list.append(MeshPly(data[f'mesh{mesh_id}']))

    if isinstance(data["mesh"], str) and single_cls:
        if data["mesh"].endswith(".ply"):
            mesh = MeshPly(data["mesh"])
            vertices = np.c_[
                np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))
            ].transpose()
            corners3D = [get_3D_corners(vertices)]
        elif data["mesh"].endswith(".glb"):
            vertices = load_points_3d_from_cad(
                data["mesh"], src_unit="meter", dst_unit="millimeter"
            )
            corners = get_3D_corners__pdebug(vertices)[
                1:, :
            ]  # [9, 3] -> [8, 3] skip center
            corners = np.concatenate(
                (np.transpose(corners), np.ones((1, 8))), axis=0
            )  # [3, 8] -> [4, 8]
            corners3D = [corners]
            num_vertices = vertices.shape[0]
            vertices = np.concatenate(
                (np.transpose(vertices), np.ones((1, num_vertices))), axis=0
            )  # [3, N] -> [4, N]
        else:
            raise NotImplementedError
    else:
        corners3D = []
        for mesh_i in data["mesh"]:
            mesh = MeshPly(mesh_i)
            vertices = np.c_[
                np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))
            ].transpose()
            corners3D_i = get_3D_corners(vertices)
            corners3D.append(corners3D_i)

    try:
        diam = float(data["diam"])
    except:
        diam = calc_pts_diameter(np.array(mesh.vertices))

    wandb_images = []
    count = 0

    for batch_i, (img, targets, intrinsics, paths, shapes) in enumerate(
        tqdm(dataloader)
    ):
        t = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        with torch.no_grad():
            # Run model
            # with autocast():
            with torch.amp.autocast("cuda"):
                t1.append(time_synchronized() - t)

                out, train_out = model(
                    img, augment=augment
                )  # inference and training outputs
                t2.append(time_synchronized() - t)

            # Compute loss
            if compute_loss:
                # _, loss_items = pose_loss([x.float() for x in train_out], targets, imgs_size=list(img.shape[2:]))
                batch_loss, batch_loss_items = pose_loss(
                    [x.float() for x in train_out], targets
                )
                loss_items += batch_loss_items
                loss += batch_loss

            t3.append(time_synchronized() - t)

            # Using confidence threshold, eliminate low-confidence predictions
            out = box_filter(
                out, conf_thres=conf_thres, multi_label=(not opt.single_cls)
            )
            t4.append(time_synchronized() - t)

            # Statistics per image
            for si, pred in enumerate(out):

                path, shape = Path(paths[si]), shapes[si][0]
                im_native_width, im_native_height = shape[1], shape[0]
                # Predictions
                if len(pred) == 0:
                    continue
                if single_cls:
                    pred[:, 19] = 0
                predn = pred.clone().cpu()
                scale_coords(
                    img[si].shape[1:], predn[:, :18], shape, shapes[si][1]
                )  # native-space pred
                labels = targets[targets[:, 0] == si, 1:].cpu()
                tbox = labels[:, 1:19]
                tbox[:, ::2] = tbox[:, ::2] * width
                tbox[:, 1::2] = tbox[:, 1::2] * height
                scale_coords(
                    img[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels

                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target classes

                seen += 1
                # Iterate through each prediction and ground-truth object

                for k in range(nl):
                    box_gt = tbox[k]
                    if (predn[:, 19] == tcls[k]).sum() == 0:
                        continue
                    else:
                        full_pr = predn[
                            torch.where(predn[:, 19] == tcls[k]), :
                        ]

                    if (
                        len(full_pr) == 0
                        or not full_pr.shape[0]
                        or full_pr.nelement() == 0
                    ):
                        continue

                    tcls_i = tcls[k]

                    box_pr = full_pr[0, :18]
                    prediction_confidence = full_pr[0, 18]
                    # Denormalize the corner predictions
                    corners2D_gt = np.array(
                        np.reshape(
                            box_gt[: num_keypoints * 2], [num_keypoints, 2]
                        ),
                        dtype="float32",
                    )
                    corners2D_pr = np.array(
                        np.reshape(
                            box_pr[: num_keypoints * 2], [num_keypoints, 2]
                        ),
                        dtype="float32",
                    )

                    # Compute corner prediction error
                    corner_norm = np.linalg.norm(
                        corners2D_gt - corners2D_pr, axis=1
                    )
                    corner_dist = np.mean(corner_norm)
                    errs_corner2D.append(corner_dist)

                    u0, v0, fx, fy = (
                        intrinsics[k][4],
                        intrinsics[k][5],
                        intrinsics[k][0],
                        intrinsics[k][1],
                    )
                    internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

                    # Compute [R|t] by pnp
                    R_gt, t_gt = pnp(
                        np.array(
                            np.transpose(
                                np.concatenate(
                                    (
                                        np.zeros((3, 1)),
                                        corners3D[int(tcls_i)][:3, :],
                                    ),
                                    axis=1,
                                )
                            ),
                            dtype="float32",
                        ),
                        corners2D_gt,
                        np.array(internal_calibration, dtype="float32"),
                    )
                    t_temp = time_synchronized()
                    R_pr, t_pr = pnp(
                        np.array(
                            np.transpose(
                                np.concatenate(
                                    (
                                        np.zeros((3, 1)),
                                        corners3D[int(tcls_i)][:3, :],
                                    ),
                                    axis=1,
                                )
                            ),
                            dtype="float32",
                        ),
                        corners2D_pr,
                        np.array(internal_calibration, dtype="float32"),
                    )
                    t6.append(time_synchronized() - t_temp)

                    # Compute errors
                    # Compute translation error
                    trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                    errs_trans.append(trans_dist)

                    # Compute angle error
                    angle_dist = calcAngularDistance(R_gt, R_pr)
                    errs_angle.append(angle_dist)

                    # Compute pixel error
                    Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
                    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
                    proj_2d_gt = compute_projection(
                        vertices, Rt_gt, internal_calibration
                    )
                    proj_2d_pred = compute_projection(
                        vertices, Rt_pr, internal_calibration
                    )
                    norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
                    pixel_dist = np.mean(norm)
                    errs_2d.append(pixel_dist)

                    # Compute 3D distances
                    transform_3d_gt = compute_transformation(vertices, Rt_gt)
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)
                    if symetric:
                        from utils.compute_overlap import (
                            wrapper_c_min_distances,  # for computing ADD-S metric
                        )

                        norm3d = wrapper_c_min_distances(
                            transform_3d_gt, transform_3d_pred
                        )
                    else:
                        norm3d = np.linalg.norm(
                            transform_3d_gt - transform_3d_pred, axis=0
                        )
                    vertex_dist = np.mean(norm3d)
                    errs_3d.append(vertex_dist)

                    # Sum errors
                    testing_error_trans += trans_dist
                    testing_error_angle += angle_dist
                    testing_error_pixel += pixel_dist
                    testing_samples += 1

                    # test_plotting = False
                    # W&B logging
                    if (
                        test_plotting
                        or (plots and len(wandb_images)) < log_imgs
                    ):

                        local_img = (
                            img[si, :, :, :].cpu().numpy().transpose(1, 2, 0)
                        )
                        local_img = retrieve_image(
                            local_img,
                            img[si].shape[1:],
                            (shape[0], shape[1]),
                            shapes[si][1],
                        )  #  im_native_width, im_native_height
                        figsize = (im_native_width / 96, im_native_height / 96)
                        fig = plt.figure(frameon=False, figsize=figsize)

                        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                        ax.set_axis_off()
                        fig.add_axes(ax)

                        image = np.uint8(
                            local_img * 255
                        )  # .resize((im_native_width, im_native_height)))
                        ax.imshow(image, cmap="gray", aspect="auto")

                        corn2D_pr = corners2D_pr[1:, :]
                        corn2D_gt = corners2D_gt[1:, :]

                        # Projections
                        for edge in edges_corners:
                            ax.plot(
                                corn2D_gt[edge, 0],
                                corn2D_gt[edge, 1],
                                color="g",
                                linewidth=0.5,
                            )  #  if test_plotting else None
                            ax.plot(
                                corn2D_pr[edge, 0],
                                corn2D_pr[edge, 1],
                                color="b",
                                linewidth=0.5,
                            )
                        ax.scatter(
                            corners2D_gt.T[0],
                            corners2D_gt.T[1],
                            c=colormap,
                            s=10,
                        )  # if not test_plotting else None
                        ax.scatter(
                            corners2D_pr.T[0],
                            corners2D_pr.T[1],
                            c=colormap,
                            s=10,
                        )

                        # draw on image
                        # Create a Rectangle patch
                        min_x = np.amin(corners2D_pr.T[0])
                        min_y = np.amin(corners2D_pr.T[1])

                        # vx_threshold = diam * 0.1
                        # facecolor = 'green' if vertex_dist <=vx_threshold else 'red'
                        # ax.text(min_x, min_y-30, f"conf: {prediction_confidence:.3f}", style='italic', bbox={'facecolor': facecolor, 'alpha': 0.5, 'pad': 2})
                        # ax.text(min_x, min_y-10, f"2d err: {pixel_dist:.3f}, vertex_dist: {vertex_dist:.3f}", style='italic', bbox={'facecolor': facecolor, 'alpha': 0.5, 'pad': 2})

                        filename = f'foo_{count}_{datetime.now().strftime("%H_%M_%S")}.png'
                        file_path = os.path.join(save_dir, filename)
                        fig.savefig(
                            file_path,
                            dpi=96,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        plt.close()
                        wandb_images.append(
                            wandb.Image(file_path)
                        ) if not test_plotting else None

                        count += 1
            t5.append(time_synchronized() - t)

    # Compute 2D projection, 6D pose and 5cm5degree scores
    px_threshold = 5  # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works
    vx_threshold = diam * 0.1
    eps = 1e-5
    acc_value = len(np.where(np.array(errs_2d) <= px_threshold)[0])
    acc = acc_value * 100.0 / (len(errs_2d) + eps)
    acc3d_value = len(np.where(np.array(errs_3d) <= vx_threshold)[0])
    acc3d = acc3d_value * 100.0 / (len(errs_3d) + eps)
    acc5cm5deg_value = len(
        np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[
            0
        ]
    )
    acc5cm5deg = acc5cm5deg_value * 100.0 / (len(errs_trans) + eps)
    corner_acc = (
        len(np.where(np.array(errs_corner2D) <= px_threshold)[0])
        * 100.0
        / (len(errs_corner2D) + eps)
    )
    mean_err_2d = np.mean(errs_2d)
    mean_corner_err_2d = np.mean(errs_corner2D)
    nts = float(testing_samples)

    t1 = np.array(t1)
    t2 = np.array(t2)
    t3 = np.array(t3)
    t4 = np.array(t4)
    t5 = np.array(t5)
    t6 = np.array(t6)

    num_itr = 5  # first couple of passes are slow
    if True:
        print("-----------------------------------")
        print("  tensor to cuda : %f" % (np.mean(t1[-num_itr:])))
        print("         predict : %f" % (np.mean((t2 - t1)[-num_itr:])))
        print("    compute loss : %f" % (np.mean((t3 - t2)[-num_itr:])))
        print("get_region_boxes : %f" % (np.mean((t4 - t3)[-num_itr:])))
        print("            eval : %f" % (np.mean((t5 - t4)[-num_itr:])))
        print("             pnp : %f" % (np.mean(t6[-num_itr:])))
        print("           total : %f" % (np.mean(t4[-num_itr:])))
        print("-----------------------------------")

    # Print test statistics
    print("   Mean corner error is %f" % (mean_corner_err_2d))
    print(
        "   Acc using {} px 2D Projection = {:.2f}%".format(px_threshold, acc)
    )
    print(
        "   Acc using {} vx 3D Transformation = {:.2f}%".format(
            vx_threshold, acc3d
        )
    )
    print("   Acc using 5 cm 5 degree metric = {:.2f}%".format(acc5cm5deg))
    print(
        "   Translation error: %f, angle error: %f"
        % (
            testing_error_trans / (nts + eps),
            testing_error_angle / (nts + eps),
        )
    )

    # Register losses and errors for saving later on
    testing_errors_trans.append(testing_error_trans / (nts + eps))
    testing_errors_angle.append(testing_error_angle / (nts + eps))
    testing_errors_pixel.append(testing_error_pixel / (nts + eps))
    testing_accuracies.append(acc)
    # Return results
    model.float()  # for training

    # Plots
    if plots:
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})

    return (
        mean_corner_err_2d,
        acc,
        acc3d,
        acc5cm5deg,
        *(loss_items.cpu().detach() / len(dataloader)).tolist(),
        loss.cpu().numpy().item(),
    )


@app.command()
def test_main(
    repo: str = None,
    linemod_root: str = None,
    category: str = None,
    weights: str = None,
):
    global opt
    opt.project = "runs/test"  # runs/train | runs/detect | runs/test

    if repo and linemod_root:
        # add repo path to sys path, to init model from pth.
        sys.path.insert(0, repo)
        # override opt from args
        opt.data = f"{linemod_root}/{category}/{category}.yaml"
        opt.static_camera = f"{linemod_root}/{category}/linemod_camera.json"
        opt.source = f"{linemod_root}/{category}/JPEGImages"
        mesh_ext = [".ply", ".glb"]

        mesh_dir = f"{linemod_root}/{category}/{category}"
        if os.path.exists(mesh_dir):
            opt.single_cls = False
            opt.mesh_data = [
                os.path.join(mesh_dir, x)
                for x in sorted(os.listdir(mesh_dir))
                if x.endswith(".ply")
            ]
        else:
            for ext in mesh_ext:
                opt.mesh_data = f"{linemod_root}/{category}/{category}{ext}"
                if os.path.exists(opt.mesh_data):
                    break
            assert os.path.exists(opt.mesh_data), f"{opt.mesh_data} not exists"
        opt.pretrained = f"{repo}/yolov5x.pt"
        opt.cfg = f"{repo}/models/yolov5x_6dpose_bifpn.yaml"
        opt.hyp = f"{repo}/configs/hyp.single.yaml"
        # set opt.img_size from JPEGImages folder
        from pdebug.piata import Input

        opt.img_size = max(
            cv2.imread(
                Input(opt.source, name="imgdir").get_reader().imgfiles[0]
            ).shape[:2]
        )
        opt.weights = weights

    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements()

    if opt.task in ["val", "test"]:  # run normally
        test(
            opt.data,
            opt.weights,
            opt.batch_size_test,
            opt.img_size,
            opt.conf_thres_test,
            opt.num_keypoints,
            opt.save_json,
            opt.single_cls,
            False,
            opt.verbose,
            save_txt=opt.save_txt | opt.save_hybrid,
            save_hybrid=opt.save_hybrid,
            save_conf=opt.save_conf,
            symetric=opt.symetric,
            test_plotting=opt.test_plotting,
        )


############### train.py ###############

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from typing import Dict

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch import autograd
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(
        colorstr("hyperparameters: ")
        + ", ".join(f"{k}={v}" for k, v in hyp.items())
    )
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size_train,
        opt.total_batch_size,
        opt.pretrained,
        opt.global_rank,
    )

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / "last.pt"
    best = wdir / "best.pt"
    results_file = save_dir / "results.txt"

    # Save run settings
    with open(save_dir / "hyp.yaml", "w") as f:
        yaml.dump(edict2dict(hyp), f, sort_keys=False)
    with open(save_dir / "opt.yaml", "w") as f:
        yaml.dump(edict2dict(vars(opt)), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != "cpu"
    init_seeds(42 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict["train"]
    if "background_path" in data_dict:
        background_path = data_dict["background_path"]
        bg_files = get_all_files(background_path)
    else:
        background_path = None
        bg_files = None

    test_path = data_dict["val"]
    nc = 1 if opt.single_cls else int(data_dict["nc"])  # number of classes
    names = (
        ["item"]
        if opt.single_cls and len(data_dict["names"]) != 1
        else data_dict["names"]
    )  # class names
    assert len(names) == nc, "%g names found for nc=%g dataset in %s" % (
        len(names),
        nc,
        opt.data,
    )  # check

    # Model
    if opt.resume:
        # model = attempt_load(opt.resume, map_location=device) # Load model
        ckpt = torch.load(
            opt.resume, map_location=device, weights_only=False
        )  # load checkpoint
        model = ckpt["model"]
        state_dict = ckpt["model"].float().state_dict()
        logger.info("Resume from %s", opt.resume)

    elif opt.pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(opt.pretrained)  # download if not found locally
        ckpt = torch.load(
            opt.pretrained, map_location=device, weights_only=False
        )  # load checkpoint
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(
                hyp["anchors"]
            )  # force autoanchor
        # init model
        # model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=nc).to(device)
        model = Model(opt.cfg, ch=3, nc=nc).to(device)
        exclude = (
            ["anchor"] if opt.cfg or hyp.get("anchors") else []
        )  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )  # report
        print(
            "Transferred %g/%g items from %s"
            % (len(state_dict), len(model.state_dict()), weights)
        )
    else:
        print("Creating model from scratch")
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(
            device
        )  # create

    # Freeze
    freeze = (
        []
    )  # ["model.0.", "model.1.", "model.2.", "model.3.", "model.4.", "model.5.", "model.6.",  "model.7.",  "model.8."]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        # print(k)
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print("freezing %s" % k)
            v.requires_grad = False

    # Optimizer
    nbs = 32  # nominal batch size
    accumulate = max(
        round(nbs / total_batch_size), 1
    )  # accumulate loss before optimizing
    hyp["weight_decay"] *= (
        total_batch_size * accumulate / nbs
    )  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.optimizer == "Adam":
        optimizer = optim.Adam(
            pg0, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999)
        )  # adjust beta1 to momentum
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)
    elif opt.optimizer == "AdamW":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyp["lr0"],
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )  # adjust beta1 to momentum

    elif opt.optimizer == "LION":
        from lion_pytorch import Lion

        optimizer = Lion(
            model.parameters(), lr=hyp["lr0"], weight_decay=1e-2
        )  # default lr = 1e-4
    else:
        optimizer = optim.SGD(
            pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": hyp["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})  # add pg2 (biases)

    logger.info(
        "Optimizer groups: %g .bias, %g conv.weight, %g other"
        % (len(pg2), len(pg1), len(pg0))
    )
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = (
            lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]
        )  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif opt.standard_lr:
        # lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=hyp["lr_factor"],
            patience=10,
            threshold=0.001,
            min_lr=1e-8,
            verbose=True,
        )
    else:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Logging
    wandb = None
    # if rank in [-1, 0] and wandb and wandb.run is None:
    #     opt.hyp = hyp  # add hyperparameters
    #     wandb_run = wandb.init(config=opt, resume="allow",
    #                            project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
    #                            name=save_dir.stem,
    #                            entity=opt.entity,
    #                            id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
    loggers = {"wandb": wandb}  # loggers dict

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if opt.pretrained or opt.resume:
        # Optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            best_fitness = ckpt["best_fitness"]

        # EMA
        if ema and ckpt.get("ema"):
            ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            ema.updates = ckpt["updates"]

        # Results
        if ckpt.get("training_results") is not None:
            results_file.write_text(
                ckpt["training_results"]
            )  # write results.txt

        # Epochs
        start_epoch = ckpt["epoch"] + 1
        if opt.resume:
            assert (
                start_epoch > 0
            ), "%s training to %g epochs is finished, nothing to resume." % (
                weights,
                epochs,
            )
        if epochs < start_epoch:
            logger.info(
                "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                % (weights, ckpt["epoch"], epochs)
            )
            epochs += ckpt["epoch"]  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[
        -1
    ].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [
        check_img_size(x, gs) for x in opt.img_size
    ]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info("Using SyncBatchNorm()")

    # DDP mode
    if cuda and rank != -1:
        model = DDP(
            model, device_ids=[opt.local_rank], output_device=opt.local_rank
        )

    # Trainloader
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=opt.train_augment,
        cache=opt.cache_images,
        rect=opt.rect,
        rank=rank,
        world_size=opt.world_size,
        workers=opt.num_workers_train,
        image_weights=opt.image_weights,
        prefix=colorstr("train: "),
        bg_file_names=bg_files,
    )

    ulc = np.unique(np.concatenate(dataset.labels, 0)[:, 0]).shape[
        0
    ]  # max label class
    nb = len(dataloader)  # number of batches
    assert ulc == nc, (
        "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g"
        % (ulc, nc, opt.data, nc - 1)
    )

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(
            test_path,
            imgsz_test,
            batch_size,
            gs,
            opt,  # testloader
            hyp=hyp,
            augment=False,
            cache=opt.cache_images and not opt.notest,
            rect=opt.rect,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.num_workers_test,
            prefix=colorstr("val: "),
        )[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))

            # Anchors
            if not opt.noautoanchor:
                check_anchors(
                    dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz
                )

    # Model parameters
    hyp["box"] *= 3.0 / nl  # scale to layers
    hyp["cls"] *= nc * 3.0 / nl  # scale to classes and layers
    hyp["obj"] *= (
        (imgsz / 640) ** 2 * 3.0 / nl
    )  # scale to image size and layers

    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = (
        labels_to_class_weights(dataset.labels, nc).to(device) * nc
    )  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(
        round(hyp["warmup_epochs"] * nb), 1000
    )  # number of warmup iterations, max(3 epochs, 1k iterations)
    nw = min(
        nw, (epochs - start_epoch) / 2 * nb
    )  # limit warmup to < 1/2 of training

    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.amp.GradScaler("cuda")

    pose_loss = PoseLoss(
        model, num_keypoints=9, pretrain_num_epochs=hyp["pretrain_epochs"]
    )  # init loss class

    logger.info(
        f"Image sizes {imgsz} train, {imgsz_test} test\n"
        f"Using {dataloader.num_workers} dataloader workers\n"
        f"Logging results to {save_dir}\n"
        f"Starting training for {epochs} epochs..."
    )
    if wandb:
        wandb.watch(model, log_freq=1)

    for epoch in range(
        start_epoch, epochs
    ):  # epoch ------------------------------------------------------------------

        model.train()
        mloss = torch.zeros(3, device=device)  # mean losses

        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(
            ("\n" + "%10s" * 7)
            % (
                "Epoch",
                "gpu_mem",
                "l_obj",
                "l_box",
                "l_cls",
                "n_targets",
                "img_size",
            )
        )

        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar

        optimizer.zero_grad()

        for i, (
            imgs,
            targets,
            intrinsics,
            paths,
            _,
        ) in (
            pbar
        ):  # batch -------------------------------------------------------------
            ni = (
                i + nb * epoch
            )  # number integrated batches (since train start)

            imgs = (
                imgs.to(device, non_blocking=True).float() / 255.0
            )  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(
                    1, np.interp(ni, xi, [1, nbs / total_batch_size]).round()
                )

                for j, x in enumerate(optimizer.param_groups):
                    if opt.standard_lr:
                        # x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni,
                                xi,
                                [hyp["warmup_momentum"], hyp["momentum"]],
                            )
                    else:
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni,
                            xi,
                            [
                                hyp["warmup_bias_lr"] if j == 2 else 0.0,
                                x["initial_lr"] * lf(epoch),
                            ],
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(
                                ni,
                                xi,
                                [hyp["warmup_momentum"], hyp["momentum"]],
                            )

            # Multi-scale
            if hyp["multi_scale"]:
                sz = (
                    random.randrange(
                        imgsz * (1 - hyp["multi_scale"]),
                        imgsz * (1 + hyp["multi_scale"]) + gs,
                    )
                    // gs
                    * gs
                )  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [
                        math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]
                    ]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(
                        imgs, size=ns, mode="bilinear", align_corners=False
                    )
            else:
                ns = [x for x in imgs.shape[2:]]

            # Forward
            with torch.amp.autocast("cuda"):
                pred = model(imgs)  # forward
                loss, loss_items = pose_loss(
                    pred, targets.to(device), epoch
                )  # loss
            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (
                    i + 1
                )  # update mean losses
                mem = "%.3gG" % (
                    torch.cuda.memory_reserved() / 1e9
                    if torch.cuda.is_available()
                    else 0
                )  # (GB)
                s = ("%10s" * 2 + "%10.4g" * 5) % (
                    "%g/%g" % (epoch, epochs - 1),
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for tensorboard
        if not opt.standard_lr:
            scheduler.step()

        # Log
        train_tags = [
            "train/obj_loss",
            "train/box_loss",
            "train/cls_loss",  # train loss
            "x/lr0",
            "x/lr1",
            "x/lr2",
        ]  # params

        for x, tag in zip(list(mloss) + lr, train_tags):
            if tb_writer:
                tb_writer.add_scalar(tag, x, epoch)  # tensorboard

            if wandb:
                wandb.log({tag: x})  # W&B

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:

            if ema:
                ema.update_attr(
                    model,
                    include=[
                        "yaml",
                        "nc",
                        "hyp",
                        "gr",
                        "names",
                        "stride",
                        "class_weights",
                    ],
                )
            final_epoch = epoch + 1 == epochs
            if (
                not opt.notest
                and epoch > 0
                and epoch % opt.eval_interval == 0
                or not opt.notest
                and epoch > 0.9 * epochs
                and epoch % 50 == 0
                or final_epoch
            ):  # Calculate accuracies
                results = test(
                    opt.data,
                    batch_size=opt.batch_size_test,
                    imgsz=imgsz_test,
                    model=model,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=save_dir,
                    verbose=nc < 50 and final_epoch,
                    plots=plots,
                    log_imgs=opt.log_imgs if wandb else 0,
                    compute_loss=True,
                    symetric=opt.symetric,
                )

                print(results)
                # Write
                with open(results_file, "a") as f:
                    f.write(
                        s + "%10.4g" * 8 % results + "\n"
                    )  # append metrics, val_loss

                # Log
                val_tags = [
                    "val/mean_corner_err_2d",
                    "val/acc",
                    "val/acc3d",
                    "val/acc5cm5deg",  # val metrics
                    "val/obj_loss",
                    "val/box_loss",
                    "val/cls_loss",
                    "val/total_loss",
                ]  # val loss

                for x, tag in zip(list(results), val_tags):

                    if tb_writer:
                        tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                    if wandb:
                        wandb.log({tag: x})  # W&B
                # Update best
                fi = fitness(
                    np.array(results).reshape(1, -1)
                )  # weighted combination of  [mean_corner_err_2d, acc, acc3d, acc5cm5deg]
                if fi > best_fitness:

                    best_fitness = fi

                if opt.standard_lr:
                    scheduler.step(list(results)[-1])
                # Save model
                if (not opt.nosave) or (
                    final_epoch and not opt.evolve
                ):  # if save

                    ckpt = {
                        "epoch": epoch,
                        "best_fitness": best_fitness,
                        "training_results": results_file.read_text(),
                        "model": deepcopy(
                            model.module if is_parallel(model) else model
                        ).half(),
                        "ema": deepcopy(ema.ema).half(),
                        "updates": ema.updates,
                        "optimizer": optimizer.state_dict(),
                        "wandb_id": wandb_run.id if wandb else None,
                    }

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        print(f"Saving best model with fitness: {fi}")
                        torch.save(ckpt, best)
                    del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)
        if opt.bucket:
            os.system(f"gsutil cp {final} gs://{opt.bucket}/weights")  # upload

        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                files = [
                    "results.png",
                    "confusion_matrix.png",
                    *[f"{x}_curve.png" for x in ("F1", "PR", "P", "R")],
                ]
                wandb.log(
                    {
                        "Results": [
                            wandb.Image(str(save_dir / f), caption=f)
                            for f in files
                            if (save_dir / f).exists()
                        ]
                    }
                )
                if opt.log_artifacts:
                    wandb.log_artifact(
                        artifact_or_path=str(final),
                        type="model",
                        name=save_dir.stem,
                    )

        # Test best.pt
        logger.info(
            "%g epochs completed in %.3f hours.\n"
            % (epoch - start_epoch + 1, (time.time() - t0) / 3600)
        )

    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()
    return results


@app.command()
def train_main(
    repo: str = None,
    linemod_root: str = None,
    category: str = None,
    resume: str = None,
    weights: str = None,
):
    global opt
    opt.project = "runs/train"

    if repo and linemod_root:
        # add repo path to sys path, to init model from pth.
        sys.path.insert(0, repo)
        # override opt from args
        opt.data = f"{linemod_root}/{category}/{category}.yaml"
        opt.static_camera = f"{linemod_root}/{category}/linemod_camera.json"
        opt.source = f"{linemod_root}/{category}/JPEGImages"
        mesh_ext = [".ply", ".glb"]

        mesh_dir = f"{linemod_root}/{category}/{category}"
        if os.path.exists(mesh_dir):
            opt.single_cls = False
            opt.mesh_data = [
                os.path.join(mesh_dir, x)
                for x in sorted(os.listdir(mesh_dir))
                if x.endswith(".ply")
            ]
        else:
            for ext in mesh_ext:
                opt.mesh_data = f"{linemod_root}/{category}/{category}{ext}"
                if os.path.exists(opt.mesh_data):
                    break
            assert os.path.exists(opt.mesh_data), f"{opt.mesh_data} not exists"

        opt.pretrained = f"{repo}/yolov5x.pt"
        opt.cfg = f"{repo}/models/yolov5x_6dpose_bifpn.yaml"
        opt.hyp = f"{repo}/configs/hyp.single.yaml"
        opt.resume = resume

    # Set DDP variables
    opt.world_size = (
        int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    )
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    if opt.resume:  # resume an interrupted run
        resume_ckpt = (
            opt.resume if isinstance(opt.resume, str) else get_latest_run()
        )  # specified or most recent path
        assert os.path.isfile(
            resume_ckpt
        ), "ERROR: opt.resume checkpoint does not exist"
        with open(Path(resume_ckpt).parent.parent / "opt.yaml") as f:
            opt = argparse.Namespace(
                **yaml.load(f, Loader=yaml.SafeLoader)
            )  # replace
        opt.resume = resume_ckpt
        opt.batch_size_train = opt.total_batch_size
        logger.info("Resuming training from %s" % opt.resume)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = (
            check_file(opt.data),
            check_file(opt.cfg),
            check_file(opt.hyp),
        )  # check files
        assert len(opt.cfg) or len(
            opt.weights
        ), "either --cfg or --weights must be specified"
        if isinstance(opt.img_size, int):
            opt.img_size = [opt.img_size, opt.img_size]
        opt.img_size.extend(
            [opt.img_size[-1]] * (2 - len(opt.img_size))
        )  # extend to 2 sizes (train, test)
        opt.name = "evolve" if opt.evolve else opt.name
        opt.save_dir = increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve
        )  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size_train
    device = select_device(opt.device, batch_size=opt.batch_size_train)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
        )  # distributed backend
        assert (
            opt.batch_size_train % opt.world_size == 0
        ), "--batch-size must be multiple of CUDA device count"
        opt.batch_size_train = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Override hyp from opt.hyp
    for k, v in getattr(opt, "hyp_cfg", {}).items():
        if k in hyp:
            print(f"Set {k} to {v}")
            hyp[k] = v
        else:
            print(f"Hyp {k} not found, skip")

    # Train
    logger.info(opt)
    try:
        # wandb = None
        import wandb
    except ImportError:
        wandb = None
        prefix = colorstr("wandb: ")
        logger.info(
            f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)"
        )
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(
                f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/'
            )
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer, wandb)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            "lr0": (
                1,
                1e-5,
                1e-1,
            ),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (
                1,
                0.01,
                1.0,
            ),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (1, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (1, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (1, 0.0, 0.2),  # warmup initial bias lr
            "box": (1, 0.02, 0.2),  # box loss gain
            "cls": (1, 0.2, 4.0),  # cls loss gain
            "cls_pw": (1, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (1, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (0, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (1, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (
                0,
                0.0,
                2.0,
            ),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (
                1,
                0.0,
                0.9,
            ),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (1, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (1, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (1, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (1, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (
                0,
                0.0,
                0.001,
            ),  # image perspective (+/- fraction), range 0-0.001
        }

        assert opt.local_rank == -1, "DDP mode not implemented for --evolve"
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = (
            Path(opt.save_dir) / "hyp_evolved.yaml"
        )  # save best result here
        if opt.bucket:
            os.system(
                "gsutil cp gs://%s/evolve.txt ." % opt.bucket
            )  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path(
                "evolve.txt"
            ).exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = (
                    "single"  # parent selection method: 'single' or 'weighted'
                )
                x = np.loadtxt("evolve.txt", ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == "single" or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[
                        random.choices(range(n), weights=w)[0]
                    ]  # weighted selection
                elif parent == "weighted":
                    x = (x * w.reshape(n, 1)).sum(
                        0
                    ) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(
                    v == 1
                ):  # mutate until a change occurs (prevent duplicates)
                    v = (
                        g
                        * (npr.random(ng) < mp)
                        * npr.randn(ng)
                        * npr.random()
                        * s
                        + 1
                    ).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, wandb=wandb)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(
            f"Hyperparameter evolution complete. Best results saved as: {yaml_file}\n"
            f"Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}"
        )


def detect(opt):
    source, weights, view_img, imgsz, mesh_data, cam_intrinsics = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.img_size,
        opt.mesh_data,
        opt.static_camera,
    )

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()

    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    assert os.path.exists(weights), f"{weights} not exists"
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # torch.save(model.state_dict(), "state_dict_model.pt")
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    if cam_intrinsics:
        with open(cam_intrinsics) as f:
            cam_intrinsics = yaml.load(f, Loader=yaml.FullLoader)

        dtx = np.array(cam_intrinsics["distortion"])
        mtx = np.array(cam_intrinsics["intrinsic"])

        fx = mtx[0, 0]
        fy = mtx[1, 1]
        u0 = mtx[0, 2]
        v0 = mtx[1, 2]

        internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # load train/test flag
    if os.path.exists(os.path.join(source, "../train.txt")):
        train_lst = [
            l.strip()
            for l in open(os.path.join(source, "../train.txt")).readlines()
        ]
        test_lst = [
            l.strip()
            for l in open(os.path.join(source, "../test.txt")).readlines()
        ]
    else:
        train_lst = test_lst = []

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if isinstance(mesh_data, str) and opt.single_cls:
        if mesh_data.endswith(".ply"):
            mesh = MeshPly(mesh_data)
            vertices = np.c_[
                np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))
            ].transpose()
            corners3D = [get_3D_corners(vertices)]
        elif mesh_data.endswith(".glb"):
            vertices = load_points_3d_from_cad(
                mesh_data, src_unit="meter", dst_unit="millimeter"
            )
            corners = get_3D_corners__pdebug(vertices)[
                1:, :
            ]  # [9, 3] -> [8, 3] skip center
            corners = np.concatenate(
                (np.transpose(corners), np.ones((1, 8))), axis=0
            )  # [3, 8] -> [4, 8]
            corners3D = [corners]
            num_vertices = vertices.shape[0]
            vertices = np.concatenate(
                (np.transpose(vertices), np.ones((1, num_vertices))), axis=0
            )  # [3, N] -> [4, N]
        else:
            raise NotImplementedError
    else:
        corners3D = []
        for mesh_i in data["mesh"]:
            mesh = MeshPly(mesh_i)
            vertices = np.c_[
                np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))
            ].transpose()
            corners3D_i = get_3D_corners(vertices)
            corners3D.append(corners3D_i)

    # edges_corners = [[0, 1], [0, 3], [0, 7], [1, 2], [1, 6], [2, 3], [2, 4], [3, 5], [4, 5], [4, 6], [5, 7], [6, 7]]
    edges_corners = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]
    colormap = np.array(
        ["r", "g", "b", "c", "m", "y", "k", "w", "xkcd:sky blue"]
    )

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz)
            .to(device)
            .type_as(next(model.parameters()))
        )  # run once

    predictions = []
    t0 = time.time()
    for path, img, im0s, intrinsics, shapes in dataset:
        count = 0
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Compute intrinsics
        if cam_intrinsics is None:
            (
                fx,
                fy,
                det_height,
                u0,
                v0,
                im_native_width,
                im_native_height,
            ) = intrinsics
            # fx, fy  = # calculate_focal_length(float(focal_len), int(im_native_width), int(im_native_height), float(det_width), float(det_height))
            internal_calibration = get_camera_intrinsic(u0, v0, fx, fy)

        # Inference
        t1 = time_synchronized()
        pred, train_out = model(img, augment=False)  # img: 640x480
        # pred = model(img, augment=False)[0]

        # Using confidence threshold, eliminate low-confidence predictions
        pred = box_filter(
            pred,
            conf_thres=opt.conf_thres_infer,
            max_det=opt.detect_max_det,
            multi_label=(not opt.single_cls),
        )
        t2 = time_synchronized()

        if pred[0].shape[0] == 0 and opt.save_img:
            # no detect result
            filename = (
                os.path.basename(path).split(".")[0]
                + "_"
                + str(count)
                + "_predicted.png"
            )
            file_path = os.path.join(save_dir, filename)
            shutil.copy(path, file_path)
            continue

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            print(det.cpu().numpy())
            p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            (Path(str(save_dir / "labels"))).mkdir(parents=True, exist_ok=True)
            s += "%gx%g " % img.shape[2:]  # print string
            if len(det):
                det = det.cpu()
                scale_coords(
                    img.shape[2:], det[:, :18], shapes[0], shapes[1]
                )  # native-space pred

                num_dets = det.shape[0]
                for j in range(num_dets):
                    # Rescale boxes from img_size to im0 size
                    prediction_confidence = det[j, 18]

                    # box_pred = det.clone().cpu()
                    box_predn = det[j, :18].clone().cpu()
                    # Denormalize the corner predictions
                    corners2D_pr = np.array(
                        np.reshape(box_predn, [9, 2]), dtype="float32"
                    )
                    # Calculate rotation and tranlation in rodriquez format
                    mesh_idx = int(det[j, -1])
                    R_pr, t_pr = pnp(
                        np.array(
                            np.transpose(
                                np.concatenate(
                                    (
                                        np.zeros((3, 1)),
                                        corners3D[mesh_idx][:3, :],
                                    ),
                                    axis=1,
                                )
                            ),
                            dtype="float32",
                        ),
                        corners2D_pr,
                        np.array(internal_calibration, dtype="float32"),
                    )
                    pose_mat = cv2.hconcat((R_pr, t_pr))
                    euler_angles = cv2.decomposeProjectionMatrix(pose_mat)[6]
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    if opt.save_img:
                        # convert bgr to rgb

                        local_img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        figsize = (shapes[0][1] / 96, shapes[0][0] / 96)
                        fig = plt.figure(frameon=False, figsize=figsize)
                        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
                        ax.set_axis_off()
                        fig.add_axes(ax)
                        # image = np.uint8(local_img) # .resize((im_native_width, im_native_height)))
                        ax.imshow(local_img, aspect="auto")
                        corn2D_pr = corners2D_pr[1:, :]

                        # Plot projection corners
                        for edge in edges_corners:
                            ax.plot(
                                corn2D_pr[edge, 0],
                                corn2D_pr[edge, 1],
                                color="b",
                                linewidth=0.5,
                            )
                        ax.scatter(
                            corners2D_pr.T[0],
                            corners2D_pr.T[1],
                            c=colormap,
                            s=10,
                        )

                        min_x, min_y = np.amin(corners2D_pr.T[0]), np.amin(
                            corners2D_pr.T[1]
                        )

                        max_x, max_y = np.amax(corners2D_pr.T[0]), np.amax(
                            corners2D_pr.T[1]
                        )

                        if True:
                            # Create a bounding box around the object
                            rect = patches.Rectangle(
                                (min_x, min_y),
                                max_x - min_x,
                                max_y - min_y,
                                linewidth=1,
                                edgecolor="r",
                                facecolor="none",
                            )
                            # Add the patch to the Axes
                            ax.add_patch(rect)

                        ax.text(
                            min_x,
                            min_y - 10,
                            f"Conf: {prediction_confidence.cpu().numpy():.3f}, Rot: {euler_angles}",
                            style="italic",
                            bbox={
                                "facecolor": "blue",
                                "alpha": 0.5,
                                "pad": 10,
                            },
                        )
                        if os.path.basename(path) in train_lst:
                            phase = "_train_"
                        elif os.path.basename(path) in test_lst:
                            phase = "_test_"
                        else:
                            phase = "_"
                        filename = (
                            os.path.basename(path).split(".")[0]
                            + phase
                            + str(count)
                            + "_predicted.png"
                        )
                        file_path = os.path.join(save_dir, filename)
                        fig.savefig(
                            file_path,
                            dpi=96,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        # fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
                        plt.close()

                        count += 1

                with open(txt_path + ".txt", "a") as f:
                    f.write(str(det.numpy()) + "\n")

            # Print time (inference + NMS)
            print(f"{s}Done. ({t2 - t1:.3f}s)")
            # # Save results (image with detections)
            # if opt.save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)

    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
    print(f"Results saved to {save_dir}{s}")
    print(f"Done. ({time.time() - t0:.3f}s)")


@app.command()
def detect_main(
    repo: str = None,
    linemod_root: str = None,
    category: str = None,
    weights: str = None,
    source: str = None,
    max_det: int = 1,
):
    global opt
    opt.project = "runs/detect"

    if repo and linemod_root:
        # add repo path to sys path, to init model from pth.
        sys.path.insert(0, repo)
        # override opt from args
        opt.data = f"{linemod_root}/{category}/{category}.yaml"
        opt.static_camera = f"{linemod_root}/{category}/linemod_camera.json"
        if source:
            opt.source = source
        else:
            opt.source = f"{linemod_root}/{category}/JPEGImages"
        opt.pretrained = f"{repo}/yolov5x.pt"
        opt.cfg = f"{repo}/models/yolov5x_6dpose_bifpn.yaml"
        opt.hyp = f"{repo}/configs/hyp.single.yaml"

        mesh_ext = [".ply", ".glb"]
        mesh_dir = f"{linemod_root}/{category}/{category}"
        if os.path.exists(mesh_dir):
            opt.single_cls = False
            opt.mesh_data = [
                os.path.join(mesh_dir, x)
                for x in sorted(os.listdir(mesh_dir))
                if x.endswith(".ply")
            ]
        else:
            for ext in mesh_ext:
                opt.mesh_data = f"{linemod_root}/{category}/{category}{ext}"
                if os.path.exists(opt.mesh_data):
                    break
            assert os.path.exists(opt.mesh_data), f"{opt.mesh_data} not exists"

        opt.weights = weights
        opt.detect_max_det = max_det

    print(opt)
    with torch.no_grad():
        detect(opt)


@app.command()
def pt2onnx(
    repo: str = None,
    weights_path: str = None,
    onnx_path: str = None,
    batch_size: int = 1,
    simplify: bool = True,
    dynamic: bool = False,
    device: str = "0",
):
    """
    Convert PyTorch model to ONNX format for yolo pose model.

    Args:
        weights_path: Path to PyTorch .pt weights file
        onnx_path: Output ONNX file path (auto-generated if None)
        img_size: Input image size (square)
        batch_size: Batch size for ONNX model
        simplify: Whether to simplify ONNX model using onnxsim
        dynamic: Whether to export with dynamic axes
        device: Device to run conversion on ("cpu" or "0")

    Returns:
        tuple: (onnx_path, success_flag)
    """
    import onnx
    import torch.onnx

    try:
        import onnxsim

        has_onnxsim = True
    except ImportError:
        has_onnxsim = False
        simplify = False
        print("Warning: onnxsim not found, skipping model simplification")

    global opt
    sys.path.insert(0, repo)
    opt.pretrained = f"{repo}/yolov5x.pt"
    opt.cfg = f"{repo}/models/yolov5x_6dpose_bifpn.yaml"
    opt.hyp = f"{repo}/configs/hyp.single.yaml"

    # Set device
    if device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )

    img_size = (480, 640)

    # Initialize model
    model = Model(opt.cfg).to(device)

    # Load model
    if weights_path:
        print(f"Loading model from {weights_path}")
        ckpt = torch.load(
            weights_path, map_location=device, weights_only=False
        )
        state_dict = (
            ckpt["ema" if ckpt.get("ema") else "model"].float().state_dict()
        )
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(
        device
    )

    # Generate output path
    if onnx_path is None:
        onnx_path = weights_path.replace(".pt", ".onnx")

    print(f"Exporting to ONNX: {onnx_path}")

    # Dynamic axes configuration
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 1: "anchors"},
        }
    else:
        dynamic_axes = None

    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False,
        )

        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")

        # Simplify if requested
        if simplify and has_onnxsim:
            print("Simplifying ONNX model...")
            model_simp, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(model_simp, onnx_path)
                print("ONNX model simplified successfully")
            else:
                print("Warning: ONNX model simplification failed")

        print(f"ONNX export completed successfully: {onnx_path}")
        return onnx_path, True

    except Exception as e:
        print(f"ONNX export failed: {e}")
        return None, False


from pdebug.otn import manager as otn_manager
from pdebug.utils.fileio import do_system


@otn_manager.NODE.register(name="yolo-pose")
def node(
    name: str = None,  # train_main, test_main, detect_main, verify_data, pt2onnx
    **kwargs,
):
    """Launch yolo-pose from otn-cli.

    Example:
        >>> oth-cli --node yolo-pose --name train-main --repo ${YOLO_POSE_REPO} --linemod-root ${OUTPUT} --category ${CATEGORY}
    """
    cur_path = os.path.abspath(__file__)
    main_cmd = f"python3 {cur_path} {name}"
    for k, v in kwargs.items():
        if "*" in v:
            v = f"'{v}'"
        main_cmd += f" --{k.replace('_', '-')} {v}"

    retry_args = (
        "--resume runs/train/latest/weights/last.pt"
        if name == "train_main"
        else ""
    )

    trials = 10 if os.getenv("OTN_RETRY_YOLOPOSE", "0") == "1" else 1
    ret = do_system(
        main_cmd, with_return=True, trials=trials, retry_args=retry_args
    )
    if not ret:
        raise RuntimeError(f"{main_cmd} failed")


if __name__ == "__main__":
    app()
