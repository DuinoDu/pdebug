"""source reader """
import glob
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from json import JSONEncoder
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.io as sio
import tqdm

from ...data_types import (
    Camera,
    CameraExtrinsic,
    CameraIntrinsic,
    PointcloudTensor,
    Pose,
    PoseList,
    Tensor,
)
from ...utils.cache import FileCache
from ...utils.ddd import align_timestamp, generate_pcd_from_depth
from ...utils.env import SCIPY_INSTALLED
from ...utils.fileio import load_rgb_from_yuv
from ...utils.waicai_utils import (
    get_sensor_yaml_roomplan,
    prepare_biaoding_info,
)
from ..registry import SOURCE_REGISTRY
from .source import Reader

if SCIPY_INSTALLED:
    from scipy.spatial.transform import Rotation as R


@dataclass
class Frame:
    """
    Frame means a view frame.

    TODO: support multiple view.
    """

    rgb: Optional[Tensor] = None
    rgb_file: Optional[str] = None

    depth: Optional[Tensor] = None
    depth_file: Optional[str] = None

    pcd: Optional[PointcloudTensor] = None
    pcd_file: Optional[str] = None

    pose: Optional[CameraExtrinsic] = None
    pose_file: Optional[str] = None

    intrinsic: Optional[CameraIntrinsic] = None
    intrinsic_file: Optional[str] = None

    camera: Optional[Camera] = None
    camera_file: Optional[str] = None

    rgb_camera: Optional[Camera] = None
    rgb_camera_file: Optional[str] = None

    depth_camera: Optional[Camera] = None
    depth_camera_file: Optional[str] = None

    semseg: Optional[Tensor] = None
    semseg_file: Optional[str] = None

    instseg: Optional[Tensor] = None

    boxes2d: Optional[Tensor] = None

    boxes3d: Optional[Tensor] = None

    timestamp: Optional[float] = None

    extras: Optional[Dict[str, Any]] = None

    ### loading func
    load_rgb_file: Callable = lambda x: cv2.imread(x, -1)
    load_depth_file: Callable = lambda x: cv2.imread(x, -1)
    load_semseg_file: Callable = lambda x: cv2.imread(x, -1)

    def load_data(
        self,
        load_rgb=True,
        load_depth=True,
        load_semseg=True,
        generate_pcd=True,
        depth_rgb_scale=None,
    ):
        if self.rgb_file is not None and self.rgb is None and load_rgb:
            self.rgb = self.load_rgb_file(self.rgb_file)

        if self.depth_file is not None and self.depth is None and load_depth:
            self.depth = self.load_depth_file(self.depth_file)

        if (
            self.semseg_file is not None
            and self.semseg is None
            and load_semseg
        ):
            self.semseg = self.load_semseg_file(self.semseg_file)

        if (
            self.pose is not None
            and self.intrinsic is not None
            and self.camera is None
        ):
            assert isinstance(self.pose, CameraExtrinsic)
            assert isinstance(self.intrinsic, CameraIntrinsic)
            self.camera = Camera(self.pose, self.intrinsic)

        if (
            self.pcd_file is None
            and self.pcd is None
            and self.depth is not None
            and self.camera is not None
            and generate_pcd
        ):
            self.pcd = generate_pcd_from_depth(
                self.depth,
                self.camera,
                rgb=self.rgb,
                depth_rgb_scale=depth_rgb_scale,
            )

        if self.rgb_camera is None and self.rgb_camera_file is not None:
            assert os.path.exists(self.rgb_camera_file)
            self.rgb_camera = Camera.from_yamlfile(self.rgb_camera_file)

        if self.depth_camera is None and self.depth_camera_file is not None:
            assert os.path.exists(self.depth_camera_file)
            self.depth_camera = Camera.from_yamlfile(self.depth_camera_file)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["load_rgb_file"]
        del state["load_depth_file"]
        del state["load_semseg_file"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.load_rgb_file = lambda x: cv2.imread(x, -1)
        self.load_depth_file = lambda x: cv2.imread(x, -1)
        self.load_semseg_file = lambda x: cv2.imread(x, -1)

        # TODO: refactor to a better method
        if self.semseg_file and self.semseg_file.endswith(".mat"):
            self.load_semseg_file = DDDDataSUNRGBD.load_semseg_file


@dataclass
class HawkFrame(Frame):
    """
    Hawk Frame, 2 rgb + 1 tof

    self.rgb is rgb_left

    """

    rgb_right: Optional[Tensor] = None

    depth_conf: Optional[Tensor] = None

    left_rgb_camera: Optional[Camera] = None
    left_rgb_camera_pose: Optional[Pose] = None

    right_rgb_camera: Optional[Camera] = None
    right_rgb_camera_pose: Optional[Pose] = None

    tof_camera: Optional[Camera] = None
    tof_camera_pose: Optional[Pose] = None

    undistort_rgb: Optional[Tensor] = None
    depth_aligned_to_rgb: Optional[Tensor] = None

    T_WH_rgb: Optional[Tensor] = None
    T_rgb_tof: Optional[Tensor] = None

    def load_data(
        self,
        load_rgb=True,
        load_depth=True,
        load_semseg=True,
        generate_pcd=True,
        depth_rgb_scale=None,
        load_rgb_right=False,
        load_depth_conf=False,
    ):
        extras = self.extras
        if (
            extras.get("rgb_left_y", None) is not None
            and extras.get("rgb_left_uv", None) is not None
            and self.rgb is None
            and load_rgb
        ):
            self.rgb = load_rgb_from_yuv(
                extras["rgb_left_y"], extras["rgb_left_uv"]
            )

            if load_rgb_right:
                self.rgb_right = load_rgb_from_yuv(
                    extras["rgb_right_y"], extras["rgb_right_uv"]
                )

        if (
            extras.get("tof_depth", None) is not None
            and self.depth is None
            and load_depth
        ):
            self.depth = self.load_depth_file(extras["tof_depth"])

        if (
            extras.get("tof_conf", None) is not None
            and self.depth_conf is None
            and load_depth_conf
        ):
            self.depth_conf = self.load_depth_file(extras["tof_conf"])

        if (
            extras.get("left_rgb_camera", None) is not None
            and extras["data_batch"].left_rgb_camera_image.IsInitialized()
        ):
            self.left_rgb_camera = extras["left_rgb_camera"]
            x, y, z, rw, rx, ry, rz = extras[
                "data_batch"
            ].left_rgb_camera_image.tracking_info.pose.pose
            self.left_rgb_camera_pose = Pose.fromRt(
                [rx, ry, rz, rw], [x, y, z]
            )

        if (
            extras.get("right_rgb_camera", None) is not None
            and extras["data_batch"].right_rgb_camera_image.IsInitialized()
        ):
            self.right_rgb_camera = extras["right_rgb_camera"]
            x, y, z, rw, rx, ry, rz = extras[
                "data_batch"
            ].right_rgb_camera_image.tracking_info.pose.pose
            self.right_rgb_camera_pose = Pose.fromRt(
                [rx, ry, rz, rw], [x, y, z]
            )

        if (
            extras.get("tof_camera", None) is not None
            and extras["data_batch"].tof_depth_camera_image.IsInitialized()
        ):
            self.tof_camera = extras["tof_camera"]
            x, y, z, rw, rx, ry, rz = extras[
                "data_batch"
            ].tof_depth_camera_image.tracking_info.pose.pose
            self.tof_camera_pose = Pose.fromRt([rx, ry, rz, rw], [x, y, z])

        if (
            self.semseg_file is not None
            and self.semseg is None
            and load_semseg
        ):
            self.semseg = self.load_semseg_file(self.semseg_file)

        if (
            self.pcd_file is None
            and self.pcd is None
            and self.depth is not None
            and self.camera is not None
            and generate_pcd
        ):
            self.pcd = generate_pcd_from_depth(
                self.depth,
                self.camera,
                rgb=self.rgb,
                depth_rgb_scale=depth_rgb_scale,
            )


class FrameEncoder(json.JSONEncoder):
    def default(self, o):
        ret = {}
        ret["rgb_file"] = o.__dict__["rgb_file"]
        ret["depth_file"] = o.__dict__["depth_file"]
        ret["semseg_file"] = o.__dict__["semseg_file"]
        return ret


@dataclass
class Room:
    name: str

    frames: List[Frame]


class DDDReader(Iterator):

    """Base Reader Class"""

    def __init__(self, data_path, logger=None):
        self._cur = 0
        self._cur_filename = None

        self.logging = True
        self.logger = logger if logger else logging

        self._frames = []
        self._rooms = []

        self.data_path = data_path
        self.load_info_from_data_path()

    def load_info_from_data_path(self):
        """Load data info from path."""
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        try:
            frame = self.frames[self._cur]
            self._cur += 1
            return frame
        except:
            raise StopIteration

    def read(self, idx):
        frame = self.frames[idx]
        return frame

    next = __next__

    def __len__(self):
        return len(self.frames)

    @property
    def index(self):
        return self._cur

    @property
    def rooms(self):
        return self._rooms

    @property
    def frames(self):
        return self._frames

    def __add__(self, other):
        if not isinstance(other, DDDReader):
            raise NotImplementedError
        self._frames.extend(other._frames)
        self._rooms.extend(other._rooms)
        return self


"""
ScanNet
"""


@SOURCE_REGISTRY.register(name="ddddata_scannet")
class DDDDataScanNet(DDDReader):

    """ScanNet data reader.

    ScanNet
    |---scene0000_0
    |          |---color
    |          |     |---0.jpg
    |          |     |---...
    |          |---depth
    |          |     |---0.png
    |          |     |---...
    |          |---pose
    |          |     |---0.txt
    |          |     |---...
    |          |---intrinsic
    |                |---extrinsic_color.txt
    |                |---extrinsic_depth.txt
    |                |---intrinsic_color.txt
    |                |---intrinsic_depth.txt
    |---scene0000_1

    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=None,
        quiet=False,
        frame_downsample=None,
    ):
        if ratio is not None:
            assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.quiet = quiet
        self.frame_downsample = frame_downsample
        super(DDDDataScanNet, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        videos = os.listdir(self.data_path)
        rooms = defaultdict(list)

        if self.ratio:
            keep_num = max(int(len(videos) * self.ratio), 1)
            old_length = len(videos)
            videos = videos[:keep_num]
            print(f"{old_length} => {len(videos)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(videos),
            )
        for video in videos:
            if self.logging and (not self.quiet):
                t.update()
            room_name = video.split("_")[0]

            rgb_path = os.path.join(self.data_path, video, "color")
            depth_path = os.path.join(self.data_path, video, "depth")
            pose_path = os.path.join(self.data_path, video, "pose")
            intrinsic_path = os.path.join(self.data_path, video, "intrinsic")

            rgb_files = sorted(
                [
                    os.path.join(rgb_path, x)
                    for x in sorted(os.listdir(rgb_path))
                    if x.endswith(".jpg")
                ]
            )
            # depth_files = sorted([os.path.join(depth_path, x)
            #   for x in sorted(os.listdir(depth_path)) if x.endswith('.png')])
            # pose_files = sorted([os.path.join(pose_path, x)
            #   for x in sorted(os.listdir(pose_path)) if x.endswith('.txt')])

            file_indexes = [
                os.path.basename(f).split(".")[0] for f in rgb_files
            ]

            if self.frame_downsample:
                random.shuffle(file_indexes)
                topk = int(len(file_indexes) * self.frame_downsample)
                file_indexes = file_indexes[:topk]

            for idx in file_indexes:
                rgb_file = os.path.join(rgb_path, f"{idx}.jpg")
                depth_file = os.path.join(depth_path, f"{idx}.png")
                pose_file = os.path.join(pose_path, f"{idx}.txt")

                if not os.path.exists(depth_file):
                    __import__("ipdb").set_trace()
                    print(f"{depth_file} not found")

                if not os.path.exists(pose_file):
                    __import__("ipdb").set_trace()
                    print(f"{pose_file} not found")

                frame = Frame(
                    rgb_file=rgb_file,
                    depth_file=depth_file,
                    pose_file=pose_file,
                )
                self._frames.append(frame)
                rooms[room_name].append(frame)

        for name, frames in rooms.items():
            room = Room(name, frames)
            self._rooms.append(room)


"""
APPLE RoomPlan
"""


@SOURCE_REGISTRY.register(name="ddddata_apple_roomplan")
class DDDDataAppleRoomPlan(DDDReader):

    """Apple RoomPlan data reader.

        2023_03_15_23_41_28 <capture timestamp>
        |---conf_*.png
        |---depth_*.png
        |---frame_*.jpg
        |---frame_*.json
        |---info.json
        |---points.ply
        |---textured_output.mtl
        |---textured_output.jpg
        |---textured_output.obj
        |---export.obj
        |---export.mtl
        |---export_refined.obj
        |---roomplan
                |---room.json
                |---room.mtl
                |---room.obj
                |---room_no_objects.mtl
                |---room_no_objects.obj
                |---room.usdz

    Args:
        data_path: input data path.
        logger: given logger.
        ratio: not used.
        quiet: remove tqdm progress.
        frame_downsample: downsample frame number, float, 0~1
        depth_first: search index from rgb or depth. Default is `false`, means rgb first.
    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=None,
        quiet=False,
        frame_downsample=None,
        depth_first=False,
    ):
        self.quiet = quiet
        # assert 0.0 <= ratio <= 1.0
        # self.ratio = ratio
        self.frame_downsample = frame_downsample
        self.depth_first = depth_first
        super(DDDDataAppleRoomPlan, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        rooms = defaultdict(list)
        room_name = os.path.basename(self.data_path)

        if self.depth_first:
            index_files = sorted(
                [
                    x
                    for x in sorted(os.listdir(self.data_path))
                    if x.endswith(".png") and x.startswith("depth_")
                ]
            )
        else:
            index_files = sorted(
                [
                    x
                    for x in sorted(os.listdir(self.data_path))
                    if x.endswith(".jpg") and x.startswith("frame_")
                ]
            )

        if self.frame_downsample:
            random.shuffle(index_files)
            topk = int(len(index_files) * self.frame_downsample)
            index_files = index_files[:topk]

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(index_files),
            )

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()
            index = index_file.split(".")[0].split("_")[1]

            depth_file = os.path.join(self.data_path, f"depth_{index}.png")
            conf_file = os.path.join(self.data_path, f"conf_{index}.png")
            camera_file = os.path.join(self.data_path, f"frame_{index}.json")
            rgb_file = os.path.join(self.data_path, f"frame_{index}.jpg")

            if not os.path.exists(rgb_file):
                """For apple roomplan data, rgb data number is less than depth."""
                rgb_file = None

            if not os.path.exists(depth_file):
                __import__("ipdb").set_trace()
                print(f"{depth_file} not found")

            if not os.path.exists(camera_file):
                __import__("ipdb").set_trace()
                print(f"{camera_file} not found")
            camera, timestamp = self.parse_apple_camera_json(camera_file)

            frame = Frame(
                rgb_file=rgb_file,
                depth_file=depth_file,
                camera_file=camera_file,
                camera=camera,
                pose=camera.extrinsic,
            )
            frame.extras = {"conf_file": conf_file}
            self._frames.append(frame)
            rooms[room_name].append(frame)

        for name, frames in rooms.items():
            room = Room(name, frames)
            self._rooms.append(room)

    def parse_apple_camera_json(self, jsonfile) -> Tuple[Camera, float]:
        """Load apple camera json."""
        with open(jsonfile, "r") as fid:
            data = json.load(fid)
        intrinsic = np.asarray(data["intrinsics"]).reshape(3, 3)
        pose = np.asarray(data["cameraPoseARFrame"]).reshape(4, 4)
        camera = Camera(pose, intrinsic)
        timestamp = data["time"]
        return camera, timestamp


@dataclass
class ImageInfo:
    image_name: str = None
    raw_image_path: str = None
    raw_depth_path: str = None
    raw_pose: Pose = (
        None  # IMU pose, but timestamp aligned to rgb (raw_image_path)
    )
    raw_depth_pose: Pose = (
        None  # IMU pose, but timestamp aligned to depth (raw_depth_path)
    )

    def search_depth_file(self):
        """
        Find raw_depth_path given raw_image_path.
        """
        depth_dir = os.path.dirname(self.raw_image_path).replace(
            "cam5", "tof_depth"
        )
        assert os.path.exists(depth_dir), f"{depth_dir} not exists"
        depth_file_ext = os.listdir(depth_dir)[0].split("_")[1]
        # find depth timestamp
        name_to_ts = lambda x: int(os.path.basename(x).split("_")[0])
        rgb_timestamp = name_to_ts(self.raw_image_path)
        depth_ts = [name_to_ts(x) for x in os.listdir(depth_dir)]
        depth_ts_aligned = align_timestamp([[rgb_timestamp], depth_ts], 0)
        depth_timestamp = depth_ts_aligned[0][0]
        # find depth file
        depth_filename = f"{depth_timestamp}_{depth_file_ext}"
        depth_filepath = os.path.join(depth_dir, depth_filename)
        assert os.path.exists(depth_filepath), f"{depth_filepath} not exists"
        self.raw_depth_path = depth_filepath

    def search_pose(self, pose_list):
        name_to_ts = lambda x: int(os.path.basename(x).split("_")[0])

        def _find_pose(seed_ts):
            pose_ts = pose_list.timestamps
            pose_ts_aligned = align_timestamp([[seed_ts], pose_ts], 0)
            pose_timestamp = pose_ts_aligned[0][0]
            pose = pose_list.data[pose_list.timestamps.index(pose_timestamp)]
            return pose

        if self.raw_image_path:
            self.raw_pose = _find_pose(name_to_ts(self.raw_image_path))
        if self.raw_depth_path:
            self.raw_depth_pose = _find_pose(name_to_ts(self.raw_depth_path))

    @staticmethod
    def get_pose_xml(raw_imgfile):
        pose_xml = os.path.dirname(raw_imgfile).replace(
            "cam5", "Save/Pose.xml"
        )
        return pose_xml

    @staticmethod
    def load_pose_list(pose_xml, use_cache=True, verbose=False):
        assert os.path.exists(pose_xml), f"{pose_xml} not exists"
        with FileCache(
            dict,
            f".tmp_cache_{pose_xml.replace('/', '__')}",
            quiet=not verbose,
            use_cache=use_cache,
        ) as _cache_pose:
            if not _cache_pose:
                pose_list = PoseList.from_xmlfile(pose_xml, verbose=verbose)
                _cache_pose[pose_xml] = pose_list
        return _cache_pose[pose_xml]


"""
NYUv2
"""


@SOURCE_REGISTRY.register(name="ddddata_nyuv2")
class DDDDataNYUv2(DDDReader):

    """NYUv2 data reader.

    nyu
    |---nyu2_train
    |           |---bathroom_0028_out   (room)
    |                   |---0.jpg   (rgb)
    |                   |---0.png   (depth)
    |                   |---...
    |---nyu2_train.csv
    |---nyu2_test
    |           |---xxxx_colors.png (rgb)
    |           |---xxxx_depth.png  (depth)
    |---nyu2_test.csv

    Args:
        imglist_file: provided image list file. If not provided, use nyu2_train
            format as default.
    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=1.0,
        quiet=False,
        frame_downsample=None,
        imglist_file=None,
        imglist_splitter=",",
    ):
        self.quiet = quiet
        assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.frame_downsample = frame_downsample
        self.imglist_file = imglist_file
        self.imglist_splitter = imglist_splitter
        super(DDDDataNYUv2, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        if self.imglist_file is None:
            self._load_info_by_folder()
        else:
            self._load_info_by_imglist()

    def _load_info_by_folder(self):
        rooms = defaultdict(list)
        room_names = sorted(os.listdir(self.data_path))
        for room_name in room_names:
            root = os.path.join(self.data_path, room_name)
            index_files = sorted(
                [x for x in sorted(os.listdir(root)) if x.endswith(".png")]
            )

            if self.frame_downsample:
                random.shuffle(index_files)
                topk = int(len(index_files) * self.frame_downsample)
                index_files = index_files[:topk]

            if self.logging and (not self.quiet):
                t = tqdm.tqdm(
                    total=len(index_files),
                )

            for index_file in index_files:
                if self.logging and (not self.quiet):
                    t.update()
                index = index_file.split(".")[0]

                depth_file = os.path.join(root, index_file)
                rgb_file = os.path.join(
                    root, index_file.replace(".png", ".jpg")
                )

                frame = Frame(rgb_file=rgb_file, depth_file=depth_file)
                self._frames.append(frame)
                rooms[room_name].append(frame)

        for name, frames in rooms.items():
            room = Room(name, frames)
            self._rooms.append(room)

    def _load_info_by_imglist(self):
        assert os.path.exists(
            self.imglist_file
        ), f"{self.imglist_file} not found."
        index_files = [
            x.strip() for x in open(self.imglist_file, "r").readlines()
        ]

        rooms = defaultdict(list)

        if self.frame_downsample:
            random.shuffle(index_files)
            topk = int(len(index_files) * self.frame_downsample)
            index_files = index_files[:topk]

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(index_files),
            )

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()

            rgb_file, depth_file = index_file.split(self.imglist_splitter)[:2]

            if not os.path.exists(rgb_file):
                # fix file path
                data_path_name = os.path.basename(self.data_path)
                rgb_file = os.path.join(
                    self.data_path, rgb_file.split(data_path_name)[1][1:]
                )  # /xxx => xxx
                depth_file = os.path.join(
                    self.data_path, depth_file.split(data_path_name)[1][1:]
                )

            assert os.path.exists(rgb_file) and os.path.exists(depth_file)

            frame = Frame(rgb_file=rgb_file, depth_file=depth_file)
            self._frames.append(frame)
            room_name = rgb_file.split("/")[-2]
            rooms[room_name].append(frame)

        for name, frames in rooms.items():
            room = Room(name, frames)
            self._rooms.append(room)


@SOURCE_REGISTRY.register(name="ddddata_nyuv2_ESANet")
class DDDDataNYUv2ESANet(DDDReader):

    """NYUv2 data reader.

    nyu
    |---train
    |     |---rgb
    |           |---0003.png
    |           |---...
    |     |---depth
    |           |---0003.png
    |           |---...
    |     |---depth_raw
    |           |---0003.png
    |           |---...
    |     |---labels_13
    |           |---0003.png
    |           |---...
    |     |---labels_40
    |           |---0003.png
    |           |---...
    |     |---labels_894
    |           |---0003.png
    |           |---...
    |---train.txt
    |---test
    |     |---rgb
    |     |---depth
    |     |---depth_raw
    |     |---labels_13
    |     |---labels_40
    |     |---labels_894
    |---test.txt

    Args:
        imglist_file: provided image list file. If not provided, use nyu2_train
            format as default.
    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=1.0,
        quiet=False,
        frame_downsample=None,
        imglist_file=None,
        num_classes=13,
    ):
        self.quiet = quiet
        assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.frame_downsample = frame_downsample
        self.imglist_file = imglist_file
        assert num_classes in [13, 40, 894]
        self.num_classes = num_classes
        super(DDDDataNYUv2ESANet, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        assert os.path.exists(
            self.imglist_file
        ), f"{self.imglist_file} not found."
        index_files = [
            x.strip() for x in open(self.imglist_file, "r").readlines()
        ]

        rooms = defaultdict(list)

        if self.frame_downsample:
            random.shuffle(index_files)
            topk = int(len(index_files) * self.frame_downsample)
            index_files = index_files[:topk]

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(index_files),
            )

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()

            rgb_file = os.path.join(self.data_path, f"rgb/{index_file}.png")
            depth_file = os.path.join(
                self.data_path, f"depth/{index_file}.png"
            )
            depth_file = os.path.join(
                self.data_path, f"depth/{index_file}.png"
            )
            semseg_file = os.path.join(
                self.data_path, f"labels_{self.num_classes}/{index_file}.png"
            )

            assert os.path.exists(rgb_file) and os.path.exists(depth_file)

            frame = Frame(
                rgb_file=rgb_file,
                depth_file=depth_file,
                semseg_file=semseg_file,
            )
            self._frames.append(frame)


"""
ARkitScenes
"""


@SOURCE_REGISTRY.register(name="ddddata_arkitscenes_raw")
class DDDDataARKitScenes_Raw(DDDReader):

    """ARKitScenes raw data reader.

    https://github.com/apple/ARKitScenes/blob/main/DATA.md
    https://github.com/apple/ARKitScenes/tree/main/raw

    Training / Validation
    |---42899756    # <room_id>
    |          |---lowres_wide      # rgb 256x192
    |          |     |---42899756_247278.827.png
    |          |     |---...
    |          |---lowres_depth     # depth 256x192
    |          |     |---42899756_247278.844.png
    |          |     |---...
    |          |---confidence       # conf 256x192, same number as lowres_depth
    |          |     |---42899756_247278.844.png
    |          |     |---...
    |          |---lowres_wide_intrinsics       # same number as lowres_wide
    |          |     |---42899756_247278.827.pincam
    |          |     |---...
    |          |---lowres_wide.traj
    |          |---42899756_3dod_annotation.json
    |          |---42899756_3dod_mesh.ply
    |          |---42899756.mov
    |          |---...
    |          |---...
    |
    |---scene0000_1

    """

    def __init__(
        self,
        data_path,
        logger=None,
        room_downsample=None,
        quiet=False,
        frame_downsample=None,
    ):
        if room_downsample is not None:
            assert 0.0 <= room_downsample <= 1.0
        self.room_downsample = room_downsample
        self.quiet = quiet
        self.frame_downsample = frame_downsample
        super(DDDDataARKitScenes_Raw, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        videos = os.listdir(self.data_path)
        rooms = defaultdict(list)

        if self.room_downsample:
            keep_num = max(int(len(videos) * self.room_downsample), 1)
            old_length = len(videos)
            videos = videos[:keep_num]
            print(f"{old_length} => {len(videos)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(videos),
            )
        for video in videos:
            if self.logging and (not self.quiet):
                t.update()
            room_name = video

            rgb_path = os.path.join(self.data_path, video, "lowres_wide")
            depth_path = os.path.join(self.data_path, video, "lowres_depth")
            conf_path = os.path.join(self.data_path, video, "confidence")
            intrinsic_path = os.path.join(
                self.data_path, video, "lowres_wide_intrinsics"
            )

            pose_file = os.path.join(self.data_path, video, "lowres_wide.traj")
            pose_list = np.loadtxt(pose_file, delimiter=" ")
            ts_pose_list = [self.parse_pose(i) for i in pose_list]
            ts_pose_dict = {x[0]: x[1] for x in ts_pose_list}

            # 42899756_247278.844.png => 247278.844
            filename_to_ts = lambda x: float(
                ".".join(x.split("_")[1].split(".")[:2])
            )
            ts_to_filename = lambda x: f"{video}_{x:.3f}"

            rgb_ts = sorted(
                [
                    filename_to_ts(x)
                    for x in sorted(os.listdir(rgb_path))
                    if x.endswith(".png")
                ]
            )
            depth_ts = sorted(
                [
                    filename_to_ts(x)
                    for x in sorted(os.listdir(depth_path))
                    if x.endswith(".png")
                ]
            )
            pose_ts = sorted(list(ts_pose_dict.keys()))

            # do timestamp alignment
            rgb_ts_aligned, depth_ts_aligned = align_timestamp(
                [pose_ts, rgb_ts, depth_ts], index=0
            )

            if self.frame_downsample:
                random.shuffle(pose_ts)
                topk = int(len(pose_ts) * self.frame_downsample)
                pose_ts = pose_ts[:topk]

            for idx, ts in enumerate(pose_ts):
                current_pose = ts_pose_dict[ts]
                rgb_idx = rgb_ts_aligned[idx]
                depth_idx = depth_ts_aligned[idx]

                rgb_file = os.path.join(
                    rgb_path, f"{ts_to_filename(rgb_idx)}.png"
                )
                depth_file = os.path.join(
                    depth_path, f"{ts_to_filename(depth_idx)}.png"
                )
                intrinsic_file = os.path.join(
                    intrinsic_path, f"{ts_to_filename(rgb_idx)}.pincam"
                )
                conf_file = os.path.join(
                    conf_path, f"{ts_to_filename(depth_idx)}.png"
                )

                if not os.path.exists(depth_file):
                    __import__("ipdb").set_trace()
                    print(f"{depth_file} not found")

                if not os.path.exists(intrinsic_path):
                    __import__("ipdb").set_trace()
                    print(f"{intrinsic} not found")

                intrinsic = self.parse_intrinsic(intrinsic_file)
                frame = Frame(
                    rgb_file=rgb_file,
                    depth_file=depth_file,
                    pose=current_pose,
                    intrinsic=intrinsic,
                )
                frame.extras = {"conf_file": conf_file}

                self._frames.append(frame)
                rooms[room_name].append(frame)

        for name, frames in rooms.items():
            room = Room(name, frames)
            self._rooms.append(room)

    def parse_pose(self, items: np.ndarray) -> Tuple[float, Pose]:
        assert SCIPY_INSTALLED, "parse_pose requiring scipy."
        assert len(items) == 7
        timestamp = items[0]
        pos_xyz = items[4:7]
        rot_angle = items[1:4]
        rot = R.from_euler("yzx", rot_angle, degrees=False)

        mat = np.ones((4, 4))
        mat[:3, 3] = pos_xyz
        mat[:3, :3] = rot.as_matrix()
        return timestamp, Pose(mat)

    def parse_intrinsic(self, intrinsic_file) -> CameraIntrinsic:
        items = np.loadtxt(intrinsic_file, delimiter=" ")
        assert len(items) == 6
        w, h, fx, fy, cx, cy = items
        data = np.eye(3, 3)
        data[0, 0] = fx
        data[1, 1] = fy
        data[0, 2] = cx
        data[1, 2] = cy
        intrinsic = CameraIntrinsic(data, width=w, height=h)
        return intrinsic


@SOURCE_REGISTRY.register(name="ddddata_arkitscenes_3dod")
class DDDDataARKitScenes_3DOD(DDDDataARKitScenes_Raw):

    """ARKitScenes 3dod data reader.

    https://github.com/apple/ARKitScenes/blob/main/DATA.md
    https://github.com/apple/ARKitScenes/tree/main/threedod

    Training / Validation
    |---42899756    # <room_id>
    |          |---42899756_3dod_annotation.json
    |          |---42899756_3dod_mesh.ply
    |          |---42899756_frames
    |                           |---lowres_wide      # rgb 256x192
    |                           |     |---42899756_247278.827.png
    |                           |     |---...
    |                           |---lowres_depth     # depth 256x192
    |                           |     |---42899756_247278.844.png
    |                           |     |---...
    |                           |---lowres_wide_intrinsics       # same number as lowres_wide
    |                           |     |---42899756_247278.827.pincam
    |                           |     |---...
    |                           |---confidence       # conf 256x192, same number as lowres_depth
    |                           |     |---42899756_247278.844.png
    |                           |     |---...
    |                           |---lowres_wide.traj
    |
    |---...

    """

    def __init__(
        self,
        data_path,
        logger=None,
        room_downsample=None,
        quiet=False,
        frame_downsample=None,
    ):
        super(DDDDataARKitScenes_3DOD, self).__init__(
            data_path, logger, room_downsample, quiet, frame_downsample
        )

    def load_info_from_data_path(self):
        videos = os.listdir(self.data_path)
        rooms = defaultdict(list)

        if self.room_downsample:
            keep_num = max(int(len(videos) * self.room_downsample), 1)
            old_length = len(videos)
            videos = videos[:keep_num]
            print(f"{old_length} => {len(videos)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(videos),
            )
        for video in videos:
            if self.logging and (not self.quiet):
                t.update()
            room_name = video

            rgb_path = os.path.join(
                self.data_path, video, f"{video}_frames", "lowres_wide"
            )
            depth_path = os.path.join(
                self.data_path, video, f"{video}_frames", "lowres_depth"
            )
            intrinsic_path = os.path.join(
                self.data_path,
                video,
                f"{video}_frames",
                "lowres_wide_intrinsics",
            )
            anno_3dod_path = os.path.join(
                self.data_path, video, f"{video}_3dod_annotation.json"
            )

            pose_file = os.path.join(
                self.data_path, video, f"{video}_frames", "lowres_wide.traj"
            )
            pose_list = np.loadtxt(pose_file, delimiter=" ")
            ts_pose_list = [self.parse_pose(i) for i in pose_list]
            ts_pose_dict = {x[0]: x[1] for x in ts_pose_list}

            # 42899756_247278.844.png => 247278.844
            get_ts_from_filename = lambda x: float(
                ".".join(x.split("_")[1].split(".")[:2])
            )
            rgb_ts = sorted(
                [
                    get_ts_from_filename(x)
                    for x in sorted(os.listdir(rgb_path))
                    if x.endswith(".png")
                ]
            )
            depth_ts = sorted(
                [
                    get_ts_from_filename(x)
                    for x in sorted(os.listdir(depth_path))
                    if x.endswith(".png")
                ]
            )
            pose_ts = sorted(list(ts_pose_dict.keys()))

            # do timestamp alignment
            rgb_ts_aligned, depth_ts_aligned = align_timestamp(
                [pose_ts, rgb_ts, depth_ts], index=0
            )

            if self.frame_downsample:
                random.shuffle(pose_ts)
                topk = int(len(pose_ts) * self.frame_downsample)
                pose_ts = pose_ts[:topk]

            for idx, ts in enumerate(pose_ts):
                current_pose = ts_pose_dict[ts]
                rgb_idx = rgb_ts_aligned[idx]
                depth_idx = depth_ts_aligned[idx]

                rgb_file = os.path.join(rgb_path, f"{video}_{rgb_idx}.png")
                depth_file = os.path.join(
                    depth_path, f"{video}_{depth_idx}.png"
                )
                intrinsic_file = os.path.join(
                    intrinsic_path, f"{video}_{rgb_idx}.pincam"
                )
                conf_file = os.path.join(conf_path, f"{video}_{depth_idx}.png")

                if not os.path.exists(depth_file):
                    __import__("ipdb").set_trace()
                    print(f"{depth_file} not found")

                if not os.path.exists(intrinsic_path):
                    __import__("ipdb").set_trace()
                    print(f"{intrinsic} not found")

                frame = Frame(
                    rgb_file=rgb_file, depth_file=depth_file, pose=current_pose
                )
                frame.extras = {"conf_file": conf_file}
                frame.intrinsic = self.parse_intrinsic(intrinsic_file)

                self._frames.append(frame)
                rooms[room_name].append(frame)

            parse_3dod_annotations(anno_3dod_path)

        for name, frames in rooms.items():
            room = Room(name, frames)
            self._rooms.append(room)

    def parse_3dod_annotations(self, anno_path):
        """Parse 3dod annotation json.

        Args:
            anno_path: 3dod annotation path.

        # TODO: In develop

        """
        with open(jsonfile, "r") as fid:
            data = json.load(fid)
        objs = data["data"]
        assert len(objs) > 0


"""
SUN-RGBD
"""


@SOURCE_REGISTRY.register(name="ddddata_sunrgbd")
class DDDDataSUNRGBD(DDDReader):

    """SUNRGBD data reader.

    SUNRGBD
    |---xx/xx/xx
    |          |---annotation
    |          |---annotation2D3D
    |          |---annotation2Dfinal
    |          |---annotation3D
    |          |---annotation3Dfinal
    |          |---annotation3Dlayout
    |          |---depth
    |          |---depth_bfx
    |          |---extrinsics
    |          |---fullres
    |          |---image
    |          |---intrinsic.txt
    |          |---label
    |          |---scene.txt
    |          |---seg.mat

    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=1.0,
        quiet=False,
        frame_downsample=None,
    ):
        self.quiet = quiet
        assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.frame_downsample = frame_downsample
        super(DDDDataSUNRGBD, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        if not self.quiet:
            print(f"scanning {self.data_path}")

        # index_files = []
        with FileCache(
            list, f".cache_{self.data_path.replace('/', '__')}.pkl"
        ) as index_files:
            if len(index_files) == 0:
                for path, dirs, files in os.walk(
                    self.data_path, followlinks=True
                ):
                    index_files += [
                        os.path.join(path, x)
                        for x in files
                        if x == "intrinsics.txt" and "/fullres/" not in path
                    ]

        if not self.quiet:
            print(f"find frames length: {len(index_files)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(index_files),
            )

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()
            if "/fullres/" in index_file:
                continue

            item_root = os.path.dirname(index_file)
            rgb_file = glob.glob(os.path.join(item_root, "image/*.jpg"))
            if len(rgb_file) != 1:
                __import__("ipdb").set_trace()
                pass
            assert len(rgb_file) == 1
            rgb_file = rgb_file[0]

            depth_file = glob.glob(os.path.join(item_root, "depth/*.png"))
            if len(depth_file) != 1:
                __import__("ipdb").set_trace()
                pass
            assert len(depth_file) == 1
            depth_file = depth_file[0]

            extrinsic_file = glob.glob(
                os.path.join(item_root, "extrinsics/*.txt")
            )
            if len(extrinsic_file) > 0:
                extrinsic_file = extrinsic_file[0]
            else:
                extrinsic_file = None

            intrinsic_file = os.path.join(item_root, "intrinsics.txt")
            if not os.path.exists(intrinsic_file):
                intrinsic_file = None

            frame = Frame(
                rgb_file=rgb_file,
                depth_file=depth_file,
                intrinsic_file=intrinsic_file,
                pose_file=extrinsic_file,
            )

            seg_mat_file = os.path.join(item_root, "seg.mat")
            if os.path.exists(seg_mat_file):
                frame.semseg_file = seg_mat_file
                frame.load_semseg_file = DDDDataSUNRGBD.load_semseg_file
            self._frames.append(frame)

    @staticmethod
    def load_semseg_file(seg_file):
        seg_mat = sio.loadmat(seg_file)
        seg_label = seg_mat["seglabel"]
        return seg_label


"""
2D-3D-Semantics
"""


@SOURCE_REGISTRY.register(name="ddddata_2d3dsemantics")
class DDDData2D3DSemantics(DDDReader):

    """2D3D-S data reader.

    2d3dsemantics
    |---area_1/data
    |            |---depth
    |                   |---camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_depth.png
    |                   |---...
    |            |---normal
    |                   |---camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_depth.png
    |                   |---...
    |            |---pose
    |                   |---camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_pose.json
    |                   |---...
    |            |---rgb
    |                   |---camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_rgb.png
    |                   |---...
    |            |---semantic
    |                   |---camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_semantic.png
    |                   |---...
    |            |---semantic_pretty
    |                   |---camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain_semantic_pretty.png
    |                   |---...
    |---area_2/data
    |---...

    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=1.0,
        quiet=False,
        frame_downsample=None,
    ):
        self.quiet = quiet
        assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.frame_downsample = frame_downsample
        super(DDDData2D3DSemantics, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        if not self.quiet:
            print(f"scanning {self.data_path}")

        with FileCache(
            list, f".cache_{self.data_path.replace('/', '__')}.pkl"
        ) as index_files:
            if len(index_files) == 0:
                for path, dirs, files in os.walk(
                    self.data_path, followlinks=True
                ):
                    index_files += [
                        os.path.join(path, x)
                        for x in files
                        if x.endswith("_rgb.png") and "/data/" in path
                    ]

        if not self.quiet:
            print(f"find frames length: {len(index_files)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(index_files),
            )

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()

            rgb_file = index_file
            depth_file = rgb_file.replace("/rgb/", "/depth/").replace(
                "_rgb.png", "_depth.png"
            )
            semseg_file = rgb_file.replace(
                "/rgb/", "/semantic_pretty/"
            ).replace("_rgb.png", "_semantic_pretty.png")
            pose_file = rgb_file.replace("/rgb/", "/pose/").replace(
                "_rgb.png", "_pose.json"
            )

            frame = Frame(
                rgb_file=rgb_file,
                depth_file=depth_file,
                semseg_file=semseg_file,
                pose_file=pose_file,
            )
            self._frames.append(frame)


"""
ADE_20K
"""


@SOURCE_REGISTRY.register(name="ADE20K")
class ADE20K(DDDReader):

    """ADE20K data reader.

    ADE20K/
    |---index_ade20k.pkl
    |---images/ADE/validation/...
    |---images/ADE/training/*/*/
    |                   |---ADE_train_xxxxx.jpg
    |                   |---ADE_train_xxxxx_seg.png
    |                   |---ADE_train_xxxxx.json
    |                   |---ADE_train_xxxxx
    |                               |---instance_000_ADE_train_xxxxx.png
    |                               |---instance_001_ADE_train_xxxxx.png
    |                               |---......

    """

    def __init__(
        self,
        data_path,
        logger=None,
        ratio=1.0,
        quiet=False,
        frame_downsample=None,
    ):
        self.quiet = quiet
        assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.frame_downsample = frame_downsample
        super(ADE20K, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        if not self.quiet:
            print(f"scanning {self.data_path}")

        with FileCache(
            list, f".cache_{self.data_path.replace('/', '__')}.pkl"
        ) as index_files:
            if len(index_files) == 0:
                for path, dirs, files in os.walk(
                    self.data_path, followlinks=True
                ):
                    index_files += [
                        os.path.join(path, x)
                        for x in files
                        if x.endswith(".jpg")
                    ]

        if not self.quiet:
            print(f"find frames length: {len(index_files)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(
                total=len(index_files),
            )

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()

            rgb_file = index_file
            semseg_file = rgb_file.replace(".jpg", "_seg.png")
            assert os.path.exists(rgb_file)
            assert os.path.exists(semseg_file)
            frame = Frame(rgb_file=rgb_file, semseg_file=semseg_file)
            self._frames.append(frame)


"""
S3DIS
"""


@SOURCE_REGISTRY.register(name="S3DIS")
class S3DIS(DDDReader):

    """S3DIS data reader.

    https://cvg-data.inf.ethz.ch/s3dis
    """


"""
ADEChallengeData2016
"""


@SOURCE_REGISTRY.register(name="ADEChallengeData2016")
class ADEChallengeData2016(DDDReader):

    """ADE20K data reader.

    ADE20K/
    |---objectInfo150.txt
    |---sceneCategories.txt
    |---images/training/*.jpg
    |---images/validation/*.jpg
    |---annotations/training/*.jpg
    |---annotations/validation/*.jpg

    """

    def __init__(
        self,
        data_path,
        phase="training",
        logger=None,
        ratio=1.0,
        quiet=False,
        frame_downsample=None,
    ):
        self.quiet = quiet
        assert 0.0 <= ratio <= 1.0
        self.ratio = ratio
        self.frame_downsample = frame_downsample
        assert phase in ("training", "validation")
        self.phase = phase
        super(ADEChallengeData2016, self).__init__(data_path, logger)

    def load_info_from_data_path(self):
        if not self.quiet:
            print(f"scanning {self.data_path}")

        root = os.path.join(self.data_path, "images", self.phase)
        index_files = [
            os.path.join(root, x)
            for x in sorted(os.listdir(root))
            if x.endswith(".jpg")
        ]

        if not self.quiet:
            print(f"find frames length: {len(index_files)}")

        if self.logging and (not self.quiet):
            t = tqdm.tqdm(total=len(index_files), desc="collecting data info")

        for index_file in index_files:
            if self.logging and (not self.quiet):
                t.update()

            rgb_file = index_file
            seg_file = rgb_file.replace(".jpg", ".png").replace(
                "images", "annotations"
            )
            assert os.path.exists(rgb_file)
            assert os.path.exists(seg_file)
            frame = Frame(rgb_file=rgb_file, semseg_file=seg_file)
            self._frames.append(frame)
