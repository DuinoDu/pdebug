"""source reader """
import glob
import logging
import os
import random
import warnings
import zipfile
from collections.abc import Iterator
from socket import IP_DROP_MEMBERSHIP
from typing import Callable

from pdebug.utils.env import SPATIALMP4_INSTALLED, TORCH_INSTALLED

import cv2
import numpy as np

from ..registry import SOURCE_REGISTRY
from .simpletxt import load_simpletxt

if SPATIALMP4_INSTALLED:
    import spatialmp4 as sm

# from .utils import cache_local


__all__ = [
    "ImgDir",
    "ImgZip",
    "Video",
    "zip_imgread",
    "RgbdSemsegReader",
    "SimpletxtReader",
    "ParquetReader",
]


class Reader(Iterator):

    """Base Reader Class"""

    def __init__(self):
        self._cur = 0
        self._cur_filename = None
        self.logging = True
        self.exts = [".jpg", ".jpeg", ".png", ".bmp"]
        self._imgfiles = []

    def __iter__(self):
        return self

    def imread(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    next = __next__

    def __len__(self):
        raise NotImplementedError

    def get_imglist(self):
        warnings.warn("deprecated, use .imglist insetad", DeprecationWarning)
        return self.imglist

    @property
    def imglist(self):
        return self._imgfiles

    @property
    def imgfiles(self):
        return self._imgfiles

    @property
    def idx(self):
        return self._cur

    @property
    def filename(self):
        return self._cur_filename

    def reset(self):
        self._cur = 0
        self._cur_filename = None


@SOURCE_REGISTRY.register(name="imgdir")
class ImgDir(Reader):

    """imgdir reader"""

    def __init__(
        self,
        imgdir,
        recursive=False,
        abspath=True,
        logger=None,
        shuffle=False,
        topk=-1,
        downsample=-1,
        to_rgb=False,
        timestamp_sample_duration=None,
        get_timestamp_fn=None,
        imread_raw=False,
        # deprecated
        rgb=False,
    ):
        """ImgDir init

        Args:
        imgdir : input image directory
        recursive : recursively search image file
        abspath: convert imgdir to abspath
        logger: set logger
        shuffle: shuffle imglist
        topk: choose topk image files
        to_rgb: read image in rgb format
        timestamp_sample_duration: sample image files by timestamp, using `get_timestamp_fn`
        get_timestamp_fn: get timestamp from image file name, used only when timestamp_sample_duration is True.

        """
        super(ImgDir, self).__init__()
        imgdir = str(imgdir)
        if abspath:
            self._imgdir = os.path.abspath(imgdir)
        else:
            self._imgdir = imgdir
        if imgdir.startswith("hdfs:"):
            raise ValueError("Not support hdfs for imgdir")

        if "*" in imgdir:
            self._imgdir = os.path.dirname(imgdir)
            self._imgfiles = sorted(glob.glob(imgdir))
        else:
            if recursive:
                self._imgfiles = []
                for path, dirs, files in os.walk(
                    self._imgdir, followlinks=True
                ):
                    self._imgfiles += [
                        os.path.join(path, x)
                        for x in files
                        if os.path.splitext(x)[1].lower() in self.exts
                    ]
            else:
                self._imgfiles = sorted(
                    [
                        os.path.join(self._imgdir, x)
                        for x in sorted(os.listdir(self._imgdir))
                        if os.path.splitext(x)[1].lower() in self.exts
                    ]
                )
        if rgb:
            print("[DEPRECATED] rgb is deprecated, use to_rgb instead.")
            to_rgb = rgb
        self.to_rgb = to_rgb
        self.timestamp_sample_duration = timestamp_sample_duration

        if timestamp_sample_duration:
            assert get_timestamp_fn
            new_imgfiles = []
            for idx, imgfile in enumerate(self._imgfiles):
                timestamp = get_timestamp_fn(os.path.basename(imgfile))
                if idx == 0:
                    new_imgfiles.append((timestamp, imgfile))
                else:
                    last_ts = new_imgfiles[-1][0]
                    if timestamp >= last_ts + timestamp_sample_duration:
                        new_imgfiles.append((timestamp, imgfile))
            print(
                f"timestamp_sample: {len(self._imgfiles)} => {len(new_imgfiles)}"
            )
            self._imgfiles = [f[1] for f in new_imgfiles]

        if shuffle:
            random.shuffle(self._imgfiles)
        if topk and topk > 0:
            new_imgfiles = self._imgfiles[:topk]
            print(f"topk: {len(self._imgfiles)} => {len(new_imgfiles)}")
            self._imgfiles = new_imgfiles
        if downsample and downsample > 0:
            new_imgfiles = self._imgfiles[::downsample]
            print(f"downsample: {len(self._imgfiles)} => {len(new_imgfiles)}")
            self._imgfiles = new_imgfiles
        if logger:
            self.logger = logger
        else:
            self.logger = logging
        self.imread_raw = imread_raw
        self.logger.info("len imgdir: %d" % len(self._imgfiles))

    def __next__(self):
        try:
            imgfile = self._imgfiles[self._cur]
            self._cur_filename = imgfile
            self._cur += 1
            if self.imread_raw:
                img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(imgfile)
            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except:
            raise StopIteration

    @property
    def imgdir(self):
        return self._imgdir

    def imread(self, imgname):
        if not imgname.startswith("/"):
            imgname = os.path.join(self._imgdir, imgname)
        if imgname not in self._imgfiles:
            self.logger.info("%s not in %s" % (imgname, self._imgdir))
            return None
        if self.logging:
            self.logger.info(imgname)

        img = cv2.imread(imgname)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return len(self._imgfiles)

    def __add__(self, other):
        if not isinstance(other, ImgDir):
            raise NotImplementedError
        self._imgfiles.extend(other._imgfiles)
        return self


@SOURCE_REGISTRY.register(name="imgzip")
class ImgZip(Reader):

    """imgzip reader.

    Direct zip image files in to zip, without folder.

    """

    def __init__(self, imgzip, logger=None):
        """ImgDir init

        Parameters
        ----------
        imgzip : input image zip file

        """
        super(ImgZip, self).__init__()
        imgzip = str(imgzip)
        # imgzip = cache_local(imgzip, logger=logger)
        self._imgzip = os.path.abspath(imgzip)
        self._zfile = zipfile.ZipFile(self._imgzip, "r")
        self._imgfiles = [
            x
            for x in self._zfile.namelist()
            if os.path.splitext(x)[1].lower() in self.exts
        ]
        self.zip_prefix = "".join(self._imgfiles[0].split("/")[:-1])
        if logger:
            self.logger = logger
        else:
            self.logger = logging
        self.logger.info("len imgzip: %d" % len(self._imgfiles))

    def __next__(self):
        try:
            imgfile = self._imgfiles[self._cur]
            self._cur_filename = imgfile
            self._cur += 1
            flags = cv2.IMREAD_COLOR
            data = self._zfile.read(imgfile)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)
        except:
            raise StopIteration

    def imread(self, imgname, flags=cv2.IMREAD_COLOR):
        if self.zip_prefix not in imgname:
            imgfile = os.path.join(self.zip_prefix, imgname)
        else:
            imgfile = imgname

        if imgfile not in self._imgfiles:
            logging.info("%s not in %s" % (imgname, self._imgzip))
            return None
        if self.logging:
            self.logger.info(imgname)
        data = self._zfile.read(imgfile)
        return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    @property
    def filename(self):
        return self._cur_filename

    def __len__(self):
        return len(self._imgfiles)


# Used by zip_imread
_zipfiles = {}


def zip_imread(imgname, flags=cv2.IMREAD_COLOR):
    global _zipfiles
    assert "@" in imgname
    imgzip = imgname.split("@")[0]
    imgname = imgname.split("@")[1]

    if imgzip not in _zipfiles:
        # cache to global
        zfile = zipfile.ZipFile(imgzip, "r")
        _zipfiles[imgzip] = zfile
    zfile = _zipfiles[imgzip]
    data = zfile.read(imgname)
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


@SOURCE_REGISTRY.register(name="video")
class Video(Reader):

    """video reader"""

    def __init__(self, videofile, to_rgb=False, topk=-1):
        """video init

        Parameters
        ----------
        videofile : input video file

        """
        super(Video, self).__init__()
        videofile = str(videofile)
        self.local_videofile = ""
        self._cap = None
        videofile = str(videofile)
        if videofile.startswith("hdfs:"):
            self.local_videofile = os.path.basename(videofile)
            cmd = "hdfs dfs -get %s %s" % (videofile, self.local_videofile)
            logging.info(cmd)
            os.system(cmd)
            self._cap = cv2.VideoCapture(self.local_videofile)
        else:
            if not os.path.exists(videofile):
                raise ValueError("%s not exist" % videofile)
            self._cap = cv2.VideoCapture(videofile)
        self.to_rgb = to_rgb
        self.topk = topk

    def __next__(self):
        if self.topk > 0 and self._cur >= self.topk:
            raise StopIteration

        try:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            if self.to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._cur_filename = f"{self._cur:06d}.png"
            self._cur += 1
            return frame
        except:
            raise StopIteration

    def imread(self, imgname):
        print("Video not support imread")
        raise NotImplementedError

    @property
    def frame(self):
        ret, f = self._cap.read()
        if not ret:
            logging.info("[WARNING] bad frame")
            return None
        return f

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def cnt(self):
        return self._cap.get(cv2.CAP_PROP_POS_FRAMES)

    @property
    def timestamp(self):
        """
        unit: second
        """
        return self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    @timestamp.setter
    def set_timestamp(self, timestamp):
        """
        unit: second
        """
        self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)

    @property
    def width(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        if self.topk > 0:
            return min(self.topk, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        else:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if os.path.exists(self.local_videofile):
            os.system("rm %s" % self.local_videofile)


@SOURCE_REGISTRY.register(name="simpletxt_reader")
class SimpletxtReader(Reader):

    """Simpletxt imgfiles reader"""

    def __init__(
        self,
        path,
        field,
        topk=None,
        shuffle=False,
    ):
        """Reader init"""
        super(SimpletxtReader, self).__init__()
        roidb = load_simpletxt(path, field=field, topk=topk, shuffle=shuffle)
        self._imgfiles = [roi["image_name"] for roi in roidb]


@SOURCE_REGISTRY.register(name="parquet")
class ParquetReader(Reader):

    """Parquet reader, support hdfs."""

    def __init__(
        self,
        path: str,
        transform: Callable = None,
    ):
        """Reader init"""
        super(ParquetReader, self).__init__()
        from ..data_pack.reader import ParquetReader as _ParquetReader

        self._reader = _ParquetReader(path)
        self.transform = transform

    def __len__(self):
        return len(self._reader._table_dict[self._reader._keys[0]])

    def __next__(self):
        try:
            data_dict = {
                k: self._reader._table_dict[k][self._cur]
                for k in self._reader._keys
            }
            self._cur += 1
            if self.transform:
                data_dict = self.transform(data_dict)
            return data_dict
        except IndexError:
            raise StopIteration


@SOURCE_REGISTRY.register(name="parquet_semseg_v1")
class ParquetReaderSemsegV1(ParquetReader):

    """Parquet reader, for semseg parquet."""

    def __init__(
        self,
        path: str,
    ):
        """Reader init"""

        def transform(data_dict):
            image_key = "image"
            label_key = "label"
            image = cv2.imdecode(
                np.frombuffer(data_dict[image_key], dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            data_dict[image_key] = image
            if data_dict[label_key] != b"":
                label = cv2.imdecode(
                    np.frombuffer(data_dict[label_key], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                data_dict[label_key] = label
            return data_dict

        super(ParquetReaderSemsegV1, self).__init__(path, transform=transform)


@SOURCE_REGISTRY.register(name="spatialmp4")
class SpatialMp4(Reader):

    """video reader for spatialmp4, only load left rgb."""

    def __init__(self, videofile, to_rgb=False, topk=-1, log_level="quiet"):
        """video init

        Parameters
        ----------
        videofile : input video file

        """
        super(SpatialMp4, self).__init__()
        assert SPATIALMP4_INSTALLED, "spatialmp4 is required."

        videofile = str(videofile)
        self.to_rgb = to_rgb
        self.topk = topk

        self._reader = sm.Reader(str(videofile))
        self._reader.set_read_mode(sm.ReadMode.RGB_ONLY)

    def __next__(self):
        if self.topk > 0 and self._cur >= self.topk:
            raise StopIteration

        if not self._reader.has_next():
            raise StopIteration

        try:
            rgb_frame = self._reader.load_rgb()
            frame = rgb_frame.left_rgb

            if self.to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._cur_filename = f"{self._cur:06d}.png"
            self._cur += 1
            return frame
        except:
            raise StopIteration

    def imread(self, imgname):
        print("Video not support imread")
        raise NotImplementedError

    @property
    def frame(self):
        ret, f = self._cap.read()
        if not ret:
            logging.info("[WARNING] bad frame")
            return None
        return f

    @property
    def fps(self):
        return self._reader.get_rgb_fps()

    @property
    def cnt(self):
        return self._reader.get_index()

    @property
    def timestamp(self):
        """
        unit: second
        """
        return self._reader.get_duration()

    @property
    def width(self):
        return self._reader.get_rgb_width()

    @property
    def height(self):
        return self._reader.get_rgb_height()

    def __len__(self):
        if self.topk > 0:
            return min(self.topk, self._reader.get_frame_count())
        else:
            return self._reader.get_frame_count()

    @property
    def intrinsic(self) -> "CameraIntrinsic":
        from pdebug.data_types import CameraIntrinsic

        I = self._reader.get_rgb_intrinsics_left()
        camI = CameraIntrinsic([I.fx, I.fy, I.cx, I.cy])
        camI.width = self.width
        camI.height = self.height
        return camI
