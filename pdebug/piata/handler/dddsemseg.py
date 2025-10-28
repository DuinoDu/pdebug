"""source reader """
import logging
import os
import warnings
import zipfile
from collections.abc import Iterator
from enum import Enum

import cv2
import numpy as np

from ...data_types import PointcloudTensor
from ..registry import SOURCE_REGISTRY
from .source import Reader

# TODO(min.du): In develop ...


__all__ = ["DDDSemSeg"]


class DDDSemSegSourceType(Enum):
    TXT = "txt"
    NPY = "npy"
    BIN = "bin"


@SOURCE_REGISTRY.register(name="dddsemseg")
class DDDSemSeg(Reader):

    """3d semseg data reader"""

    def __init__(self, data_path, source_type="txt", logger=None):
        """Init

        Args:
            data_path: input data path

        """
        super(DDDSemSeg, self).__init__()

        source_type = DDDSemSegSourceType(source_type)
        assert source_type == DDDSemSegSourceType.TXT

        if logger:
            self.logger = logger
        else:
            self.logger = logging
        self.logger.info("len imgdir: %d" % len(self._imgfiles))

    def __next__(self):
        try:
            imgfile = self._imgfiles[self._cur]
            self._cur_filename = imgfile
            self._cur += 1
            return cv2.imread(imgfile)
        except:
            raise StopIteration

    @property
    def filename(self):
        return self._cur_filename

    def imread(self, imgname):
        if not imgname.startswith("/"):
            imgname = os.path.join(self._imgdir, imgname)
        if imgname not in self._imgfiles:
            self.logger.info("%s not in %s" % (imgname, self._imgdir))
            return None
        if self.logging:
            self.logger.info(imgname)
        return cv2.imread(imgname)

    def __len__(self):
        return len(self._imgfiles)
