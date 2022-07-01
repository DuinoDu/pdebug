"""source reader """
import logging
import os
import warnings
import zipfile
from collections.abc import Iterator

import cv2
import numpy as np

from ..registry import SOURCE_REGISTRY

# from .utils import cache_local


__all__ = ['ImgDir', 'ImgZip', 'Video', 'zip_imgread']


class Reader(Iterator):

    """Base Reader Class"""

    def __init__(self):
        self._cur = 0
        self._cur_filename = None
        self.logging = True
        self.exts = ['.jpg', '.jpeg', '.png', '.bmp']
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


@SOURCE_REGISTRY.register(name='imgdir')
class ImgDir(Reader):

    """imgdir reader"""

    def __init__(self, imgdir, recursive=False, abspath=True, logger=None):
        """ImgDir init

        Parameters
        ----------
        imgdir : input image directory
        recursive : bool
            recursively search image file

        """
        super(ImgDir, self).__init__()
        if abspath:
            self._imgdir = os.path.abspath(imgdir)
        else:
            self._imgdir = imgdir
        if imgdir.startswith('hdfs:'):
            print('Not support hdfs for imgdir')
            import sys
            sys.exit()
        if not recursive:
            self._imgfiles = sorted([os.path.join(self._imgdir, x)
                                    for x in sorted(os.listdir(self._imgdir))
                                    if os.path.splitext(x)[1].lower() in self.exts])
        else:
            self._imgfiles = []
            for path, dirs, files in os.walk(self._imgdir, followlinks=True):
                self._imgfiles += [os.path.join(path, x) for x in files
                                   if os.path.splitext(x)[1].lower() in self.exts]
        if logger:
            self.logger = logger
        else:
            self.logger = logging
        self.logger.info('len imgdir: %d' % len(self._imgfiles))

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
        if not imgname.startswith('/'):
            imgname = os.path.join(self._imgdir, imgname)
        if imgname not in self._imgfiles:
            self.logger.info("%s not in %s" % (imgname, self._imgdir))
            return None
        if self.logging:
            self.logger.info(imgname)
        return cv2.imread(imgname)

    def __len__(self):
        return len(self._imgfiles)


@SOURCE_REGISTRY.register(name='imgzip')
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
        # imgzip = cache_local(imgzip, logger=logger)
        self._imgzip = os.path.abspath(imgzip)
        self._zfile = zipfile.ZipFile(self._imgzip, 'r')
        self._imgfiles = [x for x in self._zfile.namelist()
                          if os.path.splitext(x)[1].lower() in self.exts]
        self.zip_prefix = ''.join(self._imgfiles[0].split('/')[:-1])
        if logger:
            self.logger = logger
        else:
            self.logger = logging
        self.logger.info('len imgzip: %d' % len(self._imgfiles))

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
    assert '@' in imgname
    imgzip = imgname.split('@')[0]
    imgname = imgname.split('@')[1]

    if imgzip not in _zipfiles:
        # cache to global
        zfile = zipfile.ZipFile(imgzip, 'r')
        _zipfiles[imgzip] = zfile
    zfile = _zipfiles[imgzip]
    data = zfile.read(imgname)
    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


@SOURCE_REGISTRY.register(name='video')
class Video(Reader):

    """video reader"""

    def __init__(self, videofile):
        """video init

        Parameters
        ----------
        videofile : input video file

        """
        super(Video, self).__init__()
        self.local_videofile = ""
        self._cap = None
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

    def __next__(self):
        try:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
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
            logging.info('[WARNING] bad frame')
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
        return self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.

    @timestamp.setter
    def set_timestamp(self, timestamp):
        """
        unit: second
        """
        self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.)

    @property
    def width(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if os.path.exists(self.local_videofile):
            os.system('rm %s' % self.local_videofile)
