"""
geomentry element
"""
import cv2
import numpy as np

__all__ = ["Line"]


def _pos_neg_zero(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


class Line:
    """
    Draw line in image.

    p0: [y, x]
    p1: [y, x]
    unit_vec: [x, y]
    """

    def __init__(
        self,
        p0=None,
        p1=None,
        unit_vec=None,
        angle=None,
        scale=None,
        reverse=False,
        israd=False,
    ):
        self.p0 = p0
        self.p1 = p1
        if reverse:
            if p0 is not None:
                self.p0 = p0[::-1]
            if p1 is not None:
                self.p1 = p1[::-1]
        self.unit_vec = unit_vec
        self._angle = angle / np.pi * 180.0 if israd else angle
        if self._angle is not None and unit_vec is None:
            vec_x = 1.0 * np.cos(self._angle / 180.0 * np.pi)
            vec_y = 1.0 * np.sin(self._angle / 180.0 * np.pi)
            self.unit_vec = (vec_x, vec_y)

        if scale is not None:
            if self.p0 is not None:
                self.p0 = self.p0 * scale
            if self.p1 is not None:
                self.p1 = self.p1 * scale
        self.eps = np.finfo(self.p0.dtype).eps

    def __call__(self, x=None, y=None, x1=None, y1=None, with_direction=False):
        """
        return [y, x]
        """
        assert self.p0 is not None
        y0, x0 = int(self.p0[0]), int(self.p0[1])
        if self.unit_vec is not None:
            tan_angle = self.unit_vec[1] / self.unit_vec[0]
        elif self.p1 is not None:
            tan_angle = (self.p1[0] - self.p0[0]) / (
                self.p1[1] - self.p0[1] + self.eps
            )
        else:
            raise ValueError("Please provide p0/p1 or p0/unit_vec in Line")

        if x1 is not None or y1 is not None:
            print("x1,y1 will be deprecated, please use x,y")
            x, y = x1, y1

        if with_direction:
            assert 0 not in self.quadrant, "line is vertical or horizontal"
            if x is not None:
                if _pos_neg_zero((x - x0)) != self.quadrant[0]:
                    x = 2 * x0 - x
            if y is not None:
                if _pos_neg_zero((y - y0)) != self.quadrant[1]:
                    y = 2 * y0 - y

        if x is not None:
            y = tan_angle * (x - x0) + y0
        elif y is not None:
            x = (y - y0) / (tan_angle + self.eps) + x0
        else:
            raise ValueError
        return np.array([y, x], dtype=np.float32)

    @property
    def angle(self):
        """
        return angle
        """
        if self._angle is not None:
            return self._angle
        if self.unit_vec is not None:
            vec = self.unit_vec[::-1]  # [y, x]
        else:
            vec = self.p1 - self.p0

        vec_mod = np.sqrt(np.power(vec[0], 2) + np.power(vec[1], 2)) + self.eps
        _angle = np.arccos(vec[1] / vec_mod)
        if np.arcsin(vec[0] / vec_mod) < 0:
            _angle *= -1
        return _angle / np.pi * 180.0

    @property
    def quadrant(self):
        flag_x, flag_y = 0, 0
        if self.unit_vec is not None:
            vec_x, vec_y = self.unit_vec
            flag_x = _pos_neg_zero(vec_x)
            flag_y = _pos_neg_zero(vec_y)
        elif self.p1 is not None:
            flag_y = _pos_neg_zero(self.p1[0] - self.p0[0])
            flag_x = _pos_neg_zero(self.p1[1] - self.p0[1])
        else:
            raise ValueError("Please provide p0/p1 or p0/unit_vec in Line")
        return (flag_x, flag_y)

    @staticmethod
    def distance(p0, p1):
        return np.sqrt(np.power(p0[0] - p1[0], 2) + np.power(p0[1] - p1[1], 2))

    @staticmethod
    def inside(p, b, xywh=False, reverse=False):
        """
        p: point, [y, x]
        b: bbox: [x1, y1, x2, y2]
        """
        if xywh:
            b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])
        if reverse:
            p = p[::-1]
        py, px = p
        return b[1] <= py <= b[3] and b[0] <= px <= b[2]

    def __repr__(self):
        return "line function"

    def draw(
        self,
        _img,
        color=(0, 255, 0),
        thickness=2,
        bbox=None,
        with_direction=False,
    ):
        if self.unit_vec is not None:
            # compute border
            if bbox is not None:
                left = bbox[0]
                top = bbox[1]
                right = bbox[2]
                bottom = bbox[3]
            else:
                left = 1
                top = 1
                right = _img.shape[1] - 1
                bottom = _img.shape[0] - 1
            # get border point
            points = []
            if self.unit_vec[0] >= 0:
                points.append(self.__call__(x=right))
            else:
                points.append(self.__call__(x=left))
            if self.unit_vec[1] >= 0:
                points.append(self.__call__(y=bottom))
            else:
                points.append(self.__call__(y=top))
            dists = [Line.distance(p, self.p0) for p in points]
            dists_sort = sorted(dists)

            p0 = self.p0
            p1 = points[dists.index(dists_sort[0])]
        else:
            p0 = self.p0
            p1 = self.p1

        y0, x0 = int(p0[0]), int(p0[1])
        y1, x1 = int(p1[0]), int(p1[1])

        if with_direction:
            line_func = cv2.arrowedLine
        else:
            line_func = cv2.line
        try:
            line_func(_img, (x0, y0), (x1, y1), color, thickness)
            img = _img
        except TypeError as e:
            img = line_func(_img.copy(), (x0, y0), (x1, y1), color, thickness)
        return img
