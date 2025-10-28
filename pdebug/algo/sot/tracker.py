from typing import List, Tuple

from pdebug.piata.type_cast import boxes_to_points, points_to_boxes

import cv2
import numpy as np

__all__ = ["PointTracker", "BBoxTracker"]


class BaseTracker:
    """Point Tracker."""

    TRACKER_TYPERS = [
        "BOOSTING",
        "MIL",
        "KCF",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
        "CSRT",
    ]

    def __init__(self, tracker_type):
        self._tracker_type = tracker_type
        assert (
            tracker_type in self.TRACKER_TYPERS
        ), f"{tracker_type} not found, only support {self.TRACKER_TYPERS}"
        self._boxes = None
        self._inner_tracker = []

    @staticmethod
    def create_tracker(tracker_type: str) -> "cv2.Tracker":
        if tracker_type == "BOOSTING":
            tracker = cv2.legacy.TrackerBoosting_create()
        elif tracker_type == "MIL":
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == "TLD":
            tracker = cv2.legacy.TrackerTLD_create()
        elif tracker_type == "MEDIANFLOW":
            tracker = cv2.legacy.TrackerMedianFlow_create()
        elif tracker_type == "GOTURN":
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == "MOSSE":
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        else:
            raise NotImplementedError(
                f"Unsupported tracker type: {tracker_type}"
            )
        return tracker

    def clean_cache(self):
        """Clean inner tracker."""
        self._boxes = None
        self._inner_tracker = []

    @property
    def inner_tracker(self) -> List["cv2.Tracker"]:
        if not self._inner_tracker:
            # create tracker
            assert self._boxes is not None, "Please call `set_anchor` first."
            self._inner_tracker = []
            for bbox in self._boxes:
                self._inner_tracker.append(
                    self.create_tracker(self._tracker_type)
                )
        return self._inner_tracker

    @property
    def boxes(self):
        return self._boxes

    @property
    def tracker_type(self):
        return self._tracker_type

    def set_anchor(self, *args, **kwargs):
        """Set tracked object."""
        raise NotImplementedError

    def init(self, frame: np.ndarray):
        """Initialize inner tracker."""
        assert len(self.inner_tracker) > 0, "Please create tracker first."
        assert len(self.inner_tracker) == len(self.boxes)
        for tracker, bbox in zip(self.inner_tracker, self.boxes):
            tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> Tuple[List[bool], List[np.ndarray]]:
        rets, boxes = [], []
        for tracker in self.inner_tracker:
            ret, bbox = tracker.update(frame)
            rets.append(ret)
            boxes.append(bbox)
        return rets, boxes


class PointTracker(BaseTracker):
    def __init__(self, tracker_type: str, point_size: int):
        super(PointTracker, self).__init__(tracker_type)
        self._point_size = point_size

    def set_anchor(self, points: np.ndarray) -> None:
        """
        Set need-to-tracking points.

        Args:
            points: Points need to tracking. Shape is [N, 2], or [2,]
        """
        if points.ndim == 1:
            points = points[None, :]
        assert points.ndim == 2
        points = points[:, :2]
        boxes = points_to_boxes(points, self._point_size, layout="x1y1wh")
        self._boxes = boxes.astype(np.int)

    def update(self, frame: np.ndarray) -> Tuple[List[bool], List[np.ndarray]]:
        rets, boxes = super(PointTracker, self).update(frame)
        points = boxes_to_points(boxes, self._point_size)
        return rets, points


class BBoxTracker(BaseTracker):
    def set_anchor(self, boxes: np.ndarray) -> None:
        """
        Set need-to-tracking boundingbox.

        Args:
            boxes: Boxes need to tracking. Shape is [N, 4], or [4,]
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        assert boxes.ndim == 2
        boxes = boxes[:, :4]
        self._boxes = boxes.astype(np.int)
