from abc import ABC, abstractmethod

import numpy as np

from .contrib.OneEuroFilter import OneEuroFilter


class Smooth(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def boxes(self, boxes_list):
        """
        Smooth boxes.

        Parameter
        ---------
        boxes_list : boxes list
            XYXY
        """
        pass

    @abstractmethod
    def keypoints(self, kps_list):
        """
        Smooth keypoints.

        Parameter
        ---------
        kps_list : keypoints list
            XYV

        """
        pass


class SimpleSmooth(Smooth):
    def __init__(self, period=5, method="mean"):
        super(SimpleSmooth, self).__init__()
        self.period = period
        self.method = method

    def boxes(self, boxes_list):
        """
        Smooth boxes.

        Parameter
        ---------
        boxes_list : boxes list
            XYXY
        """
        update_boxes = list()
        if self.method == "mean":
            for ind, bbox in enumerate(boxes_list):
                if ind + 1 < self.period:
                    boxes = boxes_list[: ind + 1].mean(axis=0)
                else:
                    boxes = boxes_list[ind + 1 - self.period : ind + 1].mean(
                        axis=0
                    )
                update_boxes.append(boxes)
        else:
            raise NotImplementedError
        return np.asarray(update_boxes)

    def keypoints(self, kps_list):
        """
        Smooth keypoints.

        Parameter
        ---------
        kps_list : keypoints list
            XYV

        """
        update_keypoints = list()
        if self.method == "mean":
            for ind, _ in enumerate(kps_list):
                if ind + 1 < self.period:
                    kps = kps_list[: ind + 1].mean(axis=0)
                else:
                    kps = kps_list[ind + 1 - self.period : ind + 1].mean(
                        axis=0
                    )
                update_keypoints.append(kps)
        else:
            raise NotImplementedError
        return np.asarray(update_keypoints)


class OneEuroSmooth(Smooth):
    """
    OneEuroSmooth, only support one instance.

    """

    def __init__(self, **config):
        super(OneEuroSmooth, self).__init__()
        self.config = {
            "freq": config.get("freq", 120),
            "mincutoff": config.get("mincutoff", 1.0),
            "beta": config.get("beta", 1.0),
            "dcutoff": config.get("dcutoff", 1.0),
        }
        self.boxes_filter = self.__create_filter(4)
        num_kps = 17
        self.kps_filter = self.__create_filter(num_kps * 2)

    def __create_filter(self, number):
        return [OneEuroFilter(**self.config) for _ in range(number)]

    def boxes(self, boxes_list):
        """
        Smooth boxes.

        Parameter
        ---------
        boxes_list : boxes list
            XYXY
        """
        update_boxes = list()
        for ind, bbox in enumerate(boxes_list):
            if bbox.sum() != -1 * 4:
                bbox[0] = self.boxes_filter[0](bbox[0])
                bbox[1] = self.boxes_filter[1](bbox[1])
                bbox[2] = self.boxes_filter[2](bbox[2])
                bbox[3] = self.boxes_filter[3](bbox[3])
            update_boxes.append(bbox)
        return np.asarray(update_boxes)

    def keypoints(self, kps_list):
        """
        Smooth keypoints.

        Parameter
        ---------
        kps_list : keypoints list
            XYV

        """
        update_keypoints = list()
        for ind, kps in enumerate(kps_list):
            if kps.sum() != -1 * kps.shape[0]:
                kps = kps.reshape(-1, 3)
                for kps_ind, pt in enumerate(kps):
                    x = self.kps_filter[kps_ind * 2 + 0](pt[0])
                    y = self.kps_filter[kps_ind * 2 + 1](pt[1])
                    kps[kps_ind][0] = x
                    kps[kps_ind][1] = y
                kps = kps.reshape(-1)
            update_keypoints.append(kps)
        return np.asarray(update_keypoints)
