# -*- coding: utf-8 -*-
"""tracklet
"""
from typing import Any, Dict, List, NewType

import numpy as np

__all__ = ["Tracklets", "roidb_to_tracklets", "tracklets_to_roidb"]


class Tracklets:
    """
    Tracklets.

    Examples:
        >>> tracklets = Tracklets()
        >>> for roi in roidb:
        >>>     for ind in range(roi['trackid'])
        >>>         bbox, trackid = roi['boxes'][ind], roi['trackid'][ind]
        >>>         tracklets.update(trackid, bbox=bbox)
    """

    def __init__(self, action_dict=None):
        # key: trackid
        # value: {boxes, keypoints, action_label, action_score, action}
        self.tracklets = dict()
        self.action_dict = action_dict

    def __len__(self):
        return len(self.tracklets.keys())

    def __getitem__(self, trackid):
        assert trackid in self.ids
        return self.tracklets[trackid]

    def __str__(self):
        _str = "trackid: "
        for k in self.tracklets:
            _str += " %d" % k
        _tracklets = list(self.tracklets.values())
        if "boxes" in _tracklets[0]:
            _str += "\nmin tracklet len: %d" % min(
                [len(x["boxes"]) for x in _tracklets]
            )
            _str += "\nmax tracklet len: %d" % max(
                [len(x["boxes"]) for x in _tracklets]
            )
        return _str

    __repr__ = __str__

    @property
    def keys(self):
        """
        get tracklets keys
        """
        print("[Deprecated] Please use ids instead of keys")
        return self.tracklets.keys()

    @property
    def ids(self):
        """
        get tracklets keys, also as ids
        """
        return self.tracklets.keys()

    def update(self, trackid, **kwargs):
        """
        create and insert trackid
        """
        tracklet = self.tracklets.get(trackid, {})
        assert len(kwargs) > 0
        for key in kwargs:
            if key == "action" and self.action_dict:
                action = kwargs[key]
                action_label = action[0]
                action_score = float(action[1])
                tracklet.setdefault("action_label", []).append(
                    self.action_dict[action_label]
                )  # noqa
                tracklet.setdefault("action_score", []).append(action_score)
            else:
                tracklet.setdefault(key, []).append(kwargs[key])
        self.tracklets[trackid] = tracklet

    def create(self, trackid, bbox, keypoint=None, action=None):
        """
        create new trackid
        """
        print("Please use update interface")
        self.tracklets[trackid] = dict()
        self.tracklets[trackid]["boxes"] = [bbox]
        if keypoint:
            self.tracklets[trackid]["keypoints"] = [keypoint]
        if action and self.action_dict:
            action_label = action[0]
            action_score = float(action[1])
            self.tracklets[trackid]["action_label"] = [
                self.action_dict[action_label]
            ]  # noqa
            self.tracklets[trackid]["action_score"] = [action_score]

    def insert(self, trackid, bbox, keypoint=None, action=None):
        """
        insert one record to existing record
        """
        print("Please use update interface")
        assert (
            trackid in self.tracklets.keys()
        ), "trackid not in self.tracklets.keys()"  # noqa
        self.tracklets[trackid]["boxes"].append(bbox)
        if keypoint:
            self.tracklets[trackid]["keypoints"].append(keypoint)
        if action and self.action_dict:
            action_label = action[0]
            action_score = float(action[1])
            self.tracklets[trackid]["action_label"].append(
                self.action_dict[action_label]
            )
            self.tracklets[trackid]["action_score"].append(action_score)

    def generate_action_timestamp(self):
        """
        generate action start and end
        """
        logging.info("only support falling detection")
        assert self.action_dict
        for key in self.tracklets:
            tracklet = self.tracklets[key]
            # [[start, end, action_name], ]
            action_labels = []

            act1 = np.array(self.tracklets[key]["action_label"])
            act2 = np.array([0 for _ in act1])
            act2[1:] = act1[:-1]
            diff_act = act1 - act2

            if act1[-1] > 0:
                diff_act[-1] = -1
            action_name = "falling"
            starts = np.where(diff_act == 1)[0]
            ends = np.where(diff_act == -1)[0]

            assert starts.shape[0] == ends.shape[0]
            for i in range(starts.shape[0]):
                score = np.array(
                    self.tracklets[key]["action_score"][starts[i] : ends[i]]
                ).mean()  # noqa
                action_labels.append([starts[0], ends[0], action_name, score])
            self.tracklets[key]["action"] = action_labels


def roidb_to_tracklets(roidb: List[Dict], keys: list) -> Tracklets:
    """
    convert roidb to tracklets.
    """
    tracklets = Tracklets()
    for roi in roidb:
        assert "trackid" in roi
        for ind, trackid in enumerate(roi["trackid"]):
            kwargs = {key: roi[key][ind] for key in keys}
            if "image_name" in roi:
                kwargs.update({"image_name": roi["image_name"]})
            tracklets.update(trackid, **kwargs)
    return tracklets


def tracklets_to_roidb(
    tracklets: Tracklets, roidb: List[Dict] = 0
) -> List[Dict]:
    """
    convert tracklets to roidb.
    """
    roidb_dict = {}
    for trackid in tracklets.ids:
        image_names = tracklets[trackid]["image_name"]
        for ind, image_name in enumerate(image_names):
            if image_name not in roidb_dict:
                roidb_dict[image_name] = {
                    "image_name": image_name,
                    "trackid": [],
                }
            roidb_dict[image_name]["trackid"].append(trackid)
            for key in tracklets[trackid]:
                if key == "image_name":
                    continue
                value = tracklets[trackid][key][ind]
                if key not in roidb_dict[image_name]:
                    roidb_dict[image_name][key] = list()
                roidb_dict[image_name][key].append(value)
    new_roidb = []
    for k1 in sorted(roidb_dict.keys()):
        roi = roidb_dict[k1]
        for k2 in roi:
            if isinstance(roi[k2], list):
                roi[k2] = np.asarray(roi[k2], dtype=np.float32)
        new_roidb.append(roi)
    return new_roidb
