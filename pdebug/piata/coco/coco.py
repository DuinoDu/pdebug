"""
Extend COCO in pycocotool.
"""

import itertools
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO as MSCOCO

__all__ = ["COCO"]


def _isArrayLike(obj):
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0
    elif isinstance(obj, str):
        return False
    else:
        return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO(MSCOCO):
    """
    Extend mscoco with video and action.
    """

    def createIndex(self):
        """
        Create index.
        """
        super(COCO, self).createIndex()
        videos, actions = {}, {}
        videoToImgs, actionToImgs = defaultdict(list), defaultdict(list)

        if "videos" in self.dataset:
            for video in self.dataset["videos"]:
                videos[video["id"]] = video

        if "actions" in self.dataset:
            for action in self.dataset["actions"]:
                actions[action["id"]] = action

        if "annotations" in self.dataset and "actions" in self.dataset:
            for ann in self.dataset["annotations"]:
                if "action_id" in ann:
                    actionToImgs[ann["action_id"]].append(ann["image_id"])

        if "videos" in self.dataset and "images" in self.dataset:
            for image in self.dataset["images"]:
                videoToImgs[image["video_id"]].append(image["id"])

        print("extended index created!")

        self.videos = videos
        self.actions = actions
        self.videoToImgs = videoToImgs
        self.actionToImgs = actionToImgs

    def getAnnIds(
        self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, actionIds=[]
    ):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter.

        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        actionIds = actionIds if _isArrayLike(actionIds) else [actionIds]

        if len(imgIds) == len(catIds) == len(areaRng) == len(actionIds) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId]
                    for imgId in imgIds
                    if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )
            anns = (
                anns
                if len(actionIds) == 0
                else [ann for ann in anns if ann["action_id"] in actionIds]
            )
        if not iscrowd == None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def getImgIds(self, imgIds=[], catIds=[], videoIds=[], frameIds=[]):
        """
        Get img ids that satisfy given filter conditions.

        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        videoIds = videoIds if _isArrayLike(videoIds) else [videoIds]
        frameIds = frameIds if _isArrayLike(frameIds) else [frameIds]

        if len(imgIds) == len(catIds) == len(videoIds) == len(frameIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
            for i, videoId in enumerate(videoIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.videoToImgs[videoId])
                else:
                    ids &= set(self.videoToImgs[videoId])
            ids = (
                ids
                if len(frameIds) == 0
                else [
                    _id
                    for _id in ids
                    if self.imgs[_id]["frame_id"] in frameIds
                ]
            )
        return list(ids)

    def getVideoIds(self, videoIds=[], videoNms=[]):
        """
        Get video ids.

        Parameters
        ----------
        videoIds : list
            given video indexs
        videoNms: list
            given video names

        Returns
        -------
        video index list

        """
        videoIds = videoIds if _isArrayLike(videoIds) else [videoIds]
        videoNms = videoNms if _isArrayLike(videoNms) else [videoNms]

        if len(videoIds) == len(videoNms) == 0:
            videos = self.dataset["videos"]
        else:
            videos = self.dataset["videos"]
            videos = (
                videos
                if len(videoIds) == 0
                else [video for video in videos if video["id"] in videoIds]
            )
            videos = (
                videos
                if len(videoNms) == 0
                else [video for video in videos if video["name"] in videoNms]
            )
        ids = [video["id"] for video in videos]
        return ids

    def getActionIds(self, actionIds=[], actionNms=[]):
        """
        Get action ids.

        Parameters
        ----------
        actionIds : list
            given action indexs
        actionNms: list
            given action names

        Returns
        -------
        video index list

        """
        actionIds = actionIds if _isArrayLike(actionIds) else [actionIds]
        actionNms = actionNms if _isArrayLike(actionNms) else [actionNms]

        if len(actionIds) == len(actionNms) == 0:
            actions = self.dataset["actions"]
        else:
            actions = self.dataset["actions"]
            actions = (
                actions
                if len(actionIds) == 0
                else [
                    action for action in actions if action["id"] in actionIds
                ]
            )
            actions = (
                actions
                if len(actionNms) == 0
                else [
                    action for action in actions if action["name"] in actionNms
                ]
            )
        ids = [action["id"] for action in actions]
        return ids

    def loadVideos(self, ids=[]):
        """
        Load videos with the specified ids.
        """
        if _isArrayLike(ids):
            return [self.videos[id] for id in ids]
        elif type(ids) == int:
            return [self.videos[ids]]

    def loadActions(self, ids=[]):
        """
        Load actions with the specified ids.
        """
        if _isArrayLike(ids):
            return [self.actions[id] for id in ids]
        elif type(ids) == int:
            return [self.actions[ids]]
