"""Data entry."""
import json
import logging
import os
import pickle
import sys

import numpy as np

from .registry import ROIDB_REGISTRY, SOURCE_REGISTRY

__all__ = ["Input"]


class Input:

    """Class for Input.

    Example:
    >>> import piata
    >>> roidb = piata.Input('roidb.pkl').get_roidb()
    >>> roidb = piata.Input('coco.json', name='coco').get_roidb()
    >>> roidb = piata.Input('annots.vott', name='vott').get_roidb()
    >>> reader = piata.Input('labeldir', name='labelme').get_roidb()
    >>> roidb = piata.Input('skeleton.txt', field='boxes5_keypoints51').get_roidb()
    >>> reader = piata.Input('image_path', name='imgdir').get_reader()
    >>> reader = piata.Input('video.mp4', name='video').get_reader()
    >>> reader = piata.Input('train2017.zip', name='imgzip').get_reader()
    >>> reader = piata.Input('data_path', name='dddsemseg').get_reader()
    >>> reader = piata.Input('data_path', name='ddddata_scannet').get_reader()
    >>> reader = piata.Input('data_path', name='ddddata_apple_roomplan').get_reader()
    >>> reader = piata.Input('data_path', name='ddddata_nyuv2').get_reader()
    >>> reader = piata.Input('data_path', name='PhotoidRecording').get_reader()
    >>> roidb = piata.Input('xxx.json', name='llava_json').get_roidb()
    >>> roidb = piata.Input('xxx.csv', name='csv').get_roidb()
    """

    def __init__(self, *args, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "default"
            print("use `default` roidb handle.")

        if kwargs["name"] == "coco":
            import pdebug.piata.coco
        elif kwargs["name"] == "PhotoidRecording":
            import pdebug.piata.handler.photoid_recording
        elif kwargs["name"] == "lance_vita":
            import pdebug.piata.handler.lance_vita

        if kwargs["name"] in ROIDB_REGISTRY:
            roidb_func = ROIDB_REGISTRY.get(kwargs.pop("name"))
            self._roidb = roidb_func(*args, **kwargs)
        elif kwargs["name"] in SOURCE_REGISTRY:
            reader_func = SOURCE_REGISTRY.get(kwargs.pop("name"))
            self._reader = reader_func(*args, **kwargs)
        else:
            if kwargs["name"] == "coco":
                raise ValueError(
                    "please import pdebug.piata.coco before using coco"
                )
            if kwargs["name"] == "PhotoidRecording":
                raise ValueError(
                    "please import pdebug.piata.handler.photoid_recording "
                    "before using photoid_recording"
                )
            else:
                raise ValueError(f"Unvalid name: {kwargs['name']}")

    def get_roidb(self, as_dict=False, keepname=False):
        """
        return roidb

        Parameters
        ----------
        as_dict : bool
            return as dict
        keepname: bool
            keep image_name as dict key, only used when as_dict=True

        """
        if len(self._roidb) == 0:
            return self._roidb

        Input.trans_roidb(self._roidb)

        if "image_name" in self._roidb[0]:
            # for image-based roidb.
            if len(self._roidb) > len(
                set([x["image_name"] for x in self._roidb])
            ):
                # change to img-wise
                self._roidb = Input.roiwise_to_imgwise(self._roidb)
            if as_dict:
                if len(self._roidb) == 0:
                    return {}
                if keepname:
                    roidb_dict = {x["image_name"]: x for x in self._roidb}
                else:
                    roidb_dict = {
                        os.path.basename(x["image_name"]): x
                        for x in self._roidb
                    }  # noqa
                return roidb_dict
            else:
                return self._roidb
        else:
            return self._roidb

    def set_roidb(self, roidb):
        self._roidb = roidb

    def get_reader(self):
        return self._reader

    def __len__(self):
        if hasattr(self, "_roidb"):
            return len(self._roidb)
        if hasattr(self, "_reader"):
            return len(self._reader)

    def __str__(self):
        return "piata input"

    def _update_imgdir(self):
        if self.imgdir is not None:
            logging.info("update imgdir in roidb ...")
            for ind, roi in enumerate(self._roidb):
                if ind % 1000 == 0:
                    logging.info("%d / %d" % (ind, len(self._roidb)))
                if "image" in self._roidb[ind]:
                    self._roidb[ind]["image"] = os.path.join(
                        self.imgdir, os.path.basename(roi["image"])
                    )
                elif "image_name" in self._roidb[ind]:
                    self._roidb[ind]["image_name"] = os.path.join(
                        self.imgdir, os.path.basename(roi["image_name"])
                    )

    def get_tracklet(self, action_dict):
        """
        get all tracklets for all videos in act roidb

        Parameters
        ----------
        action_dict : dict

        Returns
        -------
        all_tracklets : list
            each item is a Tracklets object for one video

        """
        act_roidb = self.get_act_roidb()
        roidb = self.get_roidb()
        assert act_roidb
        all_tracklets = []
        imgname_to_trackid_to_kps = {}
        imgname_to_trackid_to_bbox = {}
        for line in roidb:
            imgname = line["image_name"]
            try:
                for trackid, bbox, kps in zip(
                    line["track_id"], line["boxes"], line["keypoints"]
                ):  # noqa

                    imgname_to_trackid_to_kps.setdefault(
                        imgname, {}
                    ).setdefault(trackid, []).append(kps)
                    imgname_to_trackid_to_bbox.setdefault(
                        imgname, {}
                    ).setdefault(trackid, []).append(bbox)
            except:
                continue
        for one_act_roidb in act_roidb:
            tracklets = Tracklets(action_dict)
            for i in range(len(one_act_roidb["image_names"])):
                imgname = one_act_roidb["image_names"][i]
                track_ids = one_act_roidb["track_id"][i]
                action_labels = one_act_roidb["act_label"][i]
                if len(action_labels.shape) == 2:
                    action_labels = action_labels[:, 0]
                action_labels = action_labels.tolist()
                for trackid, action_label in zip(track_ids, action_labels):
                    try:
                        kps = imgname_to_trackid_to_kps[imgname][trackid]
                        bbox = imgname_to_trackid_to_bbox[imgname][trackid]
                        tracklets.update(
                            trackid, bbox, kps, [action_label, 1.0]
                        )  # noqa

                    except:
                        # no person in this img

                        continue
            all_tracklets.append(tracklets)
        return all_tracklets

    @staticmethod
    def trans_roidb(roidb):
        """
        trans some old keys in roidb to new keys
        """
        for roi in roidb:
            if "image" in roi and "image_name" not in roi:
                roi["image_name"] = roi["image"]
            if "height" in roi and "image_height" not in roi:
                roi["image_height"] = roi["height"]
            if "width" in roi and "image_width" not in roi:
                roi["image_width"] = roi["width"]

    @staticmethod
    def imgwise_to_roiwise(roidb):
        """
        convert roidb from image-wise to roi-wise.
        """
        new_roidb = []
        for roi in roidb:
            if "boxes" in roi:
                num_boxes = roi["boxes"].shape[0]
            else:
                num_boxes = None
                for k in roi:
                    if isinstance(roi[k], np.ndarray) and roi[k].ndim == 2:
                        num_boxes = roi[k].shape[0]
                        break
                if num_boxes is None:
                    logging.warning(
                        "no boxes-like label in roidb, skip this roi"
                    )
                    continue
            for i in range(num_boxes):
                new_roi = dict()
                for key in roi:
                    if isinstance(
                        roi[key], (str, float, int, np.float32, np.float64)
                    ):
                        new_roi[key] = roi[key]
                    elif isinstance(roi[key], list):
                        try:
                            new_roi[key] = roi[key][i]
                        except IndexError as e:
                            new_roi[key] = []
                    elif isinstance(roi[key], np.ndarray):
                        try:
                            new_roi[key] = np.array(
                                roi[key][i], dtype=roi[key].dtype
                            )  # noqa
                        except IndexError as e:
                            new_roi[key] = np.array([])
                    else:
                        raise ValueError("Unknown data type")
                new_roidb.append(new_roi)
        return new_roidb

    @staticmethod
    def roiwise_to_imgwise(roidb, log_level=0):
        """
        convert roidb from roi-wise to image-wise.
        """
        roidb_dict = dict()
        for roi in roidb:
            image_name = roi["image_name"]
            if image_name not in roidb_dict:
                new_roi = dict()
                for key in roi:
                    if isinstance(roi[key], (str, float, int)):
                        new_roi[key] = roi[key]
                    elif isinstance(
                        roi[key], (np.float32, np.float64, np.int64, np.int32)
                    ):
                        new_roi[key] = np.array(
                            [roi[key]], dtype=roi[key].dtype
                        )  # noqa
                    elif isinstance(roi[key], (np.ndarray)):
                        if roi[key].ndim > 1:
                            new_roi[key] = roi[key]
                        else:
                            new_roi[key] = np.array(
                                [roi[key]], dtype=roi[key].dtype
                            )  # noqa
                    elif isinstance(roi[key], list):
                        new_roi[key] = [roi[key]]
                    else:
                        raise ValueError(
                            "Unknown data type: %s" % type(roi[key])
                        )
                roidb_dict[image_name] = new_roi
            else:
                cur_roi = roidb_dict[image_name]
                for key in roi:
                    if isinstance(
                        roi[key], (np.float32, np.float64, np.int64, np.int32)
                    ):
                        cur_roi[key] = np.concatenate(
                            (
                                cur_roi[key],
                                np.array([roi[key]], dtype=roi[key].dtype),
                            ),
                            axis=0,
                        )
                    elif isinstance(roi[key], (np.ndarray)):
                        if roi[key].ndim < cur_roi[key].ndim:
                            cur_roi[key] = np.concatenate(
                                (
                                    cur_roi[key],
                                    np.array([roi[key]], dtype=roi[key].dtype),
                                ),
                                axis=0,
                            )
                        else:
                            cur_roi[key] = np.concatenate(
                                (cur_roi[key], roi[key]), axis=0
                            )
                    elif isinstance(roi[key], list):
                        cur_roi[key].append(roi[key])
                    else:
                        pass
        new_roidb = [roidb_dict[k] for k in sorted(roidb_dict.keys())]
        # squeeze shape=1 dimension
        for roi in new_roidb:
            for key in roi:
                if (
                    isinstance(roi[key], (np.ndarray))
                    and roi[key].ndim == 2
                    and roi[key].shape[1] == 1
                ):
                    roi[key] = np.squeeze(roi[key], axis=1)
        return new_roidb

    @staticmethod
    def filter_roi(roi, valid):
        """
        filter list/array by valid index.

        Parameters
        ----------
        roi : dict
        valid : list or np.ndarray

        Returns
        -------
        new_roi : dict
            filtered roi

        """
        assert isinstance(valid, (np.ndarray, list))
        new_roi = dict()
        for key in roi:
            if isinstance(
                roi[key],
                (str, float, int, np.float32, np.float64, np.int64, np.int32),
            ):  # noqa
                new_roi[key] = roi[key]
            elif isinstance(roi[key], np.ndarray):
                new_roi[key] = roi[key][valid]
            elif isinstance(roi[key], list):
                new_roi[key] = [roi[key][i] for i in valid]
            else:
                raise ValueError("Unknown data type: %s" % type(roi[key]))
        return new_roi


@ROIDB_REGISTRY.register(name="default")
def load_roidb(roidb_file):
    with open(roidb_file, "rb") as fid:
        try:
            roidb = pickle.load(fid)
        except Exception as e:
            fid.seek(0)
            roidb = pickle.load(fid, encoding="iso-8859-1")
    return roidb


@ROIDB_REGISTRY.register(name="csv")
def load_csv_file(csv_file, **kwargs):
    from pdebug.utils.env import PANDAS_INSTALLED

    assert PANDAS_INSTALLED, "pandas is required."
    import pandas

    return_df = kwargs.get("return_df", False)
    df = pandas.read_csv(csv_file, **kwargs)
    if return_df:
        return df
    else:
        roidb = []
        for item in df.values.tolist():
            roi = dict()
            for ind, key in enumerate(df.columns):
                roi[key] = item[ind]
            if "filename" in roi:
                roi["image_name"] = roi["filename"]
            roidb.append(roi)
        return roidb
