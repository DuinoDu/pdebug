import logging
import os
import random
import re
from collections import OrderedDict

import cv2
import numpy as np

from ..registry import ROIDB_REGISTRY

__all__ = ["simpletxt_to_roidb", "roidb_to_simpletxt"]


name_mapping = {
    "boxes": "boxes",
    "skeleton": "keypoints",
    "keypoint3d": "keypoints_3d",
    "reid": "reid",
    "trackid": "trackid",
    "lmks": "lmks",
    "lmks68": "lmks68",
    "action": "action",
    "orientation": "orientation",
    "maskfile": "maskfile",
}


def parse_field(field, split_str="_"):
    """
    loading field string to dict.
    """
    field_dict = OrderedDict()
    for f in field.split(split_str):
        m = re.search("\\d+$", f)
        if m is not None:
            output = m.group(0)
        else:
            raise ValueError("field %s is invalid" % field)
        name = f.split(output)[0]
        length = output
        field_dict[name] = int(length)
    return field_dict


def simpletxt_to_roidb(
    fid,
    field,
    ignore_nobox=True,
    imgdir=None,
    xywh=False,
    ext=".jpg",
    log_level=1,
    topk=None,
    split_str=None,
    shuffle=False,
):
    """
    Convert aiottxt to roidb.

    Parameters
    ----------
    field: str
        aiottxt desc str
    ignore_nobox: bool
        if ignore null box
    imgdir: str
        image directory
    xywh: bool
        bbox layout
    roiwise: bool
        one line stands for one image, or one roi.
    ext: str
        use ext when image_name has no ext

    """
    # {'boxes': 5, 'skeleton': 51}
    if "action" in field:
        raise NotImplementedError
    field_dict = parse_field(field)
    stride = sum([v for k, v in field_dict.items()])

    if not split_str:
        split_str = ",| |;"

    roidb = []
    lines = [x.strip() for x in fid.readlines()]
    for ind, line in enumerate(lines):
        if ind % 1000 == 0 and log_level > 0:
            logging.info("%d / %d" % (ind, len(lines)))
        items = re.split(split_str, line)
        assert (len(items) - 1) % stride == 0, "invalid aiottxt"

        if topk and ind >= topk:
            break

        # if len(items) < 5:
        #     if ignore_nobox:
        #         continue
        #     else:
        #         logging.info(
        #             "find len(line) < 5, invalid aiottxt, please recheck."
        #         )  # noqa
        #         import sys
        #         sys.exit()

        roi = dict()

        # get imgname
        imgname = items[0]
        # if os.path.splitext(imgname)[1] == '':
        #     imgname += ext
        imgfile = imgname
        if imgdir is not None:
            imgdir = os.path.abspath(imgdir)
            imgfile = os.path.join(imgdir, os.path.basename(imgname))
            if not os.path.exists(imgfile):
                # search for imgname like 0000001.jpg
                imgfiles = os.listdir(imgdir)
                if len(imgfiles) == 0:
                    raise ValueError("no image in %s" % imgdir)
                _ext = os.path.splitext(imgfiles[0])[1]
                try:
                    imgname_int = int(imgname)
                except Exception as e:
                    raise ValueError("%s is not int name." % imgname)
                for i in range(4, 13):
                    _format = f"%0{i}d{_ext}"
                    imgname = _format % imgname_int
                    imgfile = os.path.join(imgdir, os.path.basename(imgname))
                    if os.path.exists(imgfile):
                        break
            assert os.path.exists(imgfile), "%s not exist" % imgfile
            # slow!!!
            # img = cv2.imread(imgfile)
            # roi['height'] = img.shape[0]
            # roi['width'] = img.shape[1]
        roi["image_name"] = imgfile

        # parse following field by field_dict, saved in roi
        item_ind = 1
        while True:
            for each_field in field_dict.items():
                try:
                    seg_name = name_mapping[each_field[0]]
                except KeyError as e:
                    seg_name = each_field[0]
                seg_len = each_field[1]
                assert len(items) >= (
                    item_ind + seg_len
                ), "items[%d] should > item_ind+seg_len[%d]" % (
                    len(items),
                    item_ind + seg_len,
                )  # noqa
                try:
                    seg_value = [
                        float(x) for x in items[item_ind : item_ind + seg_len]
                    ]
                except ValueError as e:
                    seg_value = [
                        str(x) for x in items[item_ind : item_ind + seg_len]
                    ]
                    if seg_len == 1:
                        seg_value = seg_value[0]
                if seg_name in roi:
                    roi[seg_name].append(seg_value)
                else:
                    roi[seg_name] = [seg_value]
                item_ind += seg_len
            if item_ind >= len(items):
                break
        # convert list to np.ndarray
        new_roi = roi.copy()
        for v in roi.keys():
            if v in ("image_name", "maskfile"):
                continue
            try:
                new_roi[v] = np.array(roi[v], dtype=np.float32)
            except ValueError as e:
                continue
            # squeeze shape=1 dimension
            if isinstance(new_roi[v], np.ndarray) and new_roi[v].shape[1] == 1:
                new_roi[v] = np.squeeze(new_roi[v], axis=1)
        roi = new_roi
        # generate 'boxes_score' if need
        if "boxes" in roi:
            if roi["boxes"].shape[1] > 4:
                roi["boxes_score"] = [float(_s) for _s in roi["boxes"][:, 4]]
            else:
                roi["boxes_score"] = [
                    float(1.0) for _ in range(roi["boxes"].shape[0])
                ]
            roi["boxes"] = roi["boxes"][:, :4]
            if xywh:
                w = roi["boxes"][:, 2]
                h = roi["boxes"][:, 3]
                roi["boxes"][:, 2] = roi["boxes"][:, 0] + w
                roi["boxes"][:, 3] = roi["boxes"][:, 1] + h

        if "maskfile" in roi and isinstance(roi["maskfile"], list):
            assert len(roi["maskfile"]) == 1
            roi["maskfile"] = roi["maskfile"][0]

        roidb.append(roi)

    roidb = sorted(roidb, key=lambda k: k["image_name"])
    if shuffle:
        random.shuffle(roidb)
    return roidb


def roidb_to_simpletxt(
    roidb,
    savefile=None,
    imgdir=None,
    boxes=True,
    skeleton=False,
    skeleton_num=51,
    reid=False,
    feet=False,
    trackid=False,
    lmks=False,
    pose=False,
    bodytype=False,
    action=False,
):
    """Convert roidb to aiottxt.

    Args:
        roidb (list): roidb

    Kwargs:
        imgdir (str): imgdir
        boxes (bool): if save boxes
        skeleton (bool): if save skeleton
        reid (bool): if save reid
        feet (bool): if save feet
        trackid (bool): if save trackid
        lmks (bool): if save lmks
        pose (bool): if save pose
        bodytype (bool): if save bodytype
        action (bool): if save action

    Returns: None

    """
    if imgdir is not None:
        imgdir = os.path.abspath(imgdir)

    ext = ""
    if boxes:
        ext += "_boxes5"
    if trackid:
        ext += "_trackid1"
    if skeleton:
        ext += "_skeleton{}".format(skeleton_num)
        if skeleton_num == 51:
            box_type = "person"
        elif skeleton_num == 63:
            box_type = "hand"
    if bodytype:
        ext += "_bodytype1"
    if action:
        ext += "_action1"
    if savefile is None:
        savefile = "aiottxt_" + ext + ".txt"
    fid = open(savefile, "w")

    roidb = sorted(roidb, key=lambda x: x["image_name"])
    for ind, roi in enumerate(roidb):
        line = ""
        if ind % 1000 == 0:
            logging.info("%d / %d" % (ind, len(roidb)))
        if imgdir is not None:
            imgpath = os.path.join(imgdir, os.path.basename(roi["image_name"]))
        else:
            imgpath = roi["image_name"]
        line += imgpath

        assert "boxes" in roi, "roi should have boxes"
        boxes = roi["boxes"]

        if skeleton and "gt_classes" in roi:
            # filter boxes using gt_classes
            # align keypoints to boxes
            num_boxes = boxes.shape[0]
            if roi["keypoints"].shape[0] > boxes.shape[0]:
                roi["keypoints"] = roi["keypoints"][:num_boxes]

        for box_ind, box in enumerate(boxes):
            if len(box) == 5:
                box_score = box[-1]
            elif "boxes_score" in roi:
                box_score = roi["boxes_score"][box_ind]
            else:
                box_score = 1.0
            for box_i in box[:4]:
                line += " %.3f" % box_i
            line += " %.3f" % box_score

            if skeleton:
                assert roi[name_mapping["skeleton"]].shape[1] >= skeleton_num
                keypoint = roi[name_mapping["skeleton"]][box_ind][
                    :skeleton_num
                ]
                for k in keypoint:
                    line += " %.3f" % k
        fid.write(line + "\n")
    fid.close()


@ROIDB_REGISTRY.register(name="simpletxt")
def load_simpletxt(
    txtfile,
    field,
    imgdir=None,
    xywh=False,
    topk=None,
    split_str=None,
    shuffle=False,
):
    with open(txtfile, "r") as fid:
        roidb = simpletxt_to_roidb(
            fid,
            field,
            imgdir=imgdir,
            xywh=xywh,
            topk=topk,
            split_str=split_str,
            shuffle=shuffle,
        )
    return roidb
