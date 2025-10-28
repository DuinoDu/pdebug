import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple

import mmcv
import numpy as np
from PIL import Image

from ..registry import ROIDB_REGISTRY


@ROIDB_REGISTRY.register(name="voc")
def vocxml_to_roidb(
    xmldir: str, image_set: Optional[str] = None, imgdir: str = "JPEGImages"
) -> List:
    """
    Load xml in voc-style to roidb.
    """
    if os.path.exists(xmldir):
        xmlfiles = []
        for path, dirs, files in os.walk(xmldir, followlinks=True):
            xmlfiles += [
                os.path.join(path, x) for x in files if x.endswith(".xml")
            ]
    elif xmldir.endswith(".xml"):
        xmlfiles = [xmldir]
    else:
        raise ValueError(f"Bad voc input: {xmldir}")

    if image_set is not None:
        assert os.path.exists(image_set), f"{image_set} is not available."
        assert os.path.exists(xmldir)
        img_ids = mmcv.list_from_file(image_set)
    else:
        img_ids = [os.path.basename(x)[:-4] for x in xmlfiles]

    roidb = load_annotations(img_ids, imgdir, xmldir)
    return roidb


def load_annotations(img_ids, imgdir, xmldir):
    """
    load from xmldir based on img_ids.
    """
    data_infos = list()
    for img_id in img_ids:
        filename = f"{imgdir}/{img_id}.jpg"
        xml_path = os.path.join(f"{xmldir}/{img_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        width, height = 0, 0
        if size is not None:
            width = int(size.find("width").text)
            height = int(size.find("height").text)
        else:
            img_path = os.path.join(imgdir, "{}.jpg".format(img_id))
            img = Image.open(img_path)
            width, height = img.size
        info = dict(id=img_id, image_name=filename, width=width, height=height)
        info.update(get_ann_info(root))
        data_infos.append(info)
    return data_infos


def get_ann_info(root):
    boxes = []
    classes = []
    difficults = []

    for obj in root.findall("object"):
        name = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        bnd_box = obj.find("bndbox")
        bbox = [
            int(float(bnd_box.find("xmin").text)),
            int(float(bnd_box.find("ymin").text)),
            int(float(bnd_box.find("xmax").text)),
            int(float(bnd_box.find("ymax").text)),
        ]
        classes.append(name)
        boxes.append(bbox)
        difficults.append(difficult)
    if not boxes:
        boxes = np.zeros((0, 4))
        labels = []
        difficults = []
    else:
        # convert to 0-indexed
        boxes = np.array(boxes, ndmin=2) - 1
    ann = dict(
        boxes=boxes.astype(np.float32), classes=classes, difficults=difficults
    )
    return ann
