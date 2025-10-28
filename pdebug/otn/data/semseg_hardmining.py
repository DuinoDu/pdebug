import os
from pathlib import Path
from typing import Any, Dict

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output
from pdebug.utils.decorator import mp
from pdebug.utils.semantic_types import (
    SEMANTIC_PCD_CATEGORIES,
    load_categories,
)
from pdebug.utils.types import segmentation_to_mask
from pdebug.visp import Colormap, draw

import cv2
import numpy as np
import tqdm
import typer


def compute_iou(mask1, mask2, return_area=False):
    """By devv.ai"""
    # Convert the masks to boolean arrays
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)

    # Calculate the intersection and union
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)

    # Calculate the IoU
    iou = np.sum(intersection) / np.sum(union)
    if return_area:
        return iou, mask1_bool.sum(), mask2_bool.sum()
    else:
        return iou


def compute_metric(mask1, mask2, return_area=True):
    classes = set(mask1.flatten().tolist())
    classes.update(set(mask2.flatten().tolist()))
    res = {
        cls: compute_iou(mask1 == cls, mask2 == cls, return_area=return_area)
        for cls in sorted(list(classes))
    }
    return res


@otn_manager.NODE.register(name="semseg_hardmining")
def main(
    imgdir: str,
    pred1: str,
    pred2: str,
    output: str = None,
    vis_output: str = None,
    ipdb: bool = False,
    only_ceil_wall_floor: bool = False,
    iou_threshold: float = 0.6,
    num_workers: int = 0,
):
    """Diff two semseg prediction."""
    import pdebug.piata.coco

    reader = Input(imgdir, name="imgdir").get_reader()
    roidb1 = Input(pred1, name="coco").get_roidb(as_dict=True)
    roidb2 = Input(pred2, name="coco").get_roidb(as_dict=True)

    if vis_output:
        os.makedirs(vis_output, exist_ok=True)
        classes = [l["label"] for l in SEMANTIC_PCD_CATEGORIES]
        colors = [color for color in Colormap(len(classes))]
        colors[0] = [0, 0, 0]

    if ipdb:
        __import__("ipdb").set_trace()

    @mp(nums=num_workers)
    def _process(idx_list):
        hard_roidb = []
        t = tqdm.tqdm(total=len(idx_list))
        for idx in idx_list:
            # for imgfile in reader.imglist:
            t.update()
            imgfile = reader.imglist[idx]
            image_name = os.path.basename(imgfile)
            if image_name not in roidb1 or image_name not in roidb2:
                continue
            roi1 = roidb1[image_name]
            roi2 = roidb2[image_name]

            mask1 = segmentation_to_mask(roi1, min_area=0)
            mask2 = segmentation_to_mask(roi2, min_area=0)
            ious = compute_metric(mask1, mask2)

            if only_ceil_wall_floor:
                valid = np.logical_or(
                    np.logical_or(mask1 == 1, mask1 == 2), mask1 == 3
                )
                mask1[np.logical_not(valid)] = 0
                valid = np.logical_or(
                    np.logical_or(mask2 == 1, mask2 == 2), mask2 == 3
                )
                mask2[np.logical_not(valid)] = 0
                valid_ious = {k: ious[k] for k in ious if k in [1, 2, 3]}
                ious = valid_ious

            bad_num = 0
            for idx, (cls, value) in enumerate(ious.items()):
                if isinstance(value, tuple) and len(value) == 3:
                    iou, _, _ = value
                else:
                    iou = value
                if idx == 0:
                    continue
                if iou < iou_threshold:
                    bad_num += 1

            if not bad_num:
                continue

            if vis_output:
                image = cv2.imread(imgfile)
                vis1 = draw.semseg(
                    mask1, image, classes=classes, colors=colors
                )
                vis2 = draw.semseg(
                    mask2, image, classes=classes, colors=colors
                )
                vis_all = np.concatenate((image, vis1, vis2), axis=1)

                for idx, (cls, value) in enumerate(ious.items()):
                    if isinstance(value, tuple) and len(value) == 3:
                        iou, area1, area2 = value
                    else:
                        iou = value
                    metric_str = (
                        f"{classes[cls]}: {iou:.3f}, area: {area1} / {area2}"
                    )
                    vis_all = cv2.putText(
                        vis_all,
                        metric_str,
                        (100, 100 + 50 * idx),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                savename = os.path.join(vis_output, roi1["image_name"])
                cv2.imwrite(savename, vis_all)

            hard_roidb.append(roi1)
        return hard_roidb

    idx_list = list(range(len(reader.imglist)))
    hard_roidb = _process(idx_list)
    hard_roidb = sorted(hard_roidb, key=lambda x: x["image_name"])

    print(f"hard mining: {len(roidb1)} => {len(hard_roidb)}")

    if output:
        outdir = os.path.dirname(output)
        os.makedirs(outdir, exist_ok=True)
        writer = Output(
            hard_roidb,
            name="coco",
            categories=load_categories(),
            description="semseg_hardmining",
            version="v1.0",
        )
        writer.save(output)
        print(f"Saved to {output}")

    return output


@otn_manager.NODE.register(name="semseg_hardmining_folders")
def semseg_hardmining_folders(
    root: str,
    imgdir_root: str,
    do_vis: bool = False,
    cache: bool = False,
):
    """
    ("root"/big, "root"/small) ->  "root"/hard
    """
    root = Path(root)
    pred1_root = os.path.join(root, "big")
    pred2_root = os.path.join(root, "small")
    output_root = os.path.join(root, "hard")
    vis_root = os.path.join(root, "vis_hard")

    if not os.path.exists(pred1_root) or not os.path.exists(pred2_root):
        print(f"{pred1_root} or {pred2_root} not exists")
        return
    os.makedirs(output_root, exist_ok=True)

    for jsonfile in os.listdir(pred1_root):
        if cache and jsonfile in os.listdir(output_root):
            print(f"{jsonfile} has processed, skip")
        if jsonfile not in os.listdir(pred2_root):
            print(f"{jsonfile} not found in {pred2_root}, skip")

        imgdir = os.path.splitext(jsonfile)[0]

        main(
            os.path.join(imgdir_root, imgdir),
            os.path.join(pred1_root, jsonfile),
            os.path.join(pred2_root, jsonfile),
            output=os.path.join(output_root, jsonfile),
            vis_output=os.path.join(vis_root, jsonfile) if do_vis else None,
            only_ceil_wall_floor=True,
            num_workers=8,
        )
    return output_root


if __name__ == "__main__":
    typer.run(main)
