import os
import shutil

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.piata.coco import COCOWriter, save_to_cocoDt

import numpy as np
import typer


@otn_manager.NODE.register(name="kps_roidb_to_coco")
def kps_roidb_to_coco(
    path: str,
    imgdir: str,
    output: str = None,
    cache: bool = False,
    bbox_mode: str = "center_crop_by_short_side",
    train_ratio: float = 0.8,
):
    """Convert kps roidb(pkl) to coco format.

    Args:
        path: keypoint roidb pkl file.
        imgdir: image folder
        output: coco output path
        cache: use cache if exists.
    """
    if os.path.exists(output):
        if cache:
            typer.echo(
                typer.style(f"Found {output}, skip", fg=typer.colors.WHITE)
            )
            return output
        else:
            shutil.rmtree(output)
    os.makedirs(f"{output}/annotations", exist_ok=True)
    os.makedirs(f"{output}/person_detection_results", exist_ok=True)

    roidb = Input(path, name="default").get_roidb()

    for roi in roidb:
        if roi["keypoints"].ndim == 1:
            roi["keypoints"] = roi["keypoints"][None, :]

    if bbox_mode == "center_crop_by_short_side":
        for roi in roidb:
            h, w = roi["image_height"], roi["image_width"]
            short_side = min(h, w)
            dst_shape = (short_side, short_side)
            if dst_shape[0] < h:
                y1 = h // 2 - dst_shape[0] // 2
                y2 = h // 2 + dst_shape[0] // 2 - 1
                x1, x2 = 0, dst_shape[1] - 1
            elif dst_shape[1] < w:
                x1 = w // 2 - dst_shape[1] // 2
                x2 = w // 2 + dst_shape[1] // 2 - 1
                y1, y2 = 0, dst_shape[0] - 1
            else:
                raise NotImplementedError
            roi["boxes"] = np.asarray([[x1, y1, x2, y2]]).astype(np.float32)

    train_length = int(len(roidb) * train_ratio)
    train_roidb = roidb[:train_length]
    valid_roidb = roidb[train_length:]

    imgdir = os.path.abspath(imgdir)
    os.system(f"ln -s {imgdir} {output}/train2017")
    os.system(f"ln -s {imgdir} {output}/val2017")

    writer = COCOWriter(image_ext=".png", dummy_object=True)
    writer.add_roidb(train_roidb)
    output_json = os.path.join(
        output, "annotations/person_keypoints_train2017.json"
    )
    writer.save(output_json)

    writer = COCOWriter(image_ext=".png", dummy_object=True)
    writer.add_roidb(valid_roidb)
    output_json2 = os.path.join(
        output, "annotations/person_keypoints_val2017.json"
    )
    writer.save(output_json2)

    output_json3 = os.path.join(
        output,
        "person_detection_results/COCO_val2017_detections_AP_H_56_person.json",
    )
    save_to_cocoDt(output_json3, valid_roidb)

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output
