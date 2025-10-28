import copy
import json
import os
import shutil
from functools import partial
from typing import List, Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.env import TORCH_INSTALLED, TORCHMETRICS_INSTALLED
from pdebug.utils.fileio import no_print
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer
from colorama import Back, init
from tabulate import tabulate

if TORCH_INSTALLED:
    import torch

if TORCHMETRICS_INSTALLED:
    from torchmetrics import Metric
    from torchmetrics.classification import MulticlassConfusionMatrix
    from torchmetrics.functional.classification.confusion_matrix import (
        _multiclass_confusion_matrix_update,
    )
else:
    Metric = object


try:
    from munkres import Munkres
except ImportError as e:
    Munkres = None


def find_nearest_points(query_points, refer_points, max_distance=10):
    num_query = query_points.shape[0]
    num_refer = refer_points.shape[0]

    distance_vec = np.repeat(
        query_points[:, None, :2], num_refer, axis=1
    ) - np.repeat(refer_points[None, :, :2], num_query, axis=0)
    distance_map = np.linalg.norm(distance_vec, axis=2)

    if num_query > num_refer:
        min_dist = distance_map.min(axis=1)
        query_valid_idx = min_dist.argsort()[:num_refer]
        distance_map = distance_map[query_valid_idx, :]
    elif num_query < num_refer:
        min_dist = distance_map.min(axis=0)
        refer_valid_idx = min_dist.argsort()[:num_query]
        distance_map = distance_map[:, refer_valid_idx]

    assert distance_map.shape[0] == distance_map.shape[1]
    matched = np.asarray(Munkres().compute(distance_map.copy())).astype(int)

    # remove bad matched pair
    for item in matched:
        if distance_map[item[0], item[1]] > max_distance:
            item[1] = -1

    # recover to raw point index
    if num_query > num_refer:
        for m in matched:
            m[0] = query_valid_idx[m[0]]
    elif num_query < num_refer:
        for m in matched:
            m[1] = refer_valid_idx[m[1]]

    return matched


class KeypointsMetric(Metric):
    """Keypoints metric."""

    def __init__(self, **kwargs):
        super(KeypointsMetric, self).__init__(**kwargs)
        if TORCHMETRICS_INSTALLED:
            self.add_state(
                "num_instance", torch.tensor(0), dist_reduce_fx="sum"
            )
            self.add_state(
                "err", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
            )
            self.add_state("TP", torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("TN", torch.tensor(0), dist_reduce_fx="sum")
            self.add_state("FP", torch.tensor(0), dist_reduce_fx="sum")
        else:
            self.num_instance = 0
            self.err = 0
            self.TP = 0
            self.TN = 0
            self.FP = 0

    def update(self, pred, target, num_examples=None):
        if hasattr(pred, "device") and self.num_instance.device != pred.device:
            self.num_instance = self.num_instance.to(pred.device)

        if num_examples:
            self.num_instance += num_examples
        else:
            self.num_instance += pred.shape[0]

        assert pred.ndim == 2
        assert target.ndim == 2

        num_pred = pred.shape[1] // 3
        pred = pred.reshape(-1, 3)
        pred = pred[pred[:, 2] > 0.2]
        if pred.size == 0:
            self.TN += max(0, target.shape[0])
            return

        num_target = target.shape[1] // 3
        target = target.reshape(-1, 3)

        matched = find_nearest_points(pred, target, max_distance=10)
        errs = []
        for i in matched:
            err = np.linalg.norm(pred[i[0]][:2] - target[i[1]][:2])
            errs.append(err)

        self.err += np.asarray(errs).mean()
        self.TP += matched.shape[0]
        self.FP += max(0, pred.shape[0] - matched.shape[0])
        self.TN += max(0, target.shape[0] - matched.shape[1])
        return errs

    def compute(self):
        if self.err.sum() == 0:
            raise RuntimeError("Please call `update(...)` before `compute()`.")
        err = self.err / self.num_instance
        TP = self.TP
        TN = self.TN
        FP = self.FP
        precision = TP / (TP + FP)
        recall = TP / (TP + TN)
        return err.item(), precision.item(), recall.item()


@otn_manager.NODE.register(name="kps_eval")
def kps_eval(
    pred_file: str,
    gt_file: str,
    output: str = None,
    vis_output: str = None,
    cache: bool = False,
    print_result: bool = True,
):
    """Evaluate keypoints result."""
    if output:
        if cache:
            if os.path.exists(output):
                return output
        else:
            if os.path.exists(output):
                shutil.rmtree(output)

    assert pred_file.endswith(".json")
    assert gt_file.endswith(".json")
    with no_print():
        pred_roidb = Input(pred_file, name="coco").get_roidb()
        reader = Input(gt_file, name="coco", return_coco=True)
        assert isinstance(reader._roidb, tuple)
        reader._roidb, coco_inst = reader._roidb
        gt_roidb = reader.get_roidb(as_dict=True, keepname=True)
    assert len(pred_roidb) == len(gt_roidb)

    if vis_output and os.path.exists(vis_output):
        shutil.rmtree(vis_output)

    metric = KeypointsMetric()

    t = tqdm.tqdm(
        total=len(pred_roidb), desc=f"eval {os.path.basename(pred_file)}"
    )
    for pred_roi in pred_roidb:
        t.update()
        image_name = coco_inst.imgs[pred_roi["image_id"]]["file_name"]
        gt_roi = gt_roidb[image_name]

        metric.update(
            pred_roi["keypoints"], gt_roi["keypoints"], num_examples=1
        )
        if vis_output:
            pass
        # if t.n > 10: break

    err, precision, recall = metric.compute()
    if print_result:
        print("error: \t", err)
        print("prec: \t", precision)
        print("rec: \t", recall)

    if not output:
        return {"err": err, "precision": precision, "recall": recall}
    else:
        return output


if __name__ == "__main__":
    typer.run(kps_eval)
