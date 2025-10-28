import copy
import json
import os
import shutil
from functools import partial
from typing import List, Optional

from pdebug.otn import manager as otn_manager
from pdebug.otn.data.cvat_utils import merge_polygon
from pdebug.piata import Input
from pdebug.utils.env import TORCH_INSTALLED, TORCHMETRICS_INSTALLED
from pdebug.utils.fileio import no_print
from pdebug.utils.semantic_types import load_categories, load_colors
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


class CropMask:

    """Crop sementations by given mask."""

    def __init__(self, mask_file=None, maskout_value=0):
        self.maskout_value = 0
        if mask_file is not None:
            assert os.path.exists(mask_file), f"{mask_file} not exists"
            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            if len(np.unique(mask)) != 2:
                threshold = np.unique(mask).mean()
                print(
                    f"Found multiple value in mask file, threshold by {threshold} to 0, 255"
                )
                mask[mask <= threshold] = 0
                mask[mask > threshold] = 255
            assert len(np.unique(mask)) == 2
            self.mask = mask
        else:
            self.mask = None

    def __call__(self, mask_array):
        if self.mask is None:
            return mask_array

        if mask_array.shape[:2] != self.mask.shape[:2]:
            print(
                f"pred mask shape {mask_array.shape} != maskout file shape {self.mask.shape}, skip"
            )
            return mask_array

        mask_array[self.mask == 0] = self.maskout_value
        return mask_array


class SemSegMetric(Metric):
    """Semantic segmentation metric, supporting IoU and AP."""

    def __init__(
        self, classes, ignore_classes=(), classes_info=None, **kwargs
    ):
        super(SemSegMetric, self).__init__(**kwargs)
        self.classes = classes
        self.num_classes = len(classes)
        self.ignore_classes = ignore_classes
        self.classes_info = classes_info

        self.compute_iou = lambda x: x.diag() / (
            x.sum(dim=1) + x.sum(dim=0) - x.diag() + 1e-15
        )
        self.iou_threshold = np.arange(0.5, 1.0, 0.1)
        self.reset()

    def reset(self, device=None):
        assert TORCHMETRICS_INSTALLED
        self.add_state(
            "num_instance", default=torch.tensor(0), dist_reduce_fx="sum"
        )
        self.add_state(
            "confmat",
            torch.zeros(self.num_classes, self.num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "TP", torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "TN", torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "FP", torch.zeros(self.num_classes), dist_reduce_fx="sum"
        )
        # TODO: check dist_reduce result

    def update(self, pred, target, num_examples=None):
        self.update_inner(pred, target, num_examples)

    def update_inner(self, pred, target, num_examples=None):
        if hasattr(pred, "device") and self.num_instance.device != pred.device:
            self.num_instance = self.num_instance.to(pred.device)
            self.confmat = self.confmat.to(pred.device)
            self.TP = self.TP.to(pred.device)
            self.TN = self.TN.to(pred.device)
            self.FP = self.FP.to(pred.device)

        if num_examples:
            self.num_instance += num_examples
        else:
            self.num_instance += pred.shape[0]
        # print(f"num_exmples: {num_examples}, num_instance: {self.num_instance}")

        ious = self.update_ious(pred, target)

        TPs, TNs, FPs, TN, FP = self.update_average_percision(
            pred, target, ious
        )
        ap = self.update_ap_by_iou_threashold(
            TPs, TNs, FPs, TN, FP, iou_threshold=0.5
        )
        self.TP += ap["TP"]
        self.TN += ap["TN"]
        self.FP += ap["FP"]

        return {"ious": ious, "ap": ap}

    def update_ious(self, pred, target):
        is_numpy_input = False
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(np.asarray([pred])).flatten()
            target = torch.tensor(np.asarray([target])).flatten()
            is_numpy_input = False

        confmat = _multiclass_confusion_matrix_update(
            pred, target, self.num_classes
        )
        self.confmat += confmat

        ious = self.compute_iou(confmat)
        if is_numpy_input:
            ious = ious.cpu().numpy()
        return ious

    def update_average_percision(self, pred, target, ious):

        if isinstance(pred, np.ndarray):
            pred_classes = np.unique(pred)
            target_classes = np.unique(target)
            zeros_fn = np.zeros
            dtype = np.int64
            make_tensor = np.asarray
        else:
            pred_classes = pred.unique()
            target_classes = target.unique()
            zeros_fn = partial(torch.zeros, device=pred.device)
            dtype = torch.int64
            make_tensor = partial(torch.tensor, device=pred.device)
            pred_classes = pred_classes.to(torch.long)
            target_classes = target_classes.to(torch.long)

        TPs = zeros_fn(
            (self.num_classes, len(self.iou_threshold)), dtype=dtype
        )
        TNs = zeros_fn(
            (self.num_classes, len(self.iou_threshold)), dtype=dtype
        )
        FPs = zeros_fn(
            (self.num_classes, len(self.iou_threshold)), dtype=dtype
        )
        TN = zeros_fn(self.num_classes, dtype=dtype)
        FP = zeros_fn(self.num_classes, dtype=dtype)

        for cls in target_classes:
            if cls in self.ignore_classes:
                continue
            if cls not in pred_classes:
                TN[cls] += 1
                continue
            tp = make_tensor([ious[cls] >= t for t in self.iou_threshold])
            tn = make_tensor([ious[cls] < t for t in self.iou_threshold])
            TPs[cls] += tp
            TNs[cls] += tn

        for cls in pred_classes:
            if cls in self.ignore_classes:
                continue
            if cls not in target_classes:
                FP[cls] += 1
                continue
            fp = make_tensor([ious[cls] < t for t in self.iou_threshold])
            FPs[cls] += fp

        return TPs, TNs, FPs, TN, FP

    def update_ap_by_iou_threashold(
        self, TPs, TNs, FPs, TN, FP, iou_threshold=0.5
    ):
        iou_threshold_idx = np.where(self.iou_threshold == iou_threshold)[0][0]
        TP_high_iou = TPs[:, iou_threshold_idx]
        TN_low_iou = TNs[:, iou_threshold_idx]
        FP_low_iou = FPs[:, iou_threshold_idx]
        TN += TN_low_iou
        FP += FP_low_iou
        return {"TP": TP_high_iou, "TN": TN, "FP": FP}

    def compute(self):
        if self.confmat.sum() == 0:
            raise RuntimeError("Please call `update(...)` before `compute()`.")

        ious = self.compute_iou(self.confmat)

        if isinstance(self.TP[0], np.ndarray):
            TP = np.asarray(self.TP).sum(axis=0)
            TN = np.asarray(self.TN).sum(axis=0)
            FP = np.asarray(self.FP).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                precision = TP / (TP + FP)
                recall = TP / (TP + TN)
            precision = np.nan_to_num(precision)
            recall = np.nan_to_num(recall)
        else:
            TP = self.TP
            TN = self.TN
            FP = self.FP
            precision = TP / (TP + FP)
            recall = TP / (TP + TN)

        return ious, precision, recall

    def summary(
        self, logger=None, save_to_file=None, previous_result=None
    ) -> float:
        ious, precision, recall = self.compute()

        if logger:
            logger.info(f"num instance: {self.num_instance}")
        else:
            print(f"num instance: {self.num_instance}")

        is_tensor_input = False
        if isinstance(ious, np.ndarray):
            make_tensor = np.asarray
        else:
            make_tensor = torch.tensor
            is_tensor_input = True

        ious = make_tensor(
            [iou for i, iou in enumerate(ious) if i not in self.ignore_classes]
        )
        classes = [
            cls
            for i, cls in enumerate(self.classes)
            if i not in self.ignore_classes
        ]
        precision = make_tensor(
            [
                prec
                for i, prec in enumerate(precision)
                if i not in self.ignore_classes
            ]
        )
        recall = make_tensor(
            [
                rec
                for i, rec in enumerate(recall)
                if i not in self.ignore_classes
            ]
        )

        if save_to_file:
            result = {
                "classes": classes,
                "ious": ious.numpy().tolist(),
                "precision": precision.numpy().tolist(),
                "recall": recall.numpy().tolist(),
            }
            with open(save_to_file, "w") as fid:
                json.dump(result, fid, indent=2)
                print(f"save eval result to {save_to_file}")

        self.print_result(
            classes,
            ious,
            precision,
            recall,
            logger=logger,
            previous_result=previous_result,
            classes_info=self.classes_info,
        )
        return ious.mean().item() if is_tensor_input else ious.mean()

    @staticmethod
    def print_result(
        classes,
        ious,
        precision,
        recall,
        logger=None,
        previous_result=None,
        classes_info=None,
    ):
        is_tensor_input = TORCH_INSTALLED and isinstance(ious, torch.Tensor)

        prev_result = {}
        if previous_result and os.path.exists(previous_result):
            init()
            with open(previous_result, "r") as fid:
                data = json.load(fid)
                for idx, cls in enumerate(data["classes"]):
                    prev_result[cls] = {
                        "iou": data["ious"][idx],
                        "prec": data["precision"][idx],
                        "rec": data["recall"][idx],
                    }

        def _color_cell(value, ref_value):
            value_str = f"{value:.3f},{ref_value:.3f}"
            if ref_value >= 0.01:  # good
                return f"{Back.GREEN}{value_str}{Back.RESET}"
            elif ref_value <= -0.01:  # bad
                return f"{Back.RED}{value_str}{Back.RESET}"
            else:  # keep same
                return f"{value_str}"

        table_data = []
        for cls, iou, prec, rec in zip(classes, ious, precision, recall):
            if classes_info:
                cls += f"({classes_info[cls]})"
            if is_tensor_input:
                iou = iou.item()
                prec = prec.item()
                rec = rec.item()

            if prev_result:
                diff_iou = iou - prev_result[cls]["iou"]
                diff_prec = prec - prev_result[cls]["prec"]
                diff_rec = rec - prev_result[cls]["rec"]
                iou_str = _color_cell(iou, diff_iou)
                prec_str = _color_cell(prec, diff_prec)
                rec_str = _color_cell(rec, diff_rec)
                table_data.append((cls, iou_str, prec_str, rec_str))
            else:
                table_data.append((cls, str(iou), str(prec), str(rec)))

        if is_tensor_input:
            table_data.append(
                (
                    "Mean",
                    str(ious.mean().item()),
                    str(precision.mean().item()),
                    str(recall.mean().item()),
                )
            )
        else:
            table_data.append(
                (
                    "Mean",
                    f"{ious.mean():.3f}",
                    f"{precision.mean():.3f}",
                    f"{recall.mean():.3f}",
                )
            )

        if logger:
            logger.info(
                "\n"
                + tabulate(
                    table_data,
                    headers=["Label", "IoU", "Precision", "Recall"],
                    tablefmt="fancy_grid",
                )
            )
        else:
            print(
                "\n"
                + tabulate(
                    table_data,
                    headers=["Label", "IoU", "Precision", "Recall"],
                    tablefmt="fancy_grid",
                )
            )


@otn_manager.NODE.register(name="semseg_eval")
def semseg_eval(
    pred_file: str = None,
    gt_file: str = None,
    output: str = None,
    vis_output: str = None,
    undistort_mask_file: str = None,
    ignore_classes: List[int] = (
        0,
        6,
        7,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    ),
    alias_to_pred: bool = False,
    subtract_one_in_pred: bool = False,
    previous_result: str = None,
    cache: bool = False,
):
    """Evaluate semseg result."""

    if cache:
        if os.path.exists(output):
            with open(output, "r") as fid:
                data = json.load(fid)
            print(f"using cache: {output}")
            SemSegMetric.print_result(
                data["classes"],
                np.asarray(data["ious"]),
                np.asarray(data["precision"]),
                np.asarray(data["recall"]),
                previous_result=previous_result,
            )
            return output
    else:
        if os.path.exists(output):
            shutil.rmtree(output)

    if not TORCHMETRICS_INSTALLED:
        raise RuntimeError("torchmetrics is required")

    assert pred_file.endswith(".json")
    with no_print():
        import pdebug.piata.coco

        pred_roidb = Input(pred_file, name="coco").get_roidb(
            as_dict=True, keepname=True
        )

    if gt_file.endswith(".txt"):
        gt_files = {item[0]: item for item in np.loadtxt(gt_file, dtype="str")}
        gt_type = "simpletxt"
    else:
        # load from parquet
        gt_files = Input(
            gt_file, name="cruise_rgbd_semseg", batch_size=1
        ).get_reader()
        gt_type = "parquet"

    if not alias_to_pred:
        assert len(pred_roidb) == len(
            gt_files
        ), f"{len(pred_roidb)} != {len(gt_files)}"

    if gt_type == "simpletxt" and len(gt_files) > len(pred_roidb):
        print(
            f"gt files num({len(gt_files)}) > pred files({len(pred_roidb)}), reduce to pred files"
        )
        gt_files = {k: gt_files[k] for k in gt_files if k in pred_roidb.keys()}

    classes = load_categories(name="semantic_pcd", simple=True)
    colors = load_colors(name="semantic_pcd", as_tuple=True)

    if vis_output and os.path.exists(vis_output):
        shutil.rmtree(vis_output)

    if undistort_mask_file:
        crop_mask = CropMask(undistort_mask_file)

    summary_txt = os.path.join(os.path.dirname(gt_file), "summary.txt")
    if os.path.exists(summary_txt):
        summary = np.loadtxt(summary_txt, dtype="str", delimiter=",")[:, 1:]
        classes_info = {s[0].split("|")[1].strip(): int(s[1]) for s in summary}
    else:
        classes_info = None
    metric = SemSegMetric(
        classes=classes,
        ignore_classes=ignore_classes,
        classes_info=classes_info,
    )

    t = tqdm.tqdm(
        total=len(pred_roidb), desc=f"eval {os.path.basename(pred_file)}"
    )
    for image_name in gt_files:
        t.update()

        if gt_type == "parquet":
            data_dict = image_name
            image_name = data_dict["image_name"][0]
            gt = data_dict["label"][0].numpy()
        else:
            gt = cv2.imread(gt_files[image_name][1], cv2.IMREAD_UNCHANGED)

        if alias_to_pred and image_name not in pred_roidb:
            continue

        roi = pred_roidb[image_name]
        pred = merge_polygon(roi)

        if subtract_one_in_pred:
            pred[pred > 0] = pred[pred > 0] - 1  # trick

        # convert 17 to 34 (others, which is last category in classes)
        pred[pred == 17] = len(classes) - 1

        if undistort_mask_file:
            pred = crop_mask(pred)
            gt = crop_mask(gt)

        res = metric.update_inner(pred, gt, num_examples=1)

        if vis_output:
            if gt_type == "parquet":
                image = data_dict["image"][0].numpy()
            else:
                image = cv2.imread(gt_files[image_name][0])

            vis_pred = draw.semseg(pred, image, colors=colors, classes=classes)
            vis_pred = cv2.putText(
                vis_pred,
                "pred",
                (100, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            vis_gt = draw.semseg(gt, image, colors=colors, classes=classes)
            vis_gt = cv2.putText(
                vis_gt,
                "gt",
                (100, 100),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            line_i = 0
            for cat_idx, (cat, iou, tp, tn, fp) in enumerate(
                zip(
                    classes,
                    res["ious"],
                    res["ap"]["TP"],
                    res["ap"]["TN"],
                    res["ap"]["FP"],
                )
            ):
                if cat_idx == 0:
                    continue  # skip VOID class
                # if iou == 0: continue
                if cat_idx not in np.unique(gt):
                    continue
                image = cv2.putText(
                    image,
                    f"{cat}: {iou:.4f} tp({tp}) tn({tn}) fp({fp})",
                    (50, line_i * 50 + 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    colors[cat_idx],
                    2,
                )
                line_i += 1

            line_i = 0
            ap_colors = {
                "TP": (0, 210, 0),
                "TN": (0, 0, 255),
                "FP": (0, 0, 255),
            }
            right_padding = 300
            for ap_type, ap_value in res["ap"].items():
                image = cv2.putText(
                    image,
                    f"{ap_type}: {ap_value.sum()}",
                    (image.shape[1] - right_padding, line_i * 100 + 100),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    ap_colors[ap_type],
                    2,
                )
                line_i += 1

            vis_all = np.concatenate((image, vis_gt, vis_pred), axis=1)
            os.makedirs(vis_output, exist_ok=True)
            savename = os.path.join(vis_output, os.path.basename(image_name))
            cv2.imwrite(savename, vis_all)

        # if t.n > 10: break

    metric.summary(save_to_file=output, previous_result=previous_result)
    return output


if __name__ == "__main__":
    typer.run(semseg_eval)
