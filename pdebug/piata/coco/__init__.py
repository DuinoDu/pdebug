"""
Data interface for mscoco format.
"""

from .coco import COCO
from .utils import coco_to_roidb, compute_iou, compute_oks, lighten_json
from .writer import COCOWriter, save_to_cocoDt, save_to_cocoGt
