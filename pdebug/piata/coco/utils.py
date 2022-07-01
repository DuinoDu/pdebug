import logging
import os
import random
from typing import Dict, List, Optional, Union

import json_tricks as json
import numpy as np

from ..registry import ROIDB_REGISTRY
from .coco import COCO

try:
    import commands
except Exception as e:
    import subprocess as commands


__all__ = ['save_to_cocoDt', 'save_to_cocoGt', 'coco_to_roidb',
           'compute_oks', 'lighten_json']


@ROIDB_REGISTRY.register(name='coco')
def coco_to_roidb(inputfile: str) -> List[Dict]:
    """Load coco json and convert to roidb.

    Args:
        inputfile: str

    Returns:
        roidb: output roidb list
    """
    with open(inputfile, 'r') as fid:
        data = json.load(fid)
    if isinstance(data, dict):
        return _coco_gt_to_roidb(data, inputfile)
    elif isinstance(data, list):
        return _coco_dt_to_roidb(data)
    else:
        raise ValueError("Bad coco json")


def box_transform(box):
    """
    x1,y1,x2,y2 -> x1,y1,w,h
    """
    _box = [0, 0, 0, 0]
    _box[0] = box[0]
    _box[1] = box[1]
    _box[2] = box[2] - box[0] + 1
    _box[3] = box[3] - box[1] + 1
    return _box


def fetch_imgsize(imgfile):
    """
    fetch image size using image-magick
    """
    if not os.path.exists(imgfile):
        raise ValueError("%s not exixt" % imgfile)
    cmd = "identify -format '%%h:%%w' %s" % imgfile
    (status, output) = commands.getstatusoutput(cmd)
    output = output.split('\n')[0]
    return int(output.split(':')[0]), int(output.split(':')[1])


def roidb_to_cocoGt(roidb, save_json_path,
        base_ann_Id=1000, base_img_Id=0,
        supercategories=[u'person'], names=[u'person'], ids=[1],
        videofile=None):
    """
    convert roidb to cocoGt format
    """
    logging.warnings('Deprecated, please use COCOWriter instead.')

    ann_index = base_ann_Id
    img_index = base_img_Id

    # add category
    js_dict = dict()
    js_dict[u'categories'] = []
    js_dict[u'images'] = []
    js_dict[u'annotations'] = []
    if videofile:
        js_dict[u'videos'] = [{'id':1, 'file_name': videofile}]

    for i in range(len(names)):
        rec = dict()
        rec[u'supercategory'] = supercategories[i]
        rec[u'id'] = ids[i]
        rec[u'name'] = names[i]
        # you may have other parameters
        # rec.update()
        if rec[u'name'] == 'person':
            rec[u'keypoints'] = [u'nose', u'left_eye', u'right_eye', u'left_ear', u'right_ear', u'left_shoulder', u'right_shoulder', u'left_elbow', u'right_elbow', u'left_wrist', u'right_wrist', u'left_hip', u'right_hip', u'left_knee', u'right_knee', u'left_ankle', u'right_ankle']
            rec[u'skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        js_dict[u'categories'].append(rec)

    roidb = sorted(roidb, key=lambda x: x['image_name'])

    for epoch, roi in enumerate(roidb):
        # epoch += 1
        if epoch % 1000 == 0:
            logging.info('%d / %d' % (epoch, len(roidb)))

        image_name = os.path.split(roi['image_name'])[1]
        image_info = dict()
        image_info[u'file_name'] = image_name
        try:
            image_info[u'id'] = int(os.path.splitext(os.path.basename(roi['image_name']))[0])
        except ValueError as e:
            image_info[u'id'] = img_index
        if 'image_height' not in roi:
            try:
                roi['image_height'], roi['image_width'] = fetch_imgsize(roi['image_name'])
            except ValueError as e:
                roi['image_height'], roi['image_width'] = -1, -1
        image_info[u'image_height'] = roi['image_height']
        image_info[u'image_width'] = roi['image_width']
        if videofile:
            image_info[u'video_id'] = 1
            image_info[u'prev_image_id'] = img_index - 1
            image_info[u'next_image_id'] = img_index + 1
            # frame_id is 1-index
            image_info[u'frame_id'] = img_index + 1 if base_img_Id == 0 else img_index
        js_dict[u'images'].append(image_info)

        if 'boxes' in roi:
            if 'gt_classes' in roi:
                valid_index = np.where(roi['gt_classes'] == 1)[0]
            else:
                valid_index = range(len(roi['boxes']))
            for ix in valid_index:
                ann = dict()
                ann[u'bbox'] = box_transform(roi['boxes'][ix].tolist())
                if 'area' in roi:
                    ann[u'area'] = float(roi['area'][ix])
                else:
                    ann[u'area'] = ann[u'bbox'][2]*ann[u'bbox'][3]
                if 'iscrowd' in roi:
                    ann[u'iscrowd'] = int(roi['iscrowd'][ix])
                else:
                    ann[u'iscrowd'] = 0
                ann[u'category_id'] = 1
                ann[u'id'] = ann_index
                ann_index += 1
                try:
                    # only coco dataset using number as img name, which is also image_id
                    ann[u'image_id'] = int(os.path.splitext(os.path.basename(roi['image_name']))[0])
                except ValueError as e:
                    ann[u'image_id'] = img_index

                ann[u'segmentation'] = []

                if 'keypoints' in roi:
                    if ann[u'category_id'] == 1:
                        scores = set(roi['keypoints'][ix][2::3].tolist())
                        key_points = roi['keypoints'][ix].tolist()
                        num_keypoint = int(0)
                        for ixx in range(2, len(key_points), 3):
                            if int(key_points[ixx]) == 3 or int(key_points[ixx]) == 0:
                                key_points[ixx] = int(0)
                            else:
                                num_keypoint += 1
                        ann[u'keypoints'] = key_points
                        ann[u'num_keypoints'] = num_keypoint
                    else:
                        key_points = [int(0)]*51
                        num_keypoint = int(0)
                        ann[u'keypoints'] = key_points
                        ann[u'num_keypoints'] = num_keypoint
                js_dict[u'annotations'].append(ann)
        img_index += 1

    with open(save_json_path, 'w') as new_json:
        json.dump(js_dict, new_json)


def get_kps_score(kps, bbox_score):
    """
    kps: list, [51, ]
    bbox: list, [5, ]
    """
    kps = np.array(kps)
    kps_each_scores = kps[2::3].copy()
    kps_each_scores[kps_each_scores < 0.1] = 0
    kps_scores_sum = kps_each_scores.sum(axis=0)
    kps_each_scores[kps_each_scores > 0] = 1
    kps_scores_count = kps_each_scores.sum(axis=0)
    #kps_scores_count[kps_scores_count == 0] = 1
    if kps_scores_count == 0:
        kps_score = 0.0
    else:
        kps_score = kps_scores_sum / kps_scores_count
    kps_score = (kps_score + bbox_score) / 2
    return kps_score


def roidb_to_cocoDt(roidb, save_json_path, base_img_Id=0, no_bbox=False):
    """
    convert roidb to cocoDt format
    """
    results = []
    roidb = sorted(roidb, key=lambda x: x['image_name'])
    img_index = base_img_Id
    for roi in roidb:
        if img_index % 1000 == 0:
            logging.info('%d / %d' % (img_index, len(roidb)))
        try:
            # only coco dataset using number as img name, which is also image_id
            image_id = int(os.path.splitext(os.path.basename(roi['image_name']))[0])
        except ValueError as e:
            image_id = img_index

        if 'gt_classes' in roi:
            valid_index = np.where(roi['gt_classes'] == 1)[0]
        elif 'boxes' in roi:
            valid_index = range(len(roi['boxes']))
        elif 'keypoints' in roi:
            valid_index = range(len(roi['keypoints']))
        for i in valid_index:

            if 'scores' in roi:
                bbox_score = roi['scores'][i]
            elif 'boxes_score' in roi:
                bbox_score = roi['boxes_score'][i]
            elif roi['boxes'][i].shape[0]==5:
                bbox_score = roi['boxes'][i][-1]
            else:
                bbox_score = 1.0

            res = dict()
            res['image_id'] = image_id
            res['category_id'] = 1

            if not no_bbox and 'boxes' in roi:
                x = roi['boxes'][i][0]
                y = roi['boxes'][i][1]
                w = roi['boxes'][i][2] - roi['boxes'][i][0]
                h = roi['boxes'][i][3] - roi['boxes'][i][1]
                res['bbox'] = [float(v) for v in [x, y, w, h]]

            if 'keypoints' in roi:
                res['keypoints'] = [round(float(v), 3) for v in roi['keypoints'][i].tolist()]
                if 'scores' in roi:
                    res['score'] = float(bbox_score)
                else:
                    res['score'] = float(get_kps_score(roi['keypoints'][i], bbox_score))
            else:
                res['score'] = float(bbox_score)

            results.append(res)
        img_index += 1
    with open(save_json_path, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


SUPPORTED_IMAGE_KEYS = ['calib', 'timestamp', 'distCoeffs', 'ignore_mask']


def _coco_gt_to_roidb(data, jsonfile):
    """
    Convert coco-gt to roidb.
    """
    coco = COCO(jsonfile)

    roidb = dict()
    category_id = 1

    for ind, image_id in enumerate(coco.imgs):
        if ind % 10000 == 0:
            logging.info('%d / %d' % (ind, len(coco.imgs)))
        image = coco.imgs[image_id]
        imgname = image['file_name']
        roidb[imgname] = dict()
        roidb[imgname]['image_name'] = imgname
        if 'id' in image:
            roidb[imgname]['image_id'] = image['id']
        if 'height' in image:
            roidb[imgname]['image_height'] = image['height']
        if 'width' in image:
            roidb[imgname]['image_width'] = image['width']
        for key in SUPPORTED_IMAGE_KEYS:
            if key in image:
                roidb[imgname][key] = image[key]

        annoIds = coco.getAnnIds(imgIds=image_id)
        for annoId in annoIds:
            res = coco.anns[annoId]
            res.pop('id')
            res.pop('image_id')
            if 'height' in res: res.pop('height')
            if 'width' in res: res.pop('width')

            if 'bbox' in res:
                bbox = [res['bbox'][0], res['bbox'][1],
                        res['bbox'][0]+res['bbox'][2],
                        res['bbox'][1]+res['bbox'][3]]
                res['bbox'] = bbox
            for res_k in res:
                if res_k not in roidb[imgname]:
                    roidb[imgname][res_k] = [res[res_k]]
                elif not isinstance(roidb[imgname], list):
                    # skip duplicate keys in annotations and images
                    continue
                else:
                    roidb[imgname][res_k].append(res[res_k])
    for k in roidb:
        if 'bbox' in roidb[k]:
            roidb[k]['boxes'] = np.asarray(roidb[k]['bbox'], dtype=np.float32)
            roidb[k].pop('bbox')
        if 'keypoints' in roidb[k]:
            roidb[k]['keypoints'] = np.asarray(roidb[k]['keypoints'], dtype=np.float32)
    return [roidb[k] for k in sorted(roidb.keys())]


def _coco_dt_to_roidb(data):
    """
    Convert coco-dt to roidb.
    """
    roidb = dict()
    category_id = 1
    for ind, res in enumerate(data):
        if ind % 10000 == 0:
            logging.info('%d / %d' % (ind, len(data)))
        try:
            imgname = "%012d.jpg" % res['image_id']
        except Exception as e:
            print('only support coco dataset.')
            raise e

        assert 'category_id' in res
        if res['category_id'] != category_id:
            continue

        score = res['score']
        if 'bbox' in res:
            bbox = [res['bbox'][0], res['bbox'][1],
                    res['bbox'][0]+res['bbox'][2],
                    res['bbox'][1]+res['bbox'][3]]

        if imgname not in roidb:
            roidb[imgname] = dict()
            roidb[imgname]['image_name'] = imgname
            roidb[imgname]['scores'] = []
            if 'bbox' in res:
                roidb[imgname]['boxes'] = []
            if 'keypoints' in res:
                roidb[imgname]['keypoints'] = []
            if 'segmentation' in res:
                roidb[imgname]['segmentation'] = []
        roidb[imgname]['scores'].append(score)
        if 'bbox' in res:
            roidb[imgname]['boxes'].append(bbox)
        if 'keypoints' in res:
            roidb[imgname]['keypoints'].append(res['keypoints'])
        if 'segmentation' in res:
            roidb[imgname]['segmentation'].append(res['segmentation'])

    for k in roidb:
        roidb[k]['scores'] = np.asarray(roidb[k]['scores'], dtype=np.float32)
        if 'bbox' in res:
            roidb[k]['boxes'] = np.asarray(roidb[k]['boxes'], dtype=np.float32)
        if 'keypoints' in res:
            roidb[k]['keypoints'] = np.asarray(roidb[k]['keypoints'], dtype=np.float32)

    return [roidb[k] for k in sorted(roidb.keys())]


def compute_oks(gt_kps, dt_kps, gt_area, return_each_oks=False):
    """Compute oks between gt kps and dt kps.

    Parameters
    ----------
    gt_kps : list or array, [51,]
        gt keypoints
    dt_kps : list or array, [51,]
        dt keypoints
    gt_area: float
        gt area
    """

    sigmas = np.array([.26, .25, .25, .35, .35,
                       .79, .79, .72, .72, .62,
                       .62, 1.07, 1.07, .87, .87,
                       .89, .89]) / 10.0
    vars = (sigmas * 2)**2
    g = np.asarray(gt_kps)
    xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
    k1 = np.count_nonzero(vg > 0)
    d = np.asarray(dt_kps)
    xd = d[0::3]; yd = d[1::3]

    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / (gt_area+np.spacing(1)) / 2
        ious = np.exp(-e)
        ious[vg == 0] = -1
        e=e[vg > 0]
        ious_mean = np.sum(np.exp(-e)) / e.shape[0]
    else:
        ious = -1
        ious_mean = -1

    if return_each_oks:
        return ious_mean, ious
    else:
        return ious_mean


def compute_iou(gt_bbox, dt_bbox):
    """Compute iou between gt bbox and dt bbox.

    Parameters
    ----------
    gt_bbox : list or array, [4,], x1y1x2y2
        gt bbox
    dt_bbox : list or array, [4,], x1y1x2y2
        dt bbox
    """
    from pycocotools import mask as maskUtils
    g = [[gt_bbox[0], gt_bbox[1], gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]]]
    d = [[dt_bbox[0], dt_bbox[1], dt_bbox[2]-dt_bbox[0], dt_bbox[3]-dt_bbox[1]]]
    iscrowd = [0]
    iou = maskUtils.iou(g, d, iscrowd)
    iou = iou[0, 0]
    return iou


def lighten_json(
    inputfile: str,
    shuffle: bool = True,
    amount: Union[int, float] = 100,
    output_file: Optional[str] = None,
) -> List:
    """
    lighten coco json file.
    """
    with open(inputfile, 'r') as fid:
        data = json.load(fid)
    assert isinstance(data, dict), "inputfile should be coco-format json"
    assert 'images' in data, "inputfile should be coco-format json"
    assert 'annotations' in data, "inputfile should be coco-format json"

    length = len(data['annotations'])
    if shuffle:
        random.shuffle(data['annotations'])
    if isinstance(amount, float):
        assert 0 <= amount <= 1.0
        choosed_length = int(length * amount)
    elif isinstance(amount, int):
        assert 0 <= amount <= length
        choosed_length = amount
    else:
        raise ValueError
    data['annotations'] = data['annotations'][:choosed_length]

    if 'image_id' in data['annotations'][0] and 'images' in data:
        image_ids = set([anno['image_id'] for anno in data['annotations']])
        data['images'] = [image for image in data['images'] if image['id'] in image_ids]

    if output_file is not None:
        with open(output_file, 'w') as fid:
            json.dump(data, fid)
        print(f"saved to {output_file}")
    return data
