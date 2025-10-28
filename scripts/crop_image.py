#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crop roi from imgdir, using roidb and condition.
feet_pred:score:>:0.5
"""

from __future__ import print_function
import argparse
import os

import cPickle as pickle
import cv2
import numpy as np
import vispy1


def draw_legend(img, legend_str, color, x=10, y=10, font_size=2):
    img = cv2.putText(img, legend_str, (x+10, y), cv2.FONT_HERSHEY_COMPLEX, font_size, color, 2)
    return img


def resize(img, minsize):
    cur_minsize = min(img.shape[:2])
    factor = cur_minsize * 1.0 / minsize
    img = cv2.resize(img, (int(img.shape[1]/factor), int(img.shape[0]/factor)))
    return img


def valid(item, condition):
    assert 'score' == condition[0], 'only support score now'
    scores = item[2::3]
    threshold = float(condition[2])
    if '>' == condition[1]:
        valid_fn = lambda x: np.mean(scores) > threshold
    elif '<' == condition[1]:
        valid_fn = lambda x: np.mean(scores) < threshold
    else:
        raise ValueError, "Not support %s" % condition[1]
    return valid_fn(scores)


def render_crop_and_save(_img, roi, roi_ind, keyname, padding, dst_imgfile, args):
    img = _img.copy()
    h, w = img.shape[:2]
    box = roi['boxes'][roi_ind]

    if 'feet' not in keyname:
        raise NotImplementedError

    pred_feet = roi[keyname][roi_ind][np.newaxis, :]
    feet = roi['feet'][roi_ind][np.newaxis, :]
    color = {'pred':(0, 255,255), 'gt':(0,255,0)}

    if 'kps_pred' in roi and not args.nokps:
        img = vispy1.kps(img, roi['kps_pred'], vis_threshold=0.5)
    if 'head' in roi and not args.nohead:
        img = vispy1.head(img, roi['head'], color=color['gt'])
    img = vispy1.feet(img, pred_feet, color=color['pred'], vis_threshold=0.5)
    img = vispy1.feet(img, feet, color=color['gt'], vis_threshold=0.5)
    img = vispy1.boxes(img, box[np.newaxis, :])

    target_x1 = min(pred_feet[:, 0::3].min(), feet[:, 0::3].min())
    target_y1 = min(pred_feet[:, 1::3].min(), feet[:, 1::3].min())
    target_x2 = max(pred_feet[:, 0::3].max(), feet[:, 0::3].max())
    target_y2 = max(pred_feet[:, 1::3].max(), feet[:, 1::3].max())

    x1 = min(box[0], target_x1)
    y1 = min(box[1], target_y1)
    x2 = max(box[2], target_x2)
    y2 = max(box[3], target_y2)

    x1 = int(max(x1 - padding, 0))
    y1 = int(max(y1 - padding, 0))
    x2 = int(min(x2 + padding, w))
    y2 = int(min(y2 + padding, h))

    roi = img[y1:y2, x1:x2]
    roi = resize(roi, minsize=300)
    roi = draw_legend(roi, 'pred', color['pred'], x=10, y=30, font_size=1)
    roi = draw_legend(roi, 'gt', color['gt'], x=10, y=60, font_size=1)
    cv2.imwrite(dst_imgfile, roi)


def main(args):
    if args.output is None:
        args.output = 'output'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print('loading %s' % args.roidb)
    with open(args.roidb, 'r') as fid:
        roidb = pickle.load(fid)

    if args.condition is None:
        args.condition = "pred_feet:score:>:0.0"

    assert len(args.condition.split(':')) == 4
    keyname = args.condition.split(':')[0]
    target = args.condition.split(':')[1:]
    assert 'boxes' in roidb[0]
    assert keyname in roidb[0]

    for ind, roi in enumerate(roidb):
        if ind % 100 == 0:
            print('%d / %d' % (ind, len(roidb)))
        imgname = os.path.basename(roi['image'])
        imgfile = os.path.join(args.imgdir, imgname)
        img = cv2.imread(imgfile)

        num_box = roi['boxes'].shape[0]
        assert roi[keyname].shape[0] == num_box
        for i in range(num_box):
            box = roi['boxes'][i]
            item = roi[keyname][i]
            if valid(item, target):
                new_imgname = os.path.splitext(imgname)[0] + '_%03d'%i + os.path.splitext(imgname)[1]
                new_imgfile = os.path.join(args.output, new_imgname)
                render_crop_and_save(img, roi, i, keyname, args.padding, new_imgfile, args)
    print('saved to %s' % args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roidb', default=None, type=str, help='', required=True)
    parser.add_argument('--imgdir', default=None, type=str, help='', required=True)
    parser.add_argument('--condition', default=None, type=str, help='crop condition, use double quote')
    parser.add_argument('--padding', default=30, type=int, help='padding when crop')
    parser.add_argument('--nokps', dest='nokps', action='store_true', help='description')
    parser.add_argument('--nohead', dest='nohead', action='store_true', help='description')
    parser.add_argument('--output', default=None, type=str, help='output')
    args = parser.parse_args()
    main(args)
