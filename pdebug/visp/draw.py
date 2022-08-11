# -*- coding: utf-8 -*-
"""
simple vis utilities
"""
import math
import os
import sys
from random import random
from typing import List

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import PIL.Image as pil
from mpl_toolkits.mplot3d import Axes3D

from .colormap import Colormap

try:
    from pycocotools import mask as maskUtils
except (ImportError, ValueError) as e:
    maskUtils = None

try:
    import torch
except ImportError as e:
    torch = None


__all__ = ['boxes', 'kps', 'skeleton', 'skeleton_3d', 'skeleton_car',
           'skeleton_hand', 'lmks', 'lmks68', 'mask',
           'feet_center', 'feet', 'head', 'pose',
           'quality', 'bodytype', 'luid', 'flag', 'orientation',
           'skeleton_score', 'skeleton_dog',
           'semseg', 'depth', 'points']


KPS_SKELETON = np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                         [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]) - 1

KPS_SKELETON_HUMAN36M = np.array([[1, 2], [2, 3], [3, 7], [7, 4], [4, 5], [5, 6],
                                  [7, 8], [8, 9],
                                  [9, 10], [11, 12], [12, 13], [13, 9], [9, 14], [14, 15], [15, 16]]) - 1

SKELETON_COLOR = [[255, 85, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255]]

KPS_SKELETON_CARFUSION = np.array([[1, 2], [2, 3], [3, 4], [4, 1],
                                   [1, 5], [2, 6], [3, 7], [4, 8],
                                   [5, 6], [6, 7], [7, 8], [8, 5],
                                   [5, 10], [6, 11], [7, 12], [8, 13],
                                   [10, 11],[11, 12], [12, 13], [13, 10]]) - 1

SKELETON_COLOR_CARFUSION = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255],
                  ]

KPS_SKELETON_DOG = np.array([[ 0,  1], [ 1,  2], [ 3,  4], [ 4,  5],
                             [ 6,  7], [ 7,  8], [ 9, 10], [10, 11],
                             [12, 13], [14, 15], [15, 16], [16, 14],
                             [14, 18], [15, 19], [16, 17]])
SKELETON_COLOR_DOG = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255],
                            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 255],
                           ]



HAND_SKELETON_CMU = np.array([[0,1], [1,2], [2,3], [3,4], [0,5], [5,6], [6,7], [7,8], [0,9], [9,10], [10,11], [11,12],
                              [0,13], [13,14], [14,15], [15,16], [0,17], [17,18], [18,19], [19,20]])

HAND_SKELETON_COLOR_CMU = [[0, 0, 255], [0, 76, 255], [0, 153, 255], [0, 229, 255], [0, 255, 203], [0, 255, 127],
                           [0, 255, 51], [25, 255, 0], [102, 255, 0], [178, 255, 0], [255, 255, 0], [255, 178, 0],
                           [255, 102, 0], [255, 25, 0], [255, 0, 50], [255, 0, 127], [255, 0, 204], [229, 0, 255],
                           [152, 0, 255], [76, 0, 255]]


RANDOM1_FIXED = [random() for _ in range(65535)]
RANDOM2_FIXED = [random() for _ in range(65535)]
RANDOM3_FIXED = [random() for _ in range(65535)]

RANDOM1_FIXED_SMALL = [random() for _ in range(255)]
RANDOM2_FIXED_SMALL = [random() for _ in range(255)]
RANDOM3_FIXED_SMALL = [random() for _ in range(255)]


SKELETON_TEMPLATE_IMGFILE = os.path.join(os.path.dirname(__file__), "res/body_bar.jpg")

SKELETON_TEMPLATE = np.array([
    [198, 68+10], [206, 56-30], [188, 56-30], [220+30, 70], [176-30, 70],
    [256, 136], [142, 134], [264, 220], [128, 222],
    [290, 308], [108, 302], [238, 258], [158, 260], [228, 410], [168, 410], [220, 530], [172, 528]])


def gen_colors(num, fixed=False, small=False):
    colors = []
    for i in range(num):
        if fixed:
            if small:
                random_num1 = RANDOM1_FIXED_SMALL[i]
                random_num2 = RANDOM2_FIXED_SMALL[i]
                random_num3 = RANDOM3_FIXED_SMALL[i]
            else:
                random_num1 = RANDOM1_FIXED[i]
                random_num2 = RANDOM2_FIXED[i]
                random_num3 = RANDOM3_FIXED[i]
        else:
            random_num1 = random()
            random_num2 = random()
            random_num3 = random()
        color = (int(random_num1 * 255),
                 int(random_num2 * 255),
                 int(random_num3 * 255))
        colors.append(color)
    return colors


def to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, pil.Image):
        return np.array(x)[:,:,::-1].copy()
    elif isinstance(x, list):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Unknown data type: {type(x)}")


def to_tensor(x):
    assert torch is not None
    if x is None:
        return x
    if isinstance(x, np.ndarray):
        return torch.tensor(x)


'''
https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
'''
def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)
    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1
def drawpoly(img,pts,color,thickness=1,style='dotted'):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)
def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    drawpoly(img,pts,color,thickness,style)


def boxes(_img, boxes, mask=False, dot=False, thickness=2, colors=None,
          color=(0,255,255), trackid=None, threshold=None,
          num_classes=None):
    """Draw boxes on image

    Parameters
    ----------
    img : cv2 image
    boxes : np.ndarray, [N, 4]
    mask : draw mask in box, default is False

    Returns
    -------
    img

    """
    img = _img.copy()
    boxes = to_numpy(boxes)
    if boxes.ndim == 1:
        boxes = np.array([boxes])

    if threshold is not None and boxes.shape[1] >= 5:
        scores = boxes[:, 4]
        valid = scores > threshold
        boxes = boxes[valid]

    if num_classes:
        assert boxes.shape[1] >= 5
        class_colors = gen_colors(num_classes, fixed=True, small=True)

    for ind, box in enumerate(boxes):
        if colors is not None:
            assert len(colors) == len(boxes)
            color = colors[ind]
        elif num_classes:
            class_id = box[-1].astype(int)
            assert class_id < num_classes
            color = class_colors[class_id]
        if dot:
            drawrect(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=thickness)
        else:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=thickness)
        if trackid is not None:
            cv2.putText(img, str(trackid[ind]), (int(box[0]+2), int(box[1]+25)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    return img


def kps(img, kps, skeleton=True, vis_threshold=0.2, custom_skeleton_color=None, vis_occlued=False, drawpoint=False):
    print('Please use skeleton instead.')
    sys.exit()


def keypoints(_img, kps, color=(0,255,255), colors=None,
              drawscore=False, vis_threshold=0.2, draworder=False):
    """
    common keypoints visualization.

    img: np.array
    kps: [N, num_kps*3]
    """
    img = _img.copy()
    if kps.ndim == 1:
        kps = np.array([kps])
    num_kps = int(kps.shape[1] / 3)
    isvis = lambda x: x > vis_threshold
    for ind, k in enumerate(kps):
        if colors is not None:
            color = colors[ind]
        for pt_ind, pt in enumerate(k.reshape(num_kps, 3)):
            if isvis(pt[2]):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1)
            if drawscore:
                cv2.putText(img, "%.2f" % pt[2], (int(pt[0]+10), int(pt[1]+10)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
            if draworder:
                cv2.putText(img, "%d" % pt_ind, (int(pt[0]+10), int(pt[1]+10)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
    return img


def points(_img, points, color=(0,255,255), colors=None):
    """
    common keypoints visualization.

    img: np.array
    points: [N, num_kps*2]
    """
    img = _img.copy()
    if points.ndim == 1:
        points = np.array([points])
    if points.shape[1] == 2:
        points = points.flatten()[None, :]

    num_kps = int(points.shape[1] / 2)
    for ind, k in enumerate(points):
        if colors is not None:
            color = colors[ind]
        for pt_ind, pt in enumerate(k.reshape(num_kps, 2)):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, color, -1)
    return img


def skeleton(_img, kps, num_kps=17, draw_skeleton=True, vis_threshold=0.2,
        custom_skeleton=KPS_SKELETON,
        custom_skeleton_color=SKELETON_COLOR,
        vis_occlued=False, vis_occlusion=False, only_occlusion=False,
        drawpoint=False, pointcolor=(0,255,255),
        drawscore=False):
    """Draw skeleton on image

    Parameters
    ----------
    img : cv2 image
    kps : np.ndarray, [N, num_kps*3]

    Returns
    -------
    img
    """
    img = _img.copy()
    if kps.ndim == 1:
        kps = np.array([kps])
    num_kps_cur = kps.shape[1] // 3
    if num_kps_cur != num_kps:
        draw_skeleton = False
        drawpoint = True
        num_kps = num_kps_cur
    if vis_occlued:
        print('please use draw.skeleton(..., vis_occlusion=True)')
        vis_occlusion = vis_occlued

    if np.asarray(custom_skeleton_color).ndim == 1:
        custom_skeleton_color = [custom_skeleton_color for _ in range(len(SKELETON_COLOR))]

    for k in kps:
        if k.ndim == 1:
            k = k[:num_kps*3].reshape(num_kps, 3)
        if set(k[:, -1].astype(np.float32).tolist()).issubset(set([0.,1.,2.,3.])):
            if vis_occlusion:
                isvalid = lambda x: x == 2.0 or x == 1.0
            elif only_occlusion:
                isvalid = lambda x: x == 1.0
            else:
                isvalid = lambda x: x == 2.0
            isvis = lambda x: x == 2.0
            isocc = lambda x: x == 1.0
        else:
            isvalid = lambda x: x > vis_threshold
            isvis = isvalid
            isocc = lambda x: x <= vis_threshold

        if draw_skeleton:
            for j in range(custom_skeleton.shape[0]):
                p1 = custom_skeleton[j, 0]
                p2 = custom_skeleton[j, 1]
                x1 = int(k[p1, 0] + 0.5)
                y1 = int(k[p1, 1] + 0.5)
                x2 = int(k[p2, 0] + 0.5)
                y2 = int(k[p2, 1] + 0.5)
                if isvalid(k[p1, 2]) and isvalid(k[p2, 2]):
                    color = custom_skeleton_color[j % len(custom_skeleton_color)]
                    cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=2)
        if drawpoint:
            for pt in k:
                if isvis(pt[2]):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 4, pointcolor, -1)
                if isocc(pt[2]):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 4, pointcolor, 1)
        if drawscore:
            for pt in k:
                cv2.putText(img, "%.2f" % pt[2], (int(pt[0]+10), int(pt[1]+10)), cv2.FONT_HERSHEY_COMPLEX, 1, pointcolor, 1)
    return img


def hand_skeleton(*args, **kwargs):
    print('Please use draw.skeleton_hand instead of hand_skeleton.')
    skeleton_hand(*args, **kwargs)


def skeleton_hand(_img,
                  kps,
                  num_kps=21,
                  draw_skeleton=True,
                  vis_threshold=0.2,
                  custom_skeleton_color=HAND_SKELETON_COLOR_CMU,
                  custom_skeleton=HAND_SKELETON_CMU,
                  vis_occlued=False,
                  vis_occlusion=False,
                  only_occlusion=False,
                  drawpoint=False,
                  pointcolor=(0,0,255),
                  drawscore=False,
                  thickness=1):
    """Draw skeleton on image

    Parameters
    ----------
    img : cv2 image
    kps : np.ndarray, [N, num_kps*3]

    Returns
    -------
    img
    """
    img = _img.copy()
    if kps.ndim == 1:
        kps = np.array([kps])
    num_kps_cur = kps.shape[1] // 3
    if num_kps_cur != num_kps:
        draw_skeleton = False
        drawpoint = True
        num_kps = num_kps_cur
    if vis_occlued:
        print('please use draw.hand_skeleton(..., vis_occlusion=True)')
        vis_occlusion = vis_occlued

    for k in kps:
        if k.ndim == 1:
            k = k[:num_kps*3].reshape(num_kps, 3)
        if set(k[:, -1].astype(np.float32).tolist()).issubset(set([0.,1.,2.,3.])):
            if vis_occlusion:
                isvalid = lambda x: x == 2.0 or x == 1.0
            elif only_occlusion:
                isvalid = lambda x: x == 1.0
            else:
                isvalid = lambda x: x == 2.0
            isvis = lambda x: x == 2.0
            isocc = lambda x: x == 1.0
        else:
            isvalid = lambda x: x > vis_threshold
            isvis = isvalid
            isocc = lambda x: x <= vis_threshold

        if draw_skeleton:
            for j in range(custom_skeleton.shape[0]):
                p1 = custom_skeleton[j, 0]
                p2 = custom_skeleton[j, 1]
                x1 = int(k[p1, 0] + 0.5)
                y1 = int(k[p1, 1] + 0.5)
                x2 = int(k[p2, 0] + 0.5)
                y2 = int(k[p2, 1] + 0.5)
                if isvalid(k[p1, 2]) and isvalid(k[p2, 2]):
                    color = custom_skeleton_color[j % len(custom_skeleton_color)]
                    cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if drawpoint:
            for pt in k:
                if isvis(pt[2]):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 1, pointcolor, -1, cv2.LINE_AA)
                if isocc(pt[2]):
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 1, pointcolor, 1, cv2.LINE_AA)
        if drawscore:
            for pt in k:
                cv2.putText(img, "%.2f" % pt[2], (int(pt[0]+4), int(pt[1]+4)), cv2.FONT_HERSHEY_COMPLEX, 0.4, pointcolor, 1)
    return img


def skeleton_3d(_img, kps, depth, savedir, num_kps=16,
        draw_skeleton=True, vis_threshold=0.2,
        custom_skeleton_color=SKELETON_COLOR,
        custom_skeleton=KPS_SKELETON,
        vis_occlued=False, vis_occlusion=False, only_occlusion=False,
        drawpoint=False, pointcolor=(0,255,255),
        drawscore=False):
    """Draw skeleton 3d on image

    Parameters
    ----------
    img : cv2 image
    kps : np.ndarray, [N, num_kps*2]
    depth: np.ndarray, [N, num_kps]

    Returns
    -------
    img
    """
    num_kps = depth.shape[1]
    assert int(kps.shape[1]/3) == num_kps and num_kps == 16

    fig = plt.figure(figsize=(20, 20))
    ax_2d = fig.add_subplot((121))
    ax_2d.axis('off')
    img_2d = skeleton(_img, kps, num_kps=num_kps, draw_skeleton=draw_skeleton,
                      vis_threshold=vis_threshold,
                      custom_skeleton_color=custom_skeleton_color,
                      custom_skeleton=custom_skeleton,
                      vis_occlued=vis_occlued,
                      vis_occlusion=vis_occlusion,
                      only_occlusion=only_occlusion,
                      drawpoint=drawpoint,
                      pointcolor=pointcolor,
                      drawscore=drawscore)
    plt.imshow(img_2d)

    ax_3d = fig.add_subplot((122), projection='3d')
    ax_3d.grid(False)
    oo = 1e10
    xmax, ymax, zmax = -oo, -oo, -oo
    xmin, ymin, zmin = oo, oo, oo
    kps = kps.reshape((1, 16, 3))
    depth = depth.reshape((1, 16, 1))
    kps_3d = np.concatenate((kps[:, :, 0:-1], depth), axis=-1)
    kps_3d = kps_3d.reshape(-1, 3)
    x, y, z = np.zeros((3, kps_3d.shape[0]))
    for j in range(kps_3d.shape[0]):
        x[j] = kps_3d[j, 0].copy()
        y[j] = kps_3d[j, 2].copy()
        z[j] = -kps_3d[j, 1].copy()
        xmax = max(x[j], xmax)
        ymax = max(y[j], ymax)
        zmax = max(z[j], zmax)
        xmin = min(x[j], xmin)
        ymin = min(y[j], ymin)
        zmin = min(z[j], zmin)
    ax_3d.scatter(x, y, z, s = 200, c = 'b', marker = 'o')
    for e, c in zip(custom_skeleton, custom_skeleton_color):
        ax_3d.plot(x[e], y[e], z[e], c = [one_c/255.0 for one_c in c])
    plt.savefig(savedir)
    plt.close()


def skeleton_car(_img, kps, **kwargs):
    """Draw skeleton on image in carfusion way

    Parameters
    ----------
    img : cv2 image
    kps : np.ndarray, [N, num_kps*3]

    Returns
    -------
    img
    """
    kwargs['num_kps'] = 14
    kwargs['custom_skeleton_color'] = SKELETON_COLOR_CARFUSION
    kwargs['custom_skeleton'] = KPS_SKELETON_CARFUSION
    return skeleton(_img, kps, **kwargs)


def skeleton_dog(_img, kps, **kwargs):
    """Draw dog skeleton on image. (StandfordExtra dog dataset)
    Parameters
    ----------
    img : cv2 image
    kps : np.ndarray, [N, num_kps*3]

    Returns
    -------
    img
    """
    kwargs['num_kps'] = 24
    kwargs['custom_skeleton_color'] = SKELETON_COLOR_DOG
    kwargs['custom_skeleton'] = KPS_SKELETON_DOG
    return skeleton(_img, kps, **kwargs)


def skeleton_rectangle(_img, kps, color=(255, 0, 0), **kwargs):
    """Draw skeleton on image in rectangle way.

    Parameters
    ----------
    img : cv2 image
    kps : np.ndarray, [N, num_kps*3]

    Returns
    -------
    img
    """
    kwargs['num_kps'] = 4
    kwargs['custom_skeleton'] = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    kwargs['custom_skeleton_color'] = [color for _ in range(4)]
    kwargs['drawpoint'] = True
    kwargs['pointcolor'] = color
    return skeleton(_img, kps, **kwargs)


skeleton_keyboard = skeleton_rectangle


def lmks(_img, _lmks, color=(0, 255, 255), vis_threshold=0.5, size=2, factor=1.0):
    img = _img.copy()
    lmks = _lmks.copy()
    if lmks.ndim == 1:
        lmks = np.array([lmks])

    img = cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)))
    lmks *= factor

    for ind, lmk in enumerate(lmks):
        if lmk.shape[0] % 3 == 0:
            for pt in lmk.reshape(-1, 3):
                x = int(pt[0])
                y = int(pt[1])
                if vis_threshold == -1:
                    if pt[2] < 0.5:
                        color_unvis = [int(c/2) for c in color]
                        cv2.circle(img, (x, y), 2, color_unvis, size)
                    else:
                        cv2.circle(img, (x, y), 2, color, size)
                elif pt[2] > vis_threshold:
                    cv2.circle(img, (x, y), 2, color, size)
        else:
            for pt in lmk.reshape(-1, 2):
                x = int(pt[0])
                y = int(pt[1])
                cv2.circle(img, (x, y), 2, color, size)
    return img


def lmks68(_img, _lmks, color=(0, 255, 255), vis_threshold=0.5, size=2, factor=1.0):
    """Draw 68 keypoints

    Parameters
    ----------
    _img : cv2image
        input image
    _lmks: np.array, [68*3] or [N, 68*3]

    """
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
    img = _img.copy()
    lmks = _lmks.copy()
    if lmks.ndim == 1:
        lmks = np.array([lmks])

    img = cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)))
    lmks *= factor
    lmks = np.round(lmks).astype(np.int32)

    for ind, lmk in enumerate(lmks):
        if lmk.shape[0] % 3 == 0:
            lmk_reshape = lmk.reshape(-1, 3)
        else:
            lmk_reshape = lmk.reshape(-1, 2)
        for pt_ind, pt in enumerate(lmk_reshape):
            if len(pt) < 3 or pt[2] > vis_threshold:
                cv2.circle(img, (pt[0], pt[1]), 2, color, size)
            if pt_ind in end_list:
                continue
            end = lmk_reshape[pt_ind+1]
            cv2.line(img, (pt[0], pt[1]), (end[0], end[1]), (255, 255, 255), 1)
    return img


def mask(img, segms, colors=None):
    """
    vis segmentation, both for semantic and instance.
    """
    from .contrib.visualizer import VisImage

    h, w = img.shape[0], img.shape[1]
    vis = VisImage(img[:,:,::-1])

    if isinstance(segms, np.ndarray):
        mask_labels = set(segms.flatten())
        if colors is None:
            colors = gen_colors(256, fixed=True)

        mask_img = np.zeros_like(img)
        for mask_label in mask_labels:
            mask_img[segms == mask_label] = colors[mask_label]
        vis_img = img * 0.5 + mask_img * 0.5
        return vis_img

    for ind, segm in enumerate(segms):
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif 'counts' in segm and type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        mask_img = maskUtils.decode(rle)
        if colors is not None:
            vis.add_mask(mask_img, color=colors[ind])
        else:
            vis.add_mask(mask_img)
    return vis.get_image()[:,:,::-1]


def feet_center(img, feetcenter, radius=4, color=(0, 255, 0), colors=None, vis_threshold=0.5):
    for ind, pt in enumerate(feetcenter):
        thickness = -1 if pt[2] > vis_threshold else 2
        if colors is not None:
            color = colors[ind]
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, thickness)
    return img


def feet(img, feet, drawcenter=False, color=(0, 255, 0), colors=None, radius=4, drawscore=None, vis_threshold=0.5):
    """Draw feet on image.

    Args:
        img (cv2 image): input image
        feet (ndarray): feet data, [N, 6]

    Kwargs:
        drawcenter (bool): if draw feet center
        drawscore (str): how to draw score, '>[number]', '<[number]', '=[number]'
        color (tuple): render color
        radius (int): render point radius

    Returns:
        rendered image
    """
    valid_vis = None
    if drawscore is not None:
        if '>' in drawscore:
            valid_vis = lambda x: x > float(drawscore.split('>')[1])
        elif '<' in drawscore:
            valid_vis = lambda x: x < float(drawscore.split('<')[1])
        elif '=' in drawscore:
            valid_vis = lambda x: x == float(drawscore.split('<')[1])

    for ind, pt in enumerate(feet):
        if colors is not None:
            assert len(colors) == len(feet)
            color = colors[ind]

        if drawcenter:
            pt_c = [(pt[0]+pt[3])/2, (pt[1]+pt[4])/2]
            img = cv2.circle(img, (int(pt_c[0]), int(pt_c[1])), radius, color, -1)
        else:
            if set(pt[2::3].astype(float).tolist()).issubset(set([0.0,1.0,2.0,3.0])):
                valid_feet = lambda x: x == 2
            else:
                valid_feet = lambda x: x > vis_threshold
            thickness1= -1 if valid_feet(pt[2]) else 2
            thickness2= -1 if valid_feet(pt[5]) else 2
            img = cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, thickness1)
            img = cv2.circle(img, (int(pt[3]), int(pt[4])), radius, color, thickness2)

            if valid_vis is not None:
                x, y, v = int(pt[0]), int(pt[1]), pt[2]
                if valid_vis(v):
                    img = cv2.putText(img, '%.2f' % v, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
                x, y, v = int(pt[3]), int(pt[4]), pt[5]
                if valid_vis(v):
                    img = cv2.putText(img, '%.2f' % v, (x+3, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color, 1)
    return img


def head(img, head, color=(0, 255, 255), colors=None, thickness=2):
    for ind, each in enumerate(head):
        if colors is not None:
            color = colors[ind]
        x1 = int(each[0])
        y1 = int(each[1])
        x2 = int(each[3])
        y2 = int(each[4])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img


def pose(img, pose, boxes=None, size=100.):
    '''
    Parameters
    ----------
    img : np.array
        cv2image
    pose : list or np.array, [N, 3]
        [yaw, pitch, roll]
    return:
        img:
    '''
    if boxes is None:
        if pose.ndim == 1:
            pose = pose[np.newaxis, :]
        tdx = np.zeros((pose.shape[0]))
        tdy = np.zeros((pose.shape[0]))
        tdy.fill(img.shape[0] / 2)
        tdx.fill(img.shape[1] / 2)
    else:
        tdx = (boxes[:, 0] + boxes[:, 2] ) / 2
        tdy = (boxes[:, 1] + boxes[:, 3] ) / 2
    face_Xs = tdx - 0.5 * size
    face_Ys = tdy - 0.5 * size

    for ind, this_pose in enumerate(pose):
        face_x = face_Xs[ind]
        face_y = face_Ys[ind]
        yaw = this_pose[0]
        pitch = this_pose[1]
        roll = this_pose[2]
        p = pitch * np.pi / 180
        y = -(yaw * np.pi / 180)
        r = roll * np.pi / 180

        x1 = size * (math.cos(y) * math.cos(r)) + face_x
        y1 = size * (math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y)) + face_y
        x2 = size * (-math.cos(y) * math.sin(r)) + face_x
        y2 = size * (math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r)) + face_y
        x3 = size * (math.sin(y)) + face_x
        y3 = size * (-math.cos(y) * math.sin(p)) + face_y

        # Draw base in red
        cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0,0,255), 2)
        cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0,0,255), 1)
        cv2.line(img, (int(x2), int(y2)), (int(x1+x2-face_x), int(y1+y2-face_y)), (0,0,255), 1)
        cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x), int(y1+y2-face_y)), (0,0,255), 1)
        # Draw pillars in blue
        cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (0,255,255), 1) # 0
        cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x), int(y1+y3-face_y)), (0,255,255), 1) # 1
        cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x), int(y2+y3-face_y)), (0,255,255), 1) # 2
        cv2.line(img, (int(x1+x2-face_x), int(y1+y2-face_y)), (int(x3+x1+x2-2*face_x), int(y3+y2+y1-2*face_y)), (0,255,255), 1) # 3
        # Draw top in green
        cv2.line(img, (int(x3+x1-face_x), int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x), int(y3+y2+y1-2*face_y)), (0,255,0), 2)
        cv2.line(img, (int(x2+x3-face_x), int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x), int(y3+y2+y1-2*face_y)), (0,255,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x), int(y3+y1-face_y)), (0,255,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x), int(y3+y2-face_y)), (0,255,0), 2)

    return img


def quality(_img, quality, boxes, method="score", offset=0, color=(0,255,0),
            legend=None, legend_offset=0):
    img = _img.copy()
    assert(len(quality) == len(boxes))

    for ind, q_str in enumerate(quality):
        x = boxes[ind][0] + offset
        y = boxes[ind][1] - 10
        if method == 'flag':
            if q_str == 'Clear':
                flag = 1
            elif q_str == 'Blurry':
                flag = 0
            else:
                flag = -1
        elif method == 'score':
            flag = q_str
        else:
            raise ValueError("Unknown method: %s" % method)
        cv2.putText(img, str(flag), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        if legend is not None:
            cv2.putText(img, str(legend), (50, 100+legend_offset), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return img


def flag(_img, flag, boxes, offset=0, color=(0,255,0),
            legend=None, legend_offset=0, colors=None):
    img = _img.copy()
    assert(len(flag) == len(boxes))

    for ind, f in enumerate(flag):
        x = boxes[ind][0] + offset
        y = boxes[ind][1] - 10
        if isinstance(f, (float, np.float32)):
            f_str = '%.3f' % f
        else:
            f_str = str(f)
        if colors is not None:
            color = colors[ind]
        cv2.putText(img, str(f_str), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        if legend is not None:
            cv2.putText(img, str(legend), (50, 100+legend_offset), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return img


bodytype = flag

scores = flag


def luid(_img, luid, boxes, offset=0, color=(0,0,255),
         legend=None, legend_offset=0):
    img = _img.copy()
    assert(len(luid) == len(boxes))

    for ind, q_str in enumerate(luid):
        x = boxes[ind][0] + offset
        y = boxes[ind][1] - 10
        q_str = str(q_str)
        if len(q_str) > 5:
            q_str = q_str[-5:]
        flag = q_str
        cv2.putText(img, str(flag), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    return img


def orientation(_img, orientation, boxes,
        scale=0.3, ratio=0.5,
        color_ellipse=(0,255,255), color_arrow=(255,255,0),
        thick_ellipse=1, thick_arrow=3):
    orientation_ignore = -10000
    img = _img.copy()
    assert(len(orientation) == len(boxes))

    for ind, box in enumerate(boxes):
        center = (int((box[0]+box[2])/2), int(box[1]))
        a = int((box[2] - box[0]) * scale)
        b = int(a * ratio)
        angle = orientation[ind]
        if angle == orientation_ignore:
            continue

        def ellipse(center, a, b, theta):
            theta = -1 * (theta - 90) / 180. * np.pi
            x = a * np.cos(theta) + center[0]
            y = b * np.sin(theta) + center[1]
            return int(x), int(y)

        arrow = ellipse(center, a, b, angle)
        cv2.arrowedLine(img, center, arrow, color_arrow, thick_arrow)
        cv2.ellipse(img,center,(a, b),0,0,360, color_ellipse, thick_ellipse)
    return img


def skeleton_score(_img, kps1, kps2=None):
    """Draw score beside image

    Parameters
    ----------
    img : cv2 image
    kps1 : np.ndarray, [51,]
        pred
    kps2 : np.ndarray, [51,]
        label

    Returns
    -------
    img with img_score
    """
    assert kps1.ndim == 1
    img = _img.copy()
    img_side = cv2.imread(SKELETON_TEMPLATE_IMGFILE)
    def _gen_color_cube(score, is_label=False):
        cm = plt.get_cmap('jet', 16).reversed()
        if is_label:
            value = 0 if score == 0.0 else 1.0
        else:
            value = max(0.0, min(score, 1.0))
        color = [int(x) for x in cm(value, bytes=True)]
        return color[:3]

    def _draw_kps_score(img_side, kps, is_label=False):
        skeleton_t = SKELETON_TEMPLATE.copy().astype(np.float32)
        skeleton_t = np.concatenate((
                skeleton_t,
                kps.reshape(-1, 3)[:, 2:].astype(skeleton_t.dtype)), axis=1).flatten()
        for kpt in skeleton_t.reshape(-1, 3):
            x = int(kpt[0])
            y = int(kpt[1])
            score = kpt[2]
            score_color = _gen_color_cube(score, is_label=is_label)
            if is_label:
                y += 20
            radius = 10
            img_side = cv2.circle(img_side, (x, y), radius, score_color, -1)
        # legend
        img_side = cv2.circle(img_side, (20, 20), radius, (0, 0, 255), -1)
        img_side = cv2.circle(img_side, (20, 40), radius, (0, 255, 0), -1)
        return img_side

    img_side = _draw_kps_score(img_side, kps1)
    if kps2 is not None:
        img_side = _draw_kps_score(img_side, kps2, is_label=True)

    scale = img_side.shape[0] / img.shape[0]
    img_side = cv2.resize(img_side, (int(img_side.shape[1]/scale), img.shape[0]))

    img = np.concatenate((img, img_side), axis=1)
    return img


def colorize(img, colors):
    """
    colorize img based on colors
    """
    if img.ndim == 2:
        _img = np.concatenate((img[:, :, np.newaxis],
                               img[:, :, np.newaxis],
                               img[:, :, np.newaxis]), axis=2)
        img_shape = _img.shape
        _img = _img.reshape(-1, 3)
    else:
        img_shape = img.shape
        _img = img.reshape(-1, 3)
    img_c = np.zeros_like(_img)
    for i in range(_img.shape[0]):
        img_c[i] = colors[_img[i][0]]
    return img_c.reshape(img_shape).astype(np.uint8)


def semseg(_seg, image=None, colors=None, hwc=False, alpha=0.5,
           keep_shape_as_image=False):

    _seg = to_numpy(_seg).squeeze()
    seg = _seg.copy()
    assert isinstance(seg, np.ndarray)

    if seg.ndim == 2:
        c = seg.max()
    elif seg.ndim == 3:
        if hwc:
            seg = np.transpose(seg, (2, 0, 1))
        c = seg.shape[0]
        seg = np.argmax(seg, axis=0)
    else:
        raise ValueError("bad input ndim: %d" % seg.ndim)
    h, w = seg.shape

    if colors is None:
        colors = Colormap(c+1)
    mask_color = colorize(seg, colors)

    if image is not None:
        if keep_shape_as_image:
            image_h, image_w = image.shape[:2]
            if image_h != h or image_w != w:
                mask_color = cv2.resize(mask_color, (image_w, image_h))
            h, w = image_h, image_w
        if image.shape[0] != h:
            image_resize = cv2.resize(image, (w, h))
        else:
            image_resize = image
        blended = image_resize * alpha + mask_color * (1 - alpha)
    else:
        blended = mask_color
    return np.ascontiguousarray(blended, dtype=np.uint8)


def depth(_depth, image=None, vertical=True, min_depth=0.0, max_depth=None, cmap="magma", hist=False):
    """
    visualize depth map.

    Args:
        cmap: magma / jet / magma_r
    """
    depth_np = to_numpy(_depth).squeeze()
    if max_depth is not None:
        depth_clip = np.clip(depth_np, min_depth, max_depth)
    else:
        depth_clip = depth_np

    vmax = np.percentile(depth_clip, 95)
    normalizer = matplotlib.colors.Normalize(vmin=depth_clip.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = (mapper.to_rgba(depth_clip)[:, :, :3] * 255).astype(np.uint8)

    if image is not None:
        image = to_numpy(image).squeeze()

        if colormapped_im.shape[0] != image.shape[0]:
            colormapped_im = cv2.resize(colormapped_im, (image.shape[1], image.shape[0]))

        if vertical:
            colormapped_im = np.concatenate((image, colormapped_im), axis=0)
        else:
            colormapped_im = np.concatenate((image, colormapped_im), axis=1)

    if hist:
        histogram, bin_edges = np.histogram(depth_np, bins=256)

        fig = plt.figure()
        plt.xlabel("depth")
        plt.ylabel("pixel count")
        plt.plot(bin_edges[0:-1], histogram)  # <- or here
        fig.canvas.draw()
        hist_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        hist_image = hist_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if vertical:
            w = colormapped_im.shape[1]
            scale = 1.0 * w / hist_image.shape[1]
            hist_image = cv2.resize(hist_image, (int(hist_image.shape[1]*scale), int(hist_image.shape[0]*scale)))
            colormapped_im = np.concatenate((colormapped_im, hist_image), axis=0)
        else:
            h = colormapped_im.shape[0]
            scale = 1.0 * h / hist_image.shape[0]
            hist_image = cv2.resize(hist_image, (int(hist_image.shape[1]*scale), int(hist_image.shape[0]*scale)))
            colormapped_im = np.concatenate((colormapped_im, hist_image), axis=1)
    return colormapped_im


def tensor_hist(
    data: List,
    bins: int = 256,
    xlabel: str = "value",
    ylabel: str = "count",
) -> np.ndarray:
    """Draw tensor value in hist."""
    if not isinstance(data, list):
        data = [data]
    
    fig = plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for data_i in data:
        data_np = to_numpy(data_i)
        histogram, bin_edges = np.histogram(data_np, bins=bins)
        plt.plot(bin_edges[0:-1], histogram)

    fig.canvas.draw()
    hist_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    hist_image = hist_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return hist_image
