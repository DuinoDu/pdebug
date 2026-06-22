#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
put many small images in a big image
1. padding roi
2. resize big roi to larger size <= 500
3. jigsaw
"""

from __future__ import division, print_function
import argparse
import os
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import cv2
import numpy as np


def limit(h, w, max_h, max_w):
    ratio = h / w
    if ratio > (max_h / max_w):
        if h > max_h:
            scale = h / max_h
        else:
            return False, h, w
    else:
        if w > max_w:
            scale = w / max_w
        else:
            return False, h, w
    new_h = int(h / scale)
    new_w = int(w / scale)
    return True, new_h, new_w


def pad(box, img_h, img_w, padding=50):
    x1 = max(0, box[0] - padding)
    y1 = max(0, box[1] - padding)
    x2 = min(img_w - 1, box[2] + padding)
    y2 = min(img_h - 1, box[3] + padding)
    return [x1, y1, x2, y2]


def jigsaw(imgs, rois):
    new_rois = []
    assert len(imgs) == len(rois)
    assert len(imgs) == 4
    dst_w = 300
    dst_h = 400
    bgs = []
    rois_bg = []
    imgnames = []
    for img, roi in zip(imgs, rois):
        # cv2.imshow('img_src', img)
        # ch = cv2.waitKey(0) & 0xff
        img_h, img_w = img.shape[:2]
        assert roi["boxes"].shape[0] == 1
        imgnames.append(roi["image"])
        for ind, box in enumerate(roi["boxes"]):
            lmks = roi["lmks"][ind]
            assert box[2] > box[0] and box[3] > box[1]
            bg = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
            box_pad = pad(box, img_h, img_w, padding=50)
            offset_x = box_pad[0]
            offset_y = box_pad[1]
            # crop
            box_pad = [int(x) for x in box_pad]
            img_crop = img[box_pad[1] : box_pad[3], box_pad[0] : box_pad[2]]
            box[0::2] -= offset_x
            box[1::2] -= offset_y
            lmks[0::3] -= offset_x
            lmks[1::3] -= offset_y
            # resize
            crop_h, crop_w = img_crop.shape[:2]
            _, new_h, new_w = limit(crop_h, crop_w, dst_h, dst_w)
            scale = crop_h / new_h
            img = cv2.resize(img_crop, (new_w, new_h))
            box /= scale
            lmks[0::3] /= scale
            lmks[1::3] /= scale
            # put
            if new_h < dst_h:
                start_h = np.random.randint(dst_h - new_h)
                end_h = start_h + new_h
            else:
                start_h = 0
                end_h = new_h
            if new_w < dst_w:
                start_w = np.random.randint(dst_w - new_w)
                end_w = start_w + new_w
            else:
                start_w = 0
                end_w = new_w
            bg[start_h:end_h, start_w:end_w, :] = img
            bgs.append(bg)
            box[0::2] += start_w
            box[1::2] += start_h
            lmks[0::3] += start_w
            lmks[1::3] += start_h
            roi["boxes"][ind] = box
            roi["lmks"][ind] = lmks
        rois_bg.append(roi)

    bg1 = np.concatenate((bgs[0], bgs[1]), axis=1)
    bg2 = np.concatenate((bgs[2], bgs[3]), axis=1)
    bg3 = np.concatenate((bg1, bg2), axis=0)
    new_roi = dict()
    new_roi["sub_images"] = imgnames
    new_roi["height"] = bg3.shape[0]
    new_roi["width"] = bg3.shape[1]
    new_roi["boxes"] = np.concatenate([x["boxes"] for x in rois_bg], axis=0)
    new_roi["lmks"] = np.concatenate([x["lmks"] for x in rois_bg], axis=0)
    if "face_pose" in rois_bg[0]:
        new_roi["face_pose"] = np.concatenate(
            [x["face_pose"] for x in rois_bg], axis=0
        )
    elif "pose" in rois_bg[0]:
        new_roi["pose"] = np.concatenate([x["pose"] for x in rois_bg], axis=0)

    new_roi["boxes"][1, 0::2] += dst_w
    new_roi["lmks"][1, 0::3] += dst_w
    new_roi["boxes"][2, 1::2] += dst_h
    new_roi["lmks"][2, 1::3] += dst_h
    new_roi["boxes"][3, 0::2] += dst_w
    new_roi["boxes"][3, 1::2] += dst_h
    new_roi["lmks"][3, 0::3] += dst_w
    new_roi["lmks"][3, 1::3] += dst_h

    # import vispy1
    # bg3 = vispy1.boxes(bg3, new_roi['boxes'])
    # bg3 = vispy1.lmks(bg3, new_roi['lmks'])
    # cv2.imshow('img', bg3)
    # ch = cv2.waitKey(0) & 0xff
    # if ch == 27: #ord('q')
    #    import sys; sys.exit()
    return bg3, new_roi


def main(args):
    assert args.jigsaw_num == 4
    if args.output is None:
        args.output = "output"
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    save_imgdir = os.path.join(args.output, "images")
    if not os.path.exists(save_imgdir):
        os.makedirs(save_imgdir)

    print("loading %s" % args.roidb)
    try:
        roidb = pickle.load(open(args.roidb, "rb"))
    except Exception as e:
        roidb = pickle.load(open(args.roidb, "rb"), encoding="iso-8859-1")

    imgs = []
    rois = []
    new_roidb = []
    cnt = 0
    bad_imglist = []

    for ind, roi in enumerate(roidb):
        if ind % 1000 == 0:
            print("%d / %d" % (ind, len(roidb)))
        if args.imgdir is not None:
            imgfile = os.path.join(args.imgdir, roi["image"])
        else:
            imgfile = roi["image"]
        img = cv2.imread(imgfile)
        h, w = roi["height"], roi["width"]
        imgs.append(img)
        rois.append(roi)
        if len(imgs) >= args.jigsaw_num:
            new_imgfile = os.path.join(save_imgdir, "%08d.jpg" % cnt)
            bigimg, roi = jigsaw(imgs, rois)
            # try:
            #    bigimg, roi = jigsaw(imgs, rois)
            # except Exception as e:
            #    bad_imglist.extend([x['image'] for x in rois])
            #    continue
            roi["image"] = os.path.basename(new_imgfile)
            cv2.imwrite(new_imgfile, bigimg)
            new_roidb.append(roi)
            cnt += 1
            imgs = []
            rois = []
    print("saved in %s" % args.output)
    roidbname = os.path.basename(args.roidb)
    new_roidbname = os.path.splitext(roidbname)[0] + "_jigsaw.pkl"
    with open(os.path.join(args.output, new_roidbname), "wb") as fid:
        pickle.dump(new_roidb, fid)
    if len(bad_imglist) > 0:
        with open("bad_imglist.txt", "w") as fid:
            for imgname in bad_imglist:
                fid.write(imgname + "\n")
        print("bad imgname saved in bad_imglist.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roidb", default=None, type=str, help="", required=True
    )
    parser.add_argument("--imgdir", default=None, type=str, help="")
    parser.add_argument(
        "--jigsaw_num",
        default=4,
        type=int,
        help="small image num in one big image",
    )
    parser.add_argument("--output", default=None, type=str, help="output")
    args = parser.parse_args()
    main(args)
