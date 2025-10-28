#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
limit image size to [600, 1000]
"""

from __future__ import division, print_function
import argparse
import os

import cPickle as pickle
import cv2


def limit(h, w):
    max_size = max(h, w)
    min_size = min(h, w)
    ratio = max_size / min_size
    if ratio > (1000 / 600):
        if max_size > 1000:
            scale = max_size / 1000
        else:
            return False, None, None
    else:
        if min_size > 600:
            scale = min_size / 600
        else:
            return False, None, None
    new_h = int(h / scale)
    new_w = int(w / scale)
    return True, new_h, new_w


def main(args):
    if args.output is None:
        args.output = os.path.splitext(args.roidb)[0] + "_limit.pkl"

    print("loading %s" % args.roidb)
    with open(args.roidb, "r") as fid:
        roidb = pickle.load(fid)
    args.imgdir = os.path.abspath(args.imgdir)

    cnt = 0
    begin, end = 0, 0

    for ind, roi in enumerate(roidb):
        if ind % 1000 == 0:
            print("%d / %d" % (ind, len(roidb)))
        if args.save_split is not None:
            if ind % args.save_split == 0 and ind > 0:
                cnt = int(ind / args.save_split)
                begin = ind - args.save_split
                end = ind
                save_roidb = roidb[begin:end]
                savepkl = os.path.splitext(args.roidb)[0] + "_%d.pkl" % (
                    cnt - 1
                )
                print("saved in %s" % savepkl)
                with open(savepkl, "w") as fid:
                    pickle.dump(save_roidb, fid)

        imgfile = os.path.join(args.imgdir, roi["image"])
        h = roi["height"]
        w = roi["width"]
        boxes = roi["boxes"]
        lmks = roi["lmks"]
        if args.add_keypoints:
            roi["keypoints"] = lmks

        ok, new_h, new_w = limit(h, w)
        if not ok:
            continue
        scale = h / new_h
        roi["height"] = new_h
        roi["width"] = new_w
        # resize image
        img = cv2.imread(imgfile)
        img = cv2.resize(img, (new_w, new_h))
        new_imgfile = (
            os.path.splitext(imgfile)[0]
            + "_small"
            + os.path.splitext(imgfile)[1]
        )
        new_imgname = os.path.basename(new_imgfile)
        roi["image"] = new_imgname
        if args.ignore_cache:
            cv2.imwrite(new_imgfile, img)
        else:
            if not os.path.exists(new_imgfile):
                cv2.imwrite(new_imgfile, img)
            else:
                pass
        # resize boxes
        boxes = boxes / scale
        roi["boxes"] = boxes
        # resize lmks
        lmks_x = lmks[:, 0::3].copy() / scale
        lmks_y = lmks[:, 1::3].copy() / scale
        lmks[:, 0::3] = lmks_x
        lmks[:, 1::3] = lmks_y
        roi["lmks"] = lmks
        roidb[ind] = roi

    if args.save_split:
        save_roidb = roidb[end:]
        savepkl = os.path.splitext(args.roidb)[0] + "_%d.pkl" % (cnt)
        print("saved in %s" % savepkl)
        with open(savepkl, "w") as fid:
            pickle.dump(save_roidb, fid)

    print("saved in %s" % args.output)
    with open(args.output, "w") as fid:
        pickle.dump(roidb, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roidb", default=None, type=str, help="", required=True
    )
    parser.add_argument(
        "--imgdir", default=None, type=str, help="", required=True
    )
    parser.add_argument(
        "--add_keypoints",
        dest="add_keypoints",
        action="store_true",
        help="copy lmks to keypoints",
    )
    parser.add_argument(
        "--save_split",
        default=None,
        type=int,
        help="save pkl in splitted file",
    )
    parser.add_argument(
        "--ignore_cache",
        dest="ignore_cache",
        action="store_true",
        help="write image even if it already exist",
    )
    parser.add_argument("--output", default=None, type=str, help="output")
    args = parser.parse_args()
    main(args)
