#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transform imgdir, from one imgdir to another. \n
Transform method: \n
1. cp/ln from src to dst, by imglist. \n
2. cp/ln from src to dst, by roidb. \n
"""

from __future__ import print_function
import argparse
import os

import cPickle as pickle
import cv2
from PIL import Image

try:
    import commands
except Exception as e:
    import subprocess as commands

import random


def valid(imgfile):
    try:
        assert os.path.exists(imgfile)
        img = cv2.imread(imgfile)
        assert img is not None
        assert len(img.shape) > 1
        height, width = img.shape[:2]
    except Exception as e:
        return False
    return True


def reset_ext(imgfile):
    cmd = "identify %s" % imgfile
    (status, output) = commands.getstatusoutput(cmd)
    output = output.split("\n")
    if len(output) > 0:
        ext = output[0].split(" ")[1].lower()
        if ext == "jpeg":
            ext = "jpg"
        imgfile = os.path.splitext(imgfile)[0]
        imgfile += ".%s" % ext
    return imgfile


def convert_to_jpg(imgfile):
    dst_imgfile = os.path.splitext(imgfile)[0] + ".jpg"
    cmd = "convert %s %s" % (imgfile, dst_imgfile)
    try:
        (status, output) = commands.getstatusoutput(cmd)
        output = output.split("\n")
    except Exception as e:
        print(output)
        import sys

        sys.exit()


def flip(imgfile, dst_imgfile):
    cmd = "convert %s -flop %s" % (imgfile, dst_imgfile)
    try:
        (status, output) = commands.getstatusoutput(cmd)
        output = output.split("\n")
    except Exception as e:
        print(output)
        import sys

        sys.exit()


# from parallel import mp
# @mp(nums=12)
def cleanimages(imgfiles):
    for ind, imgfile in enumerate(imgfiles):
        if ind % 1000 == 0:
            print("%d / %d" % (ind, len(imgfiles)))
        # rm invalid image
        if not valid(imgfile):
            os.system("rm -f %s" % imgfile)
        # rm multi image ext
        imgdir = os.path.dirname(imgfile)
        imgname = os.path.basename(imgfile)
        items = imgname.split(".")

        # rm .imagename.jpg file
        if items[0] == "":
            os.system("rm %s" % imgfile)
            continue
        # rename imagename.png.jpg to imagename.jpg
        new_imgfile = "%s/%s.%s" % (imgdir, items[0], items[-1])
        if len(items) > 2:
            print("%s => %s" % (imgfile, new_imgfile))
            os.system("mv %s %s" % (imgfile, new_imgfile))
            continue
        ## reset image ext, useless
        # new_imgfile = reset_ext(imgfile)
        # if imgfile != new_imgfile:
        #    os.system('mv %s %s' % (imgfile, new_imgfile))
        # convert_to_jpg(imgfile)


def main(args):
    args.imgdir = os.path.abspath(args.imgdir)
    imgext = (".jpg", ".jpeg", ".bmp", ".png", ".tiff", ".JPG", ".PNG")
    imgfiles = sorted(
        [
            os.path.join(args.imgdir, x)
            for x in sorted(os.listdir(args.imgdir))
            if os.path.splitext(x)[1] in imgext
        ]
    )

    if args.output is None:
        if args.imgdir.endswith("/"):
            args.imgdir = args.imgdir[:-1]
        args.output = args.imgdir + "_output"
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    sh = "ln -s" if args.link else "cp"

    if args.imglist is not None:
        print("loading %s" % args.imglist)
        with open(args.imglist, "r") as fid:
            imgnames = [x.strip().split(" ")[0] for x in fid.readlines()]
        for ind, item in enumerate(imgnames):
            if ind % 1000 == 0:
                print("%d / %d" % (ind, len(imgnames)))
            cmd = "%s %s %s" % (
                sh,
                os.path.join(args.imgdir, item),
                args.output,
            )
            os.system(cmd)
        print("output images are saved in %s" % args.output)

    if args.roidb is not None:
        print("loading %s" % args.roidb)
        with open(args.roidb, "r") as fid:
            roidb = pickle.load(fid)
        new_roidb = []
        for ind, roi in enumerate(roidb):
            if ind % 1000 == 0:
                print("%d / %d" % (ind, len(roidb)))
            imgname = os.path.basename(roi["image"])
            if (
                args.to_jpg
                and "jpg" not in os.path.splitext(imgname)[1].lower()
            ):
                convert_to_jpg(os.path.join(args.imgdir, imgname))
                imgname = os.path.splitext(imgname)[0] + ".jpg"
                roi["image"] = imgname
            new_roidb.append(roi)
            imgfile = os.path.join(args.imgdir, imgname)
            cmd = "%s %s %s" % (sh, imgfile, args.output)
            os.system(cmd)
        if args.to_jpg:
            print("update %s" % args.roidb)
            with open(args.roidb, "w") as fid:
                pickle.dump(new_roidb, fid)
        print("output images are saved in %s" % args.output)

    # if args.cleanimage:
    #    imgext = ('jpg', 'jpeg', 'bmp', 'png', 'tiff')
    #    old_sum = len(imgfiles)
    #    cleanimages(imgfiles)
    #    new_sum = len([os.path.join(args.imgdir, x) for x in sorted(os.listdir(args.imgdir)) if x.endswith('.jpg')])
    #    print("remove bad images: %d => %d" % (old_sum, new_sum))

    if args.flip:
        for ind, imgfile in enumerate(imgfiles):
            if ind % 50 == 0:
                print("%d / %d" % (ind, len(imgfiles)))
            dst_imgfile = os.path.join(args.output, os.path.basename(imgfile))
            flip(imgfile, dst_imgfile)

    if args.random_num is not None:
        assert args.random_num < len(imgfiles), "%d < %d" % (
            args.random_num,
            len(imgfiles),
        )
        random.shuffle(imgfiles)
        imgfiles = imgfiles[: args.random_num]
        for ind, imgfile in enumerate(imgfiles):
            if ind % 1000 == 0:
                print("%d / %d" % (ind, len(imgfiles)))
            cmd = "%s %s %s" % (sh, imgfile, args.output)
            os.system(cmd)

    len1 = len(os.listdir(args.imgdir))
    len2 = len(os.listdir(args.output))
    print("%d => %d" % (len1, len2))
    print("output images are saved in %s" % args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imgdir", default=None, type=str, help="imgdir", required=True
    )
    parser.add_argument(
        "--link",
        dest="link",
        action="store_true",
        help="if choose, using ln instead of cp",
    )
    parser.add_argument("--output", default=None, type=str, help="output")

    parser.add_argument("--imglist", default=None, type=str, help="image list")
    parser.add_argument("--roidb", default=None, type=str, help="input roidb")
    parser.add_argument(
        "--to_jpg",
        dest="to_jpg",
        action="store_true",
        help="convert to jpg format",
    )
    parser.add_argument(
        "--cleanimage",
        dest="cleanimage",
        action="store_true",
        help="clean images, rm bad images",
    )
    parser.add_argument(
        "--flip", dest="flip", action="store_true", help="if hflip image"
    )
    parser.add_argument(
        "--random_num",
        default=None,
        type=int,
        help="random sample from imgdir to output",
    )
    args = parser.parse_args()
    main(args)
