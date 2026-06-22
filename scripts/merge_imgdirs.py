#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os

import cPickle as pickle


def main(args):
    if args.output is None:
        args.output = "imgdir_merged"
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.mapping_pkl is None:
        args.mapping_pkl = "imgnames_merge_imgdirs.pkl"

    imgdirs = args.imgdirs.split(":")
    cnt = 1
    imgname_mapping = {}

    for imgdir in imgdirs:
        imgfiles = sorted(
            [
                os.path.join(imgdir, x)
                for x in sorted(os.listdir(imgdir))
                if x.endswith(".jpg")
            ]
        )
        print(imgdir)
        for ind, imgfile in enumerate(imgfiles):
            if ind % 1000 == 0:
                print("%d / %d" % (ind, len(imgfiles)))

            sh = "ln -s" if args.link else "cp"
            dstname = (
                os.path.basename(imgfile)
                if args.keepname
                else "%08d.jpg" % (cnt)
            )
            cmd = "%s %s %s/%s" % (sh, imgfile, args.output, dstname)
            os.system(cmd)

            imgkey = imgfile.split("/")[-2:]
            imgkey = "%s/%s" % (imgkey[0], imgkey[1])
            imgname_mapping[imgkey] = "%08d.jpg" % cnt
            cnt += 1
    print("Saved in %s" % args.output)
    with open(args.mapping_pkl, "w") as fid:
        pickle.dump(imgname_mapping, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge imgdirs")
    parser.add_argument(
        "--imgdirs",
        default=None,
        type=str,
        help="input imgdirs",
        required=True,
    )
    parser.add_argument(
        "--mapping_pkl",
        default=None,
        type=str,
        help="output image mapping pkl",
    )
    parser.add_argument(
        "--link",
        dest="link",
        action="store_true",
        help="make soft link, instead of copy",
    )
    parser.add_argument(
        "--keepname",
        dest="keepname",
        action="store_true",
        help="keep image name",
    )
    parser.add_argument(
        "--output", default=None, type=str, help="output imgdir"
    )
    args = parser.parse_args()
    main(args)
