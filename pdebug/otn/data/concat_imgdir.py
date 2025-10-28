#!/usr/bin/env python
import argparse
import copy
import os
import shutil
import sys
from collections import OrderedDict as odict
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input, Output

import cv2
import numpy as np
import tqdm
import typer
from termcolor import cprint


class Concat:
    """
    Concat images.
    """

    def __init__(
        self,
        length,
        titles=None,
        vertical=False,
        horizontal=False,
        fisheye=False,
        title_pos="lr",
        add_index=True,
        mask_by_alpha=True,
    ):
        self.length = length
        self.titles = titles
        self.title_pos = title_pos
        assert title_pos in ("tl", "bl", "tr", "br")
        if titles:
            assert len(titles) == length
        self.concat_axis = None
        if vertical:
            self.concat_axis = 0
        if horizontal:
            self.concat_axis = 1
        self._idx = 0
        self.add_index = add_index
        self.mask_by_alpha = mask_by_alpha

        if fisheye:
            assert length == 4 or length == 5
            self.concat_fn = eval("self.concat_fisheye")
        else:
            try:
                self.concat_fn = eval("self.concat_%d" % length)
            except Exception as e:
                cprint("imgdir len %d NotImplementedError" % length, "red")
                sys.exit()

    def __call__(self, *x):
        imgs = list()
        for ind, imgfile in enumerate(x):
            if isinstance(imgfile, np.ndarray):
                img = imgfile
            else:
                img = cv2.imread(imgfile, -1)
                if img is None:
                    return None
                if img.ndim == 3 and img.shape[2] == 4 and self.mask_by_alpha:
                    mask = img[:, :, 3]
                    img = copy.deepcopy(img[:, :, :3])
                    img[np.logical_not(mask)] = 0
            if self.titles:
                title = self.titles[ind]
                if self.title_pos == "tl":
                    text_pos = (50, 100)
                elif self.title_pos == "bl":
                    text_pos = (50, img.shape[0] - 100)
                elif self.title_pos == "tr":
                    text_pos = (img.shape[1] - 50, 100)
                elif self.title_pos == "br":
                    text_pos = (img.shape[1] - 50, img.shape[0] - 100)
                else:
                    raise ValueError

                if self.add_index:
                    title += f", {self._idx}"

                img = cv2.putText(
                    img,
                    title,
                    text_pos,
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            imgs.append(img)
        self._idx += 1
        res = self.concat_fn(*imgs)
        res = np.ascontiguousarray(res)
        return res

    def concat_2(self, *x, axis=1):
        assert len(x) == 2
        if self.concat_axis is not None:
            axis = self.concat_axis
        return np.concatenate((x[0], x[1]), axis=axis)

    def concat_3(self, *x, axis=1):
        assert len(x) == 3
        if self.concat_axis is not None:
            return np.concatenate(x, axis=self.concat_axis)
        return np.concatenate((x[0], x[1], x[2]), axis=axis)
        # bg = np.zeros_like(x[0])
        # x = list(x)
        # x.append(bg)
        # return self.concat_4(*x)

    def concat_4(self, *x):
        assert len(x) == 4
        if self.concat_axis is not None:
            return np.concatenate(x, axis=self.concat_axis)
        row1 = self.concat_2(*x[0:2], axis=1)
        row2 = self.concat_2(*x[2:4], axis=1)
        return self.concat_2(row1, row2, axis=0)

    def concat_6(self, *x):
        assert len(x) == 6
        if self.concat_axis is not None:
            return np.concatenate(x, axis=self.concat_axis)
        row1 = np.concatenate(x[:2], axis=1)
        row2 = np.concatenate(x[2:4], axis=1)
        row3 = np.concatenate(x[4:], axis=1)
        return np.concatenate((row1, row2, row3), axis=0)

    def concat_7(self, *x):
        assert len(x) == 7
        if self.concat_axis is not None:
            axis = self.concat_axis
            return np.concatenate(x, axis=axis)
        bg = np.zeros_like(x[0])
        x = list(x)
        x.append(bg)
        return self.concat_8(*x)

    def concat_8(self, *x):
        assert len(x) == 8
        if self.concat_axis is not None:
            return np.concatenate(x, axis=self.concat_axis)
        row1 = np.concatenate(x[:4], axis=1)
        row2 = np.concatenate(x[4:], axis=1)
        return np.concatenate((row1, row2), axis=0)

    def concat_9(self, *x):
        assert len(x) == 9
        if self.concat_axis is not None:
            return np.concatenate(x, axis=self.concat_axis)
        row1 = self.concat_3(*x[0:3], axis=1)
        row2 = self.concat_3(*x[3:6], axis=1)
        row3 = self.concat_3(*x[6:9], axis=1)
        return self.concat_3(row1, row2, row3, axis=0)

    def concat_fisheye(self, *x):
        bg = np.zeros_like(x[0])
        if self.length == 4:
            all_x = [bg, x[0], bg, x[1], bg, x[2], bg, x[3], bg]
        elif self.length == 5:
            # a bit hard code
            if x[2].shape != x[0].shape:
                padding_x2 = bg.copy()
                start_x = (bg.shape[1] - x[2].shape[1]) // 2
                end_x = start_x + x[2].shape[1]
                padding_x2[:, start_x:end_x] = x[2]
            else:
                padding_x2 = x[2]
            all_x = [bg, x[0], bg, x[1], padding_x2, x[3], bg, x[4], bg]
        return self.concat_9(*all_x)


@otn_manager.NODE.register(name="concat_video")
def concat_video(
    path: str = None,
    videos: str = None,
    titles: str = None,
    title_pos: str = "tl",
    vertical: bool = False,
    horizontal: bool = False,
    add_index: bool = True,
    mask_by_alpha=True,
    output: str = "tmp_concat_output",
):
    """Concat two videos from two.

    Args:
        path: used for dag, not used in concat_imgdir code.
        videos: input video file list, concat by comma.
        titles: title of each imgdir.
    """
    if output is None:
        output = "result_concat.mp4"
    if os.path.exists(output):
        os.remove(output)

    if "," in videos:
        split_flag = ","
    elif ":" in videos:
        split_flag = ":"
    else:
        cprint("Please use , or : to split multi imgdirs.", "red")
        sys.exit()

    if titles:
        titles = titles.split(split_flag)
        if titles == "None":
            titles = None
    else:
        titles = [video.strip() for video in videos.split(split_flag)]
    videos = [x.strip() for x in videos.split(split_flag) if x.strip() != ""]
    readers = [Input(video, name="video").get_reader() for video in videos]

    concat = Concat(
        len(videos),
        titles=titles,
        title_pos=title_pos,
        vertical=vertical,
        horizontal=horizontal,
        add_index=add_index,
        mask_by_alpha=mask_by_alpha,
    )

    t = tqdm.tqdm(total=len(readers[0]))
    output_frames = []
    for ind in range(len(readers[0])):
        t.update()
        img_list = [r.__next__() for r in readers]
        img_concat = concat(*img_list)
        output_frames.append(img_concat[:, :, ::-1])  # bgr to rgb

    Output(output_frames, name="video", fps=readers[0].fps).save(output)
    return output


@otn_manager.NODE.register(name="concat_imgdir")
def main(
    path: str = None,
    imgdirs: str = None,
    titles: str = None,
    title_pos: str = "tl",
    ext: str = ".jpg",
    vertical: bool = False,
    horizontal: bool = False,
    fisheye: bool = False,
    clean: bool = False,
    by_order: bool = False,
    add_index: bool = True,
    mask_by_alpha: bool = True,
    output: str = "tmp_concat_output",
    save_video: bool = False,
    remove_imgdirs: bool = False,
):
    """Concat two image from two.

    Args:
        path: used for dag, not used in concat_imgdir code.
        imgdirs: input imgdir list, concat by comma.
        titles: title of each imgdir.
    """
    if output is None:
        output = "result_concat"
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    if save_video:
        writer = Output(
            os.path.join(output, "concat.mp4"), name="video_ffmpeg"
        ).get_writer()

    if "," in imgdirs:
        split_flag = ","
    elif ":" in imgdirs:
        split_flag = ":"
    else:
        cprint("Please use , or : to split multi imgdirs.", "red")
        sys.exit()

    if titles:
        titles = titles.split(split_flag)
        if titles == "None":
            titles = None
    else:
        titles = [imgdir.strip() for imgdir in imgdirs.split(split_flag)]
    imgdirs = [x.strip() for x in imgdirs.split(split_flag) if x.strip() != ""]
    num_imgdir = len(imgdirs)

    all_imgfiles = []
    all_imglength = []
    for imgdir in imgdirs:
        imgfiles = Input(imgdir, name="imgdir").get_reader().imgfiles
        if len(imgfiles) == 0:
            imgfiles = sorted(
                [os.path.join(imgdir, x) for x in sorted(os.listdir(imgdir))]
            )
        imgfiles = odict({os.path.basename(x): x for x in imgfiles})
        all_imgfiles.append(imgfiles)
        all_imglength.append(len(imgfiles))

    min_idx = np.argmin(all_imglength)
    cur_imgfiles = all_imgfiles[min_idx]
    if num_imgdir * len(cur_imgfiles) != np.sum(all_imglength):
        cprint("found different imgdir length, use min length imgdir", "red")

    if clean:
        cprint("clean image names ...", "green")
        img_names_cnt = {}
        for imgfiles in all_imgfiles:
            for imgname in imgfiles:
                if imgname in img_names_cnt:
                    img_names_cnt[imgname] += 1
                else:
                    img_names_cnt[imgname] = 1
        img_names = [
            k for k in img_names_cnt if img_names_cnt[k] == len(all_imgfiles)
        ]

    elif by_order:
        propor_list = [i // min(all_imglength) for i in all_imglength]
        for ind, imgfiles in enumerate(all_imgfiles):
            propor = propor_list[ind]
            select_imgnames = list(imgfiles.keys())[0::propor]
            all_imgfiles[ind] = odict(
                {key: imgfiles[key] for key in select_imgnames}
            )
            cprint(
                f"downsample: {len(imgfiles)} => {len(all_imgfiles[ind])}",
                "green",
            )
        img_names = list(cur_imgfiles.keys())
    else:
        img_names = list(cur_imgfiles.keys())

    concat = Concat(
        num_imgdir,
        titles=titles,
        title_pos=title_pos,
        vertical=vertical,
        horizontal=horizontal,
        fisheye=fisheye,
        add_index=add_index,
    )

    t = tqdm.tqdm(total=len(img_names))
    for ind, key in enumerate(img_names):
        t.update()
        imgfile_list = list()
        for imgfiles in all_imgfiles:
            if by_order:
                imgfile_list.append(list(imgfiles.values())[ind])
            else:
                imgfile_list.append(imgfiles[key])
        img_concat = concat(*imgfile_list)
        if img_concat is None:
            continue
        if save_video:
            writer.write_frame(img_concat)
        else:
            cv2.imwrite(
                os.path.join(output, os.path.basename(imgfile_list[0])),
                img_concat,
            )

    if remove_imgdirs:
        for imgdir in imgdirs:
            shutil.rmtree(imgdir)

    if save_video:
        writer.save()

    print(f"saved to {output}")
    return output


if __name__ == "__main__":
    typer.run(main)
