import inspect
import os
from typing import Callable, List, Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.piata.handler.ddddata import Frame
from pdebug.utils.env import MMSEG_INSTALLED

import cv2
import numpy as np
import tqdm
import typer

if MMSEG_INSTALLED:
    from mmseg.datasets import ADE20KDataset


__all__ = ["ADE20KDatasetCreator", "load_semseg_file_fn"]


def _add_linebreak_if_need(
    input_str: str, split_by: str, split_num: int, extra_str: str = ""
) -> str:
    """
    Add linebreak to input_str if need.

    For example: aaa,aaa,aaa => aaa,aaa,\naaa
    """
    assert split_num > 0
    assert len(split_by) == 1
    new_str = ""
    split_cnt = 0
    for s in input_str:
        if s == split_by:
            split_cnt += 1
        if split_cnt >= split_num:
            s += "\n" + extra_str
            split_cnt = 0
        new_str += s
    return new_str


load_semseg_file_fn: Callable = lambda x: cv2.imread(x, 0)


class ADE20KDatasetCreator:
    """
    Create ADE20K format semseg data for InternImage codebase.

        ADE20K
            |--- images
                    |--- training
                            |--- xxx.png
                            |--- ...
                    |--- validation
                            |--- xxx.png
                            |--- ...
            |--- annotations
                    |--- training
                            |--- xxx.png
                            |--- ...
                    |--- validation
                            |--- xxx.png
                            |--- ...

    Args:
        name: task name
        code: intern_image codebase root
        output: data in ade20k format output
        reduce_zero_label: change 0 to 255 for background, done in mmseg.
        num_classes: semseg num classes. If not set, it will find from mask files.

    Example:
        >> from pdebug.otn.train.semseg_intern import ADE20KDatasetCreator
        >> intern_code = "/root/ws/projects/teeth_2d/InternImage/segmentation"
        >> creator = ADE20KDatasetCreator("new_task", intern_code, output)
        >> creator.dump_data(train_data, valid_data)
        >> creator.dump_code(code_root=InternImageCode)
        >> creator.update_intern_image_code(code_root=InternImageCode)
    """

    def __init__(
        self,
        name: str,
        code: str,
        output: str,
        *,
        reduce_zero_label: bool = False,
        num_classes: int = None,
    ):
        assert MMSEG_INSTALLED, "mmseg is required."
        self.name = name
        self.data_root = os.path.abspath(output)
        self.code_root = os.path.abspath(code)
        self.reduce_zero_label = reduce_zero_label
        self.new_config_file = os.path.join(
            output, f"ade20k_{self.name}_config.py"
        )
        self.new_dataset_file = os.path.join(
            output, f"ade20k_{self.name}_dataset.py"
        )

        self._train_data_dst = []
        self._valid_data_dst = []

    def dump_data(
        self,
        train_data: List[Frame],
        valid_data: List[Frame],
        method: str = "link",
        mask_func: Callable = None,
        output=None,
    ):
        self.train_data = train_data
        self.valid_data = valid_data

        assert method in ["link", "copy"]
        if method == "link":
            self.func = "ln -s {} {}"
        elif method == "copy":
            self.func = "cp {} {}"
        elif callable(method):
            self.func = method
        else:
            raise ValueError(
                f"Unknown method: {method}, only support link and copy."
            )
        self.mask_func = mask_func

        if not output:
            assert self.data_root is not None, "Please set output."
            output = self.data_root
        imgdir_train = os.path.join(output, "images/training")
        imgdir_valid = os.path.join(output, "images/validation")
        maskdir_train = os.path.join(output, "annotations/training")
        maskdir_valid = os.path.join(output, "annotations/validation")
        os.makedirs(imgdir_train, exist_ok=True)
        os.makedirs(imgdir_valid, exist_ok=True)
        os.makedirs(maskdir_train, exist_ok=True)
        os.makedirs(maskdir_valid, exist_ok=True)

        t = tqdm.tqdm(total=len(self.train_data), desc="process train")
        for f in self.train_data:
            t.update()
            dst_rgb = os.path.join(imgdir_train, os.path.basename(f.rgb_file))
            dst_mask = os.path.join(
                maskdir_train, os.path.basename(f.semseg_file)
            )

            if callable(self.func):
                self.func(f.rgb_file, dst_rgb)
                if self.mask_func:
                    self.mask_func(f.semseg_file, dst_mask)
                else:
                    self.func(f.semseg_file, dst_mask)
            else:
                func = self.func.format(f.rgb_file, dst_rgb)
                os.system(func)
                if self.mask_func:
                    self.mask_func(f.semseg_file, dst_mask)
                else:
                    func = self.func.format(f.semseg_file, dst_mask)
                    os.system(func)

        t = tqdm.tqdm(total=len(self.train_data), desc="process valid")
        for f in self.valid_data:
            t.update()
            dst_rgb = os.path.join(imgdir_valid, os.path.basename(f.rgb_file))
            dst_mask = os.path.join(
                maskdir_valid, os.path.basename(f.semseg_file)
            )

            if callable(self.func):
                self.func(f.rgb_file, dst_rgb)
                if self.mask_func:
                    self.mask_func(f.semseg_file, dst_mask)
                else:
                    self.func(f.semseg_file, dst_mask)
            else:
                func = self.func.format(f.rgb_file, dst_rgb)
                os.system(func)
                if self.mask_func:
                    self.mask_func(f.semseg_file, dst_mask)
                else:
                    func = self.func.format(f.semseg_file, dst_mask)
                    os.system(func)

        if not self.data_root:
            self.data_root = os.path.abspath(output)

    def dump_code(self):
        """
        Create new config and dataset code.
        """
        output = self.data_root
        os.makedirs(output, exist_ok=True)

        # config.py
        config_file = os.path.join(
            self.code_root, "configs/_base_/datasets/ade20k.py"
        )
        new_config_fid = open(self.new_config_file, "w")
        with open(config_file, "r") as fid:
            for line in fid:
                if line.startswith("data_root = 'data/ADEChallengeData2016'"):
                    line = f"data_root = '{self.data_root}'\n"
                if line.startswith("dataset_type = 'ADE20KDataset'"):
                    line = f"dataset_type = '{self.dataset_name}'\n"
                new_config_fid.write(line)
        new_config_fid.close()
        print(f"Save config to {self.new_config_file}")

        # create dataset file
        if self.num_classes:
            num_classes = self.num_classes
            classes = np.array([_ for _ in range(num_classes)])
        else:
            classes = self.gather_classes()
            num_classes = len(classes)
        classes_str = ",".join([f"'type_{i}'" for i in classes])

        if num_classes > len(ADE20KDataset.PALETTE):
            print(
                f"classes_num ({num_classes}) > ADE20k (150), you need to edit {self.new_dataset_file} manually."
            )
        palette_str = str(ADE20KDataset.PALETTE[:num_classes])

        classes_str = _add_linebreak_if_need(classes_str, ",", 6, " " * 15)
        palette_str = _add_linebreak_if_need(palette_str, ",", 12, " " * 14)

        img_suffix = os.path.splitext(self.train_data_dst[0].rgb_file)[1]
        seg_map_suffix = os.path.splitext(self.train_data_dst[0].semseg_file)[
            1
        ]

        dataset_file = inspect.getfile(ADE20KDataset)
        new_ds_fid = open(self.new_dataset_file, "w")

        with open(dataset_file, "r") as fid:
            for idx, line in enumerate(fid):
                line_id = idx + 1
                # CLASSES: L21 - 45
                if 21 <= line_id <= 45:
                    if line_id == 21:
                        line = f"    CLASSES = ({classes_str})\n"
                    else:
                        continue
                # PALETTE: L47 - 84
                if 47 <= line_id <= 84:
                    if line_id == 47:
                        line = f"    PALETTE = {palette_str}\n"
                    else:
                        continue

                # update class name and args.
                line = line.replace("ADE20KDataset", f"{self.dataset_name}")
                line = line.replace(
                    "img_suffix='.jpg'", f"img_suffix='{img_suffix}'"
                )
                line = line.replace(
                    "seg_map_suffix='.png'",
                    f"seg_map_suffix='{seg_map_suffix}'",
                )
                line = line.replace(
                    "reduce_zero_label=True",
                    f"reduce_zero_label={self.reduce_zero_label}",
                )

                # update import code.
                line = line.replace(".builder", "mmseg.datasets.builder")
                line = line.replace(".custom", "mmseg.datasets.custom")

                new_ds_fid.write(line)
        new_ds_fid.close()
        print(f"Save config to {self.new_dataset_file}")

    def update_intern_image_code(self):
        """
        Move new config and dataset file to internimage codebase.
        """
        config_root = os.path.join(self.code_root, "configs/_base_/datasets/")
        dataset_root = os.path.join(self.code_root, "mmseg_custom/datasets/")
        os.system(f"mv {self.new_config_file} {config_root}")
        os.system(f"mv {self.new_dataset_file} {dataset_root}")

        dataset_init_file = os.path.join(dataset_root, "__init__.py")
        with open(dataset_init_file, "a") as fid:
            fid.write(
                f"\nfrom .ade20k_{self.name}_dataset import {self.dataset_name}"
            )
        print(f"Update {dataset_init_file}")

        info_msg = f"""
        New config and dataset file has been created.

        You need to update `../_base_/datasets/ade20k.py` to
        `../_base_/datasets/{os.path.basename(self.new_config_file)}` in your config file.

        """
        typer.echo(typer.style(info_msg, fg=typer.colors.YELLOW))

    @property
    def train_data_dst(self) -> List[Frame]:
        """Get ade train data info."""
        if not self._train_data_dst:
            assert self.data_root
            imgdir = os.path.join(self.data_root, "images/training")
            maskdir = os.path.join(self.data_root, "annotations/training")
            image_files = Input(imgdir, name="imgdir").get_reader().imgfiles
            for _f in image_files:
                image_file = _f
                mask_file = os.path.join(maskdir, os.path.basename(_f))
                self._train_data_dst.append(
                    Frame(
                        rgb_file=image_file,
                        semseg_file=mask_file,
                        load_semseg_file=load_semseg_file_fn,
                    )
                )
        return self._train_data_dst

    @property
    def valid_data_dst(self) -> List[Frame]:
        """Get ade valid data info."""
        if not self._valid_data_dst:
            assert self.data_root
            imgdir = os.path.join(self.data_root, "images/validation")
            maskdir = os.path.join(self.data_root, "annotations/validation")
            image_files = Input(imgdir, name="imgdir").get_reader().imgfiles
            for _f in image_files:
                image_file = _f
                mask_file = os.path.join(maskdir, os.path.basename(_f))
                self._valid_data_dst.append(
                    Frame(
                        rgb_file=image_file,
                        semseg_file=mask_file,
                        load_semseg_file=load_semseg_file_fn,
                    )
                )
        return self._valid_data_dst

    def gather_classes(self):
        """
        Get classes info from mask file.
        """
        all_classes = []

        t = tqdm.tqdm(
            total=len(self.train_data_dst), desc="collect label info"
        )
        for f in self.train_data_dst:
            t.update()
            f.load_data(
                load_rgb=False,
                load_depth=False,
                load_semseg=True,
                generate_pcd=False,
            )
            classes = np.unique(f.semseg)
            all_classes.extend(classes)
            all_classes = list(set(all_classes))
        return all_classes

    @property
    def dataset_name(self):
        return f"ADE20KDataset_{self.name}"


@otn_manager.NODE.register(name="semseg")
def main(
    args: str,
    output: str = None,
):
    """Tool description."""
    typer.echo(typer.style(f"WIP", fg=typer.colors.GREEN))


if __name__ == "__main__":
    typer.run(main)
