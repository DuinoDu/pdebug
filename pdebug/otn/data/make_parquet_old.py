import glob
import os
import shutil
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.fileio import iter_to_parquet
from pdebug.utils.semantic_types import load_categories, load_colors
from pdebug.visp import draw

import cv2
import numpy as np
import typer


def file_to_bytes(filepath, process_func=None):
    assert os.path.exists(filepath), f"{filepath} not exists"
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if process_func:
        image = process_func(image)
    _, image_encoded = cv2.imencode(
        ".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    )
    bytes = image_encoded.tobytes()
    return bytes


@otn_manager.NODE.register(name="make_parquet_old")
def main(
    path: str,
    # summary_txt_file: str = None,
    output: str = None,
    num_workers: int = 1,
    cache: bool = False,
    topk: int = -1,
    version: str = "None",
    prefix: str = "None",
    description: str = "None",
    vis_output: str = None,
):
    """Make parquet from path."""
    if not output:
        output = "updated_parquet/data_pack"
    output = os.path.abspath(output)

    if cache and output and os.path.exists(output):
        if os.path.isfile(output):
            print(f"{output} exists, skip")
            return output
        elif os.path.isdir(output):
            if len(os.listdir(output)) == num_workers:
                print(f"{output} exists, skip")
                return output
            else:
                shutil.rmtree(output)

    if path.endswith(".txt"):
        roidb = Input(
            path, name="simpletxt", field="maskfile1", split_str=" "
        ).get_roidb()
    elif os.path.isdir(path):
        txt_file = glob.glob(f"{path}/*image_anno_info.txt")
        assert len(txt_file) > 0
        txt_file = txt_file[0]
        print(f"Found {txt_file}")
        roidb = Input(
            txt_file, name="simpletxt", field="maskfile1", split_str=" "
        ).get_roidb()
    else:
        raise ValueError(f"Unknown path: {path}")

    if topk > 0:
        old_length = len(roidb)
        roidb = roidb[:topk]
        print(f"topk: {old_length} => {len(roidb)}")

    if vis_output:
        if os.path.exists(vis_output):
            shutil.rmtree(vis_output)
        os.makedirs(vis_output, exist_ok=True)

    def data_iter():
        for roi in roidb:
            data_dict = {}
            image_func = lambda x: cv2.resize(
                x, (940, 705), interpolation=cv2.INTER_LINEAR
            )
            label_func = lambda x: cv2.resize(
                x, (940, 705), interpolation=cv2.INTER_NEAREST
            )
            data_dict["image"] = file_to_bytes(
                roi["image_name"], process_func=image_func
            )
            data_dict["label"] = file_to_bytes(
                roi["maskfile"], process_func=label_func
            )
            data_dict["pose"] = "None"
            data_dict["ori_pose"] = "None"
            data_dict["depth"] = "None"
            data_dict["prefix"] = prefix
            data_dict["image_name"] = roi["image_name"]
            data_dict["pat_version"] = version
            data_dict["description"] = description  # "Manually annotation."

            if vis_output:
                colors = load_colors(as_tuple=True)
                classes = load_categories(simple=True)
                image = cv2.imdecode(
                    np.frombuffer(data_dict["image"], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                label = cv2.imdecode(
                    np.frombuffer(data_dict["label"], dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                vis1 = draw.semseg(
                    label, image=image, colors=colors, classes=classes
                )
                savename = os.path.join(
                    vis_output, os.path.basename(roi["image_name"])
                )
                cv2.imwrite(savename, vis1)

            yield data_dict

    print(f"Saving to {output} ...")
    iter_to_parquet(data_iter, output, 32)
    print("Done")

    if os.path.exists(output) and output.endswith(".parquet"):
        reader = Input(output, name="cruise_rgbd_semseg").get_reader()
        for data_item in reader:
            break

    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output


if __name__ == "__main__":
    typer.run(main)
