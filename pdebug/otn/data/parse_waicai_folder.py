import datetime
import glob
import json
import os
import random
from collections import defaultdict
from typing import Optional

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input

import tqdm
import typer


def get_date(input_str, root=None):
    """Convert 20230601 or 20230601_* to datetime format."""
    if root:
        input_str = input_str.split(root)[1]
        input_str = input_str.split("/")[1]
        date_str = input_str.split("_")[0]
    else:
        date_str = input_str

    if len(date_str) != 8:
        print(f"Unvalid date string: {date_str}, skip.")
        return datetime.datetime(2020, 1, 1)

    assert len(date_str) == 8
    year = int(date_str[:4])
    mon = int(date_str[4:6])
    day = int(date_str[6:])
    return datetime.datetime(year, mon, day)


def bfs_walk(root, check_file=None):
    """
    By devv.ai
    """
    dirs = [root]
    while len(dirs):
        nextDirs = []
        for parent in dirs:
            if check_file and check_file in os.listdir(parent):
                yield os.path.join(parent, check_file)
                continue
            for f in os.listdir(parent):
                ff = os.path.join(parent, f)
                if os.path.isdir(ff):
                    nextDirs.append(ff)
                # else:
                #     yield ff
        dirs = nextDirs


@otn_manager.NODE.register(name="parse_waicai_folder")
def main(
    root: str,  #  = "/mnt/bn/depth-data-bn/K1H/raw",
    output: str = "imagefiles_list.txt",
    cache: bool = False,
    show_room: bool = False,
    save_room: str = None,
    shuffle_image: bool = False,
    topk_image_per_location: int = None,
    topk_location: int = None,
    max_image: int = None,
    timestamp_sample_duration: int = None,
):
    """Parse waicai data directory.

    waicai directory:

        root/.../
                |--hawk
                    |--cam5
                |--kinect [optional]
                |--metainfo_new.json

    Type: data_root -> imagefiles_list.txt

    """
    if cache and os.path.exists(output):
        typer.echo(typer.style(f"Use cache {output}", fg=typer.colors.GREEN))
        return output

    # info_files = os.path.join(root, "*/*/metainfo_new.json")
    # info_files = glob.glob(info_files)

    info_files = []
    for f in bfs_walk(root, check_file="metainfo_new.json"):
        info_files.append(f)

    # # select date regon
    # # >= 20230601
    # info_files = [f for f in info_files if get_date(f, root) >= get_date("20230601")]

    # order by locations
    locations = defaultdict(list)
    for f in info_files:
        try:
            with open(f, "r") as fid:
                data = json.load(fid)
            location = data["location"]
            locations[location].append(f)
        except Exception as e:
            print(e)
            pass

    if show_room:
        for loc in locations:
            print(loc)
            for f in locations[loc]:
                print(f"  {f}")
        print(f"imgdir sum: {len(locations)}")
        return

    if False:
        locations_list = list(locations.keys())
        locations_list = [l for l in locations_list if "京仪" not in l]
        # random.shuffle(locations_list)
        # locations_val = locations_list[:10]
        new_locations = {k: locations[k] for k in locations_list}
        locations = new_locations

    if save_room:
        # 2000, 10 * 200
        with open(save_room, "w") as fid:
            for l in locations_val:
                fid.write(f"{l}\n")

    if topk_location:
        locations_list = list(locations.keys())[:topk_location]
        new_locations = {k: locations[k] for k in locations_list}
        print("topk_location: {len(locations)} => {len(new_locations)}")
        locations = new_locations

    all_rgb_files = []
    t = tqdm.tqdm(total=len(locations))
    for loc in locations:
        t.update()
        rgb_files = []
        for f in sorted(locations[loc]):
            dirname = os.path.dirname(f)
            if "biaoding" in dirname:
                continue
            imgdir_kwargs = {}
            imgdir_kwargs["name"] = "imgdir"

            if timestamp_sample_duration:
                imgdir_kwargs[
                    "timestamp_sample_duration"
                ] = timestamp_sample_duration
                imgdir_kwargs["get_timestamp_fn"] = lambda x: int(
                    x.split("_")[0][:7]
                )  # ms
            if shuffle_image:
                imgdir_kwargs["shuffle"] = True
            if topk_image_per_location:
                imgdir_kwargs["topk"] = topk_image_per_location

            reader = Input(
                os.path.join(dirname, "hawk/cam5"), **imgdir_kwargs
            ).get_reader()
            rgb_files.extend(reader.imgfiles)

        if max_image:
            random.shuffle(rgb_files)
            rgb_files = rgb_files[:max_image]
            rgb_files = sorted(rgb_files)

        all_rgb_files.extend(rgb_files)

    if len(all_rgb_files) == 0:
        typer.echo(
            typer.style(f"no image found in {root}", fg=typer.colors.RED)
        )
        raise RuntimeError

    with open(output, "w") as fid:
        for f in all_rgb_files:
            fid.write(f"{f}\n")
    typer.echo(typer.style(f"saved to {output}", fg=typer.colors.GREEN))
    return output


if __name__ == "__main__":
    typer.run(main)
