import json
import os
import pickle
import shutil
import zipfile
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, List, Optional

from pdebug.data_types import Segmentation
from pdebug.otn import manager as otn_manager
from pdebug.otn.analysis.visualize import Roidb
from pdebug.piata import Input, Output
from pdebug.utils.decorator import mp
from pdebug.utils.env import IMANTICS_INSTALLED, RICH_INSTALLED
from pdebug.utils.fileio import no_print
from pdebug.utils.semantic_types import (
    SEMANTIC_PCD_CATEGORIES,
    load_categories,
    load_colors,
)
from pdebug.visp import draw

import cv2
import numpy as np
import tqdm
import typer

try:
    from cvat_sdk import make_client, models
    from cvat_sdk.core.proxies.tasks import ResourceType, Task

except ModuleNotFoundError as e:
    make_client = None

HOST = None
USER = None
PASSWORD = None


def download(task_id, output=None, include_images=False):
    if make_client is None:
        raise RuntimeError(
            "cvat_sdk is required, please install cvat_sdk using pip."
        )

    with make_client(host=HOST, credentials=(USER, PASSWORD)) as client:
        print(f"Downloading {task_id} ...")
        filename = "download.zip"
        client.tasks.retrieve(task_id).export_dataset(
            # "COCO 1.0",                   # coco.json format
            "CVAT for images 1.1",  # xml format
            filename,
            include_images=include_images,
        )
        print("Unzip ...")
        os.system("unzip -q download.zip")
        os.system(f"rm {filename}")
        if not output:
            output = "."
        if os.path.exists("annotations.xml"):
            os.system(f"mv annotations.xml {output}/task_{task_id}.xml")
        if os.path.exists("annotations/instances_default.json"):
            os.system(
                f"mv annotations/instances_default.json {output}/task_{task_id}.json"
            )
            os.system(f"rm -r annotations")
        if include_images:
            os.system(f"mv images {output}/images_{task_id}")


@otn_manager.NODE.register(name="download_cvat")
def download_cvat(
    task_ids: str,
    include_images: bool = False,
    cache: bool = False,
    output: str = None,
):
    """Download annotation from cvat.

    Example:
        >> python download_anno.py 43,44,45,53
    """
    if "-" in task_ids:
        start, end = [int(i) for i in task_ids.split("-")]
        ids = list(range(start, end + 1))
    else:
        ids = [int(i) for i in task_ids.split(",")]

    if not output:
        output = f"download_{task_ids}"

    os.makedirs(output, exist_ok=True)

    for task_id in ids:
        if cache:
            anno_file = os.path.join(output, f"task_{task_id}.xml")
            if os.path.exists(anno_file):
                print(f"{anno_file} exists, skip")
                continue

        download(task_id, output, include_images)
    print(f"Downloaded to {output}")
    return output


@otn_manager.NODE.register(name="delete_cvat")
def delete_cvat(
    task_ids: str,
    previous_node: str = None,
):
    """Delete annotation task in cvat.

    !!! BE CAUTION WITH THIS OPERATION

    Usage:
        >> otn-cli --node delete_cvat --task-ids 213-220
    """
    if make_client is None:
        raise RuntimeError(
            "cvat_sdk is required, please install cvat_sdk using pip."
        )

    if "-" in task_ids:
        start, end = [int(i) for i in task_ids.split("-")]
        ids = list(range(start, end + 1))
    else:
        ids = [int(i) for i in task_ids.split(",")]

    with make_client(host=HOST, credentials=(USER, PASSWORD)) as client:
        for task_id in ids:
            task = client.tasks.retrieve(int(task_id))
            task.remove()
            print(f"task {task_id} has been removed")
    return ""


def get_roidb_from_zipfile(file_path, return_extra=False):
    """Get roidb from zipfile."""
    with zipfile.ZipFile(file_path, "r") as fid:
        namelist = fid.namelist()
        jsonfile = [n for n in namelist if n.endswith(".json")]
        if not jsonfile:
            raise FileNotFoundError
        json_data = json.loads(fid.read(jsonfile[0]))

        with no_print():
            import pdebug.piata.coco

            roidb = Input(json_data, name="coco").get_roidb(as_dict=True)

    if return_extra:
        categories = [c["name"] for c in json_data["categories"]]
        desc = json_data["info"]["description"]
        version = json_data["info"]["version"]
        return roidb, categories, desc, version
    else:
        return roidb


@otn_manager.NODE.register(name="merge_cvat_zip")
def merge_cvat_zip(
    zip_dir: str,
    merge_func: Callable = None,
    skip_list: List[str] = None,
    do_vis: bool = False,
    remove_raw_zip: bool = False,
    output: str = None,
):
    """Merge zip files by date."""
    if not merge_func:
        print("merge_func not found, use default.")
        merge_func = lambda x: output
    if not output:
        output = "merge_cvat_zip_output"
    os.makedirs(output, exist_ok=True)
    # typer.echo(typer.style(f"hello, tool", fg=typer.colors.GREEN))

    zipfiles = sorted(
        [
            os.path.join(zip_dir, x)
            for x in sorted(os.listdir(zip_dir))
            if x.endswith(".zip")
        ]
    )

    zips = defaultdict(list)
    for f in zipfiles:
        key = merge_func(f)
        if skip_list and key in skip_list:
            print(f"skip {f}")
            continue
        zips[key].append(f)

    t = tqdm.tqdm(total=len(zips))
    for key in zips:
        t.update()
        output_zip = os.path.join(output, f"{key}.zip")
        f_write = zipfile.ZipFile(output_zip, "w")

        all_roidb = {}
        coco_categories, coco_description, coco_version = None, None, None

        tt = tqdm.tqdm(total=len(zips[key]), desc=key)
        for f in zips[key]:
            tt.update()
            try:
                (
                    roidb,
                    coco_categories,
                    coco_description,
                    coco_version,
                ) = get_roidb_from_zipfile(f, return_extra=True)
            except FileNotFoundError as e:
                print(f"No json found in {f}, skip")
                continue
            try:
                reader = Input(f, name="imgzip").get_reader()
            except IndexError as e:
                print(f"{f} has no image files, skip")
                continue
            if len(reader) == 0:
                print(f"{f} has no image files, skip")
                continue

            old_len = len(all_roidb)
            all_roidb.update(roidb)
            if old_len + len(roidb) != len(all_roidb):
                __import__("ipdb").set_trace()
                pass
            print(f"append roidb: {old_len} => {len(all_roidb)}")

            for imgfile in reader.imglist:
                img = reader.imread(imgfile)
                retval, img_buf = cv2.imencode(".png", img)
                f_write.writestr(f"{key}/{os.path.basename(imgfile)}", img_buf)

                if do_vis:
                    roi = roidb[os.path.basename(imgfile)]
                    vis_img = Roidb.vis_segmentation(roi, img, "contour")
                    cv2.imwrite(os.path.basename(imgfile), vis_img)

        roidb = [all_roidb[k] for k in all_roidb]
        roidb = sorted(roidb, key=lambda x: x["image_name"])
        save_jsonfile = os.path.join(output, f"{key}.json")
        Output(
            roidb,
            name="coco",
            categories=coco_categories,
            description=coco_description,
            version=coco_version,
        ).save(save_jsonfile)
        f_write.write(save_jsonfile, arcname=os.path.basename(save_jsonfile))
        os.remove(save_jsonfile)
        f_write.close()

        if remove_raw_zip:
            for f in zips[key]:
                os.remove(f)
                print(f"delete {f}")

    return output


def merge_polygon(
    roi, categories=None, label_info=None, min_area: float = 100
):

    if "label" in roi and "category_id" not in roi:
        if not categories:
            categories = load_categories()
        category_id = [categories.index(l) for l in roi["label"]]
        roi["category_id"] = category_id

    if label_info:
        for l in set(roi["label"]):
            label_info[l] += 1

    assert IMANTICS_INSTALLED, "imantics is requred to run process_cvat."
    return Segmentation.to_mask(
        roi["segmentation"],
        roi["category_id"],
        roi["image_height"],
        roi["image_width"],
        min_area=100,
    )


@otn_manager.NODE.register(name="process_cvat")
def process_cvat(
    root: str,
    output: str = "tmp_cvat",
    imgdir_prefix: str = None,
    save_in_files: bool = True,
    do_statistic: bool = False,
    cache: bool = False,
    vis_output: str = None,
    num_workers: int = 0,
    rename_mask_name: bool = False,
    skip_xmlfiles: List[str] = (),
    topk: int = None,
    frate: float = None,
    semantic_name: str = "semantic_pcd",
):
    """Process cvat segmentation annotations.

    Args:
        root: xml folder.
        output: processed folder,
        imgdir_prefix: add to imgdir prefix
        save_in_files: save mask and image info to output.
        do_statistic: show statistic info.
        cache: keep mask_output as cache.
        vis_output: vis output folder. If not set, skip vis.
        num_workers: num workers.
        rename_mask_name: append task id to mask png filename.
        skip_xmlfiles: skip xmlfiles for debug.
        topk: topk image for debug.
        frate: do frate to crop image.
    """
    xmlfiles = [
        os.path.join(root, x)
        for x in sorted(os.listdir(root))
        if x.endswith(".xml") and os.path.basename(x) not in skip_xmlfiles
    ]
    output = os.path.abspath(output)
    mask_output = os.path.join(output, "mask")

    if cache and os.path.exists(mask_output):
        print("")
        image_length = sum(
            [
                len(Input(x, name="cvat_segmentation").get_roidb())
                for x in xmlfiles
            ]
        )
        if image_length == len(os.listdir(mask_output)):
            print(f"{mask_output} exists, skip")
            return output

    if os.path.exists(mask_output) and not cache:
        shutil.rmtree(mask_output)
    os.makedirs(mask_output, exist_ok=True)

    categories = load_categories(name=semantic_name)
    classes = [obj.split("|")[1] for obj in categories]
    colors = load_colors(name=semantic_name, as_tuple=True)

    do_process = save_in_files

    if vis_output:
        if os.path.exists(vis_output):
            shutil.rmtree(vis_output)
        os.makedirs(vis_output, exist_ok=True)

    if frate:
        assert frate <= 1.0
        assert do_process, "`frate` is only used when `do_process` is True."
        image_output = os.path.join(output, "image")
        if os.path.exists(image_output) and not cache:
            shutil.rmtree(image_output)
        os.makedirs(image_output, exist_ok=True)

    @mp(nums=num_workers)
    def _process(roidb, xmlfile_ext):
        info = []
        label_info = OrderedDict({name: 0 for name in categories})

        t = tqdm.tqdm(total=len(roidb), desc=xmlfile_ext)
        for roi in roidb:
            t.update()
            mask = merge_polygon(roi, categories, label_info)

            imgdir = os.path.join(root, f"images_{xmlfile_ext}")
            if imgdir_prefix:
                imgdir = imgdir_prefix + imgdir
            assert os.path.exists(imgdir), f"{imgdir} not exists"

            image_file = os.path.join(imgdir, roi["image_name"])
            if not os.path.exists(image_file):
                print(
                    f"{image_file} not exists, please check imgdir processing code in cvat_process.py"
                )
                return

            image_file = os.path.abspath(image_file)
            assert "image_name" in roi
            image_basename = os.path.splitext(
                os.path.basename(roi["image_name"])
            )[0]
            if rename_mask_name:
                image_name = f"task{xmlfile_ext}_{image_basename}.png"
            else:
                image_name = f"{image_basename}.png"

            if frate:
                image = cv2.imread(image_file)
                assert image.shape[:2] == mask.shape[:2]
                h, w = image.shape[:2]
                new_h, new_w = int(h * frate), int(w * frate)
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                new_image = image[y1 : y1 + new_h, x1 : x1 + new_w]
                mask = mask[y1 : y1 + new_h, x1 : x1 + new_w]
                image_file = os.path.join(
                    image_output, f"task{xmlfile_ext}_{image_basename}.png"
                )
                if save_in_files and not os.path.exists(image_file):
                    cv2.imwrite(image_file, new_image)

            savename = os.path.join(mask_output, f"{image_name}")
            if save_in_files and not os.path.exists(savename):
                cv2.imwrite(savename, mask)

            info.append([image_file, savename])

            if vis_output:
                assert os.path.exists(image_file), f"{image_file} not exists."
                image = cv2.imread(image_file)
                vis_image = draw.semseg(
                    mask, image, classes=classes, colors=colors
                )
                os.makedirs(vis_output, exist_ok=True)
                vis_savename = os.path.join(vis_output, f"{image_name}")
                cv2.imwrite(vis_savename, vis_image)
        # TODO: tricky, append label_info to info
        info.append(label_info)
        return info

    info = []
    label_info = OrderedDict({name: 0 for name in categories})

    for xmlfile in xmlfiles:
        roidb = Input(
            xmlfile, name="cvat_segmentation", append_task_name=False
        ).get_roidb()
        if topk:
            print(f"topk: {len(roidb)} => {topk}")
            roidb = roidb[:topk]
        # task_110.xml => 110
        xmlfile_ext = os.path.basename(xmlfile).split("_")[1].split(".")[0]
        res = _process(roidb, xmlfile_ext)
        info_i = [r for r in res if isinstance(r, list)]
        label_info_i = [r for r in res if isinstance(r, dict)]

        assert len(info_i) == len(roidb)
        info.extend(info_i)
        if save_in_files and not cache:
            assert len(os.listdir(mask_output)) == len(
                info
            ), f"{len(os.listdir(mask_output))} != {len(info)}"

        for label_info_ii in label_info_i:
            for name, count in label_info_ii.items():
                label_info[name] += count

    if save_in_files:
        info_txt = os.path.join(output, "image_anno_info.txt")
        with open(info_txt, "w") as fid:
            for each_info in info:
                image_file, mask_file = each_info
                fid.write(f"{image_file} {mask_file}\n")
        print(f"saved to {output}")

    if do_statistic:
        typer.echo(
            typer.style(f"images num sum: {len(info)}", fg=typer.colors.GREEN)
        )
        with open(f"{output}/summary.txt", "w") as fid:
            for i, k in enumerate(label_info):
                line = str(i) + ", " + k + ", " + str(label_info[k])
                fid.write(line + "\n")

        if RICH_INSTALLED:
            from rich.console import Console
            from rich.table import Table

            table = Table(title=f"{os.path.basename(output)} label info")
            table.add_column("id", justify="right", style="cyan", no_wrap=True)
            table.add_column("name", style="magenta")
            table.add_column("num images", justify="right", style="green")
            for i, k in enumerate(label_info):
                if "VOID" in k:
                    continue
                table.add_row(str(i), k, str(label_info[k]))
            console = Console()
            console.print(table)

    return output


@otn_manager.NODE.register(name="upload_cvat")
def upload_cvat(
    imgdir: str,
    anno: str = "",
    name: str = None,
    ipy: bool = False,
    organization: str = None,
    objects: List[Dict] = None,
    palette_file: str = None,
):
    """Upload data to cvat.

    Args:
        imgdir: input image dir.
        anno: annotation file, default is "".
        name: task name, default is None.
        ipy: whether to use ipython, default is False.
        organization: organization name, default is None.
        objects: objects list, default is None.
        palette_file: palette file, default is None.

    Returns:
        output: output path.
    """

    if not name:
        name = "__".join(imgdir.split("/")[-2:])

    # DeferredTqdmProgressReporter requiring high version cvat_sdk and python3.8
    from cvat_sdk.core.helpers import DeferredTqdmProgressReporter

    """Add data to cvat."""
    with make_client(host=HOST, credentials=(USER, PASSWORD)) as client:
        if ipy:
            __import__("IPython").embed()
            return

        imgdir = os.path.abspath(imgdir)
        imgfiles = Input(imgdir, name="imgdir").get_reader().imgfiles
        typer.echo(
            typer.style(f"Found {len(imgfiles)} images", fg=typer.colors.GREEN)
        )

        print(f"task name: {name}")

        if objects is None:
            # set color from cvat_palette.json
            if not palette_file:
                palette_file = os.path.expanduser(
                    "~/code/pat/projects/semantic_pcd/cvat_annotation/cvat_palette.json"
                )
            cvat_palette = json.load(open(palette_file, "r"))
            cvat_palette = {k["name"]: k for k in cvat_palette}
            objects = SEMANTIC_PCD_CATEGORIES
            for obj in objects:
                key_name = obj["label_zh"] + " | " + obj["label"]
                obj["color"] = cvat_palette[key_name]["color"]

        task_spec = {
            "name": name,
            "labels": [
                {
                    "name": obj["label_zh"] + " | " + obj["label"],
                    "color": obj["color"],
                    "type": "polygon",
                    "attributes": [],
                }
                for obj in objects
            ],
            "segment_size": 100,
        }

        if organization:
            client.organization_slug = organization

        print("Creating task ...")
        task = client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.LOCAL,
            resources=imgfiles,
            pbar=DeferredTqdmProgressReporter(),
            annotation_path=anno,
            annotation_format="COCO 1.0",
            data_params=None,
        )
        assert task.size == len(imgfiles)
        print(f"Done ({HOST}/tasks/{task.id}, {task.name}).")
    return ""


@otn_manager.NODE.register(name="upload_cvat_kps")
def upload_cvat_kps(
    imgdir: str,
    anno: str = "",
    name: str = None,
    ipy: bool = False,
    organization: str = None,
    topk: int = None,
    annotation_format: str = "CVAT 1.1",
):
    """Upload kps in roidb pickle to cvat."""
    # DeferredTqdmProgressReporter requiring high version cvat_sdk and python3.8
    from cvat_sdk.core.helpers import DeferredTqdmProgressReporter

    if not name:
        name = os.path.basename(imgdir)

    """Add data to cvat."""
    with make_client(host=HOST, credentials=(USER, PASSWORD)) as client:
        imgdir = os.path.abspath(imgdir)
        imgfiles = (
            Input(imgdir, name="imgdir", topk=topk).get_reader().imgfiles
        )
        typer.echo(
            typer.style(f"Found {len(imgfiles)} images", fg=typer.colors.GREEN)
        )

        if anno.endswith(".pkl"):
            annotation_format = "CVAT 1.1"
            tmp_anno_file = f"/tmp/{os.path.basename(anno)}".replace(
                ".pkl", ".xml"
            )
            roidb = Input(anno).get_roidb()
            if topk and topk > 0:
                roidb = [
                    roi
                    for roi in roidb
                    if os.path.join(
                        imgdir, os.path.basename(roi["image_name"])
                    )
                    in imgfiles
                ]
            Output(roidb, name="cvat_1_1").save(tmp_anno_file)
            anno = tmp_anno_file

        print(f"task name: {name}")
        task_spec = {
            "name": name,
            "labels": [
                {
                    "name": "point",
                    "color": "#f078f0",
                    "type": "points",
                    "attributes": [],
                }
            ],
        }

        if organization:
            client.organization_slug = organization

        print("Creating task ...")
        task = client.tasks.create_from_data(
            spec=task_spec,
            resource_type=ResourceType.LOCAL,
            resources=imgfiles,
            pbar=DeferredTqdmProgressReporter(),
            annotation_path=anno,
            annotation_format=annotation_format,
            data_params=None,
        )
        assert task.size == len(imgfiles)
        print(f"Done ({HOST}/tasks/{task.id}, {task.name}).")
    return ""


@otn_manager.NODE.register(name="download_cvat_kps")
def download_cvat_kps(
    task_ids: str,
    output: str,
    cache: bool = False,
):
    """Download cvat kps.

    Args:
        task_ids: task ids, split by ",".
        output: output path, default is "tmp_download_cvat_kps.pkl".
        cache: whether to cache the prediction results, default is False.

    Returns:
        output: output path.
    """
    if not output:
        output = "tmp_download_cvat_kps.pkl"

    if cache and os.path.exists(output):
        typer.echo(typer.style(f"Found {output}, skip", fg=typer.colors.WHITE))
        return output

    cvat_download_output = "/tmp/cvat_download"
    download_cvat(task_ids, False, cache, cvat_download_output)
    cvat_anno_file = os.path.join(cvat_download_output, f"task_{task_ids}.xml")
    roidb = Input(cvat_anno_file, name="cvat_keypoints").get_roidb()
    with open(output, "wb") as fid:
        pickle.dump(roidb, fid)
    typer.echo(typer.style(f"save to {output}", fg=typer.colors.WHITE))
    return output


if __name__ == "__main__":
    typer.run(download_cvat)
