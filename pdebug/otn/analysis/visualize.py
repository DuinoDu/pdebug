import os
from typing import Optional

from pdebug.data_types import Camera, Segmentation
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.ddd import generate_pcd_from_depth
from pdebug.utils.decorator import mp
from pdebug.utils.semantic_types import load_categories, load_colors
from pdebug.utils.types import as_list_or_tensor
from pdebug.visp import Colormap, draw

import cv2
import mmcv
import numpy as np
import tqdm


class Roidb:
    """Utility class for region of interest database operations."""

    @staticmethod
    def print_info(roidb):
        """Print information about the roidb structure."""
        print("======= roidb info =======")
        print(f"length: {len(roidb)}")
        print(f"keys: {roidb[0].keys()}")
        print("==========================")

    @staticmethod
    def vis_segmentation(roi, image, mask_or_contour):
        cats = list(set(roi["category_id"]))
        classes = [x.split("|")[1].strip() for x in load_categories()]
        colors = load_colors(as_tuple=True)

        image = copy.deepcopy(image)

        if mask_or_contour == "contour":
            for cat_id, contour in zip(
                roi["category_id"], roi["segmentation"]
            ):
                color = colors[cats.index(cat_id)]
                if mmcv.is_list_of(contour, list) and len(contour) == 1:
                    contour = contour[0]
                image = draw.contour(contour, image, color=color)
        elif mask_or_contour == "mask":
            mask = Segmentation.to_mask(
                roi["segmentation"],
                roi["category_id"],
                roi["image_height"],
                roi["image_width"],
                min_area=100,
            )
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            image = draw.semseg(mask, image, colors=colors, classes=classes)

            # bbox = Polygons([contour]).bbox()
            # area = bbox.area()
            # x = (bbox.min_point[0] + bbox.max_point[0]) // 2
            # y = (bbox.min_point[1] + bbox.max_point[1]) // 2
            #
        return image

    @staticmethod
    def vis_data_dict(
        data_dict,
        batch_idx=0,
        return_image=False,
        semantic_name="semantic_pcd",
    ):
        rgb_resize_factor = 1.0
        if "rgb_resize_factor" in data_dict:
            rgb_resize_factor = as_list_or_tensor(
                data_dict["rgb_resize_factor"]
            )[batch_idx]
            rgb_resize_factor = float(rgb_resize_factor)

        assert "image" in data_dict
        image = as_list_or_tensor(data_dict["image"])[batch_idx]
        if hasattr(image, "numpy"):
            image = image.numpy()
        else:
            image = cv2.imdecode(
                np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
        if rgb_resize_factor != 1.0:
            image = cv2.resize(
                image,
                (
                    int(image.shape[1] / rgb_resize_factor),
                    int(image.shape[0] / rgb_resize_factor),
                ),
            )

        vis_img = copy.deepcopy(image)

        if (
            "pose" in data_dict
            and data_dict["pose"] != ["None"]
            and data_dict["pose"] != b""
            and data_dict["pose"] != [b""]
        ):
            pose = np.frombuffer(
                as_list_or_tensor(data_dict["pose"])[batch_idx]
            )
            vis_pose = draw.pose(image, pose, draw_text=True)
            vis_img = vis_pose

        mask_color = None
        if (
            "label" in data_dict
            and data_dict["label"] != b""
            and data_dict["label"][batch_idx] != b""
            and data_dict["label"][batch_idx] != [b""]
        ):
            label = as_list_or_tensor(data_dict["label"])[batch_idx]
            if hasattr(label, "numpy"):
                label = label.numpy()
            else:
                label = cv2.imdecode(
                    np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
            if rgb_resize_factor != 1.0:
                label = cv2.resize(
                    label,
                    (
                        int(label.shape[1] / rgb_resize_factor),
                        int(label.shape[0] / rgb_resize_factor),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            if label.shape[:2] != image.shape[:2]:
                label = cv2.resize(
                    label,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            colors = load_colors(name=semantic_name, as_tuple=True)
            classes = load_categories(name=semantic_name, simple=True)
            vis_seg, mask_color = draw.semseg(
                label,
                image=image,
                colors=colors,
                classes=classes,
                return_mask_color=True,
            )
            vis_img = np.concatenate((vis_img, vis_seg), axis=1)

        if (
            "depth" in data_dict
            and data_dict["depth"] != ["None"]
            and data_dict["depth"] != b""
            and data_dict["depth"] != [b""]
        ):
            depth = as_list_or_tensor(data_dict["depth"])[batch_idx]
            if hasattr(depth, "numpy"):
                depth = depth.numpy()
            else:
                assert "depth_shape" in data_dict
                depth_shape = eval(
                    as_list_or_tensor(data_dict["depth_shape"])[batch_idx]
                )
                depth = np.frombuffer(depth, dtype=np.uint16).reshape(
                    depth_shape
                )

            depth = depth.astype(np.float32) * 0.001
            if rgb_resize_factor != 1.0:
                depth = cv2.resize(
                    depth,
                    (
                        int(depth.shape[1] / rgb_resize_factor),
                        int(depth.shape[0] / rgb_resize_factor),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            vis_depth = draw.depthpoint(depth, image=image)
            vis_img = np.concatenate((vis_img, vis_depth), axis=1)

        if "K_rgb" in data_dict and "T_WH_rgb" in data_dict:
            assert "depth" in data_dict
            K_rgb = as_list_or_tensor(data_dict["K_rgb"])[batch_idx]
            K_rgb = np.frombuffer(K_rgb).reshape(3, 3)
            if rgb_resize_factor != 1.0:
                K_rgb = copy.deepcopy(K_rgb)
                K_rgb[:2, :3] /= rgb_resize_factor
            T_WH_rgb = as_list_or_tensor(data_dict["T_WH_rgb"])[batch_idx]
            T_WH_rgb = np.frombuffer(T_WH_rgb).reshape(4, 4)
            camera = Camera(T_WH_rgb, K_rgb)
            pcd_bgr = mask_color if mask_color is not None else image
            pcd_rgb = cv2.cvtColor(pcd_bgr, cv2.COLOR_BGR2RGB)
            pcd = generate_pcd_from_depth(
                depth, camera, rgb=pcd_rgb, coordinate_type="photoid"
            )
            vis_img = {"image": vis_img, "pointcloud": pcd}

        if return_image:
            return vis_img, image
        else:
            return vis_img

    @staticmethod
    def vis_keypoints(roi, image):
        image = copy.deepcopy(image)
        image = draw.keypoints(image, roi["keypoints"], radius=10)
        return image

    @staticmethod
    def vis_boxes(roi, image):
        image = copy.deepcopy(image)
        image = draw.boxes(image, roi["boxes"])
        return image


@otn_manager.NODE.register(name="vis_roidb")
def main(
    path: str,
    imgdir: str = None,
    name: str = None,
    output: str = "vis_roidb_output",
    num_workers: int = 0,
    ####
    keys: str = None,
    #### segmentation ####
    mask_or_contour: str = "mask",
):
    """Visualize roidb."""
    if name == "coco":
        import pdebug.piata.coco

    roidb = Input(path, name=name).get_roidb()
    if not imgdir:
        # guess imgdir from path
        imgdir = os.path.splitext(os.path.basename(path))[0].replace("__", "/")

    if not os.path.exists(imgdir):
        typer.echo(
            typer.style(
                f"imgdir not found, use 'image_file' in roidb",
                fg=typer.colors.YELLOW,
            )
        )

    Roidb.print_info(roidb)
    keys = keys.split(",") if keys else []
    if not keys:
        keys = ["boxes", "keypoints", "segmentation"]

    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)

    @mp(nums=int(num_workers))
    def _process(roidb):
        t = tqdm.tqdm(total=len(roidb))
        for roi in roidb:
            t.update()
            if os.path.exists(imgdir):
                image_file = os.path.join(imgdir, roi["image_name"])
            else:
                assert "image_file" in roi
                image_file = roi["image_file"]
            assert os.path.isfile(image_file) and os.path.exists(
                image_file
            ), f"{image_file} not exists."
            image = cv2.imread(image_file)
            vis_res = [image]
            for key in keys:
                common_vis = None
                if key == "boxes" and "boxes" in roi:
                    vis_boxes = Roidb.vis_boxes(roi, image)
                    vis_res = [vis_boxes]
                if key == "keypoints" and "keypoints" in roi:
                    vis_kps = Roidb.vis_keypoints(roi, vis_res[0])
                    vis_res = [vis_kps]
                if key == "segmentation" and "segmentation" in roi:
                    vis_segm = Roidb.vis_segmentation(
                        roi, image, mask_or_contour
                    )
                    vis_res.append(vis_segm)
            vis_all = np.concatenate(vis_res, axis=1)
            savename = os.path.join(output, os.path.basename(image_file))
            cv2.imwrite(savename, vis_all)

    _process(roidb)
    return output


@otn_manager.NODE.register(name="vis_semseg_parquet")
def vis_semseg_parquet(
    path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    num_readers: int = 0,
    output: str = None,
    roidb_path: str = None,
    roidb_name: str = "coco",
    savename_by_iou: bool = False,
    semantic_name: str = "semantic_pcd",
    ##### select data #####
    iou_threshold: float = None,
    save_raw_image: bool = False,
    raw_image_output: str = None,
):
    """Visualize semseg parquet files, used to check data."""
    reader = Input(
        path,
        name="cruise_rgbd_semseg",
        num_workers=num_workers,
        num_readers=num_readers,
        batch_size=batch_size,
        resize_image=False,
    ).get_reader()

    if not output:
        output = f"tmp_vis_{os.path.basename(path)}"

    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output, exist_ok=True)
    colors = load_colors(as_tuple=True)
    classes = load_categories(simple=True)

    roidb = None
    if roidb_path and os.path.exists(roidb_path):
        if roidb_name == "coco":
            import pdebug.piata.coco
        roidb = Input(roidb_path, name=roidb_name).get_roidb()
        Roidb.print_info(roidb)
        # convert to dict
        roidb = {roi["image_name"]: roi for roi in roidb}

    iou = None

    t = tqdm.tqdm(total=len(reader))
    for i in range(len(reader)):
        data_dict = next(reader)
        t.update()

        if "image" not in data_dict:
            print("`image` not found in parquet file.")
            continue

        for batch_idx in range(batch_size):
            vis_img, raw_img = Roidb.vis_data_dict(
                data_dict,
                batch_idx,
                return_image=True,
                semantic_name=semantic_name,
            )
            if roidb is not None:
                assert "image_name" in data_dict
                roi = roidb[data_dict["image_name"][batch_idx]]
                vis_roi = copy.deepcopy(raw_img)
                for key in roi:
                    if key == "segmentation":
                        roi["category_id"] = [
                            cat - 1 for cat in roi["category_id"]
                        ]
                        roi["category_id"] = [
                            34 if cat == 17 else cat
                            for cat in roi["category_id"]
                        ]
                        vis_roi = Roidb.vis_segmentation(roi, raw_img, "mask")
                    elif key == "iou":
                        iou = roi[key][0]
                        vis_roi = cv2.putText(
                            vis_roi,
                            f"iou: {iou:.3f}",
                            (100, 100),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                vis_img = np.concatenate((vis_img, vis_roi), axis=1)
                if "iou" in roi and iou_threshold and iou > iou_threshold:
                    continue

            if "image_name" in data_dict:
                image_name = os.path.basename(
                    data_dict["image_name"][batch_idx]
                )
            else:
                image_name = (
                    f"{os.path.basename(path)}_index_{i}_{batch_idx}.png"
                )
            if savename_by_iou and iou is not None:
                image_name = f"iou_{iou:.3f}_{image_name}"
            vis_savename = os.path.join(output, image_name)

            if isinstance(vis_img, np.ndarray):
                cv2.imwrite(vis_savename, vis_img)
            else:
                cv2.imwrite(vis_savename, vis_img["image"])
                pcd_savename = os.path.join(output, image_name[:-4] + ".ply")
                vis_img["pointcloud"].to_ply(pcd_savename)

            if save_raw_image:
                assert raw_image_output
                os.makedirs(raw_image_output, exist_ok=True)
                raw_savename = os.path.join(raw_image_output, image_name)
                cv2.imwrite(raw_savename, raw_img)

    reader.loader.terminate()

    if save_raw_image:
        if not os.path.exists(raw_image_output):
            print("no raw image found, exit program.")
            import sys

            sys.exit()
        return raw_image_output
    else:
        return output


@otn_manager.NODE.register(name="vis_semseg_parquet_roidb")
def vis_semseg_parquet_roidb(
    path: str,
    roidb_path: str,
    **kwargs,
):
    """Visualize semseg parquet files, used to check data."""
    return vis_semseg_parquet(path=path, roidb_path=roidb_path, **kwargs)


if __name__ == "__main__":
    typer.run(main)
