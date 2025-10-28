import glob
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import List, Optional, Union

from pdebug.data_types import Camera
from pdebug.otn import manager as otn_manager
from pdebug.piata import Input
from pdebug.utils.decorator import mp
from pdebug.utils.fileio import load_content
from pdebug.utils.waicai_utils import (
    get_sensor_yaml_depth,
    get_sensor_yaml_roomplan,
    prepare_biaoding_info,
)

import cv2
import tqdm
import typer


def get_sensor_yaml_default(imgdir):
    yamlfile = os.path.join(
        os.path.dirname(__file__), "undistort_cam5_sensor.yaml"
    )
    assert os.path.exists(yamlfile)
    return yamlfile


_UNDISTORT_CAMERAS = {}


@otn_manager.NODE.register(name="undistort_image")
def undistort_image(
    image,
    camera_yaml,
    output: str = None,
    frate: float = 0.5,
    noodlesoup_root: str = "/mnt/bn/picoroomplan/wbw/ws/roomplan/3rdparty",
    cache: bool = False,
):
    """Undistort image using photoid.

    Args:
        image: input rgb or rgb file.
        camera_yaml: rgb camera calib yaml.
        output: optional, output filename is image is filename
        frate: frate used in undistortion.
        noodlesoup_root: noodlesoup root.
    Returns:
        undistort image
    Type: image -> image

    """
    src, loaded_from_file = load_content(image)
    if src is None:
        print(f"Bad image file: {image}")
        return

    if cache and loaded_from_file:
        savename = os.path.join(output, os.path.basename(image))
        if os.path.exists(savename):
            return savename

    def __undistort_photoid(src):
        import photoid

        global _UNDISTORT_CAMERAS
        if camera_yaml not in _UNDISTORT_CAMERAS:
            camera1 = photoid.Camera(
                camera_yaml, photoid.DeviceParameters().T_B_I
            )
            camera1.initUndistortRectifyMap(
                frate
            )  # may cause ratio-mismatch problem
            camera2 = Camera.from_yamlfile(camera_yaml)
            _UNDISTORT_CAMERAS[camera_yaml] = (camera1, camera2)
        PC = _UNDISTORT_CAMERAS.get(camera_yaml)

        height, width = PC[1].I.height, PC[1].I.width
        if src.shape[0] != height or src.shape[1] != width:
            src = cv2.resize(src, (width, height))

        dst = PC[0].undistortImage(src)
        return dst

    def __undistort_noodlesoup(src):
        sys.path.append(noodlesoup_root)
        from noodlesoup.cv.distort import undistort_image_with_Knew
        from noodlesoup.io.calibration import read_factory_cam_params

        global _UNDISTORT_CAMERAS
        if camera_yaml not in _UNDISTORT_CAMERAS:
            camera1 = read_factory_cam_params(camera_yaml)
            camera2 = Camera.from_yamlfile(camera_yaml)
            _UNDISTORT_CAMERAS[camera_yaml] = (camera1, camera2)
        PC = _UNDISTORT_CAMERAS.get(camera_yaml)
        (
            xi,
            K,
            dist_coeffs,
            T_B_cam,
            image_width_height,
            distort_type,
        ) = PC[0]

        height, width = PC[1].I.height, PC[1].I.width
        if src.shape[0] != height or src.shape[1] != width:
            src = cv2.resize(src, (width, height))

        f_scale = frate * 6.0
        dst, Knew = undistort_image_with_Knew(
            distort_type, src, K, dist_coeffs, xi, f_scale=f_scale
        )
        return dst

    try:
        import photoid

        dst = __undistort_photoid(src)
    except ImportError as e:
        dst = __undistort_noodlesoup(src)

    if loaded_from_file:
        if not output:
            name, ext = os.path.splitext(image)
            savename = name + "_undistort" + ext
        else:
            os.makedirs(output, exist_ok=True)
            savename = os.path.join(output, os.path.basename(image))
        cv2.imwrite(savename, dst)
        return savename
    else:
        return dst


@otn_manager.NODE.register(name="undistort_images")
def main(
    path: str,
    output: str = "undistort_output",
    frate: float = 0.5,
    num_workers: int = 0,
    cache: bool = True,
    debug_check_mav0: bool = False,
    debug_ipdb: bool = False,
    debug_imgdirs: List[str] = None,
    debug_check_rgb: bool = False,
    sensor_yaml: str = None,
):
    """Undistort images.

    Args:
        path: input image files. It can be image file list txt, or image folder path.

    Example:
        >> otn-cli --node undistort_images --path cam5  --sensor-yaml mav0/cam5/sensor.yaml
    """
    if os.path.isfile(path) and os.path.exists(path):
        with open(path, "r") as fid:
            datalist = [l.strip() for l in fid.readlines()]
    elif os.path.isdir(path) and os.path.exists(path):
        datalist = Input(path, name="imgdir").get_reader().imgfiles
    else:
        raise ValueError(f"Unknown path: {path}")

    if not cache and os.path.exists(output):
        shutil.rmtree(output)

    imgdirs = defaultdict(list)
    for imagefile in datalist:
        imgdir = os.path.dirname(imagefile)
        imgdirs[imgdir].append(imagefile)

    typer.echo(
        typer.style(f"Found {len(imgdirs)} imgdirs.", fg=typer.colors.GREEN)
    )

    prepare_biaoding_info(imgdirs)

    if debug_check_mav0:
        num_workers = 0
    if debug_ipdb:
        num_workers = 0
    if debug_imgdirs:
        num_workers = 0
    if debug_check_rgb:
        num_workers = 0

    num_workers = min(num_workers, len(imgdirs))

    @mp(nums=num_workers)
    def _process(process_id, imgdirs_keys):
        t = tqdm.tqdm(total=len(imgdirs_keys), desc=f"process-{process_id}")
        for imgdir in imgdirs_keys:
            t.update()
            if debug_imgdirs and not imgdir in debug_imgdirs:
                continue

            try:
                if sensor_yaml:
                    _sensor_yaml = sensor_yaml
                else:
                    # _sensor_yaml = get_sensor_yaml_default(imgdir)
                    # _sensor_yaml = get_sensor_yaml_depth(imgdir)
                    _sensor_yaml = get_sensor_yaml_roomplan(
                        imgdir, debug_ipdb=debug_ipdb
                    )
            except Exception as e:
                print(f"When processing {imgdir}, meet bug ...")
                raise e

            if debug_check_mav0:
                continue

            imglist = imgdirs[imgdir]
            if len(imgdirs_keys) == 1:
                undistort_output_i = output
            else:
                # create one nested folder if multiple imgdirs need to process.
                undistort_output_i = os.path.join(
                    output, imgdir.replace("/", "__")
                )
            if (
                cache
                and os.path.exists(undistort_output_i)
                and len(os.listdir(undistort_output_i)) == len(imglist)
            ):
                continue

            tt = tqdm.tqdm(
                total=len(imglist),
                desc=f"undistort {os.path.basename(imgdir)}",
            )
            try:
                for imgfile in imglist:
                    tt.update()
                    if debug_check_rgb:
                        img = cv2.imread(imgfile)
                        if img is None:
                            __import__("ipdb").set_trace()
                            pass
                        continue
                    undistort_image(
                        imgfile,
                        _sensor_yaml,
                        undistort_output_i,
                        frate,
                        cache=cache,
                    )
            except Exception as e:
                print(f"When processing {imgdir}, meet bug ...")
                raise e

    imgdirs_keys = list(imgdirs.keys())
    _process(imgdirs_keys)

    return output


if __name__ == "__main__":
    typer.run(main)
