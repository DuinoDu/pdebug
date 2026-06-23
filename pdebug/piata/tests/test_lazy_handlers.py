import os

from pdebug.piata import Input, Output
from pdebug.piata import input as piata_input
from pdebug.piata import output as piata_output

import cv2
import numpy as np


def test_input_loads_imgdir_handler_on_demand(tmpdir):
    imgdir = os.path.join(tmpdir, "images")
    os.makedirs(imgdir, exist_ok=True)
    cv2.imwrite(f"{imgdir}/1.png", np.zeros((8, 8, 3), dtype=np.uint8))

    reader = Input(imgdir, name="imgdir").get_reader()

    assert len(reader) == 1


def test_output_loads_imgdir_writer_on_demand(tmpdir):
    output = os.path.join(tmpdir, "out")
    images = [np.zeros((8, 8), dtype=np.uint8)]

    Output(images, name="imgdir", ext=".png").save(output)

    assert os.path.exists(os.path.join(output, "0.png"))


def test_lazy_handler_maps_cover_external_registrations():
    assert set(piata_input._ROIDB_MODULES) == {
        "coco",
        "simpletxt",
        "llava_json",
        "cvat_segmentation",
        "cvat_keypoints",
        "voc",
        "mmpose",
        "vott",
        "labelme",
    }
    assert set(piata_input._SOURCE_MODULES) == {
        "image",
        "imgdir",
        "imgzip",
        "video",
        "simpletxt_reader",
        "parquet",
        "parquet_semseg_v1",
        "spatialmp4",
        "lance_vita",
        "dddsemseg",
        "ddddata_scannet",
        "ddddata_apple_roomplan",
        "ddddata_nyuv2",
        "ddddata_nyuv2_ESANet",
        "ddddata_arkitscenes_raw",
        "ddddata_arkitscenes_3dod",
        "ddddata_sunrgbd",
        "ddddata_2d3dsemantics",
        "ADE20K",
        "S3DIS",
        "ADEChallengeData2016",
    }
    assert set(piata_output._OUTPUT_MODULES) == {
        "imgdir",
        "video",
        "video_ffmpeg",
    }
