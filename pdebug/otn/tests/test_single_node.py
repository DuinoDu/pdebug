import os

from pdebug.otn import manager as otn_manager

import cv2
import numpy as np


def test_single_node(tmpdir):
    imgdir = os.path.join(tmpdir, "test_images")
    os.makedirs(imgdir, exist_ok=True)
    cv2.imwrite(f"{imgdir}/1.png", np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(f"{imgdir}/2.png", np.zeros((100, 100, 3), dtype=np.uint8))
    output = os.path.join(tmpdir, "result")
    config_str = """Input = {}
Input["path"] = "%s"
Input["name"] = "imgdir"

Output = {}
Output["path"] = "%s"
Output["name"] = "imgdir"
Output["ext"] = ".jpg"

Main = "to_gray"

def to_gray(**kwargs):
    image = kwargs.get("image")
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

""" % (
        imgdir,
        output,
    )
    config_file = os.path.join(tmpdir, "single_node_config.py")
    with open(config_file, "w") as fid:
        fid.write(config_str)

    otn_manager.NODE.get("single_node")(config_file)
    assert len(os.listdir(output)) == 2
