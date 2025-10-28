"""
Run label-studio serve:
    pip3 install label-studio
    label-studio --port 6150


FUCK YOU !!!
"""
import os

from pdebug.otn import manager as otn_manager
from pdebug.piata import Input

import cv2
import numpy as np
import typer

try:
    from label_studio_sdk import Client
except Exception as e:
    Client = None

LABEL_STUDIO_URL = "http://localhost:8080"
API_KEY = "09976d25977e93d3f1bfd2be4e20ae14caa93bdd"


def run_server(imgdir, port):
    import sys
    from http.server import HTTPServer, SimpleHTTPRequestHandler, test

    os.chdir(imgdir)

    class CORSRequestHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            SimpleHTTPRequestHandler.end_headers(self)

    test(CORSRequestHandler, HTTPServer, port=port)


class Choices:
    @staticmethod
    def get_label_config(classes):
        classes_label = ""
        for cls in classes:
            classes_label += f"""<Choice value="{cls}" />"""

        label_config = f"""
        <View>
          <Choices name="label" toName="image">
            {classes_label}
          </Choices>
          <Image name="image" value="$image" />
        </View>
        """
        return label_config


class RectangleLabels:
    @staticmethod
    def get_label_config(classes):
        classes_label = ""
        for cls in classes:
            classes_label += f"""<Label value="{cls}" background="green"/>"""

        label_config = f"""
        <View>
          <Image name="image" value="$image" />
          <RectangleLabels name="label" toName="image">
            {classes_label}
          </RectangleLabels>
        </View>
        """
        return label_config

    @staticmethod
    def get_result_dict(boxes, classes=None):
        """
        Args:
            boxes: np.ndarray, [N, 4]
            classes: list of class names, [N, ]
        """
        results = []
        for i in range(len(boxes)):
            bbox = boxes[i]
            cls_name = classes[i] if classes is not None else "none"
            result = {}
            result["from_name"] = "label"
            result["to_name"] = "image"
            result["type"] = "rectanglelabels"
            result["value"] = {}
            result["value"]["rectanglelabels"] = [cls_name]
            result["value"]["x"] = float(bbox[0])
            result["value"]["y"] = float(bbox[1])
            result["value"]["width"] = float(bbox[2] - bbox[0] + 1)
            result["value"]["height"] = float(bbox[3] - bbox[1] + 1)
            results.append(result)
        return results


class KeyPoint:
    @staticmethod
    def get_label_config():
        label_config = f"""
        <View>
          <Image name="image" value="$image" />
          <KeyPoint name="label" toName="image" strokewidth="5" fillcolor="#8bad00" strokecolor="#8bad00" />
        </View>
        """
        return label_config

    @staticmethod
    def get_result_dict(keypoints, image_width=None, image_height=None):
        """
        Args:
            boxes: np.ndarray, [K*3]
        """
        result = {}
        result["from_name"] = "label"
        result["to_name"] = "image"
        result["type"] = "keypoint"
        result["source"] = "$image"
        # result["original_width"] = 200
        # result["original_height"] = 200
        # result["image_rotation"] = 0
        # result["readonly"] = False
        # result["hidden"] = False
        # result["origin"] = "manual"
        # result["opacity"] = 0.5
        result["value"] = []

        K = len(keypoints) // 3
        keypoints = keypoints.reshape(K, -1)
        for pt in keypoints:
            result["value"].append(
                {"x": float(pt[0]), "y": float(pt[1]), "width": 0.3}
            )

        return result


class KeyPointLabels:
    @staticmethod
    def get_label_config(classes):
        classes_label = ""
        for cls in classes:
            classes_label += f"""<Label value="{cls}" />"""

        label_config = f"""
        <View>
          <Image name="image" value="$image" />
          <KeyPointLabels name="keypoint" toName="image" strokewidth="5" fillcolor="#8bad00" strokecolor="#8bad00">
            {classes_label}
          </KeyPointLabels>
        </View>
        """
        return label_config

    @staticmethod
    def get_result_dict(keypoints, classes):
        """
        Args:
            boxes: np.ndarray, [K*3]
        """

        result = {}
        result["from_name"] = "label"
        result["to_name"] = "image"
        result["type"] = "keypoint"
        result["source"] = "$image"
        result["value"] = []

        K = len(keypoints) // 3
        keypoints = keypoints.reshape(K, -1)
        for pt, cls_name in zip(keypoints, classes):
            result["value"].append(
                {
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "width": 0.3,
                    "keypointlabels": [cls_name],
                }
            )

        return result


@otn_manager.NODE.register(name="roidb_to_labelstudio")
def roidb_to_labelstudio(
    path: str,
    imgdir: str,
    key: str = None,
    title: str = "my_project",
    image_server_only: bool = False,
    port: int = 6151,
):
    """Create a label studio project based on input roidb."""
    if image_server_only:
        run_server(imgdir, port)
        return

    imglist = Input(imgdir, name="imgdir").get_reader().imglist
    image_height, image_width = cv2.imread(imglist[0]).shape[:2]

    imgdir_http = f"http://localhost:{port}"
    imglist = [
        l.replace(os.path.abspath(imgdir), imgdir_http) for l in imglist
    ]

    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()

    if key == "boxes":
        label_config = RectangleLabels.get_label_config()
    elif key == "keypoints":
        label_config = KeyPoint.get_label_config()
        predictions = Key
    else:
        raise NotImplementedError

    # find more label_config from https://labelstud.io/templates
    project = ls.start_project(title=title, label_config=label_config)
    project.import_tasks(
        [
            {
                "data": {"image": imgfile},
                "predictions": [
                    {
                        "result": [
                            {
                                "original_width": image_width,
                                "original_height": image_height,
                                "image_rotation": 0,
                                "from_name": "keypoints",
                                "to_name": "image",
                                "type": "keypoint",
                                "readonly": False,
                                "hidden": False,
                                "value": [
                                    {"x": 44, "y": 39, "width": 0.3},
                                    {"x": 50, "y": 52, "width": 0.3},
                                    {"x": 43, "y": 52, "width": 0.3},
                                    {"x": 51, "y": 39, "width": 0.3},
                                ],
                            }
                        ]
                    }
                ],
            }
            for imgfile in imglist
        ]
    )

    # task_ids = project.get_tasks_ids()
    # project.create_prediction(task_ids[0], result='Dog', score=0.9)

    run_server(imgdir, port)
    return ""


@otn_manager.NODE.register(name="labelstudio_debug")
def labelstudio_debug(
    project_id: str,
):
    """Create a label studio project based on input roidb."""
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    project = ls.get_project(project_id)

    # remove all projects:
    # >> [p.delete_project(p.id) for p in ls.get_projects()]

    __import__("ipdb").set_trace()
    pass

    return ""


@otn_manager.NODE.register(name="labelstudio_reset")
def labelstudio_reset():
    """Reset labelstudio. (clean all projects)"""
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    for p in ls.get_projects():
        p.delete_project(p.id)
    return ""


@otn_manager.NODE.register(name="labelstudio_demo")
def labelstudio_demo(
    task: str = "cls",
):
    """labelstudio create task demo"""
    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()

    tasks = [
        {
            "image": "https://data.heartex.net/open-images/train_0/mini/0045dd96bf73936c.jpg"
        },
        {
            "image": "https://data.heartex.net/open-images/train_0/mini/0083d02f6ad18b38.jpg"
        },
    ]
    cls_gt = ["cat", "dog"]
    det_gt = [
        np.array([[1, 1, 50, 50], [37, 28, 32, 40]], dtype=np.float32),
        np.array([[100, 50, 100, 150], [50, 100, 100, 200]], dtype=np.float32),
    ]
    kps_gt = [
        np.array([1, 1, 1, 20, 30, 1], dtype=np.float32),
        np.array([50, 20, 1, 30, 70, 1], dtype=np.float32),
    ]

    if task == "cls":
        label_config = Choices.get_label_config(["cat", "dog"])
    elif task == "det":
        label_config = RectangleLabels.get_label_config(["cat", "dog"])
    elif task == "kps":
        label_config = KeyPoint.get_label_config()
    elif task == "kps_v2":
        label_config = KeyPointLabels.get_label_config(["cat", "dog"])
    elif task == "seg":
        raise NotImplementedError
    else:
        raise NotImplementedError

    project = ls.start_project(
        title=f"demo_{task}",
        label_config=label_config,
        show_collab_predictions=True,
    )
    project.import_tasks(tasks)

    task_ids = project.get_tasks_ids()

    if task == "cls":
        for name, task_id in zip(cls_gt, task_ids):
            project.create_prediction(task_id, result=name, model_version="1")
    elif task == "det":
        for name, boxes, task_id in zip(cls_gt, det_gt, task_ids):
            result = RectangleLabels.get_result_dict(
                boxes, [name] * len(boxes)
            )
            project.create_prediction(
                task_id, result=result, model_version="1"
            )
    elif task == "kps":
        for kpt, task_id in zip(kps_gt, task_ids):
            result = KeyPoint.get_result_dict(kpt)
            project.create_prediction(
                task_id, result=[result], model_version="1"
            )
    elif task == "kps_v2":
        for name, kpt, task_id in zip(cls_gt, kps_gt, task_ids):
            num_kps = kpt.reshape(-1, 3).shape[0]
            result = KeyPointLabels.get_result_dict(kpt, [name] * num_kps)
            project.create_prediction(
                task_id, result=[result], model_version="1"
            )

    return ""


if __name__ == "__main__":
    typer.run(roidb_to_labelstudio)
