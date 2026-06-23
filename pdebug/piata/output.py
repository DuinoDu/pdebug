"""Data output"""
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

from .registry import OUTPUT_REGISTRY

__all__ = ["Output"]

_OUTPUT_MODULES = {
    "imgdir": "pdebug.piata.handler.destination",
    "video": "pdebug.piata.handler.destination",
    "video_ffmpeg": "pdebug.piata.handler.destination",
}


@OUTPUT_REGISTRY.register(name="json")
class JsonWriter:
    """Write one JSON-serializable payload."""

    def __init__(self, payload, *, indent=2, ensure_ascii=True):
        self.payload = payload
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def save(self, output_name):
        with open(output_name, "w", encoding="utf-8") as handle:
            json.dump(
                self.payload,
                handle,
                indent=self.indent,
                ensure_ascii=self.ensure_ascii,
            )


@OUTPUT_REGISTRY.register(name="lance_column")
class LanceColumnWriter:
    """Write a Lance dataset with a replaced or appended column."""

    def __init__(
        self,
        table,
        *,
        column_name: str,
        column_values: Sequence[object],
    ):
        self.table = table
        self.column_name = column_name
        self.column_values = column_values

    def save(self, output_name):
        import lance
        import pyarrow as pa

        if len(self.table) != len(self.column_values):
            raise ValueError(
                f"Table length {len(self.table)} differs from column size "
                f"{len(self.column_values)}."
            )

        arr = pa.array(self.column_values)
        table = self.table
        if self.column_name in table.column_names:
            idx = table.column_names.index(self.column_name)
            table = table.set_column(idx, self.column_name, arr)
        else:
            table = table.append_column(self.column_name, arr)

        output = Path(output_name)
        if output.exists():
            shutil.rmtree(output)
        lance.write_dataset(table, str(output))


def _ensure_registered(name: str) -> None:
    if name in OUTPUT_REGISTRY:
        return
    module_name = _OUTPUT_MODULES.get(name)
    if module_name:
        importlib.import_module(module_name)


class Output:

    """Class for output.

    Example:
    >>> import piata
    >>> writer = piata.Output(roidb, name="coco", categories=categories)
    >>> writer.save("xxx.json")
    >>> writer = piata.Output(images, name="video")
    >>> writer.save("xxx.mp4")
    >>> writer = piata.Output("save.mp4", name="video_ffmpeg")
    >>> writer.write_frame(one_frame)
    >>> # writer.save() video_ffmpeg DONT need save()

    """

    def __init__(self, *args, **kwargs):
        if "name" not in kwargs:
            raise ValueError("`name` is requred in Output.")
        self._name = kwargs["name"]
        _ensure_registered(self._name)
        if self._name in OUTPUT_REGISTRY:
            output_func = OUTPUT_REGISTRY.get(kwargs.pop("name"))
            self._output = output_func(*args, **kwargs)
        else:
            raise ValueError(f"Unknown output type: {self._name}")

    def __str__(self):
        return "piata output"

    def get_writer(self):
        return self._output

    def save(self, output_name):
        savedir = os.path.dirname(output_name)
        if savedir:
            os.makedirs(savedir, exist_ok=True)
        if hasattr(self._output, "save"):
            self._output.save(output_name)
        else:
            raise RuntimeError(
                f"{self._name} Output don't have `save` interface."
            )


@OUTPUT_REGISTRY.register(name="coco")
def output_to_coco(roidb, categories, **kwargs):
    from pdebug.piata.coco import COCOWriter

    writer = COCOWriter(**kwargs)
    writer.set_categories(categories, auto_fill=True)
    writer.add_roidb(roidb)
    return writer


@OUTPUT_REGISTRY.register(name="datumaro")
def output_to_datumaro(roidb):
    """
    Convert roidb to datumaro json.
    """
    import json

    class DatumaroWriter:
        def add_roidb(self, roidb):
            self.roidb = roidb

        def save(self, savename):
            data = {}
            data["info"] = {}
            data["categories"] = {
                "label": {
                    "labels": [
                        {"name": "point", "parent": "", "attributes": []}
                    ],
                    "attributes": [
                        "occluded",
                    ],
                },
                "points": {"items": []},
            }
            data["items"] = []
            for roi in roidb:
                annotations = []
                keypoints = roi["keypoints"].reshape(-1, 3)
                num_kps = keypoints.shape[0]
                for i in range(num_kps):
                    anno = {
                        "id": 0,
                        "type": "points",
                        "attributes": {"occluded": False},
                        "group": 0,
                        "label_id": 0,
                        "points": [int(keypoints[i][0]), int(keypoints[i][1])],
                        "z_order": 0,
                        "visibility": [2],
                    }
                    annotations.append(anno)
                _id = os.path.splitext(os.path.basename(roi["image_name"]))[0]
                _id = str(int(_id))
                data["items"].append({"id": _id, "annotations": annotations})

            with open(savename, "w", encoding="utf8") as fid:
                json.dump(data, fid, indent=2, ensure_ascii=False)

    writer = DatumaroWriter()
    writer.add_roidb(roidb)
    return writer


@OUTPUT_REGISTRY.register(name="cvat_1_1")
def output_to_datumaro(roidb):
    """
    Convert roidb to cvat xml.
    """
    import xml.etree.ElementTree as ET

    class CvatWriter:
        def add_roidb(self, roidb):
            self.roidb = roidb

        def save(self, savename):
            # data = {}
            # data["version"] = "1.1"
            # data["meta"] = {}
            root = ET.Element("annotations")
            version = ET.SubElement(root, "version")
            version.text = "1.1"
            _id = 0
            for roi in roidb:
                anno = ET.SubElement(root, "image")
                anno.set("id", str(_id))
                anno.set("name", os.path.basename(roi["image_name"]))
                anno.set("width", str(roi["image_width"]))
                anno.set("height", str(roi["image_height"]))
                keypoints = roi["keypoints"].reshape(-1, 3)
                num_kps = keypoints.shape[0]
                for i in range(num_kps):
                    points = ET.SubElement(anno, "points")
                    points.set("label", "point")
                    points.set("source", "manual")
                    points.set("occluded", "0")
                    points_str = "%d,%d" % (
                        int(keypoints[i][0]),
                        int(keypoints[i][1]),
                    )
                    points.set("points", points_str)
                    points.set("z_order", "0")
                _id += 1
            with open(savename, "wb") as fid:
                fid.write(ET.tostring(root))

    writer = CvatWriter()
    writer.add_roidb(roidb)
    return writer
