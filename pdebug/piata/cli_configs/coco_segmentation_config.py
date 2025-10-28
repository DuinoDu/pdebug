"""
type: roi-input
roi_required_key: segmentation
"""
Input = {}
Input["path"] = "/mnt/f/Downloads/task19_pat.json"
Input["name"] = "coco"

Output = {}
Output["path"] = "/mnt/f/Downloads/task19_pat_processed.json"
Output["name"] = "coco"

SourceInput = {}
SourceInput["path"] = "download_18,19,20/images_19"
SourceInput["name"] = "imgdir"

VisOutput = "tmp_vis_pat_2"

Main = "remove_small_segmentation"
# Main = "visualize_segmenation"
# Main = [
#   "visualize_segmenation:0 -> remove_small_segmentation -> visualize_segmenation:1",
#   "(visualize_segmenation:0, visualize_segmenation:1) -> concat_image"]

TopK = -1

Shuffle = False

Extras = {}
# required by coco-style output.
Extras["coco_categories"] = [
    {"label": "VOID", "color": "#CFCFCF", "label_zh": "未标注"},
    {
        "label": "ceiling",
        "color": "#804080",
        "icon": "Road",
        "label_zh": "天花板",
    },
    {"label": "wall", "color": "#F423E8", "icon": "Wall", "label_zh": "墙"},
    {
        "label": "floor",
        "color": "#FAAAA0",
        "icon": "Parking",
        "label_zh": "地板",
    },
    {"label": "door", "color": "#464646", "icon": "Train", "label_zh": "门"},
    {"label": "window", "color": "#14DC3C", "icon": "Walk", "label_zh": "窗"},
    {
        "label": "beam",
        "color": "#0088FF",
        "icon": "Motorbike",
        "label_zh": "横梁",
    },
    {"label": "column", "color": "#0000E8", "icon": "Car", "label_zh": "柱子"},
    {
        "label": "curtain",
        "color": "#000046",
        "icon": "Truck",
        "label_zh": "窗帘",
    },
    {"label": "table", "color": "#003C64", "icon": "Bus", "label_zh": "桌子"},
    {"label": "chair", "color": "#E6968C", "icon": "Train", "label_zh": "椅子"},
    {
        "label": "cabinet",
        "color": "#0000E6",
        "icon": "Motorbike",
        "label_zh": "柜子",
    },
    {"label": "bed", "color": "#770B20", "icon": "Bike", "label_zh": "床"},
    {"label": "sofa", "color": "#00005A", "icon": "Caravan", "label_zh": "沙发"},
    {
        "label": "plant",
        "color": "#00006E",
        "icon": "TruckTrailer",
        "label_zh": "植物",
    },
    {
        "label": "monitor",
        "color": "#E6968C",
        "icon": "HomeVariant",
        "label_zh": "显示器",
    },
    {
        "label": "person",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "人",
    },
    {
        "label": "kitchen-cabinet",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "橱柜",
    },
    {
        "label": "refrigerator",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "冰箱",
    },
    {
        "label": "shelf",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "架子",
    },
    {
        "label": "gas-furnace",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "燃气炉",
    },
    {
        "label": "sink",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "水槽",
    },
    {
        "label": "washer",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "洗衣机",
    },
    {
        "label": "table-lamp",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "台灯",
    },
    {
        "label": "closetool",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "马桶",
    },
    {
        "label": "bathtub",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "浴缸",
    },
    {
        "label": "oven",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "烤箱",
    },
    {
        "label": "dishwasher",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "洗碗机",
    },
    {
        "label": "fireplace",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "壁炉",
    },
    {
        "label": "lamp",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "灯",
    },
    {
        "label": "stairway",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "楼梯",
    },
    {
        "label": "radiator",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "散热器",
    },
    {
        "label": "air-conditioning",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "空调",
    },
    {
        "label": "picture",
        "color": "#FBB577",
        "icon": "Instapaper",
        "label_zh": "图片",
    },
    {"label": "others", "color": "#BE9999", "icon": "Gate", "label_zh": "其他"},
]


def visualize_segmenation(**kwargs):
    """Visualize segmentation in roi dict.

    Args:
        roi: input roi dict.
        image: input image.
        imgdir: image directory
        image_file: image file name
        vis_output: visualize output
    Returns:
        image or None.
    Type: (roi, image) -> image

    """
    roi = kwargs.get("roi")
    image = kwargs.get("image")
    imgdir = kwargs.get("imgdir")
    image_file = kwargs.get("image_file")
    vis_output = kwargs.get("vis_output")

    import copy
    import os

    from pdebug.visp import Colormap, draw

    import cv2
    from imantics import Mask, Polygons

    cats = list(set(roi["category_id"]))
    colormap = Colormap(len(cats))
    image = copy.deepcopy(image)

    for cat_id, contour in zip(roi["category_id"], roi["segmentation"]):
        color = colormap[cats.index(cat_id)]
        image = draw.contour(contour, image, color=color)

        bbox = Polygons([contour]).bbox()
        area = bbox.area()
        x = (bbox.min_point[0] + bbox.max_point[0]) // 2
        y = (bbox.min_point[1] + bbox.max_point[1]) // 2
        image = cv2.putText(
            image,
            str(area),
            (x, y),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    if vis_output:
        savename = os.path.join(vis_output, roi["image_name"])
        cv2.imwrite(savename, image)
    else:
        return image


def remove_small_segmentation(**kwargs):
    """Remove small segmentation in roi dict.

    Args:
        roi: input roi dict.
        image: input image.
        imgdir: image directory
        image_file: image file name
        vis_output: visualize output
    Returns:
        image or None.
    Type: roi -> roi

    """
    roi = kwargs.get("roi")
    image = kwargs.get("image")
    imgdir = kwargs.get("imgdir")
    image_file = kwargs.get("image_file")
    vis_output = kwargs.get("vis_output")
    ctx = kwargs.get("ctx", None)

    from pdebug.piata import Input

    import cv2
    import numpy as np
    from imantics import Mask, Polygons

    min_segmentation_bbox_area = kwargs.get("min_segmentation_bbox_area", 800)

    keep_index = []
    for idx, contour in enumerate(roi["segmentation"]):
        bbox = Polygons([contour]).bbox()
        if bbox.area() > min_segmentation_bbox_area:
            keep_index.append(idx)

    new_roi = Input.filter_roi(roi, keep_index)

    # vis1 = ctx["visualize_segmenation"](roi, image, imgdir, image_file, None)
    # vis2 = ctx["visualize_segmenation"](new_roi, image, imgdir, image_file, None)
    # savename = os.path.join(vis_output, os.path.basename(image_file))
    # image = np.concatenate((vis1, vis2), axis=1)
    # cv2.imwrite(savename, image)

    return new_roi
