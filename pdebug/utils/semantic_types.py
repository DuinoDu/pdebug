import numpy as np

__all__ = [
    "SEMANTIC_PCD_CATEGORIES",
    "load_categories",
    "load_colors",
]

SEMANTIC_PCD_CATEGORIES = [
    {"label_zh": "未标注", "label": "VOID", "color": "#CFCFCF"},
    {"label_zh": "天花板", "label": "ceiling", "color": "#787878"},
    {"label_zh": "墙", "label": "wall", "color": "#b47878"},
    {"label_zh": "地板", "label": "floor", "color": "#06e6e6"},
    {"label_zh": "门", "label": "door", "color": "#503232"},
    {"label_zh": "窗", "label": "window", "color": "#cc9933"},
    {"label_zh": "横梁", "label": "beam", "color": "#787850"},
    {"label_zh": "柱子", "label": "column", "color": "#8c8c8c"},
    {"label_zh": "窗帘", "label": "curtain", "color": "#cc05ff"},
    {"label_zh": "桌子", "label": "table", "color": "#e6e6e6"},
    {"label_zh": "椅子", "label": "chair", "color": "#04fa07"},
    {"label_zh": "柜子", "label": "cabinet", "color": "#2a7dd1"},
    {"label_zh": "床", "label": "bed", "color": "#ebff07"},
    {"label_zh": "沙发", "label": "sofa", "color": "#96053d"},
    {"label_zh": "植物", "label": "plant", "color": "#787846"},
    {"label_zh": "显示器", "label": "monitor", "color": "#8c78f0"},
    {"label_zh": "人", "label": "person", "color": "#ff0652"},
    {"label_zh": "橱柜", "label": "kitchen-cabinet", "color": "#8fff8c"},
    {"label_zh": "冰箱", "label": "refrigerator", "color": "#ccff04"},
    {"label_zh": "架子", "label": "shelf", "color": "#ff3307"},
    {"label_zh": "燃气炉", "label": "gas-furnace", "color": "#cc4603"},
    {"label_zh": "水槽", "label": "sink", "color": "#0066c8"},
    {"label_zh": "洗衣机", "label": "washer", "color": "#3de6fa"},
    {"label_zh": "台灯", "label": "table-lamp", "color": "#ff0633"},
    {"label_zh": "马桶", "label": "closetool", "color": "#0b66ff"},
    {"label_zh": "浴缸", "label": "bathtub", "color": "#ff0747"},
    {"label_zh": "烤箱", "label": "oven", "color": "#ff09e0"},
    {"label_zh": "洗碗机", "label": "dishwasher", "color": "#0907e6"},
    {"label_zh": "壁炉", "label": "fireplace", "color": "#dcdcdc"},
    {"label_zh": "灯", "label": "lamp", "color": "#ff095c"},
    {"label_zh": "楼梯", "label": "stairway", "color": "#7009ff"},
    {"label_zh": "散热器", "label": "radiator", "color": "#08ffd6"},
    {"label_zh": "空调", "label": "air-conditioning", "color": "#07ffe0"},
    {"label_zh": "图片", "label": "picture", "color": "#ffb806"},
    {"label_zh": "其他", "label": "others", "color": "#3d3df5"},
]


SEMANTIC_CUBE_CATEGORIES = [
    {"label_zh": "未标注", "label": "VOID", "color": "#CFCFCF"},
    {"label_zh": "魔方面", "label": "cube_surface", "color": "#503232"},
]


def load_categories(name="semantic_pcd", simple=False):
    if name == "semantic_pcd":
        raw_categories = SEMANTIC_PCD_CATEGORIES
    elif name in ["semantic_pcd_16_human", "photoid_semantic_pcd"]:
        raw_categories = (
            SEMANTIC_PCD_CATEGORIES[:17] + SEMANTIC_PCD_CATEGORIES[-1:]
        )
    elif name == "cube":
        raw_categories = SEMANTIC_CUBE_CATEGORIES
    else:
        raise NotImplementedError

    if simple:
        categories = [obj["label"] for obj in raw_categories]
    else:
        categories = [
            obj["label_zh"] + " | " + obj["label"] for obj in raw_categories
        ]
    return categories


def load_colors(name="semantic_pcd", as_tuple=False):
    def to_rgb_tuple(hex_str):
        hexcolor = int(hex_str.replace("#", "0x"), base=16)
        r = hexcolor >> 16 & 0xFF
        g = hexcolor >> 8 & 0xFF
        b = hexcolor & 0xFF
        return r, g, b

    if name == "photoid_semantic_pcd":
        colors_rgb = [
            (0, 0, 0),
            (0, 0, 255),
            (232, 88, 47),
            (0, 217, 0),
            (148, 0, 240),
            (222, 241, 23),
            (255, 205, 205),
            (0, 223, 228),
            (106, 135, 204),
            (116, 28, 41),
            (240, 35, 235),
            (0, 166, 156),
            (249, 139, 0),
            (225, 228, 194),
            (255, 228, 0),
            (225, 50, 0),
            (70, 0, 189),
            (255, 0, 0),
        ]
        colors_rgb = np.asarray(colors_rgb)
        colors_bgr = colors_rgb[:, ::-1].tolist()
        return colors_bgr

    if name == "semantic_pcd":
        raw_categories = SEMANTIC_PCD_CATEGORIES
    elif name == "semantic_pcd_16_human":
        raw_categories = (
            SEMANTIC_PCD_CATEGORIES[:17] + SEMANTIC_PCD_CATEGORIES[-1:]
        )
    elif name == "cube":
        raw_categories = SEMANTIC_CUBE_CATEGORIES
    else:
        raise NotImplementedError

    colors = [obj["color"] for obj in raw_categories]
    if as_tuple:
        colors = [to_rgb_tuple(c) for c in colors]
    return colors
