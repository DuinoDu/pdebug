import glob
import json
import os

__all__ = [
    "get_sensor_yaml_depth",
    "get_sensor_yaml_roomplan",
    "prepare_biaoding_info",
]


def load_vst_id_from_meta_info(meta_info_file):
    if not os.path.exists(meta_info_file):
        print(f"  meta_info file ({meta_info_file}) not exists.")
        return None
    with open(meta_info_file, "r") as fid:
        data = json.load(fid)
    vst_id = data["vst_id"]
    return vst_id


META_INFO_DEFAULT = None


def get_sensor_yaml_depth(imgdir):
    """Get sensor yaml in depth waicai data from raoqiang."""
    global META_INFO_DEFAULT
    if META_INFO_DEFAULT is None:
        default_file = "/mnt/bn/depth-data-bn/K1H/metainfo/default.json"
        with open(default_file, "r") as fid:
            META_INFO_DEFAULT = json.load(fid)["hawk"]

    meta_info_file = imgdir.replace("hawk/cam5", "metainfo_new.json")
    vst_id = load_vst_id_from_meta_info(meta_info_file)
    if not vst_id:
        return None

    calib_prefix = META_INFO_DEFAULT[vst_id]
    calib_root = "/mnt/bn/depth-data-bn/K1H/metainfo"
    mav0_path = os.path.join(calib_root, calib_prefix)
    sensor_yaml = os.path.join(mav0_path, "cam5/sensor.yaml")
    if not os.path.exists(sensor_yaml):
        print(f"  {sensor_yaml} not exists.")
        return None
    else:
        return sensor_yaml


BIAODING_INFO = {}
BIAODING_INFO_UNIQUE = {}
BIAODING_INFO_EXTRA = {}


def prepare_biaoding_info(imgdirs_dict):
    """
    imgdir: 20231101/xxxx/hawk/cam5
    data_root: 20231101
    """
    global BIAODING_INFO_UNIQUE
    if isinstance(imgdirs_dict, dict):
        date_root_list = set(
            [
                os.path.normpath(os.path.join(d, "../../../"))
                for d in imgdirs_dict.keys()
            ]
        )
    elif isinstance(imgdirs_dict, list):
        date_root_list = set(
            [
                os.path.normpath(os.path.join(d, "../../../"))
                for d in imgdirs_dict
            ]
        )

    biaoding_folders = []
    for date_root in date_root_list:
        biaoding_folders_i = glob.glob(date_root + "/*biaoding")
        if not biaoding_folders_i:
            print(f"  No **biaoding found in {date_root}")
            continue
        biaoding_folders.extend(biaoding_folders_i)

    for f in biaoding_folders:
        meta_info_file = os.path.join(f, "metainfo_new.json")
        if not os.path.exists(meta_info_file):
            continue
        biaoding_vst_id = load_vst_id_from_meta_info(meta_info_file)
        biaoding_sensor_yaml = os.path.join(f, "mav0/cam5/sensor.yaml")
        if not os.path.exists(biaoding_sensor_yaml):
            continue
        BIAODING_INFO_UNIQUE[biaoding_vst_id] = biaoding_sensor_yaml


def get_sensor_yaml_roomplan(
    imgdir, debug_ipdb=False, extra_biaoding_folders=None
):
    """Get sensor yaml in roomplan waicai data."""

    meta_info_file = imgdir.replace("hawk/cam5", "metainfo_new.json")
    vst_id = load_vst_id_from_meta_info(meta_info_file)
    if not vst_id:
        return None

    global BIAODING_INFO, BIAODING_INFO_UNIQUE, BIAODING_INFO_EXTRA

    if extra_biaoding_folders:
        # extra_biaoding_folders => BIAODING_INFO_EXTRA
        for f in extra_biaoding_folders:
            meta_info_file = os.path.join(f, "metainfo_new.json")
            if not os.path.exists(meta_info_file):
                continue
            biaoding_vst_id = load_vst_id_from_meta_info(meta_info_file)
            biaoding_sensor_yaml = os.path.join(f, "mav0/cam5/sensor.yaml")
            if not os.path.exists(biaoding_sensor_yaml):
                continue
            BIAODING_INFO_EXTRA[biaoding_vst_id] = biaoding_sensor_yaml

    date_root = "/".join(imgdir.split("/")[:-3])
    if date_root not in BIAODING_INFO:
        biaoding_folders = glob.glob(date_root + "/*biaoding")
        if not biaoding_folders:
            print(
                f"  biaoding folder of {imgdir} not found, find {vst_id} in history biaoding: {BIAODING_INFO_UNIQUE.keys()}"
            )
            assert (
                vst_id in BIAODING_INFO_UNIQUE
            ), f"vst_id({vst_id}) not found in {BIAODING_INFO_UNIQUE.keys()}. Check ***_biaoding.zip"
            sensor_yaml = BIAODING_INFO_UNIQUE[vst_id]
            return sensor_yaml
        meta_infos = {}  # vst_id: yamlfile
        for f in biaoding_folders:
            meta_info_file = os.path.join(f, "metainfo_new.json")
            biaoding_vst_id = load_vst_id_from_meta_info(meta_info_file)
            biaoding_sensor_yaml = os.path.join(f, "mav0/cam5/sensor.yaml")
            if not os.path.exists(biaoding_sensor_yaml):
                continue
            meta_infos[biaoding_vst_id] = biaoding_sensor_yaml
            BIAODING_INFO_UNIQUE[biaoding_vst_id] = biaoding_sensor_yaml
        BIAODING_INFO[date_root] = meta_infos

    # format vst_id, eg: qzd-8 -> qzd-08
    vst_id_items = vst_id.split("-")
    vst_id_items[1] = f"{int(vst_id_items[1]):02d}"
    vst_id = "-".join(vst_id_items)

    if vst_id in BIAODING_INFO[date_root]:
        sensor_yaml = BIAODING_INFO[date_root][vst_id]
    elif vst_id in BIAODING_INFO_UNIQUE:
        sensor_yaml = BIAODING_INFO_UNIQUE[vst_id]
    elif vst_id in BIAODING_INFO_EXTRA:
        sensor_yaml = BIAODING_INFO_EXTRA[vst_id]
    else:
        if debug_ipdb:
            __import__("ipdb").set_trace()
        print(f"BIAODING_INFO: {BIAODING_INFO.keys()}")
        print(f"BIAODING_INFO_UNIQUE: {BIAODING_INFO_UNIQUE.keys()}")
        print(f"BIAODING_INFO_EXTRA: {BIAODING_INFO_EXTRA.keys()}")
        raise RuntimeError(f"vst_id {vst_id} not found")

    if not os.path.exists(sensor_yaml):
        if debug_ipdb:
            __import__("ipdb").set_trace()
        pass

    return sensor_yaml
