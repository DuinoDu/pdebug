Input = {}
Input[
    "path"
] = "/mnt/bn/picoroomplan2/K1H_collect/20231106/20231106_130331/hawk/cam5"
Input["name"] = "imgdir"
# Input["topk"] = 500
Input["rgb"] = True
Input["timestamp_sample_duration"] = 1000
Input["get_timestamp_fn"] = lambda x: int(x.split("_")[0][:7])  # ms

Output = {}
Output[
    "path"
] = "/mnt/bn/picoroomplan/wbw/code/pat/projects/semantic_pcd/waicai/cam5_1sec.mp4"
Output["name"] = "video"
Output["fps"] = 1

Main = "add_filename_to_image"


def add_filename_to_image(**kwargs):
    image = kwargs.get("image")
    imgdir = kwargs.get("imgdir")
    image_file = kwargs.get("image_file")

    import cv2

    # 1,127,776,951,811_3000000 -> 1,127 s
    filename = filename.split("_")[0][:4] + " sec"
    image = cv2.putText(
        image,
        filename,
        (100, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 0, 255),
        2,
    )
    return image
