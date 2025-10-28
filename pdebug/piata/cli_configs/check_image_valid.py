Input = {}
Input[
    "path"
] = "/mnt/bn/picoroomplan2/K1H_collect/202311/20231102/20231102_114924/hawk/cam5"
Input["name"] = "imgdir"

Main = "check_image_valid"

num_workers = 8


def check_image_valid(**kwargs):
    image = kwargs.get("image")
    imgdir = kwargs.get("imgdir")
    image_file = kwargs.get("image_file")

    if image is None:
        print(f"{image_file} is unvalid.")
