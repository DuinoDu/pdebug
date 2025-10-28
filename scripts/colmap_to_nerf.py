#!/usr/bin/env python3
import math
import os
from typing import Optional

from pdebug.data_types import NerfCamera, PointcloudTensor

import cv2
import numpy as np
import open3d as o3d
import typer


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# @task(name="my-tool")
def main(
    imgdir: str = "images",
    textdir: str = "colmap_text",
    aabb_scale: int = 16,
    skip_early: int = 0,
    align_first_pose_to_origin: bool = False,
    output: Optional[str] = typer.Option("nerf_result", help="output name"),
):
    """Convert colmap pose result to nerf camera format."""
    AABB_SCALE = int(aabb_scale)
    SKIP_EARLY = int(skip_early)
    os.makedirs(output, exist_ok=True)

    with open(os.path.join(textdir, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi
    print(
        f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} "
    )

    with open(os.path.join(textdir, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(
                    " "
                )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(imgdir, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(imgdir)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                print(name, "sharpness=", b)
                image_id = int(elems[0])

                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

                c2w = np.linalg.inv(m).tolist()

                frame = {
                    "file_path": name,
                    "sharpness": b,
                    "transform_matrix": c2w,
                }
                out["frames"].append(frame)

    nerf_camera = NerfCamera(out)
    if align_first_pose_to_origin:
        t_mat = nerf_camera.align_first_pose_to_origin()

    print(f"{len(nerf_camera)} frames")
    print(f"writing {output}")
    output_file = os.path.join(output, "transforms.json")
    nerf_camera.dump(output_file)

    points_txt = os.path.join(textdir, "points3D.txt")
    if os.path.exists(points_txt):
        pcd = PointcloudTensor.from_colmap_point3d(points_txt).to_open3d()
        if align_first_pose_to_origin:
            pcd.transform(t_mat)
        output_file = os.path.join(output, "points.pcd")
        o3d.io.write_point_cloud(output_file, pcd, print_progress=True)

    print(f"save to {output}")


if __name__ == "__main__":
    typer.run(main)
