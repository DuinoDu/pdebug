import os

from pdebug.data_types import Camera
from pdebug.utils.env import RERUN_INSTALLED

if RERUN_INSTALLED:
    import rerun as rr
    import rerun.blueprint as rrb

__all__ = ["rr_rgbd", "init_rerun"]


def init_rerun(name: str, ip: str = None, port: str = "9876"):
    if not RERUN_INSTALLED:
        raise ModuleNotFoundError("rerun-sdk is required")
    if not ip:
        ip = os.getenv("RERUN_IP", None)
    if ip:
        rr.init(name, spawn=False)
        rr.connect_grpc(f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.init(name, spawn=True)


def rr_rgbd_blueprint(
    rr_root: str = "world",
    camera_name: str = "camera",
    disable_depth: bool = False,
):
    if not RERUN_INSTALLED:
        raise ModuleNotFoundError("rerun-sdk is required")
    if disable_depth:
        return rrb.Spatial2DView(
            name=camera_name,
            origin=f"{rr_root}/{camera_name}/image",
            contents=f"{rr_root}/{camera_name}/image/rgb",
        )
    else:
        return rrb.Vertical(
            rrb.Spatial2DView(
                name="RGB & Depth",
                origin=f"{rr_root}/{camera_name}/image",
                overrides={
                    f"{rr_root}/{camera_name}/image/rgb": rr.Image.from_fields(
                        opacity=0.5
                    )
                },
            ),
            rrb.Tabs(
                rrb.Spatial2DView(
                    name="RGB",
                    origin=f"{rr_root}/{camera_name}/image",
                    contents=f"{rr_root}/{camera_name}/image/rgb",
                ),
                rrb.Spatial2DView(
                    name="Depth",
                    origin=f"{rr_root}/{camera_name}/image",
                    contents=f"{rr_root}/{camera_name}/image/depth",
                ),
            ),
            name="2D",
            row_shares=[3, 3, 2],
        )


def rr_rgbd(
    rgb=None,
    depth=None,
    camera: Camera = None,
    rr_root: str = "world",
    camera_name: str = "camera",
    view_coordinate=None,
    timestamp=None,
):
    """
    Visualize rgbd data in rerun.

    Args:
        rgb(np.ndarray): rgb data in bgr
        depth(np.ndarray): depth data, in meter
        camera: camera intrinsic and extrinsic
        view_coordinate: view coordinate
        timestamp: time stamp
    """
    if not RERUN_INSTALLED:
        raise ModuleNotFoundError("rerun-sdk is required")

    if timestamp:
        # rr.set_time_seconds("time", timestamp)
        rr.set_time("frame_cnt", sequence=timestamp)
    if not view_coordinate:
        view_coordinate = rr.ViewCoordinates.RUF

    rr.log(
        f"{rr_root}/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )
    if rgb is not None:
        rr.log(
            f"{rr_root}/{camera_name}/image/rgb",
            rr.Image(rgb, color_model="BGR").compress(jpeg_quality=95),
        )
    if depth is not None:
        rr.log(
            f"{rr_root}/{camera_name}/image/depth",
            rr.DepthImage(depth, meter=1.0),
        )
    if camera:
        rr.log(
            f"{rr_root}/{camera_name}/image",
            rr.Pinhole(
                resolution=[camera.I.w, camera.I.h],
                focal_length=[camera.I.fx, camera.I.fy],
                principal_point=[camera.I.cx, camera.I.cy],
                camera_xyz=view_coordinate,
            ),
        )
        rr.log(
            f"{rr_root}/{camera_name}",
            rr.Transform3D(
                translation=camera.E.translation, mat3x3=camera.E.rotation
            ),
        )
