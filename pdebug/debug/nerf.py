import random
from functools import partial
from multiprocessing import Process
from typing import List, Optional

from pdebug.data_types import (
    Camera,
    CameraExtrinsic,
    PointcloudTensor,
    RayTensor,
    Tensor,
    x_to_ndarray,
)
from pdebug.utils.decorator import mp
from pdebug.utils.three_utils import threejs_arrow, threejs_camera, threejs_pcd
from pdebug.utils.web_engine import ImageEngine, ThreeEngine

import numpy as np

__all__ = ["vis_pose", "vis_rays", "vis_xyz", "vis_point_in_image"]


_global_engines_process = []


def _serve(
    js_code: str,
    *,
    scale: float,
    block: bool,
    port: int,
    vr: bool,
    title: str,
    res_dir: str,
    no_server: bool = False,
):
    """Serve js_code in port"""
    html_name = "simple_vr" if vr else "simple"
    engine = ThreeEngine(html_name, res_dir=res_dir)
    engine.render(js_code=js_code, scale=scale, title=title)
    if no_server:
        return
    if block:
        engine.serve(port=port, use_ssl=vr)
    else:
        p = Process(target=engine.serve, kwargs={"port": port, "use_ssl": vr})
        p.start()
        _global_engines_process.append(p)


def wait(port: int = 6151):
    """Start _global_engines in wait mode, work together with `block=False`"""
    assert (
        _global_engines_process
    ), "Found empty server process, did you call `nerf.vis_xxx`?"
    should_block = True
    while should_block:
        should_block = all([p.is_alive() for p in _global_engines_process])


def vis_pose(
    pose: Tensor,
    *,
    colors: List[str] = None,
    block: bool = True,
    port: int = 6150,
    scale: float = 1.0,
    vr: bool = False,
    title: str = "pdebug",
    res_dir: Optional[str] = None,
    no_server: Optional[bool] = False,
    js_code: Optional[str] = "",
) -> None:
    """Visualize pose (camera extrinsic).

    Args:
        pose: input camera pose matrix.

    """
    if pose.ndim == 3 and pose.shape[0] > 1:
        poses = [CameraExtrinsic(p) for p in pose]
    else:
        poses = [CameraExtrinsic(pose)]

    if colors is not None:
        assert len(colors) == len(poses)

    for ind, pose in enumerate(poses):
        color = None if colors is None else colors[ind]
        one_pose_code = threejs_camera(pose, color=color, suffix=ind)
        js_code += one_pose_code
    _serve(
        js_code,
        scale=scale,
        block=block,
        port=port,
        vr=vr,
        title=title,
        res_dir=res_dir,
        no_server=no_server,
    )


def rays_to_js_code(
    rays_o: RayTensor,
    rays_d: RayTensor,
    length: float = 1.0,
    sample=None,
) -> str:
    """Convert rays to js_code."""
    assert rays_o.ray_num == rays_d.ray_num
    ray_idx_list = list(range(rays_o.ray_num))
    if sample:
        assert sample > 0
        assert isinstance(sample, int)
        random.shuffle(ray_idx_list)
        ray_idx_list = ray_idx_list[:sample]
        print(f"sample rays num: {rays_o.ray_num} => {sample}")

    js_code = ""
    for ray_idx in ray_idx_list:
        ray_o = rays_o.data[0][ray_idx]
        ray_d = rays_d.data[0][ray_idx]
        one_ray_code = threejs_arrow(
            ray_o, ray_d, length=length, suffix=ray_idx
        )
        js_code += one_ray_code
    return js_code


def vis_rays(
    rays_o: Tensor,
    rays_d: Tensor,
    *,
    length: float = 1.0,
    sample: int = None,
    block: bool = True,
    port: int = 6150,
    scale: float = 1.0,
    vr: bool = False,
    title: str = "pdebug",
    res_dir: Optional[str] = None,
    no_server: Optional[bool] = False,
) -> None:
    """Visualize rays.

    Args:
        rays_o: input rays origin tensor.
        rays_d: input rays direction tensor.
        length: ray arrow length.
        sample: ray sample number. Default don't sample.
        block: block process with http server.
        port: http server port.
        scale: scene scale.
        vr: in vr mode.
    """
    rays_o = RayTensor(rays_o)
    rays_d = RayTensor(rays_d)
    js_code = rays_to_js_code(rays_o, rays_d, length, sample)
    _serve(
        js_code,
        scale=scale,
        block=block,
        port=port,
        vr=vr,
        title=title,
        res_dir=res_dir,
        no_server=no_server,
    )


def vis_xyz(
    xyz: Tensor,
    *,
    size: float = 0.05,
    sample: Optional[int] = 1000,
    rays_o: Optional[Tensor] = None,
    rays_d: Optional[Tensor] = None,
    rays_sample: Optional[int] = None,
    rays_length: float = 1.0,
    block: bool = True,
    port: int = 6150,
    scale: float = 1.0,
    vr: bool = False,
    title: str = "pdebug",
    res_dir: Optional[str] = None,
    no_server: Optional[bool] = False,
) -> None:
    """Visualize sampled point xyz and rays.

    Args:
        xyz: input point cloud xyz.
        size: point size.
        rays_o: input rays origin tensor.
        rays_d: input rays direction tensor.
        sample: ray sample number. Default don't sample.
        block: block process with http server.
        port: http server port.
        scale: scene scale.
        vr: in vr mode.
        title: html title.
        res_dir: engine output directory.
    """
    pcd = PointcloudTensor(xyz)

    if sample > 0:
        assert isinstance(sample, int)
        print(f"sample points num: {pcd.point_num} => {sample}")
        np.random.shuffle(pcd.data)
        pcd.data = pcd.data[:sample]
    js_code = threejs_pcd(pcd, size=size)
    if vr:
        print("Be careful when in vr mode, static/desc.png may be lost")
    if rays_o is not None and rays_d is not None:
        rays_o = RayTensor(rays_o)
        rays_d = RayTensor(rays_d)
        js_code += rays_to_js_code(rays_o, rays_d, rays_length, rays_sample)
    _serve(
        js_code,
        scale=scale,
        block=block,
        port=port,
        vr=vr,
        title=title,
        res_dir=res_dir,
        no_server=no_server,
    )


def vis_point_in_image(
    xyz: Tensor,
    extrinsic: Tensor,
    intrinsic: Tensor,
    *,
    images: Optional[List[Tensor]] = None,
    topk: Optional[int] = None,
    title: Optional[str] = None,
    res_dir: Optional[str] = None,
    num_workers: int = 0,
    port: int = 6150,
    block: bool = True,
    no_server: Optional[bool] = False,
):
    """Project point to camera plane."""
    xyz = x_to_ndarray(xyz)
    # prepare cameras
    cameras = []
    if extrinsic.ndim == 3:
        for ext in extrinsic:
            camera = Camera(ext, intrinsic)
            cameras.append(camera)
        if images is not None:
            assert len(images) == len(cameras)
        else:
            images = [None for _ in range(len(cameras))]
    else:
        cameras.append(Camera(ext, intrinsic))
        images = [images]

    if topk:
        cameras = cameras[:topk]
        images = images[:topk]

    engine = ImageEngine(
        res_dir=res_dir,
        serve_first=(not no_server),
        serve_kwargs={"port": port},
    )

    if num_workers > 0:
        raise NotImplementedError("bugs!!!!!")

    @mp(nums=num_workers)
    def _process(process_id, cameras):
        for idx, camera in enumerate(cameras):
            print(f"render image using camera: [{idx} / {len(cameras)}]")
            image = images[idx]
            rendered_image = camera.project_point_to_image(xyz, image=image)
            prefix = f"{title}"
            if num_workers > 0:
                prefix += f"_{process_id:02d}"
            if topk and topk == 1:
                pass
            else:
                prefix += f"_{idx:06d}"
            engine.add_image(rendered_image, prefix=prefix)

    _process(cameras)

    if no_server:
        return
    if block:
        engine.serve(port=port)
    else:
        p = Process(target=engine.serve, kwargs={"port": port})
        p.start()
        _global_engines_process.append(p)
