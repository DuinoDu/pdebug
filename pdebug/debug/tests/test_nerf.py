from pdebug.debug import nerf
from pdebug.tests.test_data_types import dummy_pose  # noqa: F401
from pdebug.utils.env import TORCH_INSTALLED

import pytest

pytest.skip("deprecated testc ase", allow_module_level=True)

if TORCH_INSTALLED:
    import torch


@pytest.fixture
def dummy_rays():
    rays_o = torch.tensor(
        [
            [
                [0.1628, 3.0490, -1.0378],
                [0.1628, 3.0490, -1.0378],
            ]
        ]
    )
    rays_d = torch.tensor(
        [
            [
                [0.0470, -0.9987, 0.0224],
                [0.1973, -0.8808, 0.4303],
            ]
        ]
    )
    return rays_o, rays_d


def test_vis_pose(dummy_pose):  # noqa: F401, F811
    nerf.vis_pose(dummy_pose, block=False, port=8890, vr=True)


def test_vis_rays(dummy_rays):
    rays_o, rays_d = dummy_rays
    nerf.vis_rays(rays_o, rays_d, block=False, port=8889, vr=False)


def test_wait(dummy_pose, dummy_rays):
    nerf.vis_pose(dummy_pose, block=False, port=8890)
    rays_o, rays_d = dummy_rays
    nerf.vis_rays(rays_o, rays_d, block=False, port=8889)
    nerf.wait(port=8890)


@pytest.mark.skipif(True, reason="skipped, this is used for manually debug")
def test_vis_pose_lego_and_car():  # noqa: F401, F811
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    lego_pose = np.array(
        [
            [-0.999, 0.004, -0.013, -0.053],
            [-0.013, -0.299, 0.953, 3.845],
            [0.000, 0.954, 0.299, 1.208],
            [0, 0, 0, 1],
        ]
    )
    car_pose = np.array(
        [
            [-0.832, 0.182, 0.522, 1.2],
            [0.0, 0.944, -0.329, 5.08],
            [0.55, -0.27, -0.78, 8.197],
            [0, 0, 0, 1],
        ]
    )

    # rotate car_pose according x-axis by 180 degrees.
    car_eular = R.from_matrix(car_pose[:3, :3]).as_euler("zyx", degrees=True)
    car_eular[2] += 180.0
    car_rotated = R.from_euler("zyx", car_eular, degrees=True).as_matrix()
    car_pose[:3, :3] = car_rotated

    pose = [lego_pose, car_pose]
    colors = ["0xff0000", "0x00ff00"]
    data = torch.tensor(pose)

    e_lego = R.from_matrix(data[0][:3, :3]).as_euler("zyx", degrees=True)
    e_car = R.from_matrix(data[1][:3, :3]).as_euler("zyx", degrees=True)
    print("\n")
    print(f"lego: {e_lego}")
    print(f"car: {e_car}")

    # nerf.vis_pose(
    #     data, colors=colors, block=True, port=8892, vr=True)

    def a(x, origin=(1, 1, 1), delta=0.2):
        y = np.zeros((4, 4))
        y[:3, :3] = x
        y[:3, 3] = origin
        y[:3, 3] += delta
        y[3, 3] = 1.0
        return y

    ref_pose = [
        a(R.from_euler("x", 0, degrees=True).as_matrix(), delta=-0.4),
        a(R.from_euler("x", 90, degrees=True).as_matrix(), delta=-0.2),
        a(R.from_euler("x", 180, degrees=True).as_matrix(), delta=0.0),
        a(R.from_euler("x", 270, degrees=True).as_matrix(), delta=0.2),
        a(
            R.from_euler("y", 0, degrees=True).as_matrix(),
            origin=(1, 1, -1),
            delta=-0.4,
        ),
        a(
            R.from_euler("y", 90, degrees=True).as_matrix(),
            origin=(1, 1, -1),
            delta=-0.2,
        ),
        a(
            R.from_euler("y", 180, degrees=True).as_matrix(),
            origin=(1, 1, -1),
            delta=0.0,
        ),
        a(
            R.from_euler("y", 270, degrees=True).as_matrix(),
            origin=(1, 1, -1),
            delta=0.2,
        ),
        a(
            R.from_euler("z", 0, degrees=True).as_matrix(),
            origin=(-1, 1, 1),
            delta=-0.4,
        ),
        a(
            R.from_euler("z", 45, degrees=True).as_matrix(),
            origin=(-1, 1, 1),
            delta=-0.2,
        ),
        a(
            R.from_euler("z", 90, degrees=True).as_matrix(),
            origin=(-1, 1, 1),
            delta=0.0,
        ),
        a(
            R.from_euler("z", 135, degrees=True).as_matrix(),
            origin=(-1, 1, 1),
            delta=0.2,
        ),
    ]
    ref_data = torch.tensor(ref_pose)
    ref_colors = ["0x0000ff" for _ in range(len(ref_pose))]

    data = torch.cat([data, ref_data], axis=0)
    colors = colors + ref_colors

    nerf.vis_pose(
        data, colors=colors, block=True, port=8892, vr=True, scale=0.1
    )


def test_vis_xyz():
    dummy_xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    nerf.vis_xyz(dummy_xyz, block=False, port=6006)
