from pdebug.data_types import (
    Camera,
    CameraExtrinsic,
    CameraIntrinsic,
    PointcloudTensor,
    RayTensor,
)

import numpy as np
import pytest


@pytest.fixture
def dummy_data():
    return np.random.rand(1, 1000, 3)


@pytest.fixture
def dummy_pose():
    pose = [
        [-0.9879, 0.1466, -0.0505, 0.1628],
        [0.0000, -0.3257, -0.9455, 3.0490],
        [-0.1550, -0.9340, 0.3218, -1.0378],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ]
    data = np.array(pose)
    return data


@pytest.fixture
def dummy_intrinsic():
    return np.array([627.0, 627.0, 800.0, 800.0])


def test_ray_tensor(dummy_data):
    rt = RayTensor(dummy_data)
    assert rt.batch_size == 1
    assert rt.ray_num == 1000
    assert rt.channel_num == 3


def test_camera_extrinsic(dummy_pose):
    ce = CameraExtrinsic(dummy_pose)
    assert ce.R.shape == (3, 3)
    assert ce.t.shape == (3,)


def test_camera_intrinsic(dummy_intrinsic):
    ci = CameraIntrinsic(dummy_intrinsic)
    assert ci.data.shape == (3, 3)
    assert ci.fx == dummy_intrinsic[0]
    assert ci.fy == dummy_intrinsic[1]
    assert ci.cx == dummy_intrinsic[2]
    assert ci.cy == dummy_intrinsic[3]


def test_camera(dummy_pose, dummy_intrinsic):
    camera = Camera(dummy_pose, dummy_intrinsic)
    np.testing.assert_equal(camera.E.data, dummy_pose)
    np.testing.assert_equal(camera.I.tolist(), dummy_intrinsic.tolist())
    point = np.array([[0, 0, 0]], dtype=np.float32)
    image = camera.capture(point)
    assert image.shape == (camera.I.h, camera.I.w, 3)


def test_pcd_to_html(tmpdir):
    points = np.array(([0, 0, 0], [0, 1, 0], [0, 1, 1]))
    colors = np.array(([255, 0, 0], [0, 255, 0], [0, 0, 255]))
    labels = np.array([0, 0, 1])
    pcd = PointcloudTensor(points, colors, labels)
    tmpdir = "tmp_pcd_to_html"
    pcd.to_html(title="pcd", output_dir=tmpdir + "/pcd")

    pcd.label_as_color()
    pcd.to_html(title="label_as_color", output_dir=tmpdir + "/label_as_color")
