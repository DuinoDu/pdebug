from pdebug.geometry import (
    Box2d,
    Box3d,
    Interval,
    Point2d,
    Point3d,
    Scale,
    Vector2d,
    Vector3d,
    WorldX,
    WorldY,
    WorldZ,
)

import pytest


def test_vector3d_dot_cross_angle_and_distance_for_room_axes():
    forward = Vector3d(0.0, 3.0, 0.0)
    right = Vector3d(4.0, 0.0, 0.0)

    assert right.dot(forward) == 0.0
    assert right.cross(forward) == Vector3d(0.0, 0.0, 12.0)
    assert WorldX.cross(WorldY) == WorldZ
    assert WorldX.angle_to(WorldY, degrees=True) == pytest.approx(90.0)
    assert right.distance_to(forward) == pytest.approx(5.0)


def test_vector2d_length_normalization_and_scalar_projection():
    displacement = Vector2d(6.0, 8.0)
    unit_direction = displacement.normalized()

    assert displacement.length == pytest.approx(10.0)
    assert unit_direction.length == pytest.approx(1.0)
    assert unit_direction.coords == pytest.approx((0.6, 0.8))
    assert displacement.toLength(25.0).coords == pytest.approx((15.0, 20.0))


def test_point2d_distance_and_vector_between_landmarks():
    entrance = Point2d(2.0, 1.0)
    kiosk = Point2d(8.0, 9.0)

    assert entrance.vector_to(kiosk) == Vector2d(6.0, 8.0)
    assert entrance.distance_to(kiosk) == pytest.approx(10.0)
    assert kiosk.distance_to(entrance) == pytest.approx(10.0)


def test_interval_maps_fractions_and_reports_membership_for_timeline():
    timeline = Interval(10.0, 70.0)

    assert timeline(0.0, 0.25, 1.0) == [10.0, 25.0, 70.0]
    assert timeline.fraction(10.0, 40.0, 70.0) == [0.0, 0.5, 1.0]
    assert timeline.contains(10.0)
    assert timeline.contains(69.999)
    assert not timeline.contains(70.0)


def test_scale_converts_between_sensor_and_display_ranges():
    temperature_to_pixels = Scale(domain=(20.0, 40.0), range=(100.0, 500.0))

    assert temperature_to_pixels(20.0) == pytest.approx(100.0)
    assert temperature_to_pixels(30.0) == pytest.approx(300.0)
    assert temperature_to_pixels(40.0) == pytest.approx(500.0)
    assert temperature_to_pixels.reverse(300.0) == pytest.approx(30.0)


def test_box2d_and_box3d_centers_for_viewports_and_rooms():
    viewport = Box2d(width=1920, height=1080)
    room = Box3d(width=4.0, length=6.0, height=3.0)

    assert viewport.center() == Point2d(960.0, 540.0)
    assert room.center() == Point3d(2.0, 3.0, 1.5)
    assert room.center() == Vector3d(2.0, 3.0, 1.5)
    assert room.center().z == pytest.approx(1.5)
