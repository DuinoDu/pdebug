"""
Migrated from https://github.com/bengolder/python-geometry
"""
from .boxes import Box2d, Box3d
from .core import is_zero
from .intervals import Interval, Scale
from .line import Line3d, LineSegment2d
from .matrix import Matrix, MatrixError
from .plane import Plane3d
from .point2d import Point2d
from .point3d import Point3d
from .points import PointSet
from .vector2d import PageX, PageY, Vector2d
from .vector3d import Vector3d, WorldX, WorldY, WorldZ

__all__ = [
    "Interval",
    "Scale",
    "Box2d",
    "Box3d",
    "Vector2d",
    "Vector3d",
    "Point2d",
    "Point3d",
    "PointSet",
    "PageX",
    "PageY",
    "WorldX",
    "WorldY",
    "WorldZ" "Matrix",
    "Line3d",
    "LineSegment2d",
    "Plane3d",
    "is_zero",
]
