"""This module implements a plane object and related functions
"""
from .core import is_zero
from .line import Line3d
from .point3d import Point3d
from .vector3d import Vector3d


class Plane3d(object):
    """A 3d plane object
    This plane class has an infinite size.

    A plane can be initialized with point on the plane and a vector normal to
    the plane, or with three points

    Planes are infinite and currently have no origin.
    """

    def __init__(self, *args, **kwargs):

        if len(args) == 2:
            # assume we have a point and vector
            self.point = Point3d.from_numpy(args[0])
            self.normal = Vector3d.from_numpy(args[1])

        elif len(args) == 3:
            # assume we have 3 points
            v1 = Point3d.from_numpy(args[2]) - Point3d.from_numpy(args[1])
            v2 = Point3d.from_numpy(args[1]) - Point3d.from_numpy(args[0])
            normal = v1.cross(v2)
            self.point = Point3d.from_numpy(args[0])
            self.normal = normal

        self.d = -(self.normal.dot(self.point))

    def angle_to(self, other):
        """measures the angle between this plane and another plane. This uses
        that angle between two normal vectors.
        Units expressed in radians.
        """
        if isinstance(other, Plane3d):
            other_vector = other.normal
        elif isinstance(other, Vector3d):
            other_vector = other
        return self.normal.angle_to(other_vector)

    def intersect(self, other):
        """Finds the intersection of this plane with another object."""
        if isinstance(other, Plane3d):
            # return the line intersection of two planes
            # first, get the cross product of the two plane normals
            # which is a vector parallel to L
            vector = self.normal.cross(other.normal)
            absCoords = [abs(c) for c in vector]
            if is_zero(sum(absCoords)):
                # the planes are parallel and do not intersect
                return None
            else:
                # the planes intersect in a line
                # first find the largest coordinate in the vector
                cNum = None
                cMax = 0
                for i, c in enumerate(absCoords):
                    if c > cMax:
                        cMax = c
                        cNum = i
                dims = ["x", "y", "z"]
                biggestDim = dims.pop(cNum)
                p = {}
                p[biggestDim] = 0
                if biggestDim == "x":
                    p["y"] = (
                        other.d * self.normal.z - self.d * other.normal.z
                    ) / vector.x
                    p["z"] = (
                        self.d * other.normal.y - other.d * self.normal.y
                    ) / vector.x
                elif biggestDim == "y":
                    p["x"] = (
                        self.d * other.normal.z - other.d * self.normal.z
                    ) / vector.y
                    p["z"] = (
                        other.d * self.normal.x - self.d * other.normal.x
                    ) / vector.y
                else:  # biggest dim is z
                    p["x"] = (
                        other.d * self.normal.y - self.d * other.normal.y
                    ) / vector.z
                    p["y"] = (
                        self.d * other.normal.x - other.d * self.normal.x
                    ) / vector.z
                point = Point3d(**p)
                return Line3d(vector, point)

        elif isinstance(other, Line3d):
            # return the point intersection of a line and a plane
            pass
        pass

    def __repr__(self):
        return "Plane3d( %s, %s )" % (self.point, self.normal)

    def get_nearest_point_on_normal(self, point):
        """
        https://blog.csdn.net/qq_32867925/article/details/114294753
        """
        point = Point3d.from_numpy(point)
        vec1 = self.point.vector_to(point)
        k = vec1.dot(self.normal) / sum(n**2 for n in self.normal)
        q = self.normal * k + self.point
        return q.numpy()
