from .point import PointBase
from .vector2d import Vector2d


class Point2d(Vector2d, PointBase):
    def __init__(self, *args, **kwargs):
        Vector2d.__init__(self, *args, **kwargs)
        PointBase.__init__(self)

    def __repr__(self):
        return "Point2d(%s, %s)" % self.coords

    def distance_to(self, other):
        """Find the distance between this point and another."""
        return (other - self).length


PageOrigin = Point2d(0.0, 0.0)
