class PointBase(object):
    """Should not be instantiated directly."""

    def vector_to(self, other):
        """Find the vector to another point."""
        return other - self
