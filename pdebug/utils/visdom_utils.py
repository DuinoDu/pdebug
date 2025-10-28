from pdebug.utils.env import VISDOM_INSTALLED

__all__ = ["get_global_visdom"]


if VISDOM_INSTALLED:
    import visdom
else:
    visdom = None

_global_visdom = None


def get_global_visdom():
    """Get global visdom instance."""
    assert VISDOM_INSTALLED
    global _global_visdom
    if _global_visdom is None:
        _global_visdom = visdom.Visdom()
    return _global_visdom
