from pdebug.visp.colormap import Colormap


def test_colormap():
    colormap = Colormap(10)
    assert len(colormap) == 10
    c1 = colormap[0]
    assert isinstance(c1, list)
    assert len(c1) == 3


def test_colormap_in_hex_mode():
    colormap = Colormap(10, hex_mode=True)
    assert len(colormap) == 10
    c1 = colormap[0]
    assert isinstance(c1, str)
