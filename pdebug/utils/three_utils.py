from typing import Optional, Tuple, Union

from pdebug.data_types import CameraExtrinsic, PointcloudTensor, Tensor
from pdebug.templates import STATIC_ROOT
from pdebug.utils.env import OPEN3D_INSTALLED

if OPEN3D_INSTALLED:
    import open3d as o3d

__all__ = ["threejs_arrow", "threejs_line", "threejs_camera", "threejs_pcd"]


def threejs_arrow(
    origin: Union[Tensor, Tuple[float, float, float]],
    direction: Union[Tensor, Tuple[float, float, float]],
    *,
    length: float = 1.0,
    color: str = "0xffff00",
    suffix: Union[str, int] = 0,
) -> str:
    """Threejs arrow template.

    Args:
        origin: arrow origin point.
        direction: arrow direction vector.
        length: arrow length.
        color: arrow color.
        suffix: var name suffix in js code.
    """
    x, y, z = origin
    rx, ry, rz = direction
    js_code = f"""
const dir_{suffix} = new THREE.Vector3({rx}, {ry}, {rz});
dir_{suffix}.normalize();
const origin_{suffix} = new THREE.Vector3({x}, {y}, {z});
const length_{suffix} = {length};
const color_{suffix} = {color};
const arrow_helper_{suffix} = new THREE.ArrowHelper(dir_{suffix}, origin_{suffix}, length_{suffix}, color_{suffix});
scene.add(arrow_helper_{suffix});
"""  # noqa
    return js_code


def threejs_line(
    p1: Union[Tensor, Tuple[float, float, float]],
    p2: Union[Tensor, Tuple[float, float, float]],
    *,
    color: Optional[str] = None,
    suffix: str = "0",
) -> str:
    """Threejs line template.

    Args:
        p1: line point1.
        p2: line point2.
        color: arrow color. Default is 0x0000ff (blue).
        suffix: var name suffix in js code.
    """
    # unpack data
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    if not color:
        color = "0x0000ff"
    assert isinstance(
        color, str
    ), f"color should be str, but found {type(color)}"
    js_code = f"""
const points_{suffix} = [
    new THREE.Vector3({x1}, {y1}, {z1}),
    new THREE.Vector3({x2}, {y2}, {z2}),
]
const material_{suffix} = new THREE.LineBasicMaterial({{color: {color}}});
const geometry_{suffix} = new THREE.BufferGeometry().setFromPoints(points_{suffix});
const line_{suffix} = new THREE.Line(geometry_{suffix}, material_{suffix});
scene.add(line_{suffix});
"""  # noqa
    return js_code


def threejs_camera(
    pose: Union[CameraExtrinsic, Tensor],
    *,
    size: float = 0.1,
    color: Optional[str] = None,
    suffix: Union[str, int] = 0,
) -> str:
    """Three camera template.

    A camera is visualized with 8 line segments.
    """
    if isinstance(pose, CameraExtrinsic):
        pose = pose.data
    pos = pose[:3, 3]
    a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
    b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
    c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
    d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
    segs = [
        [pos, a],
        [pos, b],
        [pos, c],
        [pos, d],
        [a, b],
        [b, c],
        [c, d],
        [d, a],
    ]

    js_code = ""
    for ind, (p1, p2) in enumerate(segs):
        js_code += threejs_line(p1, p2, color=color, suffix=f"{suffix}_{ind}")
    return js_code


def threejs_pcd(
    pcd: PointcloudTensor,
    *,
    color: Optional[str] = None,
    size: float = 0.5,
    suffix: str = "0",
) -> str:
    """Threejs pointcloud template.

    Args:
        pcd: point cloud.
        color: point color. Default is 0x0000ff (blue).
        suffix: var name suffix in js code.
    """
    sprite_texture_file = "static/disc.png"
    if not color:
        color = "0x0000ff"
    assert isinstance(
        color, str
    ), f"color should be str, but found {type(color)}"

    js_code = f"const position_{suffix} = []\n"

    if OPEN3D_INSTALLED:
        if isinstance(pcd, o3d.geometry.PointCloud):
            pcd = PointcloudTensor.from_open3d(pcd)
    for point in pcd.data:
        x, y, z = point
        js_code += f"position_{suffix}.push({x}, {y}, {z});\n"

    if pcd.color is not None:
        js_code += f"const colors_{suffix} = [];\n"
        js_code += f"const color_{suffix} = new THREE.Color();\n"

        for color_i in pcd.color:
            js_code += f"color_{suffix}.setRGB( {color_i[0]}, {color_i[1]}, {color_i[2]} );\n"
            js_code += f"colors_{suffix}.push( color_{suffix}.r, color_{suffix}.g, color_{suffix}.b );\n"

    js_code += f"""
const geometry_{suffix} = new THREE.BufferGeometry();
geometry_{suffix}.setAttribute('position', new THREE.Float32BufferAttribute(position_{suffix}, 3));
"""
    if pcd.color is not None:
        js_code += f"geometry_{suffix}.setAttribute( 'color', new THREE.Float32BufferAttribute( colors_{suffix}, 3 ) );"

    js_code += f"""
geometry_{suffix}.computeBoundingSphere();

const sprite_{suffix} = new THREE.TextureLoader().load('{sprite_texture_file}');
const material_{suffix} = new THREE.PointsMaterial( {{ size: {size}, vertexColors: true, map: sprite_{suffix}, alphaTest: 0.5}});
const points_{suffix} = new THREE.Points( geometry_{suffix}, material_{suffix}  );
scene.add( points_{suffix} )

"""  # noqa
    return js_code
