"""
Hunyuan3D-Paint infer node for PBR texture generation on 3D meshes.

Depends:
    torch, torchvision, diffusers, transformers, trimesh
"""
import json
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import types
from pathlib import Path
from typing import List, Optional

from pdebug.piata import Input, Output
from pdebug.visp import draw

import typer
from PIL import Image


def _prepend_sys_paths(paths: List[Path]) -> None:
    """Put repo-local imports ahead of installed packages deterministically."""
    for path in reversed(paths):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def _build_custom_rasterizer(extension_root: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "4")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "-e",
            str(extension_root),
        ],
        cwd=extension_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build Hunyuan3D custom_rasterizer extension with "
            f"{sys.executable}:\n{result.stdout}"
        )


def _ensure_custom_rasterizer(repo: Path) -> None:
    """Import the official repo-local rasterizer package, not its namespace dir."""
    extension_root = repo / "hy3dpaint" / "custom_rasterizer"
    _prepend_sys_paths([extension_root])

    module = sys.modules.get("custom_rasterizer")
    if module is not None and not hasattr(module, "rasterize"):
        sys.modules.pop("custom_rasterizer", None)

    try:
        import custom_rasterizer as cr
    except ModuleNotFoundError as exc:
        if exc.name != "custom_rasterizer_kernel":
            raise
        _build_custom_rasterizer(extension_root)
        importlib.invalidate_caches()
        sys.modules.pop("custom_rasterizer", None)
        import custom_rasterizer as cr

    if not all(hasattr(cr, name) for name in ("rasterize", "interpolate")):
        module_file = getattr(cr, "__file__", "<namespace package>")
        sys.modules.pop("custom_rasterizer", None)
        raise ImportError(
            "Imported an invalid custom_rasterizer module from "
            f"{module_file}; expected repo-local rasterize/interpolate APIs."
        )


def _build_mesh_inpaint_processor(extension_root: Path) -> None:
    includes = subprocess.run(
        [sys.executable, "-m", "pybind11", "--includes"],
        cwd=extension_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if includes.returncode != 0:
        raise RuntimeError(
            "Failed to query pybind11 include paths for Hunyuan3D "
            f"mesh_inpaint_processor:\n{includes.stdout}"
        )

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    result = subprocess.run(
        [
            "c++",
            "-O3",
            "-Wall",
            "-shared",
            "-std=c++11",
            "-fPIC",
            *shlex.split(includes.stdout.strip()),
            "mesh_inpaint_processor.cpp",
            "-o",
            f"mesh_inpaint_processor{suffix}",
        ],
        cwd=extension_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build Hunyuan3D mesh_inpaint_processor extension "
            f"with {sys.executable}:\n{result.stdout}"
        )


def _ensure_mesh_inpaint_processor(repo: Path) -> None:
    """Compile and import the official pybind mesh inpaint extension."""
    module_name = "DifferentiableRenderer.mesh_inpaint_processor"
    extension_root = repo / "hy3dpaint" / "DifferentiableRenderer"

    package = sys.modules.setdefault(
        "DifferentiableRenderer", types.ModuleType("DifferentiableRenderer")
    )
    package.__path__ = [str(extension_root)]

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        _build_mesh_inpaint_processor(extension_root)
        importlib.invalidate_caches()
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)

    if not hasattr(module, "meshVerticeInpaint"):
        module_file = getattr(module, "__file__", "<unknown>")
        raise ImportError(
            "Imported an invalid mesh_inpaint_processor module from "
            f"{module_file}; expected meshVerticeInpaint."
        )


def _install_mesh_utils_without_bpy(repo: Path) -> None:
    """Load official mesh IO helpers while skipping Blender-only conversion."""
    module_name = "DifferentiableRenderer.mesh_utils"
    if module_name in sys.modules:
        return

    mesh_utils_path = (
        repo / "hy3dpaint" / "DifferentiableRenderer" / "mesh_utils.py"
    )
    source = mesh_utils_path.read_text(encoding="utf-8")
    source = source.replace("import bpy\n", "")
    cutoff = source.find("\ndef _setup_blender_scene")
    if cutoff != -1:
        source = source[:cutoff]
    source += (
        "\n\ndef convert_obj_to_glb(obj_path, glb_path, *args, **kwargs):\n"
        "    return False\n"
    )

    package = sys.modules.setdefault(
        "DifferentiableRenderer", types.ModuleType("DifferentiableRenderer")
    )
    package.__path__ = [str(repo / "hy3dpaint" / "DifferentiableRenderer")]
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(mesh_utils_path)
    sys.modules[module_name] = module
    exec(compile(source, str(mesh_utils_path), "exec"), module.__dict__)


def _install_torchvision_compat() -> None:
    """Provide the legacy module path imported by BasicSR."""
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return
    try:
        import torchvision.transforms.functional as functional
    except ImportError:
        return
    sys.modules[module_name] = functional


def _apply_low_memory_config(config, face_count: int, resolution: int) -> dict:
    """Scale official paint internals down for tiny integration meshes."""
    effective = {
        "resolution": resolution,
        "render_size": config.render_size,
        "texture_size": config.texture_size,
        "max_num_view": config.max_selected_view_num,
    }
    if face_count > 1000:
        return effective

    config.resolution = min(resolution, 256)
    config.render_size = min(config.render_size, 512)
    config.texture_size = min(config.texture_size, 1024)
    effective.update(
        {
            "resolution": config.resolution,
            "render_size": config.render_size,
            "texture_size": config.texture_size,
            "max_num_view": config.max_selected_view_num,
        }
    )
    return effective


def hunyuan3d_paint_main(
    repo: str,
    mesh_path: str,
    image_path: str,
    output: str = "tmp_hunyuan3d_paint",
    model_path: str = "tencent/Hunyuan3D-2.1",
    max_num_view: int = 6,
    resolution: int = 512,
    face_count: int = 40000,
    cache: bool = True,
):
    """
    Generate PBR textures for 3D meshes using Hunyuan3D-Paint model.

    Args:
        repo: Path to Hunyuan3D repository
        mesh_path: Path to input 3D mesh file (glb/obj/ply) or folder containing meshes
        image_path: Path to reference image(s) for texture generation (file or folder)
        output: Output directory for textured 3D meshes
        model_path: Path to pretrained Hunyuan3D-Paint model
        max_num_view: Maximum number of views for texture generation (6-9)
        resolution: Texture resolution (512 or 768)
        face_count: Maximum number of faces for texture generation (1000-100000)
        cache: Whether to skip if output already exists
    """

    # Expand paths
    mesh_path = Path(mesh_path).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    repo = Path(repo).expanduser().resolve()

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Check cache
    if cache and (output / "textures.json").exists():
        print(f"Output {output} exists, skipping...")
        return str(output)

    try:
        # Add repo paths expected by the official paint scripts.
        _prepend_sys_paths(
            [
                repo / "hy3dpaint" / "custom_rasterizer",
                repo,
                repo / "hy3dpaint",
            ]
        )
        _ensure_custom_rasterizer(repo)
        _install_mesh_utils_without_bpy(repo)
        _ensure_mesh_inpaint_processor(repo)
        _install_torchvision_compat()
        from hy3dpaint.textureGenPipeline import (
            Hunyuan3DPaintConfig,
            Hunyuan3DPaintPipeline,
        )
    except ImportError as e:
        raise ImportError(f"Failed to import Hunyuan3D-Paint pipeline: {e}")

    # Initialize pipeline
    print("Loading Hunyuan3D-Paint pipeline...")

    # Configure pipeline
    config = Hunyuan3DPaintConfig(
        max_num_view=max_num_view, resolution=resolution
    )
    effective_parameters = _apply_low_memory_config(
        config, face_count, resolution
    )
    config.realesrgan_ckpt_path = str(
        repo / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    )
    config.multiview_cfg_path = str(
        repo / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
    )
    config.custom_pipeline = "hy3dpaint.hunyuanpaintpbr"
    config.multiview_pretrained_path = model_path

    paint_pipeline = Hunyuan3DPaintPipeline(config)

    # Get mesh files
    if mesh_path.is_dir():
        mesh_files = (
            list(mesh_path.glob("*.glb"))
            + list(mesh_path.glob("*.obj"))
            + list(mesh_path.glob("*.ply"))
        )
        mesh_files.sort(key=lambda x: x.stem)
    else:
        mesh_files = [mesh_path]

    if not mesh_files:
        raise ValueError("No valid mesh files found")

    # Get image files
    if image_path.is_dir():
        image_files = (
            list(image_path.glob("*.png"))
            + list(image_path.glob("*.jpg"))
            + list(image_path.glob("*.jpeg"))
        )
        image_files.sort(key=lambda x: x.stem)
    else:
        image_files = [image_path]

    if not image_files:
        raise ValueError("No valid image files found")

    print(f"Found {len(mesh_files)} meshes and {len(image_files)} images")

    # Ensure we have matching pairs (use modulo if counts don't match)
    results = []
    for i, mesh_file in enumerate(mesh_files):
        img_file = image_files[i % len(image_files)]
        print(
            f"Processing mesh {i+1}/{len(mesh_files)}: {mesh_file.name} with image: {img_file.name}"
        )

        # Load reference image
        reference_image = Image.open(img_file).convert("RGB")

        # The official paint pipeline writes OBJ first and can export GLB.
        output_mesh_path = output / f"{mesh_file.stem}_textured.obj"
        textured_mesh_path = paint_pipeline(
            mesh_path=str(mesh_file),
            image_path=str(img_file),
            output_mesh_path=str(output_mesh_path),
        )
        textured_mesh_path = Path(textured_mesh_path)
        output_glb_path = textured_mesh_path.with_suffix(".glb")
        if output_glb_path.exists():
            textured_mesh_path = output_glb_path

        results.append(
            {
                "input_mesh": str(mesh_file),
                "reference_image": str(img_file),
                "output_textured_mesh": str(textured_mesh_path),
                "parameters": {
                    "max_num_view": max_num_view,
                    "resolution": resolution,
                    "face_count": face_count,
                },
                "effective_parameters": effective_parameters,
            }
        )

        print(f"Saved textured mesh to {textured_mesh_path}")

    # Save metadata
    with open(output / "textures.json", "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(
        typer.style(
            f"Successfully generated textures for {len(results)} meshes in {output}",
            fg=typer.colors.GREEN,
        )
    )
    return str(output)


if __name__ == "__main__":
    typer.run(hunyuan3d_paint_main)
