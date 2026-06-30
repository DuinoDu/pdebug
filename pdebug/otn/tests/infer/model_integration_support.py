"""Shared helpers for opt-in real model integration tests."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import pytest
from PIL import Image


RUN_ENV = "PDEBUG_RUN_MODEL_INTEGRATION"
CASE_ENV = "PDEBUG_MODEL_CASES"
CACHE_ENV = "PDEBUG_MODEL_CACHE"
ALLOW_NETWORK_ENV = "PDEBUG_MODEL_ALLOW_NETWORK"
FORCE_CPU_ENV = "PDEBUG_MODEL_FORCE_CPU"


def integration_enabled() -> bool:
    return os.getenv(RUN_ENV) == "1"


def network_allowed() -> bool:
    return os.getenv(ALLOW_NETWORK_ENV, "1") != "0"


def model_cache_root() -> Path:
    return Path(
        os.getenv(CACHE_ENV, "~/.cache/pdebug-model-integration")
    ).expanduser()


def selected_case_ids() -> Optional[set[str]]:
    raw = os.getenv(CASE_ENV)
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def has_cuda() -> bool:
    if os.getenv(FORCE_CPU_ENV) == "1":
        return False
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def has_docker() -> bool:
    return shutil.which("docker") is not None


def pytest_skip_if_disabled(case_id: str) -> None:
    if not integration_enabled():
        pytest.skip(
            f"real model integration tests are disabled; set {RUN_ENV}=1"
        )
    selected = selected_case_ids()
    if selected is not None and case_id not in selected:
        pytest.skip(f"{case_id} not selected by {CASE_ENV}")


def require_python_modules(modules: Iterable[str]) -> None:
    missing = {}
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing[module] = exc
    if missing:
        selected = selected_case_ids()
        if selected and "langsam_predict" in selected and "lang_sam" in missing:
            pytest.fail(_missing_langsam_message(missing))
        pytest.skip("missing Python modules: " + ", ".join(missing))


def _missing_langsam_message(missing: Mapping[str, BaseException]) -> str:
    lines = [
        "langsam_predict requires installed Python modules for real inference:"
    ]
    lines.extend(
        f"- {module}: {type(exc).__name__}: {exc}"
        for module, exc in missing.items()
    )
    lines.extend(
        [
            "Install the official LangSAM package with:",
            ".venv/bin/python -m pip install -U "
            "git+https://github.com/luca-medeiros/"
            "lang-segment-anything.git",
            "If GitHub clone hangs, try the official archive path:",
            ".venv/bin/python -m pip install --no-deps -U "
            "https://github.com/luca-medeiros/lang-segment-anything/"
            "archive/918043ed4666eea04da88aa179eb8d27ef4b1a1d.zip",
        ]
    )
    return "\n".join(lines)


def require_python_modules_or_fail(case, modules: Iterable[str]) -> None:
    missing = {}
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:
            missing[module] = exc
    if not missing:
        return
    if case.case_id == "langsam_predict":
        message = _missing_langsam_message(missing)
        selected = selected_case_ids()
        if selected and case.case_id in selected:
            pytest.fail(message)
        pytest.skip("missing Python modules: " + ", ".join(missing))

    lines = [
        f"{case.case_id} requires installed Python modules for real inference:"
    ]
    lines.extend(
        f"- {module}: {type(exc).__name__}: {exc}"
        for module, exc in missing.items()
    )
    message = "\n".join(lines)

    selected = selected_case_ids()
    if selected and case.case_id in selected:
        pytest.fail(message)
    pytest.skip("missing Python modules: " + ", ".join(missing))


def require_cuda() -> None:
    if not has_cuda():
        pytest.skip("CUDA is required for this real model integration case")


def require_docker() -> None:
    if not has_docker():
        pytest.skip("Docker is required for this real model integration case")


def clone_repo(url: str, target: Path, *, ref: Optional[str] = None) -> Path:
    """Clone an official model repository into the integration cache."""
    if target.exists():
        if not _repo_clone_complete(target):
            broken = target.with_name(
                f"{target.name}.broken-{int(time.time())}"
            )
            target.rename(broken)
            print(
                "Moved incomplete cached model repository "
                f"{target} to {broken}; recloning {url}."
            )
        else:
            return target
    if not network_allowed():
        pytest.skip(
            f"network disabled by {ALLOW_NETWORK_ENV}=0; cannot clone {url}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(target)],
        check=True,
    )
    if ref:
        subprocess.run(["git", "checkout", ref], cwd=target, check=True)
    return target


def _repo_clone_complete(target: Path) -> bool:
    """Return whether a cached git clone has a checked-out commit."""
    git_dir = target / ".git"
    if not git_dir.exists():
        return False
    if any(git_dir.rglob("*.lock")):
        return False
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=target,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def repo_name_from_url(url: str) -> str:
    """Return the cache directory name for a model repository URL."""
    repo_name = url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name


def case_repo_path(case: "IntegrationCase") -> Optional[Path]:
    """Return the expected cached repo path for a case, when declared."""
    if not case.repo_url:
        return None
    return model_cache_root() / "repos" / repo_name_from_url(case.repo_url)


def ensure_case_repo(case: "IntegrationCase") -> Optional[Path]:
    """Clone a case's official repo and repo-local downloads if needed."""
    if not case.repo_url:
        return None
    repo = clone_repo(case.repo_url, case_repo_path(case))
    for rel_path, url in case.repo_downloads:
        download_url(url, repo / rel_path)
    return repo


def prepare_case_environment(case: "IntegrationCase") -> Optional[Path]:
    """Prepare import paths and external repos for a real integration case."""
    repo = ensure_case_repo(case)
    repo_root = model_cache_root() / "repos"
    if case.case_id == "genpose2" and repo is not None:
        _normalize_genpose2_checkpoints(repo)
    extra_paths = []
    if case.case_id == "sam2":
        extra_paths.append(repo_root / "sam2")
    elif case.case_id == "langsam_predict":
        extra_paths.extend(
            [
                repo_root / "lang-segment-anything",
                repo_root / "sam2",
            ]
        )
    elif case.case_id.startswith("hunyuan3d_"):
        hy_repo = repo_root / "hunyuan3d-2.1"
        extra_paths.extend([hy_repo, hy_repo / "hy3dshape", hy_repo / "hy3dpaint"])
    elif case.case_id.startswith("oneposeviagen_") and repo is not None:
        extra_paths.extend(
            [
                repo,
                repo / "oneposeviagen",
                repo / "oneposeviagen" / "trellis",
                repo / "oneposeviagen" / "Amodal3R",
                repo / "oneposeviagen" / "fpose",
                repo / "oneposeviagen" / "fpose" / "fpose",
            ]
        )
    for path in extra_paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
    return repo


def _normalize_genpose2_checkpoints(repo: Path) -> None:
    """Normalize the official Dropbox zip layout into GenPose2 defaults."""
    ckpts = repo / "results" / "ckpts"
    for name, filename in (
        ("ScoreNet", "scorenet.pth"),
        ("EnergyNet", "energynet.pth"),
        ("ScaleNet", "scalenet.pth"),
    ):
        expected = ckpts / name / filename
        if expected.exists():
            continue
        alternate = repo / "results" / name / filename
        if alternate.exists():
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alternate, expected)


def download_url(url: str, target: Path) -> Path:
    """Download a small fixture or checkpoint into the integration cache."""
    if target.exists():
        return target
    if not network_allowed():
        pytest.skip(
            f"network disabled by {ALLOW_NETWORK_ENV}=0; cannot download {url}"
        )
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    return target


@dataclass(frozen=True)
class IntegrationCase:
    """Description of one real model integration case."""

    case_id: str
    file: str
    node: str
    description: str
    args_builder: str
    official_sources: tuple[str, ...] = ()
    repo_url: Optional[str] = None
    hf_model_ids: tuple[str, ...] = ()
    repo_downloads: tuple[tuple[str, str], ...] = ()
    preferred_runtime: str = "local"
    requires_cuda: bool = False
    requires_docker: bool = False
    required_modules: tuple[str, ...] = ()
    expected_paths: tuple[str, ...] = ()
    heavy_reason: str = ""
    marks: tuple[str, ...] = field(default_factory=tuple)


class FixtureFactory:
    """Create deterministic tiny inputs for real model smoke tests."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def image(self) -> Path:
        path = self.root / "image.png"
        if not path.exists():
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[:, :] = (230, 230, 220)
            arr[24:104, 24:104] = (220, 40, 40)
            arr[44:84, 44:84] = (40, 180, 80)
            Image.fromarray(arr).save(path)
        return path

    @property
    def image_dir(self) -> Path:
        path = self.root / "images"
        path.mkdir(exist_ok=True)
        for idx in range(3):
            target = path / f"{idx:06d}.png"
            if not target.exists():
                arr = np.array(Image.open(self.image).convert("RGB"))
                arr = np.roll(arr, idx * 4, axis=1)
                Image.fromarray(arr).save(target)
        return path

    @property
    def jpg_image_dir(self) -> Path:
        path = self.root / "jpg_images"
        path.mkdir(exist_ok=True)
        for idx, image_file in enumerate(sorted(self.image_dir.glob("*.png"))):
            target = path / f"{idx:05d}.jpg"
            if not target.exists():
                Image.open(image_file).convert("RGB").save(target)
        return path

    @property
    def mask_dir(self) -> Path:
        path = self.root / "masks"
        path.mkdir(exist_ok=True)
        for idx in range(3):
            target = path / f"{idx:06d}.png"
            if not target.exists():
                mask = np.zeros((128, 128), dtype=np.uint8)
                mask[28:100, 28:100] = 255
                Image.fromarray(mask).save(target)
        return path

    @property
    def depth_dir(self) -> Path:
        path = self.root / "depth"
        path.mkdir(exist_ok=True)
        for idx in range(3):
            target = path / f"{idx:06d}.png"
            if not target.exists():
                depth = np.full((128, 128), 1000 + idx * 10, dtype=np.uint16)
                Image.fromarray(depth).save(target)
        return path

    @property
    def video(self) -> Path:
        path = self.root / "video.mp4"
        if path.exists():
            return path
        import cv2

        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (128, 128),
        )
        for image_file in sorted(self.image_dir.glob("*.png")):
            frame = np.array(Image.open(image_file).convert("RGB"))[:, :, ::-1]
            writer.write(frame)
        writer.release()
        return path

    @property
    def camera_json(self) -> Path:
        path = self.root / "camera.json"
        if not path.exists():
            payload = {
                "cam_K": [
                    100.0,
                    0.0,
                    64.0,
                    0.0,
                    100.0,
                    64.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "depth_scale": 1000.0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @property
    def genpose2_meta(self) -> Path:
        path = self.root / "genpose2_meta.json"
        if not path.exists():
            payload = {
                "objects": [
                    {
                        "mask_id": 1,
                        "meta": {
                            "oid": "fixture-cube",
                            "class_name": "cube",
                            "class_label": 1,
                            "instance_path": "",
                            "scale": [0.1, 0.1, 0.1],
                            "is_background": False,
                            "bbox_side_len": [0.1, 0.1, 0.1],
                        },
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 1.0],
                        "is_valid": True,
                        "id": 1,
                        "material": [],
                        "world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "world_translation": [0.0, 0.0, 1.0],
                    }
                ],
                "camera": {
                    "quaternion": [1.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0],
                    "intrinsics": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 64.0,
                        "cy": 64.0,
                        "width": 128.0,
                        "height": 128.0,
                    },
                    "scene_obj_path": "",
                    "background_image_path": "",
                    "background_depth_path": "",
                    "distances": [],
                    "kind": "fixture",
                },
                "scene_dataset": "real",
                "env_param": {},
                "face_up": True,
                "concentrated": False,
                "comments": "pdebug GenPose2 integration fixture",
                "runtime_seed": -1,
                "baseline_dis": 0,
                "emitter_dist_l": 0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @property
    def intrinsics_json(self) -> Path:
        path = self.root / "intrinsics.json"
        if not path.exists():
            payload = {
                str(idx): [
                    [100.0, 0.0, 64.0],
                    [0.0, 100.0, 64.0],
                    [0.0, 0.0, 1.0],
                ]
                for idx in range(3)
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @property
    def cube_obj(self) -> Path:
        path = self.root / "cube.obj"
        if not path.exists():
            path.write_text(
                "\n".join(
                    [
                        "v -0.05 -0.05 -0.05",
                        "v 0.05 -0.05 -0.05",
                        "v 0.05 0.05 -0.05",
                        "v -0.05 0.05 -0.05",
                        "v -0.05 -0.05 0.05",
                        "v 0.05 -0.05 0.05",
                        "v 0.05 0.05 0.05",
                        "v -0.05 0.05 0.05",
                        "f 1 2 3 4",
                        "f 5 8 7 6",
                        "f 1 5 6 2",
                        "f 2 6 7 3",
                        "f 3 7 8 4",
                        "f 5 1 4 8",
                    ]
                ),
                encoding="utf-8",
            )
        return path


ArgsBuilder = Callable[[IntegrationCase, FixtureFactory, Path], Dict[str, object]]


def build_common_args(
    case: IntegrationCase,
    fixtures: FixtureFactory,
    output_root: Path,
    *,
    prepare_repos: bool = True,
) -> Dict[str, object]:
    output = output_root / case.case_id
    repo_root = model_cache_root() / "repos"
    repo = None
    accepts_repo = "repo" in case_parameter_names(case)
    if prepare_repos and case.repo_url:
        repo = ensure_case_repo(case)

    builders: Mapping[str, ArgsBuilder] = {
        "depth_anything_image": _depth_anything_image_args,
        "single_image": _single_image_args,
        "image_dir": _image_dir_args,
        "langsam_predict": _langsam_predict_args,
        "video": _video_args,
        "depth_anything_video": _depth_anything_video_args,
        "repo_image_dir": _repo_image_dir_args,
        "repo_video": _repo_video_args,
        "rgb_depth_mask_mesh": _rgb_depth_mask_mesh_args,
        "genpose2": _genpose2_args,
        "oneposeviagen_3dgen": _oneposeviagen_3dgen_args,
        "oneposeviagen_pose": _oneposeviagen_pose_args,
        "oneposeviagen_scale": _oneposeviagen_scale_args,
        "sam6d": _sam6d_args,
        "sam6d_docker": _sam6d_docker_args,
        "sam2": _sam2_args,
        "hunyuan_shape": _hunyuan_shape_args,
        "hunyuan_rembg": _hunyuan_rembg_args,
        "hunyuan_paint": _hunyuan_paint_args,
        "qwen_single_image": _qwen_single_image_args,
        "foundpose_templates": _foundpose_templates_args,
    }
    if case.args_builder not in builders:
        raise KeyError(f"Unknown args_builder: {case.args_builder}")
    args = builders[case.args_builder](case, fixtures, output)
    if repo is not None:
        if accepts_repo:
            args.setdefault("repo", str(repo))
    elif accepts_repo:
        repo_name = repo_name_from_url(case.repo_url) if case.repo_url else case.case_id
        args.setdefault("repo", str(repo_root / repo_name))
    return args


def case_parameter_names(case: IntegrationCase) -> set[str]:
    """Return parameter names declared by the manifest-backed node."""
    from pdebug.otn import manager as otn_manager
    from pdebug.otn.infer.base import InferenceNode

    node = otn_manager.create(case.node)
    if not isinstance(node, InferenceNode):
        return set()
    return set(node.spec.parameters)


def case_required_args(case: IntegrationCase) -> set[str]:
    """Return parameter names that the manifest declares without defaults."""
    from pdebug.otn import manager as otn_manager
    from pdebug.otn.infer.base import InferenceNode

    node = otn_manager.create(case.node)
    if not isinstance(node, InferenceNode):
        return set()
    return {
        name
        for name, spec in node.spec.parameters.items()
        if isinstance(spec, dict) and "default" not in spec
    }


def _single_image_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {"input_path": str(fixtures.image), "output": str(output)}


def _depth_anything_image_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    return {
        "image_path": str(fixtures.image),
        "output": str(output.with_suffix(".png")),
        "do_vis": True,
    }


def _image_dir_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {"input_path": str(fixtures.image_dir), "output": str(output)}


def _langsam_predict_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.image_dir),
        "texts": "red square",
        "output": str(output),
        "sam_type": "sam2.1_hiera_tiny",
        "device": "cuda" if has_cuda() else "cpu",
        "box_threshold": 0.15,
        "text_threshold": 0.15,
        "topk": 1,
        "cache": False,
    }


def _video_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.video),
        "output": str(output),
        "vis_output": None,
        "max_frames": 3,
        "model_input_size": 3,
    }


def _depth_anything_video_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.video),
        "output": str(output),
        "vis_output": str(output) + "_vis",
        "unittest": False,
        "device": "cuda" if has_cuda() else "cpu",
        "encoder": "vits",
        "save_npz": True,
    }


def _repo_image_dir_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.image_dir),
        "output": str(output),
        "vis_output": str(output) + "_vis",
        "topk": 1,
        "device": "cuda:0" if has_cuda() else "cpu",
    }


def _repo_video_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.video),
        "output": str(output),
        "vis_output": str(output) + "_vis",
        "max_frames": 3,
        "model_input_size": 3,
        "topk": 3,
    }


def _rgb_depth_mask_mesh_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "rgb_path": str(fixtures.image_dir),
        "depth_path": str(fixtures.depth_dir),
        "masks_path": str(fixtures.mask_dir),
        "model_path": str(fixtures.cube_obj),
        "intrinsics_path": str(fixtures.intrinsics_json),
        "output": str(output),
    }


def _genpose2_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "rgb_path": str(fixtures.image_dir),
        "depth_path": str(fixtures.depth_dir),
        "mask_path": str(fixtures.mask_dir),
        "meta_path": str(fixtures.genpose2_meta),
        "output": str(output),
        "topk": 1,
        "skip_if_depth_not_exists": True,
    }


def _oneposeviagen_3dgen_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "rgb_path": str(fixtures.image_dir),
        "masks_path": str(fixtures.mask_dir),
        "output": str(output),
        "ss_sampling_steps": 1,
        "slat_sampling_steps": 1,
        "randomize_seed": False,
        "preview_resolution": 256,
        "preview_num_frames": 8,
        "preview_fps": 8,
        "generate_gaussian": False,
    }


def _oneposeviagen_pose_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "rgb_path": str(fixtures.image_dir),
        "depth_path": str(fixtures.depth_dir),
        "masks_path": str(fixtures.mask_dir),
        "model_path": str(fixtures.cube_obj),
        "intrinsics_path": str(fixtures.intrinsics_json),
        "output": str(output),
        "est_refine_iter": 1,
        "track_refine_iter": 1,
    }


def _oneposeviagen_scale_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "model_path": str(fixtures.cube_obj),
        "depth_path": str(fixtures.depth_dir),
        "rgb_path": str(fixtures.image_dir),
        "masks_path": str(fixtures.mask_dir),
        "intrinsics_path": str(fixtures.intrinsics_json),
        "output": str(output),
        "refine_iterations": 1,
    }


def _sam6d_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "rgb_path": str(fixtures.image_dir),
        "depth_path": str(fixtures.depth_dir),
        "cad_path": str(_sam6d_cube_obj(fixtures)),
        "camera_path": str(_sam6d_camera_json(fixtures)),
        "output": str(output),
        "skip_pose": True,
        "seg_first_frame": True,
    }


def _sam6d_docker_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    repo = case_repo_path(case)
    if repo is not None:
        example = repo / "SAM-6D" / "Data" / "Example"
        cad_path = example / "obj_000005.ply"
        if cad_path.exists():
            cad_path = _obj_from_ply(cad_path)
        return {
            "rgb_path": str(example / "rgb.png"),
            "depth_path": str(example / "depth.png"),
            "cad_path": str(cad_path),
            "camera_path": str(example / "camera.json"),
            "output": str(output),
            "pem_topk": 1,
        }
    return {
        "rgb_path": str(fixtures.image),
        "depth_path": str(fixtures.depth_dir / "000000.png"),
        "cad_path": str(_sam6d_cube_obj(fixtures)),
        "camera_path": str(_sam6d_camera_json(fixtures)),
        "output": str(output),
    }


def _obj_from_ply(ply_path: Path) -> Path:
    """Convert upstream PLY fixtures to OBJ for Blenderproc stability."""
    obj_path = ply_path.with_suffix(".obj")
    if obj_path.exists():
        return obj_path
    try:
        import trimesh
    except Exception as exc:
        raise RuntimeError(
            "SAM-6D Docker integration needs trimesh to convert the "
            f"official PLY CAD model to OBJ: {ply_path}"
        ) from exc

    mesh = trimesh.load(ply_path, force="mesh", process=False)
    mesh.export(obj_path)
    return obj_path


def _sam6d_camera_json(fixtures: FixtureFactory) -> Path:
    """Return camera metadata with SAM-6D's millimeter depth convention."""
    path = fixtures.root / "sam6d_camera.json"
    if not path.exists():
        payload = {
            "cam_K": [
                100.0,
                0.0,
                64.0,
                0.0,
                100.0,
                64.0,
                0.0,
                0.0,
                1.0,
            ],
            "depth_scale": 1.0,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sam6d_cube_obj(fixtures: FixtureFactory) -> Path:
    """Return a SAM-6D CAD fixture in millimeters, as expected upstream."""
    path = fixtures.root / "sam6d_cube_mm.obj"
    if not path.exists():
        path.write_text(
            "\n".join(
                [
                    "v -50 -50 -50",
                    "v 50 -50 -50",
                    "v 50 50 -50",
                    "v -50 50 -50",
                    "v -50 -50 50",
                    "v 50 -50 50",
                    "v 50 50 50",
                    "v -50 50 50",
                    "f 1 2 3 4",
                    "f 5 8 7 6",
                    "f 1 5 6 2",
                    "f 2 6 7 3",
                    "f 3 7 8 4",
                    "f 5 1 4 8",
                ]
            ),
            encoding="utf-8",
        )
    return path


def _sam2_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    points = output.parent / f"{case.case_id}_points.json"
    points.parent.mkdir(parents=True, exist_ok=True)
    if not points.exists():
        points.write_text(
            json.dumps([[[64.0, 64.0, 1], [20.0, 20.0, 0]]]),
            encoding="utf-8",
        )
    return {
        "input_path": str(fixtures.jpg_image_dir),
        "points": str(points),
        "output": str(output),
        "vis_output": str(output) + "_vis",
        "checkpoint": "tiny",
        "device": "cuda" if has_cuda() else "cpu",
        "repo": str(model_cache_root() / "repos" / "sam2"),
        "cache": False,
    }


def _hunyuan_shape_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    repo = model_cache_root() / "repos" / "hunyuan3d-2.1"
    official_demo = repo / "hy3dshape" / "demos" / "demo.png"
    input_path = official_demo if official_demo.exists() else fixtures.image
    return {
        "input_path": str(input_path),
        "output": str(output),
        "num_inference_steps": 5,
        "octree_resolution": 64,
        "num_chunks": 1000,
    }


def _hunyuan_rembg_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.image),
        "output": str(output),
        "output_format": "RGBA",
    }


def _hunyuan_paint_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    repo = model_cache_root() / "repos" / "hunyuan3d-2.1"
    official_demo = repo / "hy3dshape" / "demos" / "demo.png"
    image_path = official_demo if official_demo.exists() else fixtures.image
    return {
        "mesh_path": str(fixtures.cube_obj),
        "image_path": str(image_path),
        "output": str(output),
        "resolution": 512,
        "face_count": 1000,
    }


def _qwen_single_image_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "input_path": str(fixtures.image),
        "text": "List the visible objects in this image.",
        "output": str(output / "qwen.txt"),
        "cache": False,
    }


def _foundpose_templates_args(
    case: IntegrationCase, fixtures: FixtureFactory, output: Path
) -> Dict[str, object]:
    return {
        "model_path": str(fixtures.cube_obj),
        "output_dir": str(output),
        "min_num_viewpoints": 4,
        "num_inplane_rotations": 1,
        "num_viewspheres": 1,
        "topk": 2,
    }


def assert_expected_outputs(case: IntegrationCase, output_root: Path) -> None:
    if not case.expected_paths:
        return
    missing = []
    for rel_path in case.expected_paths:
        path = output_root / case.case_id / rel_path
        if not path.exists():
            missing.append(str(path))
    assert not missing, "missing expected outputs: " + ", ".join(missing)
    if case.case_id == "langsam_predict":
        output = output_root / case.case_id
        for name, pattern in (
            ("mask", "masks/*.png"),
            ("visualization", "vis/*.jpg"),
            ("metadata", "metadata/*.json"),
        ):
            assert list(output.glob(pattern)), (
                f"missing langsam_predict {name} outputs under {output}"
            )
