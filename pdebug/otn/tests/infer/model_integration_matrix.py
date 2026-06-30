"""Official source matrix for real OTN infer model integration tests.

This module is intentionally data-only. It must not import model code,
instantiate pipelines, download weights, or contact remote services.
"""
from __future__ import annotations


OFFICIAL_REPOS = {
    "co_tracker": {
        "repo_url": ["https://github.com/facebookresearch/co-tracker"],
        "hf_model_ids": ["facebook/cotracker"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "depth_anything": {
        "repo_url": ["https://github.com/LiheYoung/Depth-Anything"],
        "hf_model_ids": [
            "LiheYoung/depth_anything_vits14",
            "LiheYoung/depth_anything_vitb14",
            "LiheYoung/depth_anything_vitl14",
        ],
        "hf_space_ids": ["LiheYoung/Depth-Anything"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "video_depth_anything": {
        "repo_url": [
            "https://github.com/DepthAnything/Video-Depth-Anything",
        ],
        "hf_model_ids": ["depth-anything/Video-Depth-Anything-Small"],
        "hf_space_ids": ["depth-anything/Video-Depth-Anything"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "foundpose": {
        "repo_url": ["https://github.com/facebookresearch/foundpose"],
        "hf_model_ids": [],
        "docker_support": False,
        "source_status": "confirmed_upstream_uncertain_adapter",
    },
    "genpose2": {
        "repo_url": ["https://github.com/Omni6DPose/GenPose2"],
        "hf_model_ids": [],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "grounding_dino": {
        "repo_url": ["https://github.com/IDEA-Research/GroundingDINO"],
        "hf_model_ids": ["IDEA-Research/grounding-dino-base"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "hunyuan3d": {
        "repo_url": ["https://github.com/tencent-hunyuan/hunyuan3d-2.1"],
        "hf_model_ids": ["tencent/Hunyuan3D-2.1"],
        "hf_space_ids": ["tencent/Hunyuan3D-2.1"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "dcnv4": {
        "repo_url": ["https://github.com/OpenGVLab/DCNv4"],
        "hf_model_ids": ["OpenGVLab/DCNv4"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "langsam": {
        "repo_url": [
            "https://github.com/luca-medeiros/lang-segment-anything",
        ],
        "hf_model_ids": [],
        "docker_support": True,
        "source_status": "likely_lang_sam_package_source",
    },
    "ml_depth_pro": {
        "repo_url": ["https://github.com/apple/ml-depth-pro"],
        "hf_model_ids": ["apple/DepthPro-hf"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "moondream": {
        "repo_url": ["https://github.com/m87-labs/moondream"],
        "hf_model_ids": ["vikhyatk/moondream2"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "oneposeviagen": {
        "repo_url": ["https://github.com/GZWSAMA/OnePoseviaGen"],
        "hf_model_ids": ["ZhengGeng/OnePoseViaGen"],
        "docker_support": False,
        "source_status": "confirmed_upstream_uncertain_subcheckpoints",
    },
    "foundationpose": {
        "repo_url": ["https://github.com/NVlabs/FoundationPose"],
        "hf_model_ids": [],
        "docker_support": True,
        "source_status": "confirmed",
    },
    "orient_anything": {
        "repo_url": ["https://github.com/SpatialVision/Orient-Anything"],
        "hf_model_ids": ["Viglong/Orient-Anything"],
        "hf_space_ids": ["Viglong/Orient-Anything"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "dinov2": {
        "repo_url": ["https://github.com/facebookresearch/dinov2"],
        "hf_model_ids": ["facebook/dinov2-large"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "qwen2_5_vl": {
        "repo_url": ["https://github.com/QwenLM/Qwen2.5-VL"],
        "hf_model_ids": ["Qwen/Qwen2.5-VL-7B-Instruct"],
        "docker_support": False,
        "source_status": "confirmed_hf_repo_url_needs_pin_check",
    },
    "sam2": {
        "repo_url": ["https://github.com/facebookresearch/sam2"],
        "hf_model_ids": [
            "facebook/sam2.1-hiera-tiny",
            "facebook/sam2.1-hiera-small",
            "facebook/sam2.1-hiera-base-plus",
            "facebook/sam2.1-hiera-large",
        ],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "sam6d": {
        "repo_url": ["https://github.com/JiehongLin/SAM-6D"],
        "hf_model_ids": [],
        "docker_support": "manifest_only",
        "source_status": "confirmed_upstream_docker_image",
    },
    "segment_anything": {
        "repo_url": ["https://github.com/facebookresearch/segment-anything"],
        "hf_model_ids": ["facebook/sam-vit-base"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "spatracker_v2": {
        "repo_url": ["https://github.com/henry123-boy/SpaTrackerV2"],
        "hf_model_ids": [
            "Yuxihenry/SpatialTrackerV2_Front",
            "Yuxihenry/SpatialTrackerV2-Offline",
        ],
        "hf_space_ids": ["Yuxihenry/SpatialTrackerV2"],
        "docker_support": False,
        "source_status": "confirmed",
    },
    "vggt": {
        "repo_url": ["https://github.com/facebookresearch/vggt"],
        "hf_model_ids": ["facebook/VGGT-1B"],
        "hf_space_ids": ["facebook/vggt"],
        "docker_support": False,
        "source_status": "confirmed",
    },
}


def _repo_urls(*source_keys: str) -> list[str]:
    urls: list[str] = []
    for source_key in source_keys:
        urls.extend(OFFICIAL_REPOS[source_key]["repo_url"])
    return urls


def _hf_model_ids(*source_keys: str) -> list[str]:
    model_ids: list[str] = []
    for source_key in source_keys:
        model_ids.extend(OFFICIAL_REPOS[source_key]["hf_model_ids"])
    return model_ids


MODEL_INTEGRATION_MATRIX = [
    {
        "file": "pdebug/otn/infer/co-tracker.py",
        "node_names": ["cotracker", "video_kps_to_all"],
        "official_sources": ["co_tracker"],
        "repo_url": _repo_urls("co_tracker"),
        "hf_model_ids": _hf_model_ids("co_tracker"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "CoTracker3 loads torch hub or local scaled_online/offline "
            "checkpoints and runs dense video tracking on CUDA."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:file",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "cotracker", "imageio", "PIL"],
            "checkpoint_files": [
                "~/.cache/install-x/co-tracker/checkpoints/scaled_online.pth",
                "~/.cache/install-x/co-tracker/checkpoints/scaled_offline.pth",
            ],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/depth_anything.py",
        "node_names": ["depth_anything", "depth-anything-video"],
        "official_sources": ["depth_anything", "video_depth_anything"],
        "repo_url": _repo_urls("depth_anything", "video_depth_anything"),
        "hf_model_ids": _hf_model_ids(
            "depth_anything", "video_depth_anything"
        ),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Image depth uses a Gradio endpoint; video depth loads "
            "VideoDepthAnything checkpoints and processes full video tensors."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": [
                "gradio_client",
                "torch",
                "video_depth_anything",
            ],
            "checkpoint_files": [
                "checkpoints/video_depth_anything_<encoder>.pth",
                "checkpoints/metric_video_depth_anything_<encoder>.pth",
            ],
            "remote_services": ["liheyoung-depth-anything.hf.space"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/foundpose.py",
        "node_names": [
            "cad-to-templates",
            "foundpose_subprocess",
            "foundpose-to-linemod",
            "templates-to-linemod",
        ],
        "official_sources": ["foundpose"],
        "repo_url": _repo_urls("foundpose"),
        "hf_model_ids": _hf_model_ids("foundpose"),
        "preferred_runtime": "subprocess",
        "heavy_reason": (
            "FoundPose depends on external renderer and pose code; the real "
            "inference manifest already isolates it behind foundpose_infer."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module + subprocess",
            "manifest_extras": ["otn"],
            "subprocess_module": "foundpose_infer",
            "runtime_imports": ["torch", "pyrender", "trimesh", "cv2"],
            "docker": {"supported": False},
            "uncertain": [
                "foundpose_infer adapter is not in this repository",
            ],
        },
    },
    {
        "file": "pdebug/otn/infer/genpose2.py",
        "node_names": ["genpose2"],
        "official_sources": ["genpose2"],
        "repo_url": _repo_urls("genpose2"),
        "hf_model_ids": _hf_model_ids("genpose2"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "GenPose2 loads ScoreNet, EnergyNet, and ScaleNet checkpoints "
            "from an external repo and runs RGB-D pose inference."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "cutoop", "runners.infer"],
            "checkpoint_files": [
                "results/ckpts/ScoreNet/scorenet.pth",
                "results/ckpts/EnergyNet/energynet.pth",
                "results/ckpts/ScaleNet/scalenet.pth",
            ],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/groundingdino_node.py",
        "node_names": ["groundingdino"],
        "official_sources": ["grounding_dino"],
        "repo_url": _repo_urls("grounding_dino"),
        "hf_model_ids": _hf_model_ids("grounding_dino"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "GroundingDINO loads a transformer zero-shot detection model and "
            "processor for image or Lance dataset inference."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "transformers", "PIL"],
            "from_pretrained": ["IDEA-Research/grounding-dino-base"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/hunyuan3d_paint.py",
        "node_names": ["hunyuan3d_paint"],
        "official_sources": ["hunyuan3d"],
        "repo_url": _repo_urls("hunyuan3d"),
        "hf_model_ids": _hf_model_ids("hunyuan3d"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Hunyuan3D Paint loads multiview texture generation weights, "
            "RealESRGAN, and mesh processing dependencies."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["hy3dpaint", "trimesh", "torch"],
            "checkpoint_files": [
                "hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
            ],
            "from_pretrained": ["tencent/Hunyuan3D-2.1"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/hunyuan3d_rembg.py",
        "node_names": ["hunyuan3d_rembg"],
        "official_sources": ["hunyuan3d"],
        "repo_url": _repo_urls("hunyuan3d"),
        "hf_model_ids": _hf_model_ids("hunyuan3d"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "BackgroundRemover is imported from the Hunyuan3D repo and may "
            "load its own segmentation/removal weights."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["hy3dshape.rembg", "PIL", "torch"],
            "docker": {"supported": False},
            "uncertain": [
                "No explicit checkpoint or HF id is present in this node",
            ],
        },
    },
    {
        "file": "pdebug/otn/infer/hunyuan3d_shape.py",
        "node_names": ["hunyuan3d_shape"],
        "official_sources": ["hunyuan3d"],
        "repo_url": _repo_urls("hunyuan3d"),
        "hf_model_ids": _hf_model_ids("hunyuan3d"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Hunyuan3D shape generation loads a diffusion flow-matching "
            "pipeline and produces meshes from image inputs."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["hy3dshape", "diffusers", "torch"],
            "from_pretrained": [
                "tencent/Hunyuan3D-2.1:hunyuan3d-dit-v2-1",
            ],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/internimage_semseg.py",
        "node_names": ["internimage_semseg", "remove_dynamic"],
        "official_sources": ["dcnv4"],
        "repo_url": _repo_urls("dcnv4"),
        "hf_model_ids": _hf_model_ids("dcnv4"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "InternImage/DCNv4 uses mmcv/mmseg custom ops, ADE20K configs, "
            "and large semantic segmentation checkpoints."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": [
                "mmcv",
                "mmseg",
                "mmcv_custom",
                "mmseg_custom",
                "huggingface_hub",
            ],
            "hf_hub_download": ["OpenGVLab/DCNv4:<model_name>.pth"],
            "install_x_repo": "DCNv4",
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/langsam.py",
        "node_names": [
            "langsam_for_aigc",
            "langsam_predict",
            "langsam_sam",
        ],
        "official_sources": ["langsam", "grounding_dino", "sam2"],
        "repo_url": _repo_urls("langsam", "grounding_dino", "sam2"),
        "hf_model_ids": _hf_model_ids("langsam", "grounding_dino", "sam2"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "LangSAM composes GroundingDINO and SAM/SAM2 models, so real "
            "runs load multiple vision models and checkpoint variants."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["lang_sam", "torch", "PIL", "cv2"],
            "sam_types": [
                "sam2.1_hiera_tiny",
                "sam2.1_hiera_small",
                "sam2.1_hiera_base_plus",
                "sam2.1_hiera_large",
            ],
            "docker": {
                "supported": True,
                "source": "official repo Dockerfile",
                "preferred": False,
            },
            "uncertain": [
                "Several LangSAM forks exist; this maps to luca-medeiros",
            ],
        },
    },
    {
        "file": "pdebug/otn/infer/ml_depth_pro_node.py",
        "node_names": ["ml_depth_pro"],
        "official_sources": ["ml_depth_pro"],
        "repo_url": _repo_urls("ml_depth_pro"),
        "hf_model_ids": _hf_model_ids("ml_depth_pro"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "DepthPro loads Apple DepthPro through transformers and emits "
            "dense metric depth maps."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "transformers", "PIL"],
            "from_pretrained": ["apple/DepthPro-hf"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/moondream_node.py",
        "node_names": ["moondream"],
        "official_sources": ["moondream"],
        "repo_url": _repo_urls("moondream"),
        "hf_model_ids": _hf_model_ids("moondream"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Moondream uses trust_remote_code and loads a multimodal causal "
            "LM revision for image captioning."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch>=2.5.0", "transformers", "PIL"],
            "from_pretrained": ["vikhyatk/moondream2@2025-06-21"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/oneposeviagen_3dgen.py",
        "node_names": ["oneposeviagen_3dgen"],
        "official_sources": ["oneposeviagen"],
        "repo_url": _repo_urls("oneposeviagen"),
        "hf_model_ids": _hf_model_ids("oneposeviagen"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "OnePoseviaGen 3D generation loads Amodal3R or Hi3DGen/TRELLIS "
            "pipelines and writes mesh plus Gaussian outputs."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["amodal3r", "trellis", "torch", "trimesh"],
            "from_pretrained": [
                "checkpoints/OnePoseViaGen/Amodal3R",
                "checkpoints/OnePoseViaGen/Hi3DGen_Color",
            ],
            "docker": {"supported": False},
            "uncertain": [
                "The node uses repo-relative checkpoint directories",
            ],
        },
    },
    {
        "file": "pdebug/otn/infer/oneposeviagen_pose.py",
        "node_names": ["oneposeviagen_pose"],
        "official_sources": ["oneposeviagen", "foundationpose"],
        "repo_url": _repo_urls("oneposeviagen", "foundationpose"),
        "hf_model_ids": _hf_model_ids("oneposeviagen", "foundationpose"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Pose estimation imports FoundationPose, nvdiffrast, CUDA "
            "rasterization, and refinement/score predictors."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": [
                "foundationpose",
                "nvdiffrast",
                "trimesh",
                "torch",
            ],
            "docker": {
                "supported": True,
                "source": "FoundationPose official docker directory",
                "preferred": False,
            },
        },
    },
    {
        "file": "pdebug/otn/infer/oneposeviagen_scale.py",
        "node_names": ["oneposeviagen_scale"],
        "official_sources": ["oneposeviagen", "foundationpose"],
        "repo_url": _repo_urls("oneposeviagen", "foundationpose"),
        "hf_model_ids": _hf_model_ids("oneposeviagen", "foundationpose"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Scale recovery imports OnePoseviaGen/FoundationPose scale "
            "utilities and operates on RGB-D, masks, and mesh geometry."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["fpose.recover_scale", "trimesh", "cv2"],
            "docker": {
                "supported": True,
                "source": "FoundationPose official docker directory",
                "preferred": False,
            },
        },
    },
    {
        "file": "pdebug/otn/infer/orient_anything.py",
        "node_names": ["orient_anything"],
        "official_sources": ["orient_anything", "dinov2"],
        "repo_url": _repo_urls("orient_anything", "dinov2"),
        "hf_model_ids": _hf_model_ids("orient_anything", "dinov2"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Orient Anything downloads a DINOv2-MLP checkpoint and runs "
            "orientation inference with optional background preprocessing."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "transformers", "huggingface_hub"],
            "hf_hub_download": [
                "Viglong/Orient-Anything:croplargeEX2/dino_weight.pt",
            ],
            "from_pretrained": ["facebook/dinov2-large"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/qwen2_5_vl.py",
        "node_names": ["qwen2_5_vl"],
        "official_sources": ["qwen2_5_vl"],
        "repo_url": _repo_urls("qwen2_5_vl"),
        "hf_model_ids": _hf_model_ids("qwen2_5_vl"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Qwen2.5-VL-7B is a large multimodal model requiring "
            "transformers, qwen-vl-utils, and GPU memory."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": [
                "torch",
                "transformers",
                "qwen_vl_utils",
            ],
            "optional_runtime_imports": ["flash_attn"],
            "from_pretrained": ["Qwen/Qwen2.5-VL-7B-Instruct"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/sam2_node.py",
        "node_names": ["sam2", "sam_with_prompt"],
        "official_sources": ["sam2"],
        "repo_url": _repo_urls("sam2"),
        "hf_model_ids": _hf_model_ids("sam2"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "SAM2 video and prompt segmentation load SAM2.1 Hiera "
            "checkpoints and keep predictor state for image/video masks."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["sam2", "torch", "gradio", "PIL"],
            "checkpoint_files": [
                "sam2.1_hiera_tiny.pt",
                "sam2.1_hiera_small.pt",
                "sam2.1_hiera_base_plus.pt",
                "sam2.1_hiera_large.pt",
            ],
            "from_pretrained": [
                "facebook/sam2.1-hiera-<tiny|small|base-plus|large>",
            ],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/sam6d.py",
        "node_names": [
            "sam6d",
            "sam6d_docker",
            "sam6d-from-sam2",
            "sam6d-to-linemod",
            "sam6d-to-sam2",
        ],
        "official_sources": ["sam6d", "segment_anything"],
        "repo_url": _repo_urls("sam6d", "segment_anything"),
        "hf_model_ids": _hf_model_ids("sam6d", "segment_anything"),
        "preferred_runtime": "docker",
        "heavy_reason": (
            "SAM-6D invokes BlenderProc rendering plus external instance "
            "segmentation and pose estimation scripts; the manifest provides "
            "a docker backend for real inference isolation."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module + docker",
            "manifest_extras": ["otn"],
            "runtime_imports": ["blenderproc", "torch", "trimesh", "yaml"],
            "external_scripts": [
                "SAM-6D/Render/render_custom_templates.py",
                "SAM-6D/Instance_Segmentation_Model/run_inference_custom.py",
                "SAM-6D/Pose_Estimation_Model/run_inference_custom.py",
            ],
            "docker": {
                "supported": True,
                "source": "upstream README Docker Hub link",
                "image": "lihualiu/sam-6d:1.0",
                "command": "SAM-6D/demo.sh",
                "preferred": True,
            },
        },
    },
    {
        "file": "pdebug/otn/infer/segment_anything_node.py",
        "node_names": ["segment_anything"],
        "official_sources": ["segment_anything"],
        "repo_url": _repo_urls("segment_anything"),
        "hf_model_ids": _hf_model_ids("segment_anything"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "Segment Anything loads SAM ViT checkpoints and produces "
            "prompted masks with dense post-processing."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "transformers", "PIL"],
            "from_pretrained": ["facebook/sam-vit-base"],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/spatracker.py",
        "node_names": ["spatracker"],
        "official_sources": ["spatracker_v2"],
        "repo_url": _repo_urls("spatracker_v2"),
        "hf_model_ids": _hf_model_ids("spatracker_v2"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "SpatialTrackerV2 loads VGGT4Track and offline tracker HF "
            "weights, then estimates depth, cameras, and 3D tracks."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": ["torch", "torchvision", "SpaTrackV2"],
            "from_pretrained": [
                "Yuxihenry/SpatialTrackerV2_Front",
                "Yuxihenry/SpatialTrackerV2-Offline",
            ],
            "docker": {"supported": False},
        },
    },
    {
        "file": "pdebug/otn/infer/vggt_node.py",
        "node_names": ["vggt", "vggt-viser"],
        "official_sources": ["vggt"],
        "repo_url": _repo_urls("vggt"),
        "hf_model_ids": _hf_model_ids("vggt"),
        "preferred_runtime": "local",
        "heavy_reason": (
            "VGGT loads a 1B-parameter checkpoint and estimates camera "
            "parameters, depth maps, point maps, and optional tracks/BA."
        ),
        "dependency_metadata": {
            "manifest_backend": "legacy_python:module",
            "manifest_extras": ["otn"],
            "runtime_imports": [
                "torch",
                "vggt",
                "pycolmap",
                "viser",
            ],
            "checkpoint_url": [
                (
                    "https://huggingface.co/facebook/VGGT-1B/resolve/main/"
                    "model.pt"
                ),
            ],
            "from_pretrained": ["facebook/VGGT-1B"],
            "docker": {"supported": False},
        },
    },
]


UNCERTAIN_OFFICIAL_SOURCES = [
    {
        "file": "pdebug/otn/infer/foundpose.py",
        "reason": "foundpose_infer subprocess adapter is external.",
    },
    {
        "file": "pdebug/otn/infer/hunyuan3d_rembg.py",
        "reason": "Background remover has no explicit checkpoint id in code.",
    },
    {
        "file": "pdebug/otn/infer/langsam.py",
        "reason": "Multiple LangSAM forks exist; luca-medeiros is selected.",
    },
    {
        "file": "pdebug/otn/infer/oneposeviagen_3dgen.py",
        "reason": "Amodal3R/Hi3DGen checkpoint subdirectories are repo-local.",
    },
    {
        "file": "pdebug/otn/infer/sam6d.py",
        "reason": (
            "Docker manifest uses the upstream Docker Hub image and official "
            "demo.sh, but the image still needs to be pulled locally."
        ),
    },
]
