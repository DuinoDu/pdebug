"""Manifest discovery and registration for inference nodes."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, List, Tuple

from .node import InferenceNode
from .spec import NodeSpec

MANIFEST_SUFFIXES = (
    ".otn.json",
    ".otn.yaml",
    ".otn.yml",
)
MANIFEST_NAMES = (
    "otn_manifest.json",
    "otn_manifest.yaml",
    "otn_manifest.yml",
)


def _load_mapping(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json" or path.name.endswith(".otn.json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            f"PyYAML is required to load manifest file {path}"
        ) from exc
    return yaml.safe_load(text)


def is_manifest_file(path: Path) -> bool:
    if path.name in MANIFEST_NAMES:
        return True
    return any(path.name.endswith(suffix) for suffix in MANIFEST_SUFFIXES)


def discover_manifest_files(root_dirs: Iterable[str]) -> List[Path]:
    manifests: List[Path] = []
    for root_dir in root_dirs:
        root = Path(root_dir)
        if root.is_file() and is_manifest_file(root):
            manifests.append(root)
            continue
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and is_manifest_file(path):
                manifests.append(path)
    return sorted(set(manifests))


def load_node_spec(path: str) -> NodeSpec:
    manifest_path = Path(path)
    payload = _load_mapping(manifest_path)
    if payload is None:
        raise ValueError(f"Empty manifest: {manifest_path}")
    return NodeSpec.from_dict(payload, manifest_path=manifest_path)


def load_node_specs(
    root_dirs: Iterable[str],
) -> Tuple[List[NodeSpec], List[Tuple[str, Exception]]]:
    specs: List[NodeSpec] = []
    errors: List[Tuple[str, Exception]] = []
    for path in discover_manifest_files(root_dirs):
        try:
            specs.append(load_node_spec(str(path)))
        except Exception as exc:
            errors.append((str(path), exc))
    return specs, errors


def register_manifest_nodes(
    registry,
    root_dirs: Iterable[str],
    *,
    strict: bool = False,
) -> List[Tuple[str, Exception]]:
    specs, errors = load_node_specs(root_dirs)
    if strict and errors:
        raise errors[0][1]
    for spec in specs:
        if spec.name in registry:
            continue
        registry.register(name=spec.name, obj=InferenceNode(spec))
    return errors
