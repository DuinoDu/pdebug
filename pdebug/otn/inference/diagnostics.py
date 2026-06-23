"""Diagnostics helpers for manifest-backed inference nodes."""
from __future__ import annotations
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .backends import create_backend
from .manifest import discover_manifest_files, load_node_spec
from .spec import NodeSpec
from .types import HealthStatus

SUPPORTED_BACKENDS = {
    "docker",
    "http",
    "legacy_python",
    "local_python",
    "subprocess",
}
SUPPORTED_PARAMETER_TYPES = {"bool", "float", "int", "str"}


@dataclass
class ManifestIssue:
    """One schema or runtime issue found while checking a manifest."""

    level: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "message": self.message}


def manifest_schema() -> Dict[str, Any]:
    """Return the lightweight OTN inference manifest schema."""
    return {
        "type": "object",
        "required": ["name", "backend"],
        "properties": {
            "name": {"type": "string"},
            "task": {"type": "string"},
            "description": {"type": "string"},
            "backend": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_BACKENDS),
                    }
                },
                "additionalProperties": True,
            },
            "schemas": {"type": "object"},
            "parameters": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": sorted(SUPPORTED_PARAMETER_TYPES),
                        },
                        "default": {},
                        "description": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "dependencies": {"type": "object"},
            "capabilities": {"type": "object"},
        },
        "additionalProperties": True,
    }


def validate_node_spec(spec: NodeSpec) -> List[ManifestIssue]:
    """Validate a parsed manifest spec without importing model code."""
    issues: List[ManifestIssue] = []
    if not spec.name.strip():
        issues.append(ManifestIssue("error", "name must not be empty"))
    if spec.backend.type not in SUPPORTED_BACKENDS:
        issues.append(
            ManifestIssue(
                "error",
                f"unsupported backend type: {spec.backend.type}",
            )
        )
    if not isinstance(spec.schemas, Mapping):
        issues.append(ManifestIssue("error", "schemas must be a mapping"))
    if not isinstance(spec.parameters, Mapping):
        issues.append(ManifestIssue("error", "parameters must be a mapping"))
        return issues

    for name, metadata in spec.parameters.items():
        if metadata is None:
            continue
        if not isinstance(metadata, Mapping):
            issues.append(
                ManifestIssue("error", f"parameters.{name} must be a mapping")
            )
            continue
        type_name = metadata.get("type")
        if (
            type_name is not None
            and type_name not in SUPPORTED_PARAMETER_TYPES
        ):
            issues.append(
                ManifestIssue(
                    "error",
                    f"parameters.{name}.type is unsupported: {type_name}",
                )
            )
    return issues


def node_to_schema(node: Any) -> Dict[str, Any]:
    """Describe a registered node using manifest or Python introspection."""
    spec = getattr(node, "spec", None)
    if isinstance(spec, NodeSpec):
        return spec.to_dict()

    signature = inspect.signature(node)
    parameters: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        metadata: Dict[str, Any] = {}
        if parameter.annotation is not inspect.Parameter.empty:
            metadata["type"] = getattr(
                parameter.annotation,
                "__name__",
                str(parameter.annotation),
            )
        if parameter.default is not inspect.Parameter.empty:
            metadata["default"] = parameter.default
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            metadata["keyword_only"] = True
        parameters[name] = metadata

    return {
        "name": getattr(node, "__name__", type(node).__name__),
        "description": inspect.getdoc(node) or "",
        "source_file": inspect.getsourcefile(node),
        "parameters": parameters,
    }


def node_healthcheck(node: Any) -> HealthStatus:
    """Run a registered node health check when available."""
    healthcheck = getattr(node, "healthcheck", None)
    if callable(healthcheck):
        return healthcheck()
    return HealthStatus(
        ok=True,
        message="registered python node",
        details={"source_file": inspect.getsourcefile(node)},
    )


def doctor_manifest_path(
    path: str,
    *,
    check_backend: bool = True,
) -> Dict[str, Any]:
    """Load and validate all manifests under a file or directory path."""
    manifest_files = discover_manifest_files([path])
    if not manifest_files and Path(path).is_file():
        manifest_files = [Path(path)]

    checked: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for manifest_file in manifest_files:
        item: Dict[str, Any] = {"path": str(manifest_file)}
        try:
            spec = load_node_spec(str(manifest_file))
            issues = validate_node_spec(spec)
            item["name"] = spec.name
            item["backend"] = spec.backend.type
            item["issues"] = [issue.to_dict() for issue in issues]
            if check_backend:
                backend = create_backend(spec.backend)
                health = backend.healthcheck()
                item["health"] = health.to_dict()
                backend.close()
                if not health.ok:
                    item["issues"].append(
                        ManifestIssue(
                            "error",
                            f"backend healthcheck failed: {health.message}",
                        ).to_dict()
                    )
        except Exception as exc:
            errors.append(
                {
                    "path": str(manifest_file),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        checked.append(item)

    if not manifest_files:
        errors.append({"path": path, "error": "no manifest files found"})

    has_issue = any(item.get("issues") for item in checked)
    return {
        "ok": not errors and not has_issue,
        "checked": checked,
        "errors": errors,
    }


def doctor_manifest_paths(
    paths: Iterable[str],
    *,
    check_backend: bool = True,
) -> Dict[str, Any]:
    """Run manifest doctor over multiple roots."""
    reports = [
        doctor_manifest_path(path, check_backend=check_backend)
        for path in paths
    ]
    return {
        "ok": all(report["ok"] for report in reports),
        "reports": reports,
    }
