"""Manifest-backed inference node specifications."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class BackendSpec:
    """Backend adapter declaration parsed from a manifest."""

    type: str
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BackendSpec":
        if "type" not in payload:
            raise ValueError("backend.type is required")
        config = dict(payload)
        backend_type = str(config.pop("type"))
        return cls(type=backend_type, config=config)

    def to_dict(self) -> Dict[str, Any]:
        config = {
            key: value
            for key, value in self.config.items()
            if not key.startswith("_")
        }
        return {"type": self.type, **config}


@dataclass
class NodeSpec:
    """Lightweight model node metadata, independent from heavy imports."""

    name: str
    backend: BackendSpec
    task: str = "generic"
    description: str = ""
    schemas: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    manifest_path: Optional[Path] = None

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        manifest_path: Optional[Path] = None,
    ) -> "NodeSpec":
        if "name" not in payload:
            raise ValueError("name is required")
        if "backend" not in payload:
            raise ValueError("backend is required")
        backend = BackendSpec.from_dict(payload["backend"])
        if manifest_path is not None:
            backend.config.setdefault("_manifest_path", str(manifest_path))
        return cls(
            name=str(payload["name"]),
            task=str(payload.get("task", "generic")),
            description=str(payload.get("description", "")),
            backend=backend,
            schemas=dict(payload.get("schemas", {})),
            parameters=dict(payload.get("parameters", {})),
            dependencies=dict(payload.get("dependencies", {})),
            capabilities=dict(payload.get("capabilities", {})),
            manifest_path=manifest_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "task": self.task,
            "description": self.description,
            "backend": self.backend.to_dict(),
            "schemas": dict(self.schemas),
            "parameters": dict(self.parameters),
            "dependencies": dict(self.dependencies),
            "capabilities": dict(self.capabilities),
        }
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
        return payload
