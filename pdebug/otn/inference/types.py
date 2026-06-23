"""Common request/result types for inference adapters."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class HealthStatus:
    """Structured backend health status."""

    ok: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass
class InferenceRequest:
    """Normalized request passed from OTN nodes to inference backends."""

    inputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    assets: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "InferenceRequest":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, Mapping):
            known_keys = {"inputs", "parameters", "assets", "metadata"}
            if known_keys.intersection(payload.keys()):
                return cls(
                    inputs=dict(payload.get("inputs", {})),
                    parameters=dict(payload.get("parameters", {})),
                    assets=dict(payload.get("assets", {})),
                    metadata=dict(payload.get("metadata", {})),
                )
            return cls(inputs=dict(payload))
        return cls(inputs={"value": payload})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": dict(self.inputs),
            "parameters": dict(self.parameters),
            "assets": dict(self.assets),
            "metadata": dict(self.metadata),
        }


@dataclass
class InferenceResult:
    """Normalized result returned by inference backends."""

    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Any] = None

    @classmethod
    def from_payload(cls, payload: Any) -> "InferenceResult":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, Mapping):
            known_keys = {"data", "artifacts", "metadata", "raw"}
            if known_keys.intersection(payload.keys()):
                return cls(
                    data=dict(payload.get("data", {})),
                    artifacts=dict(payload.get("artifacts", {})),
                    metadata=dict(payload.get("metadata", {})),
                    raw=payload.get("raw"),
                )
            return cls(data=dict(payload), raw=payload)
        return cls(data={"value": payload}, raw=payload)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "data": dict(self.data),
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
        }
        if self.raw is not None:
            payload["raw"] = self.raw
        return payload
