"""Callable manifest-backed OTN node wrapper."""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from .backends import InferenceBackend, LegacyPythonBackend, create_backend
from .spec import NodeSpec
from .types import HealthStatus, InferenceRequest, InferenceResult


class InferenceNode:
    """OTN-compatible callable wrapper around an inference backend."""

    def __init__(
        self,
        spec: NodeSpec,
        backend: Optional[InferenceBackend] = None,
    ) -> None:
        self.spec = spec
        self.__name__ = spec.name
        self.__doc__ = spec.description or "Manifest-backed inference node."
        self._backend = backend
        self.__annotations__ = self._build_annotations()
        self.__signature__ = self._build_signature()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def description(self) -> str:
        return self.spec.description

    @property
    def manifest_path(self):
        return self.spec.manifest_path

    @property
    def backend(self) -> InferenceBackend:
        if self._backend is None:
            self._backend = create_backend(self.spec.backend)
        return self._backend

    def healthcheck(self) -> HealthStatus:
        return self.backend.healthcheck()

    def predict(self, request: InferenceRequest) -> InferenceResult:
        return self.backend.predict(request)

    def predict_batch(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        return self.backend.predict_batch(requests)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def _build_signature(self):
        import inspect

        parameters = []
        for name, meta in self.spec.parameters.items():
            meta = dict(meta or {})
            default = meta.get("default", inspect.Parameter.empty)
            annotation = self.__annotations__.get(
                name, inspect.Parameter.empty
            )
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return inspect.Signature(parameters)

    def _build_annotations(self) -> Dict[str, Any]:
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        }
        annotations: Dict[str, Any] = {}
        for name, meta in self.spec.parameters.items():
            type_name = dict(meta or {}).get("type")
            if type_name in type_map:
                annotations[name] = type_map[type_name]
        return annotations

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self.backend, LegacyPythonBackend):
            return self.backend.call(*args, **kwargs)
        if args and kwargs:
            raise TypeError("Use either positional request or keyword inputs")
        if len(args) > 1:
            raise TypeError(
                "InferenceNode accepts at most one request payload"
            )
        if args:
            request = InferenceRequest.from_payload(args[0])
        else:
            request = InferenceRequest.from_payload(kwargs)
        return self.predict(request).to_dict()
