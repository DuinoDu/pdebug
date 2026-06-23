"""Inference backend adapters."""
from __future__ import annotations
import importlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Mapping

from .spec import BackendSpec
from .types import HealthStatus, InferenceRequest, InferenceResult


class InferenceBackend(ABC):
    """Base adapter for local, containerized, or remote inference."""

    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(ok=True, message="ok")

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResult:
        """Run one inference request."""

    def predict_batch(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResult]:
        return [self.predict(request) for request in requests]

    def close(self) -> None:
        """Release resources held by the backend."""


class LocalPythonBackend(InferenceBackend):
    """Backend that calls a Python function or object lazily."""

    def __init__(self, spec: BackendSpec) -> None:
        super().__init__(spec)
        self._target = None

    def _load_target(self):
        if self._target is not None:
            return self._target

        module_name = self.spec.config.get("module")
        if not module_name:
            raise ValueError("local_python backend requires module")
        module = importlib.import_module(str(module_name))

        if "class" in self.spec.config:
            cls = getattr(module, str(self.spec.config["class"]))
            kwargs = dict(self.spec.config.get("init_kwargs", {}))
            self._target = cls(**kwargs)
        elif "function" in self.spec.config:
            self._target = getattr(module, str(self.spec.config["function"]))
        else:
            raise ValueError("local_python backend requires class or function")
        return self._target

    def predict(self, request: InferenceRequest) -> InferenceResult:
        target = self._load_target()
        if hasattr(target, "predict"):
            payload = target.predict(request)
        else:
            payload = target(request)
        return InferenceResult.from_payload(payload)

    def close(self) -> None:
        target = self._target
        if hasattr(target, "close"):
            target.close()
        self._target = None


class LegacyPythonBackend(LocalPythonBackend):
    """Backend that delegates to an existing OTN function unchanged."""

    def call(self, *args: Any, **kwargs: Any) -> Any:
        target = self._load_target()
        return target(*args, **kwargs)

    def predict(self, request: InferenceRequest) -> InferenceResult:
        kwargs = dict(request.inputs)
        kwargs.update(request.parameters)
        return InferenceResult.from_payload(self.call(**kwargs))


class SubprocessBackend(InferenceBackend):
    """Backend that exchanges JSON request/result files with a process."""

    def _command(self, input_path: Path, output_path: Path) -> List[str]:
        command = [str(x) for x in self.spec.config.get("command", [])]
        if not command:
            raise ValueError("subprocess backend requires command")
        rendered = [
            item.format(input=input_path, output=output_path)
            for item in command
        ]
        if rendered == command and "{input}" not in " ".join(command):
            rendered.extend(
                ["--input", str(input_path), "--output", str(output_path)]
            )
        return rendered

    def healthcheck(self) -> HealthStatus:
        command = self.spec.config.get("command", [])
        if not command:
            return HealthStatus(False, "subprocess command is empty")
        executable = str(command[0])
        if "/" not in executable and shutil.which(executable) is None:
            return HealthStatus(False, f"executable not found: {executable}")
        return HealthStatus(True, "ok")

    def predict(self, request: InferenceRequest) -> InferenceResult:
        timeout = self.spec.config.get("timeout_sec")
        with tempfile.TemporaryDirectory(prefix="otn-infer-") as tmpdir:
            request_path = Path(tmpdir) / "request.json"
            result_path = Path(tmpdir) / "result.json"
            request_path.write_text(
                json.dumps(request.to_dict(), ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            subprocess.run(
                self._command(request_path, result_path),
                check=True,
                timeout=timeout,
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        return InferenceResult.from_payload(payload)


class HttpBackend(InferenceBackend):
    """Backend that POSTs JSON requests to an HTTP inference service."""

    def healthcheck(self) -> HealthStatus:
        endpoint = self.spec.config.get("endpoint")
        if not endpoint:
            return HealthStatus(False, "http endpoint is required")
        return HealthStatus(True, "configured")

    def predict(self, request: InferenceRequest) -> InferenceResult:
        endpoint = self.spec.config.get("endpoint")
        if not endpoint:
            raise ValueError("http backend requires endpoint")
        data = json.dumps(request.to_dict()).encode("utf-8")
        http_request = urllib.request.Request(
            str(endpoint),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = self.spec.config.get("timeout_sec")
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return InferenceResult.from_payload(payload)


class DockerBackend(SubprocessBackend):
    """Backend that runs inference in a Docker container."""

    def build_command(self, input_path: Path, output_path: Path) -> List[str]:
        image = self.spec.config.get("image")
        if not image:
            raise ValueError("docker backend requires image")

        command = ["docker", "run", "--rm"]
        if self.spec.config.get("gpu"):
            command.extend(["--gpus", "all"])

        for item in self.spec.config.get("env", []):
            command.extend(["-e", str(item)])
        env_map = self.spec.config.get("env_map", {})
        if isinstance(env_map, Mapping):
            for key, value in env_map.items():
                command.extend(["-e", f"{key}={value}"])

        workdir = Path(input_path).parent
        command.extend(["-v", f"{workdir}:/otn_io"])
        for mount in self.spec.config.get("mounts", []):
            if isinstance(mount, Mapping):
                command.extend(["-v", f"{mount['host']}:{mount['container']}"])
            else:
                command.extend(["-v", str(mount)])

        command.append(str(image))
        inner_command = [str(x) for x in self.spec.config.get("command", [])]
        if inner_command:
            command.extend(inner_command)
        command.extend(["--input", "/otn_io/request.json"])
        command.extend(["--output", "/otn_io/result.json"])
        return command

    def _command(self, input_path: Path, output_path: Path) -> List[str]:
        return self.build_command(input_path, output_path)

    def healthcheck(self) -> HealthStatus:
        if shutil.which("docker") is None:
            return HealthStatus(False, "docker executable not found")
        if not self.spec.config.get("image"):
            return HealthStatus(False, "docker image is required")
        return HealthStatus(True, "configured")


def create_backend(spec: BackendSpec) -> InferenceBackend:
    """Instantiate a backend adapter from a backend spec."""
    backend_type = spec.type
    if backend_type == "local_python":
        return LocalPythonBackend(spec)
    if backend_type == "legacy_python":
        return LegacyPythonBackend(spec)
    if backend_type == "subprocess":
        return SubprocessBackend(spec)
    if backend_type == "http":
        return HttpBackend(spec)
    if backend_type == "docker":
        return DockerBackend(spec)
    raise ValueError(f"Unknown inference backend type: {backend_type}")
