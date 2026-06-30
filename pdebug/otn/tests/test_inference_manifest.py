from __future__ import annotations
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pdebug.otn import cli as otn_cli
from pdebug.otn import manager as otn_manager
from pdebug.otn.infer.base import (
    BackendSpec,
    DockerBackend,
    HealthStatus,
    InferenceNode,
    InferenceRequest,
    InferenceResult,
    NodeSpec,
    discover_manifest_files,
    doctor_manifest_path,
    load_node_spec,
    manifest_schema,
    node_to_schema,
    validate_node_spec,
)

import numpy as np
import pytest
from PIL import Image


def _write_manifest(path, *, name="echo_node", backend=None):
    payload = {
        "name": name,
        "task": "caption",
        "description": "Echo inference node",
        "backend": backend
        or {
            "type": "local_python",
            "module": "pdebug.otn.tests.fake_inference_backend",
            "class": "EchoBackend",
            "init_kwargs": {"prefix": "manifest"},
        },
        "schemas": {"input": "ImageRequest", "output": "CaptionResult"},
        "capabilities": {"batch": True, "lance": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_image(path):
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :5] = [30, 60, 90]
    image[:, 5:] = [120, 150, 180]
    Image.fromarray(image, mode="RGB").save(path)
    return path


def test_manifest_files_are_discovered_without_importing_model_code(tmp_path):
    manifest = tmp_path / "echo.otn.json"
    ignored = tmp_path / "plain.json"
    _write_manifest(manifest)
    ignored.write_text("{}", encoding="utf-8")

    files = discover_manifest_files([str(tmp_path)])
    spec = load_node_spec(str(manifest))

    assert files == [manifest]
    assert spec.name == "echo_node"
    assert spec.backend.type == "local_python"
    assert spec.schemas["output"] == "CaptionResult"


def test_local_python_manifest_node_predicts_through_common_contract(tmp_path):
    manifest = tmp_path / "echo.otn.json"
    _write_manifest(manifest)
    node = InferenceNode(load_node_spec(str(manifest)))

    result = node({"inputs": {"image": "frame.png"}, "parameters": {"k": 1}})

    assert result["data"] == {
        "prefix": "manifest",
        "inputs": {"image": "frame.png"},
        "parameters": {"k": 1},
    }


def test_subprocess_backend_uses_json_file_protocol(tmp_path):
    script = tmp_path / "infer.py"
    script.write_text(
        "import json, sys\n"
        "input_path = sys.argv[sys.argv.index('--input') + 1]\n"
        "output_path = sys.argv[sys.argv.index('--output') + 1]\n"
        "request = json.load(open(input_path))\n"
        "json.dump({'data': {'seen': request['inputs']['value']}}, "
        "open(output_path, 'w'))\n",
        encoding="utf-8",
    )
    spec = NodeSpec(
        name="subprocess_echo",
        backend=BackendSpec(
            type="subprocess",
            config={"command": ["python", str(script)]},
        ),
    )
    node = InferenceNode(spec)

    result = node(value=7)

    assert result["data"] == {"seen": 7}


def test_legacy_python_backend_can_load_relative_file_manifest(tmp_path):
    backend_file = tmp_path / "external_backend.py"
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "file_backend.otn.json"
    backend_file.write_text(
        "def echo(value):\n" "    return {'seen': value}\n",
        encoding="utf-8",
    )
    _write_manifest(
        manifest,
        name="file_backend",
        backend={
            "type": "legacy_python",
            "file": "../external_backend.py",
            "function": "echo",
        },
    )
    node = InferenceNode(load_node_spec(str(manifest)))

    assert node(value=5) == {"seen": 5}
    assert "_manifest_path" not in node_to_schema(node)["backend"]


class _JsonHandler(BaseHTTPRequestHandler):
    # BaseHTTPRequestHandler dispatches this exact method name.
    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response = json.dumps(
            {"data": {"path": self.path, "inputs": payload["inputs"]}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return


def test_http_backend_posts_normalized_request():
    server = HTTPServer(("127.0.0.1", 0), _JsonHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        spec = NodeSpec(
            name="http_echo",
            backend=BackendSpec(
                type="http",
                config={"endpoint": f"http://127.0.0.1:{port}/predict"},
            ),
        )
        node = InferenceNode(spec)

        result = node(text="hello")

        assert result["data"] == {
            "path": "/predict",
            "inputs": {"text": "hello"},
        }
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_docker_backend_builds_container_command_without_running_docker(
    tmp_path,
):
    spec = BackendSpec(
        type="docker",
        config={
            "image": "example/model:latest",
            "command": ["python", "/app/infer.py"],
            "gpu": True,
            "env_map": {"MODEL_CACHE": "/cache"},
            "mounts": [{"host": "/models", "container": "/models"}],
        },
    )
    backend = DockerBackend(spec)

    command = backend.build_command(
        tmp_path / "request.json", tmp_path / "result.json"
    )

    assert command[:4] == ["docker", "run", "--rm", "--gpus"]
    assert "example/model:latest" in command
    assert "MODEL_CACHE=/cache" in command
    assert "/models:/models" in command
    assert command[-2:] == ["python", "/app/infer.py"]

    spec.config["append_io_args"] = True
    command = backend.build_command(
        tmp_path / "request.json", tmp_path / "result.json"
    )
    assert command[-4:] == [
        "--input",
        "/otn_io/request.json",
        "--output",
        "/otn_io/result.json",
    ]


def test_manager_create_can_load_manifest_nodes(tmp_path, monkeypatch):
    manifest = tmp_path / "managed_echo.otn.json"
    _write_manifest(manifest, name="managed_echo")
    monkeypatch.setattr(otn_manager, "_EXTENSION_DIRS", (str(tmp_path),))
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(otn_manager, "_EXTENSION_LOAD_ERRORS", [])
    monkeypatch.setattr(otn_manager, "_MANIFEST_LOAD_ERRORS", [])

    node = otn_manager.create("managed_echo")

    assert isinstance(node.healthcheck(), HealthStatus)
    assert node(sample="frame")["data"]["inputs"] == {"sample": "frame"}


def test_manager_prefers_manifest_node_without_importing_legacy_module(
    monkeypatch,
):
    otn_manager.NODE._obj_map.pop("moondream", None)
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(
        otn_manager.NODE,
        "find_node_from_folder",
        lambda *args, **kwargs: pytest.fail(
            "legacy extension scan should not be needed"
        ),
    )

    node = otn_manager.create("moondream")

    assert isinstance(node, InferenceNode)
    assert node.spec.backend.type == "legacy_python"
    assert node.manifest_path.name == "moondream.otn.json"


def test_manifest_nodes_work_with_registry_repr_and_cli_file_print(
    tmp_path, monkeypatch, capsys
):
    manifest = tmp_path / "visible_echo.otn.json"
    _write_manifest(manifest, name="visible_echo")
    monkeypatch.setattr(otn_manager, "_EXTENSION_DIRS", (str(tmp_path),))
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(otn_manager, "_EXTENSION_LOAD_ERRORS", [])
    monkeypatch.setattr(otn_manager, "_MANIFEST_LOAD_ERRORS", [])

    node = otn_manager.create("visible_echo")

    assert "visible_echo" in repr(otn_manager.NODE)
    monkeypatch.setattr(otn_cli.otn_manager, "create", lambda name: node)
    otn_cli.main(
        ctx=type("Ctx", (), {"args": []})(),
        node="visible_echo",
        list_node=False,
        help_node=False,
        print_node_file=True,
        force_single_process=True,
    )

    assert str(manifest) in capsys.readouterr().out


def test_manifest_legacy_node_preserves_raw_python_return(tmp_path):
    manifest = Path("pdebug/otn/infer/manifests/moondream.otn.json")
    node = InferenceNode(load_node_spec(str(manifest)))
    image_path = _write_image(tmp_path / "frame.png")

    result = node(str(image_path), unittest=True)

    assert "data" not in result
    assert result["model_name"] == "moondream"
    assert result["caption"].startswith("moondream-0")


def test_cli_manifest_node_help_schema_doctor_and_bool_cast(
    tmp_path, monkeypatch, capsys
):
    manifest = Path("pdebug/otn/infer/manifests/moondream.otn.json")
    node = InferenceNode(load_node_spec(str(manifest)))
    image_path = _write_image(tmp_path / "frame.png")
    output_path = tmp_path / "caption.json"
    monkeypatch.setattr(otn_cli.otn_manager, "create", lambda name: node)

    otn_cli.main(
        ctx=type("Ctx", (), {"args": []})(),
        node="moondream",
        list_node=False,
        help_node=True,
        print_node_file=False,
        schema_node=False,
        doctor_node=False,
        force_single_process=True,
    )
    assert "unittest: bool = False" in capsys.readouterr().out

    otn_cli.main(
        ctx=type("Ctx", (), {"args": []})(),
        node="moondream",
        list_node=False,
        help_node=False,
        print_node_file=False,
        schema_node=True,
        doctor_node=False,
        force_single_process=True,
    )
    schema_payload = json.loads(capsys.readouterr().out)
    assert schema_payload["name"] == "moondream"
    assert schema_payload["parameters"]["unittest"]["type"] == "bool"

    otn_cli.main(
        ctx=type("Ctx", (), {"args": []})(),
        node="moondream",
        list_node=False,
        help_node=False,
        print_node_file=False,
        schema_node=False,
        doctor_node=True,
        force_single_process=True,
    )
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["ok"]

    otn_cli.main(
        ctx=type(
            "Ctx",
            (),
            {
                "args": [
                    "--input-path",
                    str(image_path),
                    "--output",
                    str(output_path),
                    "--unittest",
                    "true",
                ]
            },
        )(),
        node="moondream",
        list_node=False,
        help_node=False,
        print_node_file=False,
        schema_node=False,
        doctor_node=False,
        force_single_process=True,
    )
    assert json.loads(output_path.read_text())["model_name"] == "moondream"


def test_cli_can_print_generic_manifest_schema(capsys):
    otn_cli.main(
        ctx=type("Ctx", (), {"args": []})(),
        node=None,
        list_node=False,
        help_node=False,
        print_node_file=False,
        schema_node=True,
        doctor_node=False,
        force_single_process=True,
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["required"] == ["name", "backend"]
    assert (
        "docker"
        in payload["properties"]["backend"]["properties"]["type"]["enum"]
    )


def test_external_model_manifests_are_loadable_without_running_backends():
    sam6d = load_node_spec("pdebug/otn/infer/manifests/sam6d_docker.otn.json")
    foundpose = load_node_spec(
        "pdebug/otn/infer/manifests/foundpose_subprocess.otn.json"
    )

    assert sam6d.backend.type == "docker"
    assert sam6d.parameters["rgb_path"]["type"] == "str"
    assert foundpose.backend.type == "subprocess"
    assert foundpose.schemas["input"]["required"] == [
        "rgb_path",
        "mask_path",
        "cad_path",
        "camera_path",
    ]


def test_infer_model_nodes_are_manifest_backed_without_legacy_scan(
    monkeypatch,
):
    manifest_node_names = {
        "cad-to-templates",
        "cotracker",
        "depth-anything-video",
        "depth_anything",
        "foundpose-to-linemod",
        "genpose2",
        "groundingdino",
        "hunyuan3d_paint",
        "hunyuan3d_rembg",
        "hunyuan3d_shape",
        "internimage_semseg",
        "langsam_for_aigc",
        "langsam_predict",
        "langsam_sam",
        "ml_depth_pro",
        "moondream",
        "oneposeviagen_3dgen",
        "oneposeviagen_pose",
        "oneposeviagen_scale",
        "orient_anything",
        "qwen2_5_vl",
        "remove_dynamic",
        "sam2",
        "sam6d",
        "sam6d-from-sam2",
        "sam6d-to-linemod",
        "sam6d-to-sam2",
        "sam_with_prompt",
        "segment_anything",
        "spatracker",
        "templates-to-linemod",
        "vggt",
        "vggt-viser",
        "video_kps_to_all",
    }
    for name in manifest_node_names:
        otn_manager.NODE._obj_map.pop(name, None)
    monkeypatch.setattr(otn_manager, "_EXTENSIONS_LOADED", False)
    monkeypatch.setattr(
        otn_manager.NODE,
        "find_node_from_folder",
        lambda *args, **kwargs: pytest.fail(
            "legacy extension scan should not be needed"
        ),
    )

    for name in sorted(manifest_node_names):
        node = otn_manager.create(name)
        assert isinstance(node, InferenceNode), name
        assert node.spec.backend.type == "legacy_python"


def test_docker_manifest_healthcheck_reports_environment_without_crashing():
    sam6d = InferenceNode(
        load_node_spec("pdebug/otn/infer/manifests/sam6d_docker.otn.json")
    )

    health = sam6d.healthcheck()

    assert isinstance(health, HealthStatus)
    if health.ok:
        assert health.message == "configured"
    else:
        assert "docker" in health.message


def test_manifest_diagnostics_validate_and_doctor_files(tmp_path):
    valid_manifest = tmp_path / "valid.otn.json"
    _write_manifest(valid_manifest)

    bad_spec = NodeSpec(
        name="bad",
        backend=BackendSpec(type="unsupported", config={}),
        parameters={"threshold": {"type": "decimal"}},
    )
    issues = validate_node_spec(bad_spec)
    report = doctor_manifest_path(str(valid_manifest), check_backend=False)

    assert manifest_schema()["properties"]["backend"]["required"] == ["type"]
    assert issues[0].level == "error"
    assert "unsupported backend type" in issues[0].message
    assert report["ok"]
    assert report["checked"][0]["name"] == "echo_node"


def test_request_and_result_accept_plain_payloads():
    request = InferenceRequest.from_payload({"image": "a.png"})
    result = InferenceResult.from_payload({"caption": "object"})

    assert request.inputs == {"image": "a.png"}
    assert result.to_dict()["data"] == {"caption": "object"}


def test_runner_migrated_nodes_keep_single_image_and_signature_behaviour(
    tmp_path,
):
    from pdebug.otn.infer.ml_depth_pro_node import (
        _depth_infer,
        ml_depth_pro_node,
    )
    from pdebug.otn.infer.segment_anything_node import segment_anything_node

    image_path = _write_image(tmp_path / "frame.png")
    seg_output = tmp_path / "segment.json"
    depth_output = tmp_path / "depth.json"

    seg_result = segment_anything_node(
        str(image_path), output=str(seg_output), unittest=True
    )
    depth_result = ml_depth_pro_node(
        str(image_path), output=str(depth_output), unittest=True
    )
    direct_depth = _depth_infer(
        np.zeros((2, 3, 3), dtype=np.uint8), unittest=True
    )
    indexed_depth = _depth_infer(
        np.zeros((2, 3, 3), dtype=np.uint8), index=3, unittest=True
    )

    assert Path(seg_result) == seg_output
    assert json.loads(seg_output.read_text())["model_name"] == (
        "SegmentAnything"
    )
    assert Path(depth_result) == depth_output
    assert "mean_depth" in json.loads(depth_output.read_text())
    assert direct_depth["model_name"] == "ml-depth-pro"
    assert indexed_depth["model_name"] == "ml-depth-pro"

    segment_signature = inspect.signature(segment_anything_node)
    depth_signature = inspect.signature(ml_depth_pro_node)
    assert segment_signature.parameters["output_key"].default == (
        "cortexia_segmentation"
    )
    assert depth_signature.parameters["unittest"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )


def test_image_input_adapter_preserves_output_key_and_missing_output_message(
    monkeypatch,
):
    from pdebug.otn.infer.base import ImageInferenceRunner
    from pdebug.otn.infer.io_adapters import run_image_input
    from pdebug.piata import ImageBatch

    captured = {}

    def fake_read_image_batch(input_path, **kwargs):
        return ImageBatch(
            images=["image"],
            table="table",
            source_type="lance",
        )

    def fake_write_lance_column(table, column_name, values, output):
        captured.update(
            {
                "table": table,
                "column_name": column_name,
                "values": values,
                "output": output,
            }
        )
        return output

    monkeypatch.setattr(
        "pdebug.otn.infer.io_adapters.read_image_batch",
        fake_read_image_batch,
    )
    monkeypatch.setattr(
        "pdebug.otn.infer.io_adapters.write_lance_column",
        fake_write_lance_column,
    )

    runner = ImageInferenceRunner(lambda image, *, index: {"index": index})

    with pytest.raises(ValueError, match="custom missing output"):
        run_image_input(
            runner,
            "dataset.lance",
            output=None,
            output_key="default_col",
            missing_output_message="custom missing output",
        )
    assert run_image_input(
        runner,
        "dataset.lance",
        output="out.lance",
        output_key="",
    ) == ("out.lance")
    assert captured["column_name"] == ""
    assert captured["values"] == [{"index": 0}]


def test_image_inference_runner_is_storage_agnostic():
    from pdebug.otn.infer.base import ImageInferenceRunner

    runner = ImageInferenceRunner(
        lambda image, *, index, unittest: {
            "image": image,
            "index": index,
            "unittest": unittest,
        }
    )

    assert runner.run_one("decoded-image", unittest=True) == {
        "image": "decoded-image",
        "index": 0,
        "unittest": True,
    }
    assert runner.run_batch(["a", "b"], unittest=False) == [
        {"image": "a", "index": 0, "unittest": False},
        {"image": "b", "index": 1, "unittest": False},
    ]


def test_node_to_schema_describes_plain_python_nodes():
    def plain_node(path: str, *, dry_run: bool = False):
        """Plain node doc."""
        return path, dry_run

    schema = node_to_schema(plain_node)

    assert schema["name"] == "plain_node"
    assert schema["parameters"]["path"]["type"] == "str"
    assert schema["parameters"]["dry_run"]["default"] is False
