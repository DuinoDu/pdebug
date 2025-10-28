import hashlib
import json
import os
import random
import shlex
import string
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from io import StringIO
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import typer
import yaml

__all__ = [
    "load_yaml_with_head",
    "dump_yaml_with_head",
    "load_rgb_from_yuv",
    "load_python_config",
    "load_content",
    "add_folder_to_zip",
    "iter_to_parquet",
    "no_print",
    "run_with_print",
    "scp_download_if_not_exists",
    "do_system",
    "save_json",
    "sleep_with_interrupt",
    "download_file",
]


def load_yaml_with_head(yaml_file: str, return_head: bool = False) -> Dict:
    """Load yaml file with head string."""
    assert yaml_file.endswith(".yaml")
    lines = [l for l in open(yaml_file, "r").readlines()]
    head = None
    if "%YAML" in lines[0]:
        yaml_str = "".join(lines[1:])
        head = lines[0]
    else:
        yaml_str = "".join(lines)
    config = yaml.load(yaml_str, Loader=yaml.SafeLoader)

    if return_head:
        return config, head
    else:
        return config


def dump_yaml_with_head(data, yaml_file: str, head_str: str, **kwargs) -> Dict:
    """Dump yaml file with head string."""
    yaml_str = yaml.dump(data, **kwargs)
    if head_str:
        yaml_str = head_str + "\n" + yaml_str

    with open(yaml_file, "w") as fid:
        fid.write(yaml_str)


def load_rgb_from_yuv(y_file, uv_file):
    """Load rgb from yuv file, format: nv21(yuv420)"""
    img_y = cv2.imread(y_file, cv2.IMREAD_UNCHANGED)
    height, width = img_y.shape[:2]

    in_file = open(uv_file, "rb")  # opening for [r]eading as [b]inary
    data = (
        in_file.read()
    )  # if you only wanted to read 512 bytes, do .read(512)
    uv = np.frombuffer(data, dtype=np.uint8)

    img_u = uv[0::2][:, None]
    img_v = uv[1::2][:, None]
    img_u = cv2.resize(
        img_u.reshape(height // 2, width // 2),
        (width, height),
        cv2.INTER_NEAREST,
    )
    img_v = cv2.resize(
        img_v.reshape(height // 2, width // 2),
        (width, height),
        cv2.INTER_NEAREST,
    )
    yuv444 = np.concatenate(
        (img_y[:, :, None], img_u[:, :, None], img_v[:, :, None]), axis=2
    ).reshape(height, width, -1)
    rgb = cv2.cvtColor(yuv444, cv2.COLOR_YUV2RGB)
    in_file.close()
    return rgb


def update_config_from_args(config: Dict, args: List[str]) -> None:
    if not args:
        return

    def _set_dot_value_recursive(name, value, input_dict):
        if "." not in name and name in input_dict:
            input_dict[name] = value
            return
        first = name.split(".")[0]
        others = ".".join(name.split(".")[1:])
        if first not in input_dict:
            input_dict[first] = {}
        _set_dot_value_recursive(others, value, input_dict[first])

    assert len(args) % 2 == 0
    extra_args = {k: args[_i + 1] for _i, k in enumerate(args) if _i % 2 == 0}
    for k, v in extra_args.items():
        assert k.startswith("--"), f"{k} should startswith --"
        k_name = k.split("--")[1]
        _set_dot_value_recursive(k_name, v, config)


def load_python_config(
    config_file: str, args: List[str] = None, stop_string: str = None
) -> Dict:
    """Load config in python format.

    Args:
        config_file: input config file
        args: extra kv config to update
        stop_string: load part content in config_file util meet stop_string.

    """
    if stop_string:
        tmp_config_file = tempfile.NamedTemporaryFile("w", suffix=".py")
        with open(config_file, "r") as fid:
            for line in fid:
                if stop_string in line:
                    break
                else:
                    tmp_config_file.write(line)
        tmp_config_file.flush()
        config_file = tmp_config_file.name

    exec(open(config_file).read())
    config = locals()
    update_config_from_args(config, args)
    config.pop("args")
    config.pop("config_file")
    return config


def load_content(f: Union[str, Any], **kwargs) -> Tuple[Any, bool]:
    if not isinstance(f, str) or not os.path.exists(f):
        return f, False

    ext = os.path.splitext(f)[1]
    if ext in [".png", ".jpg", ".bmp", "jpeg"]:
        return cv2.imread(f, **kwargs), True
    if ext == ".py":
        return load_python_config(f), True
    if ext in [".yml", ".yaml"]:
        return load_yaml_with_head(f), True
    if ext in [".json"]:
        with open(f, "r") as file:
            data = json.load(file)
        return data
    if ext in [".txt"]:
        raise NotImplementedError


def add_folder_to_zip(zip_handle, folder_path, base_path=None):
    """Add folder to zip."""
    for root, dirs, files in os.walk(folder_path):
        relative_path = os.path.relpath(root, folder_path)
        for file_i in files:
            filepath = os.path.join(root, file_i)
            if base_path:
                arcname = os.path.relpath(filepath, base_path)
            else:
                arcname = filepath
            zip_handle.write(filepath, arcname=arcname)


def iter_to_parquet(data_iter, output, num_workers=4, check=False):
    from pdebug.piata import Input
    from pdebug.piata.data_pack.packer.base_packer import BasePacker
    from pdebug.piata.data_pack.parser.base_parser import BaseParser
    from pdebug.piata.data_pack.writer import ParquetWriter

    import pyarrow.parquet as pq

    class Parser(BaseParser):
        def get_data_iter(self):
            return data_iter()

    outdir = os.path.dirname(output)
    writer = ParquetWriter(output)
    packer = BasePacker(Parser(), writer, num_pack_workers=num_workers)
    packer.run()

    if check:
        print(f"checking {output} ...")
        if output.endswith(".parquet"):
            assert os.path.exists(output), f"{output} not exists"
            table = pq.read_table(output)
        else:
            reader = Input(outdir, name="cruise_rgbd_semseg").get_reader()
            for data_dict in reader:
                break
        print("done")


@contextmanager
def no_print():
    sys.stdout = StringIO()
    yield
    sys.stdout = sys.__stdout__


def run_with_print(
    cmd_list, use_shell_file=False, debug_shell=False, get_return_from_cmd=True
):
    """Run bash command with print to stdout.

    Args:
        cmd_list: ["bash",  tmp_shell] or "bash tmp_shell"
    """
    if isinstance(cmd_list, list):
        cmd_str = " ".join(cmd_list)
    elif isinstance(cmd_list, str):
        cmd_str = cmd_list
    else:
        raise TypeError(f"Unknown input: {cmd_list}, type: {type(cmd_list)}")
    command = shlex.split(cmd_str)

    if use_shell_file:
        file_id = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )
        tmp_shell = f"/tmp/tmp_run_{file_id}.sh"
        with open(tmp_shell, "w") as fid:
            fid.write("set -ex\n")
            fid.write(cmd_str)
        command = ["bash", tmp_shell]

    ret = []
    if get_return_from_cmd:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for c1, c2 in zip(
            iter(lambda: process.stdout.read(1), b""),
            iter(lambda: process.stderr.read(1), b""),
        ):
            sys.stdout.buffer.write(c1)
            sys.stderr.buffer.write(c2)
            ret.append(c1.decode("utf-8"))
    else:
        process = subprocess.Popen(command)

    ret = "".join(ret).strip()
    process.communicate()
    exit_code = process.wait()
    if exit_code:
        raise RuntimeError(
            f"Error met when run shell, return code ({exit_code})"
        )

    if use_shell_file and not debug_shell:
        os.system(f"rm {tmp_shell}")

    return ret


def scp_download_if_not_exists(remote_path, local_path=None, port=22):
    """
    如果本地 local_path 文件不存在，则用 scp 从 remote_path 下载到本地。

    Args:
        remote_path: 远程文件路径，如 root@115.190.130.100:/home/xxx
        local_path: 本地文件保存路径
    """
    CACHE_DIR = os.environ.get(
        "PDEBUG_CACHE_DIR", os.path.expanduser("~/.cache/pdebug")
    )
    os.makedirs(CACHE_DIR, exist_ok=True)

    if local_path is None:
        local_path = os.path.join(
            CACHE_DIR, remote_path.replace("/", "_").replace(":", "_")
        )
    if os.path.exists(local_path):
        return local_path
    try:
        print(f"Downloading {remote_path} to {local_path} ...")
        subprocess.run(
            ["scp", "-P", str(port), remote_path, local_path], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {remote_path} to {local_path}: {e}")
    return local_path


def do_system(arg, skip=False, with_return=False, trials=1, retry_args=None):
    print(f"\n==== running:")
    typer.echo(typer.style(f"  >> {arg}", fg=typer.colors.GREEN))
    if skip:
        typer.echo(typer.style(f"  skip", fg=typer.colors.GREEN))
        return

    hash_value = hashlib.sha256(arg.encode()).hexdigest()
    tmp_log = f"/tmp/do_system_temp_output_{hash_value}.txt"

    cmd_success = False
    for try_idx in range(trials):

        if try_idx == 1 and retry_args:
            arg += f" {retry_args}"

        # if "2>&1" not in arg:
        #     arg += f" 2>&1 | tee {tmp_log}"
        if os.path.exists(tmp_log):
            os.system(f"rm -rf {tmp_log}")

        os.environ["TERM"] = "xterm"
        err = os.system(arg)
        if err:
            print("FATAL: command failed")
            if try_idx < trials - 1:
                print(
                    f"Try ({try_idx} / {trials}) in 5 seconds later, or press 'Enter' to continue"
                )
                sleep_with_interrupt(5)
        else:
            cmd_success = True
            break

    if os.path.exists(tmp_log):
        os.system(f"rm -rf {tmp_log}")

    if not cmd_success:
        raise RuntimeError(f"Command: {arg} failed in {trials} trials.")

    if with_return:
        return False if err else True
    else:
        if not with_return:
            sys.exit(err)


def save_json(data, filename):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2)


def sleep_with_interrupt(duration=5):
    """
    Sleep for specified duration (default 5 seconds) with ability to exit early.
    Press Enter to exit sleep early.
    """
    interrupted = [False]

    def wait_for_enter():
        input()
        interrupted[0] = True

    # Start thread to listen for Enter key
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    start_time = time.time()
    while time.time() - start_time < duration and not interrupted[0]:
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)
        print(f"\rRemaining: {remaining:.1f} seconds", end="", flush=True)
        time.sleep(0.1)  # Small sleep to prevent busy waiting
    print()

    if interrupted[0]:
        print("Sleep interrupted by user!")
    else:
        print("Sleep completed!")


def download_file(url, local_filename, exit_when_failed=False):
    """
    Download an image from URL to local file.

    Args:
        url (str): The URL of the image to download
        local_filename (str, optional): Local filename. If None, uses URL filename.

    Returns:
        str: Path to downloaded file or None if failed
    """
    import requests

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Image downloaded successfully: {local_filename}")
        return local_filename

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        if exit_when_failed:
            sys.exit()
        return None
