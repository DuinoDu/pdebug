"""
Support Hadoop fs commands
"""
import glob
import logging
import os
import shlex
import shutil
import subprocess
import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import IO, Any, List, Union

log = logging.getLogger(__name__)


HADOOP_BIN = (
    "HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs"
)

__all__ = [
    "check_call_hdfs_command",
    "popen_hdfs_command",
    "has_hdfs_path_prefix",
    "is_hdfs_file",
    "is_hdfs_dir",
    "get_hdfs_list",
    "glob_hdfs_pattern",
    "get_hdfs_path_sizes",
    "mkdir_hdfs",
    "makedirs_local_or_hdfs",
    "download_from_hdfs",
    "batch_download_from_hdfs",
    "upload_to_hdfs",
    "copy_hdfs",
    "mv_hdfs",
    "rm_hdfs",
    "hdfs_open",
    "hlist_files",
    "hopen",
    "hexists",
    "hmkdir",
    "hglob",
    "hisdir",
    "hisfile",
    "hcountline",
    "hrm",
    "hcopy",
    "hmget",
    "hdu_dir",
    "hdu_file",
    "hput",
]


_HADOOP_COMMAND_TEMPLATE = "hadoop fs {command}"
_SUPPORTED_HDFS_PATH_PREFIXES = ("hdfs://", "ufs://")


def is_seq_of(seq, expected_type, seq_type=None):
    r"""Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    r"""Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def _get_hdfs_command(command):
    """Return hadoop fs command"""
    return _HADOOP_COMMAND_TEMPLATE.format(command=command)


def check_call_hdfs_command(command):
    """Check call hdfs command"""
    hdfs_command = _get_hdfs_command(command)
    subprocess.check_call(shlex.split(hdfs_command))


def popen_hdfs_command(command):
    """Call hdfs command with popen and return stdout result"""
    hdfs_command = _get_hdfs_command(command)
    p = subprocess.Popen(shlex.split(hdfs_command), stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout


def has_hdfs_path_prefix(filepath):
    """Check if input filepath has hdfs prefix"""
    for prefix in _SUPPORTED_HDFS_PATH_PREFIXES:
        if filepath.startswith(prefix):
            return True
    return False


def is_hdfs_file(filepath):
    """Check if input filepath is hdfs file"""
    if os.path.exists(filepath):
        # is local path, return False
        return False
    cmd = "-test -f {}".format(filepath)
    try:
        check_call_hdfs_command(cmd)
        return True
    except Exception:
        return False


def is_hdfs_dir(filepath):
    """Check if input filepath is hdfs directory"""
    if os.path.exists(filepath):
        # is local path, return False
        return False
    cmd = "-test -d {}".format(filepath)
    try:
        check_call_hdfs_command(cmd)
        return True
    except Exception:
        return False


def get_hdfs_list(filepath):
    """Glob hdfs path pattern"""
    try:
        cmd = "-ls {}".format(filepath)
        stdout = popen_hdfs_command(cmd)
        lines = stdout.splitlines()
        if lines:
            # decode bytes string in python3 runtime
            lines = [line.decode("utf-8") for line in lines]
            return [line.split(" ")[-1] for line in lines]
        else:
            return []
    except Exception:
        return []


def glob_hdfs_pattern(filepath):
    """Glob hdfs path pattern"""
    try:
        cmd = "-ls -d {}".format(filepath)
        stdout = popen_hdfs_command(cmd)
        lines = stdout.splitlines()
        if lines:
            # decode bytes string in python3 runtime
            lines = [line.decode("utf-8") for line in lines]
            return [line.split(" ")[-1] for line in lines]
        else:
            return []
    except Exception:
        return []


def get_hdfs_path_sizes(filepath):
    """Get size of all sub file/folder by globing input filepath"""
    try:
        cmd = "-du {}".format(filepath)
        stdout = popen_hdfs_command(cmd)
        lines = stdout.splitlines()
        if lines:
            # decode bytes string in python3 runtime
            paths_to_sizes = {}
            lines = [line.decode("utf-8") for line in lines]
            for line in lines:
                ret = line.split(" ")
                path = ret[-1]
                size = int(ret[0])
                paths_to_sizes[path] = size
            return paths_to_sizes
        else:
            return {}
    except Exception:
        return {}


def mkdir_hdfs(dirpath, raise_exception=False):
    """Mkdir hdfs directory"""
    try:
        cmd = "-mkdir -p {}".format(dirpath)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = "Failed to mkdir {} in HDFS: {}".format(dirpath, e)
        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return False


def makedirs_local_or_hdfs(dirpath, name="dirpath"):
    """Makdirs hdfs dir or local FS dir"""
    if has_hdfs_path_prefix(dirpath):
        if not is_hdfs_dir(dirpath):
            mkdir_hdfs(dirpath, raise_exception=True)
    elif not os.path.isdir(dirpath):
        os.makedirs(dirpath)


def download_from_hdfs(
    src_path: str,
    dst_path: str,
    overwrite: bool = False,
    raise_exception: bool = False,
):
    """Download src_path from hdfs to local dst_path

    Args:
        src_path: the source hdfs path
        dst_path: the local download destination
        overwrite: if True, the local file will be overwritten if it exists
        raise_exception: if True, error is raised when thing goes wrong
    """
    # Legality check
    assert isinstance(src_path, str) and has_hdfs_path_prefix(
        src_path
    ), src_path
    assert isinstance(dst_path, str) and not has_hdfs_path_prefix(
        dst_path
    ), dst_path

    # Get the targeted download path
    if os.path.isdir(dst_path):  # download to an existing folder
        download_path = os.path.join(dst_path, os.path.basename(src_path))
    else:  # download as a file
        download_path = dst_path

    if overwrite is True:  # Remove the targeted file/folder if it exists
        if os.path.isdir(download_path):
            shutil.rmtree(download_path)
        elif os.path.isfile(download_path):
            os.remove(download_path)
    else:  # skip downloading if the targeted file/folder exists
        if os.path.exists(download_path):
            return True

    # Download from hdfs
    try:
        cmd = "-get {} {}".format(src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = "Failed to download src {} to dst {}: {}".format(
            src_path, dst_path, e
        )
        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return False


def batch_download_from_hdfs(
    src_paths: List[str],
    dst_paths: Union[str, List[str]],
    overwrite: bool = False,
    raise_exception: bool = False,
    mp_size: int = 1,
) -> bool:
    """Batch download from hdfs with mp.Pool acceleration

    Args:
        src_paths: the source paths of hdfs files/folders
        dst_path: the local download destination
        overwrite: if True, the local file will be overwritten if it exists
        raise_exception: if True, error is raised when thing goes wrong
        mp_size: the max_workers for ProcessPoolExecutor
    Return:
        success: if True, all the downloads are successfully executed
    """
    # Legality check
    assert is_list_of(
        src_paths, str
    ), f"src_paths {src_paths} must of a list of str"
    if isinstance(dst_paths, str):
        dst_paths = [dst_paths] * len(src_paths)
    else:
        assert len(dst_paths) == len(
            src_paths
        ), f"length of dst_paths {dst_paths} mismatches with src_paths {src_paths}"

    # Multiprocess download with ProcessPoolExecutor context manager
    print(
        f"=> HDFS batch downloading... number of targets: "
        f"{len(src_paths)}; mp max_workers:{mp_size}"
    )
    with ProcessPoolExecutor(max_workers=mp_size) as executor:
        futures = [
            executor.submit(
                download_from_hdfs,
                src_path,
                dst_path,
                overwrite,
                raise_exception,
            )
            for src_path, dst_path in zip(src_paths, dst_paths)
        ]
    download_success = [future.result() for future in futures]
    return all(download_success)


def upload_to_hdfs(
    src_path, dst_path, overwrite=False, raise_exception=False, mkdir=False
):
    """Upload src_path to hdfs dst_path"""
    if not os.path.exists(src_path):
        raise IOError(
            "Input src_path {} not found in local storage".format(src_path)
        )
    if not has_hdfs_path_prefix(dst_path):
        raise ValueError(
            "Input dst_path {} is not a hdfs path".format(dst_path)
        )
    try:
        if mkdir is True:
            parent_path = os.path.dirname(dst_path)
            if not is_hdfs_dir(parent_path):
                mkdir_hdfs(parent_path)
        cmd = "-put -f" if overwrite else "-put"
        cmd = "{} {} {}".format(cmd, src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = "Failed to upload src {} to dst {}: {}".format(
            src_path, dst_path, e
        )
        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return False


def copy_hdfs(src_path, dst_path, overwrite=False, raise_exception=False):
    """Copy hdfs src_path to hdfs dst_path."""
    if not has_hdfs_path_prefix(src_path):
        raise ValueError(
            "Input src_path {} is not a hdfs path".format(src_path)
        )
    if not has_hdfs_path_prefix(dst_path):
        raise ValueError(
            "Input dst_path {} is not a hdfs path".format(dst_path)
        )

    try:
        cmd = "-cp -f" if overwrite else "-cp"
        cmd = "{} {} {}".format(cmd, src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = "Failed to copy src {} to dst {}: {}".format(
            src_path, dst_path, e
        )
        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return False


def mv_hdfs(src_path, dst_path, raise_exception=False):
    """Move hdfs src_path to hdfs dst_path."""
    if not has_hdfs_path_prefix(src_path):
        raise ValueError(
            "Input src_path {} is not a hdfs path".format(src_path)
        )
    if not has_hdfs_path_prefix(dst_path):
        raise ValueError(
            "Input dst_path {} is not a hdfs path".format(dst_path)
        )

    try:
        cmd = "-mv {} {}".format(src_path, dst_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = "Failed to copy src {} to dst {}: {}".format(
            src_path, dst_path, e
        )
        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return False


def rm_hdfs(hdfs_path, recursive=True, force=True, raise_exception=False):
    """Remove hdfs path."""
    if not has_hdfs_path_prefix(hdfs_path):
        raise ValueError("given path {} is not a hdfs path".format(hdfs_path))

    try:
        hdfs_cmd = "-rm "
        if recursive:
            hdfs_cmd += "-r "
        if force:
            hdfs_cmd += "-f "
        cmd = "{}{}".format(hdfs_cmd, hdfs_path)
        check_call_hdfs_command(cmd)
        return True
    except Exception as e:
        msg = "Failed to remove {}: {}".format(hdfs_path, e)
        if raise_exception:
            raise ValueError(msg)
        else:
            print(msg)
        return False


@contextmanager
def hdfs_open(hdfs_path: str, mode: str = "r"):
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            _get_hdfs_command(f"-text {hdfs_path}"),
            shell=True,
            stdout=subprocess.PIPE,
        )
        yield pipe.stdout
        pipe.stdout.close()
        pipe.wait()
        return
    if mode == "wa":
        pipe = subprocess.Popen(
            _get_hdfs_command(f"-appendToFile - {hdfs_path}"),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            _get_hdfs_command(f"-put -f - {hdfs_path}"),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    is_hdfs = hdfs_path.startswith("hdfs")
    if is_hdfs:
        return hdfs_open(hdfs_path, mode)
    else:
        return open(hdfs_path, mode)


@contextmanager  # type: ignore
def hdfs_open(hdfs_path: str, mode: str = "r") -> IO[Any]:
    """
    打开一个 hdfs 文件, 用 contextmanager.

    Args:
        hfdfs_path (str): hdfs文件路径
        mode (str): 打开模式，支持 ["r", "w", "wa"]
    """
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path),
            shell=True,
            stdout=subprocess.PIPE,
        )
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.wait()
        return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path),
            shell=True,
            stdin=subprocess.PIPE,
        )
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hlist_files(folders: List[str]) -> List[str]:
    """
    罗列一些 hdfs 路径下的文件。

    Args:
        folders (List): hdfs文件路径的list
    Returns:
        一个list of hdfs 路径
    """
    files = []
    for folder in folders:
        if folder.startswith("hdfs"):
            pipe = subprocess.Popen(
                "{} dfs -ls {}".format(HADOOP_BIN, folder),
                shell=True,
                stdout=subprocess.PIPE,
            )
            # output, _ = pipe.communicate()
            for line in pipe.stdout:  # type: ignore
                line = line.strip()
                # drwxr-xr-x   - user group  4 file
                if len(line.split()) < 5:
                    continue
                files.append(line.split()[-1].decode("utf8"))
            pipe.stdout.close()  # type: ignore
            pipe.wait()
        else:
            if os.path.isdir(folder):
                files.extend(
                    [os.path.join(folder, d) for d in os.listdir(folder)]
                )
            elif os.path.isfile(folder):
                files.append(folder)
            else:
                log.info("Path {} is invalid".format(folder))

    return files


def hexists(file_path: str) -> bool:
    """hdfs capable to check whether a file_path is exists"""
    if file_path.startswith("hdfs"):
        return (
            os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path)) == 0
        )
    return os.path.exists(file_path)


def hisdir(file_path: str) -> bool:
    """hdfs capable to check whether a file_path is a dir"""
    if file_path.startswith("hdfs"):
        flag1 = os.system(
            "{} dfs -test -e {}".format(HADOOP_BIN, file_path)
        )  # 0:路径存在
        flag2 = os.system(
            "{} dfs -test -f {}".format(HADOOP_BIN, file_path)
        )  # 0:是文件
        flag = (flag1 == 0) and (flag2 != 0)
        return flag
    return os.path.isdir(file_path)


def hisfile(file_name: str) -> bool:
    """hdfs capable to check whether a file_name is a file"""
    if file_name.startswith("hdfs"):
        flag1 = os.system(
            "{} dfs -test -f {}".format(HADOOP_BIN, file_name)
        )  # 0:是文件
        flag = flag1 == 0
        return flag
    return os.path.isfile(file_name)


def hmkdir(file_path: str) -> bool:
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        os.system("{} dfs -mkdir -p {}".format(HADOOP_BIN, file_path))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def hcopy(from_path: str, to_path: str) -> bool:
    """hdfs copy"""
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            os.system(
                "{} dfs -cp -f {} {}".format(HADOOP_BIN, from_path, to_path)
            )
        else:
            os.system(
                "{} dfs -copyFromLocal -f {} {}".format(
                    HADOOP_BIN, from_path, to_path
                )
            )
    else:
        if from_path.startswith("hdfs"):
            os.system(
                "{} dfs -text {} > {}".format(HADOOP_BIN, from_path, to_path)
            )
        else:
            shutil.copy(from_path, to_path)
    return True


def hglob(search_path, sort_by_time=False):
    """hdfs glob"""
    if search_path.startswith("hdfs"):
        if sort_by_time:
            hdfs_command = (
                HADOOP_BIN + " dfs -ls %s | sort -k6,7" % search_path
            )
        else:
            hdfs_command = HADOOP_BIN + " dfs -ls %s" % search_path
        path_list = []
        files = os.popen(hdfs_command).read()
        files = files.split("\n")
        for file in files:
            if "hdfs" in file:
                startindex = file.index("hdfs")
                path_list.append(file[startindex:])
        return path_list
    else:
        files = glob.glob(search_path)
        if sort_by_time:
            files = sorted(files, key=lambda x: os.path.getmtime(x))
    return files


def htext_list(files, target_folder):
    for fn in files:
        name = fn.split("/")[-1]
        hdfs_command = HADOOP_BIN + " dfs -text %s > %s/%s" % (
            fn,
            target_folder,
            name,
        )
        os.system(hdfs_command)


def hmget(files, target_folder, num_thread=16):
    """将整个hdfs 文件夹 get下来，但是不是简单的get，因为一些hdfs文件是压缩的，需要解压"""
    part = len(files) // num_thread
    thread_list = []
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:
            end = len(files)
        else:
            end = start + part
        t = threading.Thread(
            target=htext_list,
            kwargs={"files": files[start:end], "target_folder": target_folder},
        )
        thread_list.append(t)

    for t in thread_list:
        t.setDaemon(True)
        t.start()

    for t in thread_list:
        t.join()


def hcountline(path):
    """
    count line in file
    """
    count = 0
    if path.startswith("hdfs"):
        with hopen(path, "r") as f:
            for line in f:
                count += 1
    else:
        with open(path, "r") as f:
            for line in f:
                count += 1
    return count


def hrm(path):
    if path.startswith("hdfs"):
        os.system(f"{HADOOP_BIN} dfs -rm -r {path}")
    else:
        os.system(f"rm -rf {path}")


def hdu_dir(path):
    """hdfs du command to get sizes for all files under a directory. Output is List[Tuple]"""
    if path.startswith("hdfs"):
        cmd_output = os.popen(f"hdfs dfs -du {path}").read()
        files_info = cmd_output.splitlines()
        rst = []
        for info in files_info:
            size, _, file_path = info.split()
            rst.append((file_path, int(size)))
        return rst
    else:
        cmd_output = os.popen(f"du -b {path}").read()
        files_info = cmd_output.splitlines()
        rst = []
        for info in files_info:
            size, file_path = info.split()
            rst.append((file_path, int(size)))
        return rst


def hdu_file(path):
    """hdfs du command to get size for a specific file. Output is a Tuple"""
    if path.startswith("hdfs"):
        cmd = f"hdfs dfs -du {path}"
        cmd_output = os.popen(cmd).read()
        print(f">> {cmd}")
        print(f">> {cmd_output}")
        size, _, file_path = cmd_output.split()
        return (file_path, int(size))
    else:
        cmd_output = os.popen(f"du -b {path}").read()
        size, file_path = cmd_output.split()
        return (file_path, int(size))


def hput(from_path, to_path, force=True):
    """hdfs put a file/directory to remote using native hdfs command"""
    if not from_path.startswith("hdfs") and to_path.startswith("hdfs"):
        hmkdir(to_path)
        assert os.path.exists(from_path), f"{from_path} does not exist."
        force_flag = "-f" if force else ""
        os.system(
            "{} dfs -put {} {} {}".format(
                HADOOP_BIN, force_flag, from_path, to_path
            )
        )
    else:
        msg = "Invalid hput pattern, must copy from local to hdfs. "
        msg += f"Given {from_path}->{to_path}"
        raise ValueError(msg)
