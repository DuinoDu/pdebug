import gc
from typing import Dict, List, Optional, Sequence

__all__ = ["gpu_memory_tic", "gpu_memory_toc"]

_GPU_MEMORY_TOLERANCE_BYTES = 1 * 1024 * 1024


def gpu_memory_tic() -> Optional[List[Dict[str, int]]]:
    """
    Take a snapshot of current GPU memory usage across all devices.
    Returns a list of dictionaries with device, allocated, and reserved memory.
    """
    try:
        import torch
    except ImportError:
        return None

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return None
    try:
        if not cuda.is_available():
            return None
    except Exception:
        return None

    try:
        device_count = cuda.device_count()
    except Exception:
        device_count = 0
    if device_count <= 0:
        return None

    snapshot: List[Dict[str, int]] = []
    for device_idx in range(device_count):
        try:
            cuda.synchronize(device=device_idx)
        except Exception:
            pass
        try:
            with cuda.device(device_idx):
                allocated = int(cuda.memory_allocated())
                reserved = int(cuda.memory_reserved())
        except Exception:
            allocated = 0
            reserved = 0
        snapshot.append(
            {
                "device": device_idx,
                "allocated": allocated,
                "reserved": reserved,
            }
        )
    return snapshot


def gpu_memory_toc(
    stage_name: str,
    before_snapshot: Optional[List[Dict[str, int]]],
    cache_functions: Sequence[Optional[object]] = (),
) -> None:
    """
    Release GPU memory and check if memory usage is higher than baseline.

    Args:
        stage_name: Name of the processing stage for logging
        before_snapshot: Snapshot from gpu_memory_tic()
        cache_functions: Optional sequence of cache clearing functions
    """
    for cache_function in cache_functions:
        if cache_function is None:
            continue
        cache_clear = getattr(cache_function, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()

    gc.collect()

    try:
        import torch
    except ImportError:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                try:
                    device_count = cuda.device_count()
                except Exception:
                    device_count = 0
                for device_idx in range(device_count):
                    try:
                        cuda.synchronize(device=device_idx)
                    except Exception:
                        pass
                for device_idx in range(device_count):
                    try:
                        with cuda.device(device_idx):
                            cuda.empty_cache()
                            ipc_collect = getattr(cuda, "ipc_collect", None)
                            if callable(ipc_collect):
                                ipc_collect()
                    except Exception:
                        pass
                for device_idx in range(device_count):
                    try:
                        cuda.synchronize(device=device_idx)
                    except Exception:
                        pass
        except Exception:
            pass

    if before_snapshot is None:
        return

    after_snapshot = gpu_memory_tic()
    if after_snapshot is None:
        return

    before_map = {entry["device"]: entry for entry in before_snapshot}
    after_map = {entry["device"]: entry for entry in after_snapshot}
    tolerance = _GPU_MEMORY_TOLERANCE_BYTES
    for device, before in before_map.items():
        after = after_map.get(device)
        if after is None:
            continue
        if (
            after["allocated"] > before["allocated"] + tolerance
            or after["reserved"] > before["reserved"] + tolerance
        ):
            import typer

            typer.secho(
                f"[WARN] GPU memory after '{stage_name}' is higher than baseline on device {device} "
                f"(allocated {before['allocated']} -> {after['allocated']}, "
                f"reserved {before['reserved']} -> {after['reserved']}).",
                fg=typer.colors.YELLOW,
            )
