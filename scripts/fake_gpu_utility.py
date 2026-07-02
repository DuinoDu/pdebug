#!/usr/bin/env python3
# https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html

import argparse
import math
import os
import socket
import subprocess
import sys

import torch


GPU_ALLOC_CHUNK_BYTES = 256 * 1024 * 1024


def gpu_fraction(value):
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        message = "--gpu must be between 0 and 1, e.g. 0.5."
        raise argparse.ArgumentTypeError(message)
    return fraction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep CUDA devices busy and reserve GPU memory."
    )
    parser.add_argument(
        "--gpu",
        type=gpu_fraction,
        default=0.0,
        help="GPU memory fraction to reserve per GPU. 0.5 means 50%%.",
    )
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def get_local_rank(args):
    local_rank = os.getenv("LOCAL_RANK", None)
    if local_rank is not None:
        return int(local_rank)
    return args.local_rank


def format_bytes(num_bytes):
    gib = num_bytes / 1024**3
    return f"{gib:.2f} GiB"


def reserve_gpu_memory(device, fraction):
    if fraction <= 0:
        return []

    device = torch.device(device)
    if device.type != "cuda":
        print("--gpu ignored because CUDA is not available.")
        return []

    torch.cuda.set_device(device)
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    target_bytes = int(total_bytes * fraction)
    allocated_bytes = torch.cuda.memory_allocated(device)
    bytes_to_reserve = max(0, target_bytes - allocated_bytes)
    if bytes_to_reserve == 0:
        target = format_bytes(target_bytes)
        message = f"GPU memory target already reached: {target}"
        print(message)
        return []

    free_bytes, _ = torch.cuda.mem_get_info(device)
    safety_margin = min(512 * 1024**2, max(64 * 1024**2, total_bytes // 100))
    if bytes_to_reserve + safety_margin > free_bytes:
        raise RuntimeError(
            "Not enough free GPU memory to reserve "
            f"{format_bytes(bytes_to_reserve)} on {device}. "
            f"Free: {format_bytes(free_bytes)}, "
            f"safety margin: {format_bytes(safety_margin)}."
        )

    tensors = []
    remaining_bytes = bytes_to_reserve
    while remaining_bytes > 0:
        chunk_bytes = min(remaining_bytes, GPU_ALLOC_CHUNK_BYTES)
        tensor = torch.empty(chunk_bytes, dtype=torch.uint8, device=device)
        tensors.append(tensor)
        remaining_bytes -= chunk_bytes

    torch.cuda.synchronize(device)
    reserved_bytes = sum(t.numel() * t.element_size() for t in tensors)
    reserved = format_bytes(reserved_bytes)
    message = f"Reserved {reserved} ({fraction:.0%}) on {device}."
    print(message)
    return tensors


def main(args):
    dtype = torch.float
    device = "cpu"
    if torch.cuda.is_available():
        local_rank = get_local_rank(args)
        cuda_index = 0 if local_rank is None else local_rank
        device = f"cuda:{cuda_index}"
        torch.cuda.set_device(cuda_index)

    # torch.set_default_device(device)

    # Create Tensors to hold input and outputs.
    # By default, requires_grad=False, which indicates that we do not need to
    # compute gradients with respect to these Tensors during the backward pass.
    x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype).to(device)
    y = torch.sin(x).to(device)

    # Create random Tensors for weights. For a third order polynomial, we need
    # 4 weights: y = a + b x + c x^2 + d x^3
    # Setting requires_grad=True indicates that we want to compute gradients with
    # respect to these Tensors during the backward pass.
    a = torch.randn((), dtype=dtype, requires_grad=True).to(device)
    b = torch.randn((), dtype=dtype, requires_grad=True).to(device)
    c = torch.randn((), dtype=dtype, requires_grad=True).to(device)
    d = torch.randn((), dtype=dtype, requires_grad=True).to(device)
    reserved_gpu_memory = reserve_gpu_memory(device, args.gpu)
    assert reserved_gpu_memory is not None

    learning_rate = 1e-6
    t = 0
    while True:
        # Forward pass: compute predicted y using operations on Tensors.
        y_pred = a + b * x + c * x**2 + d * x**3

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
        if t % 10000 == 0:
            # print(t, loss.item())
            print("waiting ...")
            t = 0

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()

        # # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # # because weights have requires_grad=True, but we don't need to track this
        # # in autograd.
        # with torch.no_grad():
        #     a -= learning_rate * a.grad
        #     b -= learning_rate * b.grad
        #     c -= learning_rate * c.grad
        #     d -= learning_rate * d.grad

        #     # Manually zero the gradients after updating weights
        #     a.grad = None
        #     b.grad = None
        #     c.grad = None
        #     d.grad = None
        t += 1


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


if __name__ == "__main__":
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1 or os.getenv("LOCAL_RANK"):
        main(args)
    else:
        port = find_free_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            f"--master_port={port}",
            sys.argv[0],
            *sys.argv[1:],
        ]
        sys.exit(subprocess.call(cmd))
