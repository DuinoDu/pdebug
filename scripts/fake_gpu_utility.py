#!/usr/bin/env python3
# https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_autograd.html

import math
import os
import socket
import sys

import torch


def main():
    dtype = torch.float
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        local_rank = os.getenv("LOCAL_RANK", None)
        if local_rank:
            device += f":{local_rank}"

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
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1 or os.getenv("LOCAL_RANK"):
        main()
    else:
        port = find_free_port()
        cmd = f"torchrun --nproc_per_node={num_gpus} --master_port={port} {sys.argv[0]}"
        os.system(cmd)
