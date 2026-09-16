import math
import time

import torch


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    max_batches=None,
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(loader):
        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        tokens = y.numel()

        total_loss += (
            loss.item() * tokens
        )

        total_tokens += tokens

    if total_tokens == 0:
        return float("nan")

    return total_loss / total_tokens


def perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def get_memory_mb(device):
    """
    Returns an approximate allocated-memory measurement.

    MPS:
        Uses PyTorch MPS allocation APIs when available.

    CUDA:
        Uses max_memory_allocated.

    CPU:
        Returns None because PyTorch CPU allocations are not
        reliably captured by these APIs.
    """

    if device.type == "cuda":
        return (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

    if device.type == "mps":
        if hasattr(
            torch.mps,
            "driver_allocated_memory",
        ):
            return (
                torch.mps.driver_allocated_memory()
                / 1024**2
            )

        if hasattr(
            torch.mps,
            "current_allocated_memory",
        ):
            return (
                torch.mps.current_allocated_memory()
                / 1024**2
            )

    return None


def benchmark_forward(
    model,
    input_ids,
    device,
    warmup=5,
    iterations=20,
):
    model.eval()

    input_ids = input_ids.to(device)

    # Warmup.
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)

    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            model(input_ids)

    synchronize(device)

    elapsed = time.perf_counter() - start

    total_tokens = (
        input_ids.numel() * iterations
    )

    tokens_per_second = (
        total_tokens / elapsed
    )

    return {
        "elapsed_seconds": elapsed,
        "milliseconds_per_iteration": (
            elapsed / iterations * 1000
        ),
        "tokens_per_second": tokens_per_second,
        "memory_mb": get_memory_mb(device),
    }