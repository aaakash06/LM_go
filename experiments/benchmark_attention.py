import argparse
import csv
import os
import time

import torch

from src.attention import ReferenceCausalSelfAttention
from src.chunked_attention import ChunkedCausalSelfAttention
from src.evaluation import synchronize, get_memory_mb
from src.flash_attention import SDPACAusalSelfAttention
from src.training import get_device


def build_attention(
    attention_type,
    d_model,
    n_heads,
    chunk_size,
):
    if attention_type == "reference":
        return ReferenceCausalSelfAttention(
            d_model,
            n_heads,
        )

    if attention_type == "sdpa":
        return SDPACAusalSelfAttention(
            d_model,
            n_heads,
        )

    if attention_type == "chunked":
        return ChunkedCausalSelfAttention(
            d_model,
            n_heads,
            chunk_size=chunk_size,
        )

    raise ValueError(attention_type)


def benchmark(
    attention_type,
    seq_len,
    batch_size,
    d_model,
    n_heads,
    chunk_size,
    warmup,
    iterations,
    device,
):
    torch.manual_seed(1234)

    model = build_attention(
        attention_type,
        d_model,
        n_heads,
        chunk_size,
    ).to(device)

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    synchronize(device)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            model(x)

    synchronize(device)

    elapsed = (
        time.perf_counter() - start
    )

    ms = (
        elapsed
        / iterations
        * 1000
    )

    tokens = (
        batch_size
        * seq_len
        * iterations
    )

    tokens_per_second = (
        tokens / elapsed
    )

    memory = get_memory_mb(device)

    return {
        "attention_type": attention_type,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "chunk_size": chunk_size,
        "iterations": iterations,
        "milliseconds": ms,
        "tokens_per_second": tokens_per_second,
        "memory_mb": (
            memory
            if memory is not None
            else ""
        ),
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[
            128,
            256,
            512,
            1024,
        ],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--results",
        default="results/benchmark_results.csv",
    )

    args = parser.parse_args()

    device = get_device()

    os.makedirs(
        os.path.dirname(args.results)
        or ".",
        exist_ok=True,
    )

    fieldnames = [
        "attention_type",
        "seq_len",
        "batch_size",
        "d_model",
        "n_heads",
        "chunk_size",
        "iterations",
        "milliseconds",
        "tokens_per_second",
        "memory_mb",
        "device",
    ]

    with open(
        args.results,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for seq_len in args.seq_lens:
            for attention_type in [
                "reference",
                "sdpa",
                "chunked",
            ]:
                print(
                    f"\nBenchmarking "
                    f"{attention_type} "
                    f"at sequence length "
                    f"{seq_len}"
                )

                result = benchmark(
                    attention_type,
                    seq_len,
                    args.batch_size,
                    args.d_model,
                    args.n_heads,
                    args.chunk_size,
                    args.warmup,
                    args.iterations,
                    device,
                )

                writer.writerow(result)

                print(
                    f"{result['milliseconds']:.2f} ms/iter | "
                    f"{result['tokens_per_second']:,.0f} tok/s | "
                    f"memory={result['memory_mb']}"
                )


if __name__ == "__main__":
    main()