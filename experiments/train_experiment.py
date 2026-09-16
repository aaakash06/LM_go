import argparse
import csv
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data import make_dataloaders
from src.evaluation import evaluate, perplexity
from src.model import TransformerConfig, TransformerLM, count_parameters
from src.optimizer import AdamW
from src.tokenizer import CharacterTokenizer
from src.training import get_device, set_seed


CORPUS = """
The transformer architecture processes sequences using attention.
Attention allows each position to interact with earlier positions.
A language model learns to predict the next token from context.

Research in language modeling often depends on careful experiments.
A good experiment changes one variable while keeping other variables fixed.
Measurements such as validation loss, perplexity, runtime, throughput,
and memory can reveal different aspects of model behavior.

Efficient implementations can improve the amount of computation that
fits within a fixed hardware budget. However, implementation details
should be evaluated using controlled experiments rather than assumptions.

A small language model is useful for studying these ideas because the
entire system can be inspected, modified, tested, and benchmarked.
Researchers can compare reference implementations against optimized
implementations while keeping model parameters and training procedures
constant.

This repository studies causal self attention.
The reference implementation explicitly constructs attention scores.
The SDPA implementation delegates attention computation to PyTorch.
The chunked implementation processes query positions in smaller groups.

The goal is not to claim that one implementation is universally better.
The goal is to measure how implementation choices affect computation,
memory behavior, throughput, and language modeling quality.
"""


@dataclass
class TrainConfig:
    max_steps: int = 300
    warmup_steps: int = 30

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    weight_decay: float = 0.1

    grad_clip: float = 1.0

    log_every: int = 50
    eval_every: int = 100
    eval_batches: int = 20

    checkpoint_every: int = 100


def make_model(
    vocab_size,
    seq_len,
    attention_type,
    seed,
):
    set_seed(seed)

    config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        dropout=0.0,
        attention_type=attention_type,
        chunk_size=min(128, seq_len),
    )

    model = TransformerLM(config)

    return model, config


def run_experiment(
    attention_type,
    seq_len,
    seed,
    steps,
    batch_size,
    results_path,
):
    print()
    print("=" * 70)
    print(
        f"ATTENTION={attention_type} | "
        f"SEQ_LEN={seq_len} | "
        f"SEED={seed}"
    )
    print("=" * 70)

    device = get_device()

    # Make corpus large enough for many training windows.
    text = CORPUS * 300

    tokenizer = CharacterTokenizer.train(text)

    encoded = tokenizer.encode(text)

    train_loader, val_loader = make_dataloaders(
        encoded,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    model, model_config = make_model(
        tokenizer.vocab_size,
        seq_len,
        attention_type,
        seed,
    )

    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    train_config = TrainConfig(
        max_steps=steps
    )

    print(f"device: {device}")
    print(
        f"parameters: "
        f"{count_parameters(model):,}"
    )

    start = time.perf_counter()

    iterator = iter(train_loader)

    last_train_loss = None

    for step in range(1, steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        lr = (
            train_config.learning_rate
            if step <= train_config.warmup_steps
            else train_config.min_learning_rate
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()

        _, loss = model(x, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip,
        )

        optimizer.step()

        last_train_loss = loss.item()

        if (
            step == 1
            or step % train_config.log_every == 0
        ):
            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f}"
            )

    if device.type == "mps":
        torch.mps.synchronize()

    total_time = (
        time.perf_counter() - start
    )

    val_loss = evaluate(
        model,
        val_loader,
        device,
        max_batches=20,
    )

    val_ppl = perplexity(val_loss)

    total_training_tokens = (
        steps
        * batch_size
        * seq_len
    )

    tokens_per_second = (
        total_training_tokens
        / total_time
    )

    result = {
        "attention_type": attention_type,
        "seq_len": seq_len,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "parameters": count_parameters(model),
        "train_loss": last_train_loss,
        "val_loss": val_loss,
        "val_perplexity": val_ppl,
        "total_seconds": total_time,
        "tokens_per_second": tokens_per_second,
        "device": str(device),
    }

    print()
    print("RESULT")
    print(
        f"validation loss: {val_loss:.4f}"
    )
    print(
        f"perplexity:       {val_ppl:.2f}"
    )
    print(
        f"tokens/sec:       {tokens_per_second:,.0f}"
    )
    print(
        f"time:             {total_time:.2f}s"
    )

    os.makedirs(
        os.path.dirname(results_path) or ".",
        exist_ok=True,
    )

    file_exists = os.path.exists(
        results_path
    )

    with open(
        results_path,
        "a",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attention",
        choices=[
            "reference",
            "sdpa",
            "chunked",
        ],
        required=True,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--results",
        default="results/training_results.csv",
    )

    args = parser.parse_args()

    run_experiment(
        attention_type=args.attention,
        seq_len=args.seq_len,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        results_path=args.results,
    )