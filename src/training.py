from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class TrainConfig:

    max_steps: int = 1000

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    warmup_steps: int = 50

    grad_clip: float = 1.0

    eval_interval: int = 100
    eval_steps: int = 20

    log_interval: int = 10

    checkpoint_dir: str = "checkpoints"


def get_device() -> torch.device:

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_lr(
    step: int,
    config: TrainConfig,
) -> float:

    if step < config.warmup_steps:
        return (
            config.learning_rate
            * (step + 1)
            / config.warmup_steps
        )

    if step >= config.max_steps:
        return config.min_learning_rate

    progress = (
        step - config.warmup_steps
    ) / (
        config.max_steps
        - config.warmup_steps
    )

    cosine = 0.5 * (
        1 + math.cos(
            math.pi * progress
        )
    )

    return (
        config.min_learning_rate
        + cosine
        * (
            config.learning_rate
            - config.min_learning_rate
        )
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    max_steps: int,
) -> float:

    model.eval()

    total_loss = 0.0

    for step, (inputs, targets) in enumerate(
        loader
    ):

        if step >= max_steps:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        _, loss = model(
            inputs,
            targets,
        )

        total_loss += loss.item()

    model.train()

    return total_loss / min(
        max_steps,
        step + 1,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
) -> None:

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: Iterable,
    *,
    config: TrainConfig,
    val_loader: Iterable | None = None,
    device: torch.device | None = None,
) -> None:

    if device is None:
        device = get_device()

    model.to(device)

    model.train()

    train_iterator = iter(
        train_loader
    )

    start_time = time.time()

    for step in range(
        config.max_steps
    ):

        try:
            inputs, targets = next(
                train_iterator
            )

        except StopIteration:
            train_iterator = iter(
                train_loader
            )

            inputs, targets = next(
                train_iterator
            )

        inputs = inputs.to(device)
        targets = targets.to(device)

        lr = get_lr(
            step,
            config,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            inputs,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )

        optimizer.step()

        if (
            step % config.log_interval == 0
        ):
            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"step {step:5d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"time {elapsed:.1f}s"
            )

        if (
            val_loader is not None
            and step > 0
            and step
            % config.eval_interval
            == 0
        ):

            val_loss = evaluate(
                model,
                val_loader,
                device=device,
                max_steps=config.eval_steps,
            )

            perplexity = math.exp(
                min(val_loss, 20)
            )

            print(
                f"           "
                f"val loss {val_loss:.4f} | "
                f"ppl {perplexity:.2f}"
            )

            save_checkpoint(
                Path(
                    config.checkpoint_dir
                ) / "latest.pt",
                model,
                optimizer,
                step,
            )