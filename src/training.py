from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class TrainConfig:
    max_steps: int = 10_000

    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5

    warmup_steps: int = 500

    grad_clip: float = 1.0

    eval_interval: int = 500
    eval_steps: int = 100

    log_interval: int = 10

    checkpoint_dir: str = "checkpoints"


def get_device() -> torch.device:
    """
    Select the best available device.

    Apple Silicon:
        MPS

    NVIDIA:
        CUDA

    Otherwise:
        CPU
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_lr(
    step: int,
    config: TrainConfig,
) -> float:
    """
    Linear warmup followed by cosine decay.
    """

    if step < config.warmup_steps:
        return config.learning_rate * (
            step + 1
        ) / config.warmup_steps

    if step >= config.max_steps:
        return config.min_learning_rate

    progress = (
        step - config.warmup_steps
    ) / (
        config.max_steps - config.warmup_steps
    )

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return (
        config.min_learning_rate
        + cosine
        * (
            config.learning_rate
            - config.min_learning_rate
        )
    )


def set_learning_rate(
    optimizer: Optimizer,
    lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Iterable,
    *,
    device: torch.device,
    max_steps: int = 100,
) -> float:
    """
    Compute mean cross-entropy loss over evaluation batches.
    """

    model.eval()

    total_loss = 0.0
    num_steps = 0

    for batch in dataloader:
        if num_steps >= max_steps:
            break

        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)

        _, loss = model(
            inputs,
            targets,
        )

        if loss is None:
            raise RuntimeError(
                "Model did not return a loss."
            )

        total_loss += loss.item()
        num_steps += 1

    model.train()

    if num_steps == 0:
        return float("nan")

    return total_loss / num_steps


def save_checkpoint(
    path: str | os.PathLike,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    best_val_loss: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str | os.PathLike,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return checkpoint


def train(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: Iterable,
    *,
    config: TrainConfig,
    val_loader: Iterable | None = None,
    device: torch.device | None = None,
) -> None:
    """
    Basic autoregressive LM training loop.

    Expected batch format:

        inputs, targets

    where both tensors have shape [B, T].
    """

    if device is None:
        device = get_device()

    model.to(device)
    model.train()

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True,
    )

    train_iterator = iter(train_loader)

    best_val_loss = float("inf")

    start_time = time.time()

    for step in range(config.max_steps):
        try:
            inputs, targets = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            inputs, targets = next(train_iterator)

        inputs = inputs.to(device)
        targets = targets.to(device)

        lr = get_lr(
            step,
            config,
        )

        set_learning_rate(
            optimizer,
            lr,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        _, loss = model(
            inputs,
            targets,
        )

        if loss is None:
            raise RuntimeError(
                "Model did not return a loss."
            )

        loss.backward()

        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )

        optimizer.step()

        if (
            step % config.log_interval == 0
            or step == config.max_steps - 1
        ):
            elapsed = time.time() - start_time

            print(
                f"step={step:6d} "
                f"loss={loss.item():.4f} "
                f"lr={lr:.2e} "
                f"time={elapsed:.1f}s"
            )

        if (
            val_loader is not None
            and (
                step % config.eval_interval == 0
                or step == config.max_steps - 1
            )
        ):
            val_loss = evaluate(
                model,
                val_loader,
                device=device,
                max_steps=config.eval_steps,
            )

            perplexity = math.exp(
                min(val_loss, 20.0)
            )

            print(
                f"[eval] step={step:6d} "
                f"loss={val_loss:.4f} "
                f"ppl={perplexity:.2f}"
            )

            save_checkpoint(
                Path(config.checkpoint_dir)
                / "latest.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                save_checkpoint(
                    Path(config.checkpoint_dir)
                    / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                )