import math
import os
import time

import torch

from .evaluation import evaluate, perplexity


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(
    step,
    max_steps,
    warmup_steps,
    max_lr,
    min_lr,
):
    if step < warmup_steps:
        return max_lr * (
            step / max(1, warmup_steps)
        )

    progress = (
        step - warmup_steps
    ) / max(
        1,
        max_steps - warmup_steps,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    coefficient = 0.5 * (
        1.0
        + math.cos(math.pi * progress)
    )

    return (
        min_lr
        + coefficient
        * (max_lr - min_lr)
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    config,
):
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return checkpoint["step"]


def train(
    model,
    optimizer,
    train_loader,
    config,
    device,
    val_loader=None,
    checkpoint_dir=None,
):
    model.train()

    iterator = iter(train_loader)

    start_time = time.perf_counter()

    history = []

    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        lr = cosine_lr(
            step,
            config.max_steps,
            config.warmup_steps,
            config.learning_rate,
            config.min_learning_rate,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()

        _, loss = model(
            x,
            y,
        )

        loss.backward()

        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip,
            )

        optimizer.step()

        if (
            step % config.log_every == 0
            or step == 1
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            tokens_processed = (
                step
                * x.numel()
            )

            tokens_per_second = (
                tokens_processed
                / max(elapsed, 1e-9)
            )

            print(
                f"step {step:5d} | "
                f"loss {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tokens_per_second:,.0f}"
            )

        if (
            val_loader is not None
            and (
                step % config.eval_every == 0
                or step == config.max_steps
            )
        ):
            val_loss = evaluate(
                model,
                val_loader,
                device,
                max_batches=config.eval_batches,
            )

            val_ppl = perplexity(val_loss)

            print(
                f"           validation loss "
                f"{val_loss:.4f} | "
                f"perplexity {val_ppl:.2f}"
            )

            history.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                    "learning_rate": lr,
                }
            )

        if (
            checkpoint_dir is not None
            and step % config.checkpoint_every == 0
        ):
            save_checkpoint(
                os.path.join(
                    checkpoint_dir,
                    f"step_{step}.pt",
                ),
                model,
                optimizer,
                step,
                config.__dict__,
            )

    return history