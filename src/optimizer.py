from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    AdamW optimizer.

    Implements decoupled weight decay rather than adding the L2 penalty
    directly to the gradient.
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")

        if eps < 0.0:
            raise ValueError("eps must be non-negative")

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("beta2 must be in [0, 1)")

        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                if parameter.grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients."
                    )

                grad = parameter.grad

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        parameter
                    )

                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter
                    )

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # First moment.
                exp_avg.mul_(beta1)
                exp_avg.add_(grad, alpha=1.0 - beta1)

                # Second moment.
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )

                # Bias correction.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                step_size = (
                    lr
                    / bias_correction1
                    / bias_correction2**0.5
                )

                denominator = exp_avg_sq.sqrt().add_(eps)

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

                # Decoupled weight decay.
                if weight_decay != 0:
                    parameter.mul_(
                        1.0 - lr * weight_decay
                    )

        return loss