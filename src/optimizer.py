import torch


class AdamW(torch.optim.Optimizer):
    """
    Small from-scratch AdamW implementation.

    This is intentionally explicit for educational/research purposes.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        super().__init__(
            params,
            defaults,
        )

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

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients."
                    )

                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        param
                    )

                    state["exp_avg_sq"] = torch.zeros_like(
                        param
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay.
                if weight_decay != 0:
                    param.mul_(
                        1.0 - lr * weight_decay
                    )

                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1.0 - beta1,
                )

                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )

                bias_correction1 = (
                    1.0 - beta1 ** step
                )

                bias_correction2 = (
                    1.0 - beta2 ** step
                )

                step_size = (
                    lr
                    / bias_correction1
                )

                denom = (
                    exp_avg_sq.sqrt()
                    / bias_correction2**0.5
                ).add_(eps)

                param.addcdiv_(
                    exp_avg,
                    denom,
                    value=-step_size,
                )

        return loss