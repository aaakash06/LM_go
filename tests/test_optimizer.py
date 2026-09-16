import torch

from src.optimizer import AdamW


def test_adamw_reduces_simple_loss():
    torch.manual_seed(42)

    parameter = torch.nn.Parameter(
        torch.tensor([10.0])
    )

    optimizer = AdamW(
        [parameter],
        lr=0.1,
        weight_decay=0.0,
    )

    initial = parameter.item()

    for _ in range(100):
        optimizer.zero_grad()

        loss = (parameter - 3.0).pow(2).sum()

        loss.backward()

        optimizer.step()

    final = parameter.item()

    assert abs(final - 3.0) < abs(
        initial - 3.0
    )