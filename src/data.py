from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class LanguageModelDataset(Dataset):
    """
    Dataset for autoregressive next-token prediction.

    Given:

        [a b c d e]

    and seq_len=4:

        input:  [a b c d]
        target: [b c d e]
    """

    def __init__(
        self,
        tokens: Tensor,
        seq_len: int,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError(
                "tokens must be a 1D tensor"
            )

        if len(tokens) <= seq_len:
            raise ValueError(
                "Not enough tokens for requested sequence length"
            )

        self.tokens = tokens.long()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor]:

        x = self.tokens[
            index : index + self.seq_len
        ]

        y = self.tokens[
            index + 1 : index + self.seq_len + 1
        ]

        return x, y


def make_dataloaders(
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    train_fraction: float = 0.9,
) -> tuple[DataLoader, DataLoader]:

    split = int(
        len(tokens) * train_fraction
    )

    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    train_dataset = LanguageModelDataset(
        train_tokens,
        seq_len,
    )

    val_dataset = LanguageModelDataset(
        val_tokens,
        seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )

    return train_loader, val_loader