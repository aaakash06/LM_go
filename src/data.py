import torch
from torch.utils.data import Dataset, DataLoader


class LanguageModelDataset(Dataset):
    """
    Creates autoregressive language-modeling examples.

    Input:
        x[t : t+seq_len]

    Target:
        x[t+1 : t+seq_len+1]
    """

    def __init__(
        self,
        tokens,
        seq_len,
    ):
        if len(tokens) <= seq_len:
            raise ValueError(
                "Not enough tokens for the requested sequence length."
            )

        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[
            idx : idx + self.seq_len
        ]

        y = self.tokens[
            idx + 1 : idx + self.seq_len + 1
        ]

        return x, y


def make_dataloaders(
    token_ids,
    seq_len,
    batch_size,
    train_fraction=0.9,
):
    token_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
    )

    split = int(
        len(token_ids) * train_fraction
    )

    train_tokens = token_ids[:split]
    val_tokens = token_ids[split:]

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
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    return train_loader, val_loader