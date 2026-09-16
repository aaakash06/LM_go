from dataclasses import dataclass

import torch

from src.data import make_dataloaders
from src.model import (
    TransformerConfig,
    TransformerLM,
    count_parameters,
)
from src.optimizer import AdamW
from src.tokenizer import CharacterTokenizer
from src.training import (
    get_device,
    set_seed,
)
from src.evaluation import evaluate, perplexity


TEXT = """
The transformer architecture processes sequences using attention.
Attention allows each position to interact with earlier positions.
A language model learns to predict the next token from context.

Small language models are useful for understanding how modern
sequence models work. By implementing the components directly,
we can inspect the computation and measure its behavior.

Research experiments should control variables carefully.
When comparing two implementations, we should keep the model,
dataset, optimizer, learning rate, number of steps, and random seed
fixed whenever possible.

Efficient attention implementations can change runtime and memory
behavior. A reference implementation is useful for correctness.
A fused implementation can use optimized kernels. A chunked
implementation can reduce the amount of attention memory materialized
at once.

The purpose of this project is to build a language model and then
experiment with its attention implementation.
""" * 300


def main():
    print("=" * 70)
    print("LANGUAGE MODEL RESEARCH PROJECT")
    print("=" * 70)

    seed = 42

    set_seed(seed)

    device = get_device()

    print(f"Device: {device}")

    tokenizer = CharacterTokenizer.train(
        TEXT
    )

    tokens = tokenizer.encode(TEXT)

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    seq_len = 128
    batch_size = 8

    train_loader, val_loader = (
        make_dataloaders(
            tokens,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    )

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        dropout=0.0,
        attention_type="sdpa",
    )

    model = TransformerLM(config).to(
        device
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
    )

    model.train()

    iterator = iter(train_loader)

    steps = 300

    for step in range(1, steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        _, loss = model(
            x,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        if step == 1 or step % 50 == 0:
            print(
                f"step {step:4d} | "
                f"loss {loss.item():.4f}"
            )

    val_loss = evaluate(
        model,
        val_loader,
        device,
        max_batches=20,
    )

    print()
    print(
        f"Validation loss: {val_loss:.4f}"
    )

    print(
        f"Perplexity: "
        f"{perplexity(val_loss):.2f}"
    )

    prompt = "The transformer"

    prompt_tokens = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        prompt_tokens,
        max_new_tokens=300,
        temperature=0.8,
        top_k=10,
    )

    print()
    print("=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)

    print(
        tokenizer.decode(
            generated[0].tolist()
        )
    )


if __name__ == "__main__":
    main()