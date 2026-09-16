from __future__ import annotations

import torch

from src.data import make_dataloaders
from src.model import (
    TransformerConfig,
    TransformerLM,
)
from src.optimizer import AdamW
from src.tokenizer import (
    CharacterTokenizer,
)
from src.training import (
    TrainConfig,
    get_device,
    train,
)


# --------------------------------------------------
# 1. Tiny corpus
# --------------------------------------------------

TEXT = """
The language model learns to predict the next token.
A transformer reads a sequence of tokens and produces
a probability distribution over the vocabulary.

Language modeling is a simple objective with deep
consequences. Given a sequence of tokens, the model
tries to predict what comes next.

The transformer uses attention to mix information
between different positions in the sequence.

Attention allows every token to look at previous
tokens. Causal masking prevents the model from looking
into the future.

The model contains embeddings, attention layers,
feed forward networks, normalization, and an output
projection.

Training repeatedly shows the model examples and
updates its parameters using gradient descent.

The goal of this small experiment is not to build a
state of the art model. The goal is to understand the
complete language modeling pipeline.

We tokenize text, construct batches, run the model,
compute cross entropy loss, backpropagate gradients,
and update the parameters.

Once this works, the same architecture can be scaled
to larger datasets and larger models.
""" * 100


# --------------------------------------------------
# 2. Tokenizer
# --------------------------------------------------

tokenizer = CharacterTokenizer.train(
    TEXT
)

print(
    f"Vocabulary size: "
    f"{tokenizer.vocab_size}"
)


# --------------------------------------------------
# 3. Encode text
# --------------------------------------------------

token_ids = tokenizer.encode(
    TEXT
)

tokens = torch.tensor(
    token_ids,
    dtype=torch.long,
)

print(
    f"Number of tokens: "
    f"{len(tokens)}"
)


# --------------------------------------------------
# 4. Data
# --------------------------------------------------

SEQ_LEN = 128
BATCH_SIZE = 16

train_loader, val_loader = (
    make_dataloaders(
        tokens,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )
)


# --------------------------------------------------
# 5. Device
# --------------------------------------------------

device = get_device()

print(
    f"Using device: {device}"
)


# --------------------------------------------------
# 6. Model
# --------------------------------------------------

config = TransformerConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=SEQ_LEN,

    d_model=256,
    num_layers=4,
    num_heads=4,

    d_ff=1024,

    dropout=0.0,
)

model = TransformerLM(
    config
)

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Parameters: "
    f"{num_parameters:,}"
)


# --------------------------------------------------
# 7. Optimizer
# --------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
)


# --------------------------------------------------
# 8. Training
# --------------------------------------------------

train_config = TrainConfig(
    max_steps=1000,

    learning_rate=3e-4,
    min_learning_rate=3e-5,

    warmup_steps=50,

    eval_interval=100,
    eval_steps=20,

    log_interval=10,
)


train(
    model,
    optimizer,
    train_loader,
    config=train_config,
    val_loader=val_loader,
    device=device,
)


# --------------------------------------------------
# 9. Generate text
# --------------------------------------------------

model.eval()

prompt = "The transformer"

prompt_tokens = torch.tensor(
    [
        tokenizer.encode(
            prompt
        )
    ],
    dtype=torch.long,
    device=device,
)

generated = model.generate(
    prompt_tokens,
    max_new_tokens=300,
    temperature=0.8,
    top_k=10,
)

generated_text = tokenizer.decode(
    generated[0].tolist()
)

print()
print("=" * 70)
print("GENERATED TEXT")
print("=" * 70)
print()
print(generated_text)
print()