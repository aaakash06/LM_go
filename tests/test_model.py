import torch

from src.model import (
    TransformerConfig,
    TransformerLM,
    count_parameters,
)


def test_model_forward():
    config = TransformerConfig(
        vocab_size=50,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        attention_type="sdpa",
    )

    model = TransformerLM(config)

    x = torch.randint(
        0,
        50,
        (2, 32),
    )

    y = torch.randint(
        0,
        50,
        (2, 32),
    )

    logits, loss = model(x, y)

    assert logits.shape == (
        2,
        32,
        50,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_attention_variants_have_same_parameter_count():
    configs = []

    for attention_type in [
        "reference",
        "sdpa",
        "chunked",
    ]:
        configs.append(
            TransformerConfig(
                vocab_size=100,
                max_seq_len=32,
                d_model=64,
                n_layers=2,
                n_heads=4,
                d_ff=256,
                attention_type=attention_type,
            )
        )

    models = [
        TransformerLM(config)
        for config in configs
    ]

    counts = [
        count_parameters(model)
        for model in models
    ]

    assert len(set(counts)) == 1


def test_generation():
    config = TransformerConfig(
        vocab_size=30,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        attention_type="sdpa",
    )

    model = TransformerLM(config)

    x = torch.randint(
        0,
        30,
        (1, 5),
    )

    output = model.generate(
        x,
        max_new_tokens=5,
        temperature=1.0,
        top_k=5,
    )

    assert output.shape == (
        1,
        10,
    )