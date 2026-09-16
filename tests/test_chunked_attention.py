import torch

from src.attention import ReferenceCausalSelfAttention
from src.flash_attention import SDPACAusalSelfAttention


def make_models():
    torch.manual_seed(42)

    reference = ReferenceCausalSelfAttention(
        d_model=64,
        n_heads=4,
    )

    sdpa = SDPACAusalSelfAttention(
        d_model=64,
        n_heads=4,
    )

    sdpa.load_state_dict(
        reference.state_dict()
    )

    return reference, sdpa


def test_reference_matches_sdpa():
    reference, sdpa = make_models()

    x = torch.randn(
        2,
        16,
        64,
    )

    reference.eval()
    sdpa.eval()

    with torch.no_grad():
        y_reference = reference(x)
        y_sdpa = sdpa(x)

    torch.testing.assert_close(
        y_reference,
        y_sdpa,
        rtol=1e-4,
        atol=1e-5,
    )


def test_attention_is_causal():
    reference, _ = make_models()

    reference.eval()

    x1 = torch.randn(
        1,
        8,
        64,
    )

    x2 = x1.clone()

    # Change a future token.
    x2[:, 7, :] += 100.0

    with torch.no_grad():
        y1 = reference(x1)
        y2 = reference(x2)

    # Earlier positions must not change.
    torch.testing.assert_close(
        y1[:, :7],
        y2[:, :7],
        rtol=1e-4,
        atol=1e-5,
    )