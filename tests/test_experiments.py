import pytest
import torch

from mirrorshift.experiments import get_train_spec, list_train_specs


def test_default_train_spec_is_registered() -> None:
    assert "causal_lm" in list_train_specs()
    spec = get_train_spec("causal_lm")
    assert spec.name == "causal_lm"


def test_get_train_spec_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown train spec"):
        get_train_spec("missing_spec")


def test_default_loss_function_returns_scalar() -> None:
    spec = get_train_spec("causal_lm")
    logits = torch.randn(2, 4, 8)
    targets = torch.randint(0, 8, (2, 4))
    loss = spec.loss_fn(logits, targets)
    assert loss.ndim == 0
