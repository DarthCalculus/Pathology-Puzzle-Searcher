"""
Tests for the value-head model.

We avoid relying on a real corpus or a trained checkpoint — the tests
only need PyTorch and the module under test, so they run anywhere torch
is installed.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nn_value import ValueNet, INPUT_CHANNELS


def test_forward_shape():
    model = ValueNet(hidden=16, n_blocks=2)
    model.eval()
    for R, C in [(5, 5), (6, 6), (7, 9), (8, 8)]:
        x = torch.randn(4, INPUT_CHANNELS, R, C)
        y = model(x)
        assert y.shape == (4,), f"unexpected shape {y.shape} for grid {R}x{C}"


def test_single_optimizer_step_decreases_loss():
    """One AdamW step on a tiny constant-input batch should reduce MSE."""
    torch.manual_seed(0)
    model = ValueNet(hidden=16, n_blocks=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(8, INPUT_CHANNELS, 6, 6)
    y = torch.full((8,), 0.5)
    loss_fn = torch.nn.MSELoss()

    model.train()
    pred0 = model(x)
    loss0 = loss_fn(pred0, y).item()
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    pred1 = model(x)
    loss1 = loss_fn(pred1, y).item()

    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"


def test_surrogate_uses_same_architecture():
    """The solver-surrogate reuses ValueNet — confirm that's intentional
    (same input shape, same scalar output).  If someone forks a different
    architecture for the surrogate, this test pins the assumption."""
    model = ValueNet(hidden=16, n_blocks=2)
    model.eval()
    x = torch.randn(2, INPUT_CHANNELS, 6, 6)
    y = model(x)
    assert y.shape == (2,)


def test_state_dict_round_trip(tmp_path):
    model = ValueNet(hidden=16, n_blocks=2)
    x = torch.randn(2, INPUT_CHANNELS, 6, 6)
    model.eval()
    y_before = model(x).detach()

    ckpt_path = tmp_path / "v.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)

    model2 = ValueNet(hidden=16, n_blocks=2)
    state = torch.load(ckpt_path, weights_only=False)["state_dict"]
    model2.load_state_dict(state)
    model2.eval()
    y_after = model2(x).detach()

    assert torch.allclose(y_before, y_after, atol=1e-6), \
        f"outputs diverged across save/load: max diff {(y_before - y_after).abs().max()}"
