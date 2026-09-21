"""
Value-head model for back-search states.

The model takes a (B, 9, R, C) tensor (the channel layout from
corpus_features.state_to_tensor) and predicts max_descendant_depth as
a scalar per sample.

Design notes:
- Fully convolutional + global pool → handles any (R, C) at inference,
  not just the size it was trained on.  Batching still requires same-
  shape inputs within a batch.
- Residual blocks for cleanly stackable depth.
- BatchNorm after every conv; this is fine for our use (we don't run
  in single-sample online mode during inference yet; when we do, switch
  the model to eval() mode).
- Modest parameter count (~100k) — runs fast on MPS, leaves room to
  scale up later if signal is good.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_CHANNELS = 9  # matches corpus_features.CHANNELS


def _norm(ch):
    """Use GroupNorm instead of BatchNorm: numerically stable on MPS,
    no running statistics to drift across batches, equivalent quality
    at this model scale.  Group count picked so each group has ≥4 channels."""
    groups = max(1, ch // 8)
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = _norm(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = _norm(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class ValueNet(nn.Module):
    def __init__(self, in_channels=INPUT_CHANNELS, hidden=48, n_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            _norm(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # → (B, C, 1, 1)
            nn.Flatten(),                # → (B, C)
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, in_channels, R, C).  Returns (B,) predicted depth.
        h = self.stem(x)
        h = self.blocks(h)
        v = self.head(h)
        return v.squeeze(-1)


def best_device():
    """Pick the fastest available device (MPS on Apple Silicon, else CUDA, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
