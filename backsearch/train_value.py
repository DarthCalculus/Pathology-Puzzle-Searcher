#!/usr/bin/env python3
"""
Train the value-head model on a corpus of harvested states.

The CLI:
  - filters the corpus to a single grid size (so batches share shape),
  - subsamples down to --max-states if the corpus is too large for RAM,
  - splits 80/10/10 train/val/test on a deterministic seed,
  - trains a ValueNet to regress max_descendant_depth (normalised),
  - reports val MAE and Spearman rank correlation each epoch,
  - saves the best (lowest val-MAE) checkpoint to --out.

Usage:
  train_value.py [--corpus PATH] [--grid 6x6] [--out value.pt]
                 [--epochs 30] [--batch 256] [--lr 3e-4]
                 [--max-states 200000] [--seed 0]
"""

import argparse
import math
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from corpus_features import state_to_tensor, CHANNELS
from nn_value import ValueNet, best_device

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus" / "corpus.sqlite"
DEFAULT_OUT = Path(__file__).resolve().parent / "checkpoints" / "value.pt"


def parse_grid(s):
    r, c = s.lower().split("x")
    return int(r), int(c)


def load_corpus(corpus_path, grid_rows, grid_cols, max_states, seed,
                target_mode="mdd"):
    """Pull rows from the corpus, build tensor X and label y arrays.

    Returns (X: (N, 9, R, C) float32, y: (N,) float32) plus a list of
    (run_id, state_id) tuples for reproducibility / debugging.

    target_mode:
      'mdd'   — predict max_descendant_depth directly.  High Spearman in
                aggregate but dominated by the trivial mdd ≥ depth signal,
                so the model under-discriminates within a single beam level.
      'extra' — predict (max_descendant_depth - depth).  Forces the model
                to learn "how much more depth can we squeeze out of this
                branch", which is what a beam ranker actually needs.
    """
    print(f"loading corpus from {corpus_path} (grid {grid_rows}x{grid_cols}, target={target_mode})...", file=sys.stderr)
    t0 = time.time()
    conn = sqlite3.connect(corpus_path)
    cur = conn.execute(
        """SELECT s.run_id, s.state_id, s.player_pos, s.exit_pos,
                  s.committed_empty, s.blocks, s.holes,
                  s.max_descendant_depth, s.depth
           FROM states s JOIN runs r USING (run_id)
           WHERE r.grid_rows=? AND r.grid_cols=?""",
        (grid_rows, grid_cols),
    )
    rows = cur.fetchall()
    conn.close()
    print(f"  raw row count: {len(rows):,}", file=sys.stderr)

    if max_states and len(rows) > max_states:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_states)
        print(f"  subsampled to {max_states:,}", file=sys.stderr)

    N = len(rows)
    X = np.zeros((N, CHANNELS, grid_rows, grid_cols), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)
    ids = []
    for i, (run_id, state_id, player_pos, exit_pos, ce, blocks, holes, mdd, depth) in enumerate(rows):
        row_dict = {
            "player_pos":      player_pos,
            "exit_pos":        exit_pos,
            "committed_empty": ce,
            "blocks":          blocks or "",
            "holes":           holes or "",
        }
        X[i] = state_to_tensor(row_dict, grid_rows, grid_cols)
        y[i] = (mdd - depth) if target_mode == "extra" else mdd
        ids.append((run_id, state_id))

    print(f"  features built in {time.time()-t0:.1f}s; X={X.shape} y_range=[{y.min():.0f}, {y.max():.0f}]",
          file=sys.stderr)
    return X, y, ids


class InMemoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def spearman(a, b):
    """Simple Spearman rank correlation (no scipy dep)."""
    a_rank = np.argsort(np.argsort(a))
    b_rank = np.argsort(np.argsort(b))
    a_rank = a_rank - a_rank.mean()
    b_rank = b_rank - b_rank.mean()
    denom = math.sqrt((a_rank**2).sum() * (b_rank**2).sum())
    return float((a_rank * b_rank).sum() / denom) if denom > 0 else 0.0


def evaluate(model, loader, device, target_scale):
    model.eval()
    all_preds = []
    all_targs = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y_pred = model(X).cpu().numpy() * target_scale
            all_preds.append(y_pred)
            all_targs.append(y.numpy())
    p = np.concatenate(all_preds)
    t = np.concatenate(all_targs)
    mae = float(np.abs(p - t).mean())
    rho = spearman(p, t)
    return mae, rho, p, t


def train(args):
    if args.device:
        device = torch.device(args.device)
    else:
        device = best_device()
    print(f"device: {device}", file=sys.stderr)
    grid_rows, grid_cols = parse_grid(args.grid)

    X, y, ids = load_corpus(args.corpus, grid_rows, grid_cols,
                            args.max_states, args.seed,
                            target_mode=args.target_mode)

    # Normalise target for stable training.
    target_scale = max(1.0, float(y.max()))
    y_norm = y / target_scale

    # Deterministic split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * 0.10)
    n_test = int(len(X) * 0.10)
    test_idx = perm[:n_test]
    val_idx  = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    print(f"split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}",
          file=sys.stderr)

    ds = InMemoryDataset(X, y_norm)
    train_loader = DataLoader(Subset(ds, train_idx.tolist()),
                              batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(Subset(InMemoryDataset(X, y), val_idx.tolist()),
                              batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(Subset(InMemoryDataset(X, y), test_idx.tolist()),
                              batch_size=args.batch, shuffle=False)

    model = ValueNet().to(device)
    if args.init_from:
        try:
            init = torch.load(args.init_from, map_location=device, weights_only=False)
            model.load_state_dict(init["state_dict"])
            print(f"warm-start from {args.init_from} "
                  f"(prior val_mae={init.get('best_val_mae', '?')})", file=sys.stderr)
        except Exception as e:
            print(f"warning: --init-from {args.init_from} failed ({e}); training from scratch",
                  file=sys.stderr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params", file=sys.stderr)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    mse_loss = nn.MSELoss()
    pair_loss = nn.MarginRankingLoss(margin=args.pair_margin)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            pred = model(X_b)
            if args.loss_mode == "pairwise":
                # Pair each sample with the one shifted by half-batch.  Random
                # pairing gives a stable mix of easy and hard pairs without
                # depth bucketing.  Skip pairs where the targets are too close
                # to bother distinguishing (within pair_margin/2).
                n = pred.shape[0]
                half = n // 2
                if half == 0:
                    continue
                p_a, p_b = pred[:half], pred[half:half*2]
                y_a, y_b2 = y_b[:half], y_b[half:half*2]
                # sign: +1 if y_a > y_b, -1 if y_a < y_b
                target = torch.where(y_a > y_b2,
                                     torch.ones_like(y_a),
                                     -torch.ones_like(y_a))
                # Mask out pairs whose y values are essentially equal
                keep = (y_a - y_b2).abs() > (args.pair_margin * 0.5)
                if keep.any():
                    loss = pair_loss(p_a[keep], p_b[keep], target[keep])
                else:
                    loss = pred.sum() * 0  # no-op
            else:
                loss = mse_loss(pred, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        val_mae, val_rho, _, _ = evaluate(model, val_loader, device, target_scale)
        # MPS occasionally emits a Python inf as loss.item() despite a finite
        # gradient — drop them from the average so the log stays readable.
        finite_losses = [v for v in train_losses if math.isfinite(v)]
        if not finite_losses:
            finite_losses = train_losses  # all inf — show inf, surface the issue
        avg_train_loss = sum(finite_losses) / len(finite_losses)
        elapsed = time.time() - t0
        print(f"epoch {epoch+1:2d}/{args.epochs}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_mae={val_mae:6.2f}  rho={val_rho:+.3f}  "
              f"({elapsed:.1f}s)", file=sys.stderr)
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss,
                        "val_mae": val_mae, "val_rho": val_rho})

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                "state_dict":   model.state_dict(),
                "target_scale": target_scale,
                "grid_rows":    grid_rows,
                "grid_cols":    grid_cols,
                "channels":     CHANNELS,
                "history":      history,
                "best_val_mae": best_mae,
                "best_val_rho": val_rho,
                "args":         vars(args),
            }, out_path)

    print(f"\nbest val MAE: {best_mae:.2f}  → {out_path}", file=sys.stderr)

    # Final test evaluation.
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_mae, test_rho, p, t = evaluate(model, test_loader, device, target_scale)
    print(f"TEST mae={test_mae:.2f}  rho={test_rho:+.3f}  "
          f"(n={len(p):,})", file=sys.stderr)

    # Print a small ranking sanity check: among top-100 predictions, how
    # many actually live in the deepest 10% of the labels?
    top_k = 100
    deep_threshold = np.quantile(t, 0.90)
    top_pred_idx = np.argsort(-p)[:top_k]
    hits = int((t[top_pred_idx] >= deep_threshold).sum())
    print(f"ranking check: of top-{top_k} predictions, {hits} land in top-10% "
          f"of true depths (random baseline = {top_k * 0.10:.0f})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--grid", default="6x6")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-states", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="cpu | mps | cuda (default: auto-detect best available)")
    ap.add_argument("--target-mode", default="mdd", choices=["mdd", "extra"],
                    help="target = max_descendant_depth (mdd) or mdd - depth (extra). "
                         "Use 'extra' for beam-rank training: removes the trivial "
                         "depth-correlation that inflates Spearman without improving "
                         "discrimination at a fixed depth.")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a previous .pt checkpoint (skips fresh init).  "
                         "Falls back to fresh init if the file is incompatible.")
    ap.add_argument("--loss-mode", default="mse", choices=["mse", "pairwise"],
                    help="mse: regress target (default).  pairwise: margin ranking on "
                         "same-batch pairs, learns to discriminate states by relative "
                         "subtree depth (better for beam ranking).")
    ap.add_argument("--pair-margin", type=float, default=0.02,
                    help="margin for pairwise loss (in normalised target units, ~1%% of max)")
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
