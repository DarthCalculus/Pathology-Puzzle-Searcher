#!/usr/bin/env python3
"""
Train a solver-surrogate model: given a back-search state, predict the
forward-solve length (the cost the exact `sokoban_solve` would report).

We only train on rows whose `forward_solve` is known — i.e., the
shortcut-pruned states (outcome='S') where the worker called the solver
and got an actual cutoff-bounded value.  Accepted states (outcome='A')
have forward_solve=-1 ("no shortcut <= depth-2"), which is an upper-tail
censored observation; the surrogate's job in those is implicitly
"predict at least depth-1", which is fine to defer for v1.

Output: a checkpoint usable by a future fast-path in `try_successor` —
predict, and if prediction is well below `depth-2`, prune without calling
the exact solver.  At inference time we measure false-negative rate of
that decision against the corpus.

Usage:
  train_surrogate.py [--corpus PATH] [--grid 6x6] [--out surrogate.pt]
                     [--epochs N] [--batch N] [--lr F] [--max-states N]
"""

import argparse
import math
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
DEFAULT_OUT    = Path(__file__).resolve().parent / "checkpoints" / "surrogate.pt"


def parse_grid(s):
    r, c = s.lower().split("x")
    return int(r), int(c)


def load_surrogate_corpus(corpus_path, grid_rows, grid_cols, max_states, seed,
                          task="regression"):
    """Load states for surrogate training.

    Regression mode: only outcome='S' rows (the worker recorded
      forward_solve for these).  y is the forward_solve value.

    Classification mode: outcome='S' (positive) + outcome='A' (negative).
      y is 1.0 for S, 0.0 for A.  This uses the natural binary signal
      already present in the corpus and removes the "no negative class"
      problem that broke the regression-only surrogate.

    Returns (X, y, depths, ids).
    """
    print(f"loading corpus from {corpus_path} (grid {grid_rows}x{grid_cols}, task={task})...", file=sys.stderr)
    t0 = time.time()
    conn = sqlite3.connect(corpus_path)
    if task == "classification":
        cur = conn.execute(
            """SELECT s.run_id, s.state_id, s.player_pos, s.exit_pos,
                      s.committed_empty, s.blocks, s.holes,
                      s.depth, s.forward_solve, s.outcome
               FROM states s JOIN runs r USING (run_id)
               WHERE r.grid_rows=? AND r.grid_cols=?
                 AND s.outcome IN ('S', 'A')""",
            (grid_rows, grid_cols),
        )
    else:
        cur = conn.execute(
            """SELECT s.run_id, s.state_id, s.player_pos, s.exit_pos,
                      s.committed_empty, s.blocks, s.holes,
                      s.depth, s.forward_solve, s.outcome
               FROM states s JOIN runs r USING (run_id)
               WHERE r.grid_rows=? AND r.grid_cols=?
                 AND s.outcome='S' AND s.forward_solve >= 0""",
            (grid_rows, grid_cols),
        )
    rows = cur.fetchall()
    conn.close()
    print(f"  raw rows: {len(rows):,}", file=sys.stderr)
    if task == "classification":
        n_s = sum(1 for r in rows if r[9] == "S")
        n_a = sum(1 for r in rows if r[9] == "A")
        print(f"    class balance: S(prune)={n_s:,}  A(accept)={n_a:,}", file=sys.stderr)

    if max_states and len(rows) > max_states:
        rng = random.Random(seed)
        if task == "classification":
            # Stratified: half S, half A (else class imbalance dominates loss).
            s_rows = [r for r in rows if r[9] == "S"]
            a_rows = [r for r in rows if r[9] == "A"]
            per_class = max_states // 2
            s_rows = rng.sample(s_rows, min(per_class, len(s_rows)))
            a_rows = rng.sample(a_rows, min(per_class, len(a_rows)))
            rows = s_rows + a_rows
            rng.shuffle(rows)
            print(f"  stratified-sampled to {len(rows):,} ({len(s_rows):,} S + {len(a_rows):,} A)",
                  file=sys.stderr)
        else:
            rows = rng.sample(rows, max_states)
            print(f"  subsampled to {max_states:,}", file=sys.stderr)

    N = len(rows)
    X = np.zeros((N, CHANNELS, grid_rows, grid_cols), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)
    depths = np.zeros(N, dtype=np.float32)
    ids = []
    for i, (run_id, state_id, pp, ep, ce, bl, ho, d, fs, oc) in enumerate(rows):
        state = {
            "player_pos":      pp,
            "exit_pos":        ep,
            "committed_empty": ce,
            "blocks":          bl or "",
            "holes":           ho or "",
        }
        X[i] = state_to_tensor(state, grid_rows, grid_cols)
        if task == "classification":
            y[i] = 1.0 if oc == "S" else 0.0
        else:
            y[i] = fs
        depths[i] = d
        ids.append((run_id, state_id))

    print(f"  features built in {time.time()-t0:.1f}s; "
          f"X={X.shape} y_range=[{y.min():.2f}, {y.max():.2f}] depth_range=[{depths.min():.0f}, {depths.max():.0f}]",
          file=sys.stderr)
    return X, y, depths, ids


class InMemoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def evaluate(model, loader, device, target_scale, task="regression"):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            raw = model(X).cpu().numpy()
            if task == "classification":
                preds.append(1.0 / (1.0 + np.exp(-raw)))  # sigmoid
            else:
                preds.append(raw * target_scale)
            targs.append(y.numpy())
    p = np.concatenate(preds); t = np.concatenate(targs)
    if task == "classification":
        # Accuracy at threshold 0.5; also report at 0.7, 0.9 (more conservative).
        return p, t
    mae = float(np.abs(p - t).mean())
    return mae, p, t


def cutoff_decision_metrics(preds, true_fs, depths, margin):
    """For each row, the worker actually pruned (forward_solve <= depth-2).
    If we used (pred + margin <= depth-2) as our cheap pruning rule:
      - true positive  : pred says prune, real solver also prunes — saved a solver call.
      - false negative : pred says NOT prune, real solver would prune — we wasted exploration.
      - false positive : pred says prune, real solver wouldn't — would wrongly drop a real state.

    Since this corpus only has *actually-pruned* rows (true label is always
    "pruned"), we can only measure pred-prune-rate: fraction of rows where
    our cheap rule agrees.  A complete eval requires accepted-state forward
    solves too (deferred).
    """
    cutoff_per_row = depths - 2.0
    cheap_says_prune = (preds + margin) <= cutoff_per_row
    return float(cheap_says_prune.mean())


def train(args):
    device = torch.device(args.device) if args.device else best_device()
    print(f"device: {device}", file=sys.stderr)
    grid_rows, grid_cols = parse_grid(args.grid)
    X, y, depths, ids = load_surrogate_corpus(args.corpus, grid_rows, grid_cols,
                                              args.max_states, args.seed,
                                              task=args.task)

    if args.task == "classification":
        target_scale = 1.0  # y is already in [0,1]
        y_norm = y
    else:
        target_scale = max(1.0, float(y.max()))
        y_norm = y / target_scale

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * 0.10)
    n_test = int(len(X) * 0.10)
    test_idx = perm[:n_test]
    val_idx  = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    print(f"split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}",
          file=sys.stderr)

    ds_train = InMemoryDataset(X, y_norm)
    ds_eval  = InMemoryDataset(X, y)  # raw scale for eval
    train_loader = DataLoader(Subset(ds_train, train_idx.tolist()),
                              batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(Subset(ds_eval, val_idx.tolist()),
                              batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(Subset(ds_eval, test_idx.tolist()),
                              batch_size=args.batch, shuffle=False)

    model = ValueNet().to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params", file=sys.stderr)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.task == "classification":
        loss_fn = nn.BCEWithLogitsLoss()  # numerically stable; logits in/labels in
    else:
        loss_fn = nn.MSELoss()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = loss_fn(model(Xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        finite = [v for v in losses if math.isfinite(v)]
        avg = sum(finite)/len(finite) if finite else float("nan")
        if args.task == "classification":
            p, t = evaluate(model, val_loader, device, target_scale, task="classification")
            # Accuracy at threshold 0.5
            pred_cls = (p >= 0.5).astype(np.float32)
            acc = float((pred_cls == t).mean())
            # Precision @ 0.5: of predicted-prune, how often is actually-prune
            tp = float(((pred_cls == 1) & (t == 1)).sum())
            fp = float(((pred_cls == 1) & (t == 0)).sum())
            fn = float(((pred_cls == 0) & (t == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metric = 1.0 - acc  # used for "best" tracking; lower is better
            print(f"epoch {epoch+1:2d}/{args.epochs}  train_loss={avg:.5f}  "
                  f"val_acc={acc:.3f}  prec={prec:.3f}  recall={recall:.3f}  "
                  f"({time.time()-t0:.1f}s)", file=sys.stderr)
        else:
            val_mae, _, _ = evaluate(model, val_loader, device, target_scale)
            metric = val_mae
            print(f"epoch {epoch+1:2d}/{args.epochs}  train_loss={avg:.5f}  "
                  f"val_mae={val_mae:5.2f}  ({time.time()-t0:.1f}s)", file=sys.stderr)

        if metric < best_mae:
            best_mae = metric
            torch.save({
                "state_dict":   model.state_dict(),
                "target_scale": target_scale,
                "grid_rows":    grid_rows,
                "grid_cols":    grid_cols,
                "channels":     CHANNELS,
                "task":         args.task,
                "best_val_metric": best_mae,
                "args":         vars(args),
            }, out_path)

    # Final test analysis.
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    if args.task == "classification":
        p, t = evaluate(model, test_loader, device, target_scale, task="classification")
        print(f"\nTEST n={len(p):,}", file=sys.stderr)
        for thresh in [0.5, 0.7, 0.9, 0.95]:
            pred_cls = (p >= thresh).astype(np.float32)
            tp = float(((pred_cls == 1) & (t == 1)).sum())
            fp = float(((pred_cls == 1) & (t == 0)).sum())
            tn = float(((pred_cls == 0) & (t == 0)).sum())
            fn = float(((pred_cls == 0) & (t == 1)).sum())
            acc  = (tp + tn) / max(1.0, tp + tn + fp + fn)
            prec = tp / max(1.0, tp + fp)
            rec  = tp / max(1.0, tp + fn)
            # At inference, FP = wrongly-pruned-an-accept (catastrophic).
            # We want low FP rate even if recall suffers.
            print(f"  threshold={thresh:.2f}  acc={acc:.3f}  prec(prune)={prec:.3f}  "
                  f"recall(prune)={rec:.3f}  "
                  f"fp_rate(={fp:.0f}/{fp+tn:.0f})={fp/max(1.0,fp+tn):.3%}",
                  file=sys.stderr)
        print("Interpretation:", file=sys.stderr)
        print("  threshold = how confident the surrogate must be to fast-prune.", file=sys.stderr)
        print("  fp_rate = of true ACCEPTS, what fraction does the model wrongly prune.", file=sys.stderr)
        print("  Choose threshold giving acceptably-low fp_rate (this is the safety bound).", file=sys.stderr)
    else:
        test_mae, test_preds, test_targs = evaluate(model, test_loader, device, target_scale)
        print(f"\nTEST mae={test_mae:.2f}  n={len(test_preds):,}", file=sys.stderr)
        test_depths = depths[test_idx]
        print(f"\nCheap-cutoff decision rule: (pred + margin) <= depth - 2", file=sys.stderr)
        print(f"  every test row is actually-pruned, so:", file=sys.stderr)
        print(f"  margin=0   agree rate: {cutoff_decision_metrics(test_preds, test_targs, test_depths, 0.0):.1%}",
              file=sys.stderr)
        print(f"  margin=2   agree rate: {cutoff_decision_metrics(test_preds, test_targs, test_depths, 2.0):.1%}",
              file=sys.stderr)
        print(f"  margin=5   agree rate: {cutoff_decision_metrics(test_preds, test_targs, test_depths, 5.0):.1%}",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--grid", default="6x6")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-states", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--task", default="regression", choices=["regression", "classification"],
                    help="regression: predict forward_solve (needs known label, only outcome='S').  "
                         "classification: predict P(prune); uses both outcome='S' (label=1) and "
                         "outcome='A' (label=0).  Classification fixes the no-negatives problem.")
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
