#!/usr/bin/env python3
"""
Self-play orchestrator.

Loops:
  1. Search with current value model at blend α, dump --harvest binary log.
  2. Ingest harvest into the central corpus.
  3. Retrain (warm-started from current model) on the updated corpus.
  4. Export new model to TorchScript.
  5. Measure: run a fixed-config search and record best depth.
  6. Swap current → new, repeat.

The key bet: training data harvested by the current NN-guided search
teaches the next model what's *actually* promising under NN guidance.
Each iteration, the NN's predictions should become more useful for the
search regime it operates in.

Artifacts:
  selfplay_runs/<timestamp>/
    iter_0_seed/                # original checkpoint we started from
    iter_1/
      harvest.bin.gz            # the harvest from this iteration
      value.pt, value.ts        # checkpoints
      train.log                 # training stdout
      metrics.json              # depth, train MAE, etc.
    iter_2/ ...

Usage:
  selfplay.py [--seed-model checkpoints/value_v3.ts]
              [--iterations 2]
              [--harvest-seconds 60] [--harvest-blend 0.3]
              [--eval-seconds 30] [--eval-blends 0.0,0.5]
              [--train-epochs 5] [--max-states 100000]
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
WORKER = REPO / "backsearch_worker"
CORPUS = REPO / "corpus" / "corpus.sqlite"


def shell(cmd, **kw):
    """Run a command, stream output to stdout, return CompletedProcess."""
    print("$", " ".join(str(c) for c in cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, **kw)


def shell_capture(cmd, **kw):
    """Run a command, capture stdout+stderr, return (rc, combined output)."""
    print("$", " ".join(str(c) for c in cmd), file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def parse_best_depth(text):
    """Find the worker's reported best depth line."""
    m = re.search(r"best depth:\s+(\d+)", text)
    return int(m.group(1)) if m else None


def parse_train_mae(text):
    """Pick the final TEST mae from train_value.py output."""
    m = re.search(r"TEST mae=([\d.]+)", text)
    return float(m.group(1)) if m else None


def harvest_step(iter_dir, grid, exit_pos, beam, seconds, blend, model_ts):
    out = iter_dir / "harvest.bin.gz"
    cmd = [
        str(WORKER),
        "--grid", grid,
        "--exit", str(exit_pos),
        "--allow-exit-transit", "--two-tables",
        "--time", str(seconds),
        "--beam", str(beam),
        "--harvest", str(out),
        "--nn-value-model", str(model_ts),
        "--nn-blend", str(blend),
    ]
    rc, output = shell_capture(cmd)
    (iter_dir / "harvest.log").write_text(output)
    if rc != 0:
        raise RuntimeError(f"harvest failed (rc={rc})")
    best = parse_best_depth(output)
    print(f"  harvest reached depth {best}", file=sys.stderr)
    return out, best


def ingest_step(harvest_path, hostname):
    cmd = ["python3", str(REPO / "harvest_ingest.py"),
           str(harvest_path),
           "--hostname", hostname,
           "--exit-reason", "time-cap",
           "--skip-index"]
    rc, output = shell_capture(cmd)
    if rc != 0:
        raise RuntimeError(f"ingest failed (rc={rc}):\n{output[-1000:]}")
    print(f"  ingest done", file=sys.stderr)


def train_step(iter_dir, grid, max_states, epochs, init_from, seed):
    out_pt = iter_dir / "value.pt"
    cmd = ["python3", str(REPO / "train_value.py"),
           "--grid", grid,
           "--epochs", str(epochs),
           "--batch", "512",
           "--max-states", str(max_states),
           "--seed", str(seed),
           "--out", str(out_pt),
           "--device", "cpu",
           "--target-mode", "extra"]
    if init_from:
        cmd += ["--init-from", str(init_from)]
    rc, output = shell_capture(cmd)
    (iter_dir / "train.log").write_text(output)
    if rc != 0:
        raise RuntimeError(f"train failed (rc={rc}):\n{output[-1000:]}")
    mae = parse_train_mae(output)
    print(f"  train done, TEST mae={mae}", file=sys.stderr)
    return out_pt, mae


def export_step(iter_dir):
    pt = iter_dir / "value.pt"
    ts = iter_dir / "value.ts"
    shell(["python3", str(REPO / "export_torchscript.py"),
           "--checkpoint", str(pt),
           "--out", str(ts)])
    return ts


def eval_step(grid, exit_pos, beam, seconds, blend, model_ts):
    cmd = [
        str(WORKER),
        "--grid", grid,
        "--exit", str(exit_pos),
        "--allow-exit-transit", "--two-tables",
        "--time", str(seconds),
        "--beam", str(beam),
        "--nn-value-model", str(model_ts),
        "--nn-blend", str(blend),
    ]
    rc, output = shell_capture(cmd)
    if rc != 0:
        return None
    return parse_best_depth(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-model", default=str(REPO / "checkpoints" / "value_v3.ts"),
                    help="initial TorchScript model (iter 0 / seed)")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--grid", default="6x6")
    ap.add_argument("--exit", type=int, default=0)
    ap.add_argument("--beam", type=int, default=1000)
    ap.add_argument("--harvest-seconds", type=int, default=60)
    ap.add_argument("--harvest-blend", type=float, default=0.3)
    ap.add_argument("--eval-seconds", type=int, default=30)
    ap.add_argument("--eval-blends", default="0.0,0.5",
                    help="comma-separated blend values to eval at every iter")
    ap.add_argument("--train-epochs", type=int, default=5)
    ap.add_argument("--max-states", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eval_blends = [float(x) for x in args.eval_blends.split(",")]

    run_dir = REPO / "selfplay_runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_iter = run_dir / "iter_0_seed"
    seed_iter.mkdir(exist_ok=True)
    shutil.copy(args.seed_model, seed_iter / "value.ts")
    # meta.json sidecar
    meta_src = Path(args.seed_model).with_suffix(Path(args.seed_model).suffix + ".meta.json")
    if meta_src.exists():
        shutil.copy(meta_src, seed_iter / "value.ts.meta.json")

    current_ts = seed_iter / "value.ts"
    # value.pt is the warm-start source for training; map seed → value_v3.pt
    seed_pt = Path(args.seed_model).with_suffix(".pt")
    if not seed_pt.exists():
        seed_pt = REPO / "checkpoints" / "value_v3.pt"
    current_pt = seed_pt if seed_pt.exists() else None

    history = []

    # Seed-iter measurement (so we can compare improvement against the
    # starting model).
    print("\n=== ITER 0 (seed) ===", file=sys.stderr)
    seed_evals = {}
    for b in eval_blends:
        d = eval_step(args.grid, args.exit, args.beam, args.eval_seconds, b, current_ts)
        seed_evals[b] = d
        print(f"  eval blend={b}: depth={d}", file=sys.stderr)
    history.append({"iter": 0, "evals": seed_evals, "train_mae": None,
                    "harvest_depth": None, "model": str(current_ts)})

    for it in range(1, args.iterations + 1):
        print(f"\n=== ITER {it} ===", file=sys.stderr)
        iter_dir = run_dir / f"iter_{it}"
        iter_dir.mkdir(exist_ok=True)

        # 1. Harvest with current model
        h_path, h_depth = harvest_step(iter_dir, args.grid, args.exit, args.beam,
                                       args.harvest_seconds, args.harvest_blend,
                                       current_ts)
        # 2. Ingest
        ingest_step(h_path, hostname=f"selfplay-iter-{it}-{int(time.time())}")
        # 3. Train (warm-start)
        new_pt, mae = train_step(iter_dir, args.grid, args.max_states,
                                 args.train_epochs, current_pt, args.seed + it)
        # 4. Export
        new_ts = export_step(iter_dir)
        # 5. Eval at all blends
        evals = {}
        for b in eval_blends:
            d = eval_step(args.grid, args.exit, args.beam, args.eval_seconds, b, new_ts)
            evals[b] = d
            print(f"  eval blend={b}: depth={d}", file=sys.stderr)

        history.append({"iter": it, "evals": evals, "train_mae": mae,
                        "harvest_depth": h_depth, "model": str(new_ts)})
        with open(iter_dir / "metrics.json", "w") as f:
            json.dump(history[-1], f, indent=2)

        # Swap
        current_ts = new_ts
        current_pt = new_pt

        # Running summary
        print(f"\nSummary so far:", file=sys.stderr)
        for h in history:
            ev = ", ".join(f"α={b}:{h['evals'].get(b)}" for b in eval_blends)
            print(f"  iter {h['iter']}: train_mae={h['train_mae']}  harvest={h['harvest_depth']}  {ev}", file=sys.stderr)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nrun complete: {run_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
