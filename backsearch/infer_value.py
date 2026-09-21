#!/usr/bin/env python3
"""
Score a single state with a trained value-head checkpoint.

Two modes:
  1. Look up a state by (run_id, state_id) in the corpus:
       infer_value.py --checkpoint value.pt --run-id 1 --state-id 12345

  2. Score raw state JSON read from stdin (one state per line):
       echo '{"player_pos":4,"exit_pos":12,"committed_empty":1234,...}' | \
         infer_value.py --checkpoint value.pt --grid 6x6 --stdin

The stdin mode is the foundation for a sidecar inference server that a
future C-side `beam_score()` could call over a pipe.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

from corpus_features import state_to_tensor
from nn_value import ValueNet, best_device


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ValueNet().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def score(model, row, grid_rows, grid_cols, target_scale, device):
    t = state_to_tensor(row, grid_rows, grid_cols)
    x = torch.from_numpy(t).unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(x).item()
    return p * target_scale


def cmd_corpus(args):
    device = best_device()
    model, ckpt = load_model(args.checkpoint, device)
    target_scale = ckpt["target_scale"]
    conn = sqlite3.connect(args.corpus)
    row = conn.execute(
        """SELECT s.player_pos, s.exit_pos, s.committed_empty, s.blocks, s.holes,
                  s.max_descendant_depth, r.grid_rows, r.grid_cols
           FROM states s JOIN runs r USING (run_id)
           WHERE s.run_id=? AND s.state_id=?""",
        (args.run_id, args.state_id),
    ).fetchone()
    if not row:
        print(f"no such state run_id={args.run_id} state_id={args.state_id}",
              file=sys.stderr)
        return 1
    pp, ep, ce, bl, ho, true_mdd, grid_r, grid_c = row
    state = {"player_pos": pp, "exit_pos": ep, "committed_empty": int(ce),
             "blocks": bl or "", "holes": ho or ""}
    pred = score(model, state, grid_r, grid_c, target_scale, device)
    print(f"predicted max_descendant_depth: {pred:.2f}")
    print(f"true max_descendant_depth:      {true_mdd}")
    print(f"error:                          {pred - true_mdd:+.2f}")
    return 0


def cmd_stdin(args):
    device = best_device()
    model, ckpt = load_model(args.checkpoint, device)
    target_scale = ckpt["target_scale"]
    grid_rows, grid_cols = [int(x) for x in args.grid.split("x")]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
            pred = score(model, state, grid_rows, grid_cols, target_scale, device)
            print(f"{pred:.4f}")
            sys.stdout.flush()
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            print("nan")
            sys.stdout.flush()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus",
                    default=str(Path(__file__).resolve().parent / "corpus" / "corpus.sqlite"))
    sub = ap.add_subparsers(dest="cmd")

    sp = ap.add_argument_group("modes — choose one")
    sp_corpus = ap.add_argument
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--state-id", type=int)
    ap.add_argument("--stdin", action="store_true",
                    help="read JSON state objects from stdin, print one score per line")
    ap.add_argument("--grid", default="6x6", help="grid R x C (required for --stdin)")
    args = ap.parse_args()

    if args.stdin:
        return cmd_stdin(args)
    if args.run_id is not None and args.state_id is not None:
        return cmd_corpus(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
