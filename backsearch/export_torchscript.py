#!/usr/bin/env python3
"""
Export a trained ValueNet checkpoint to TorchScript for libtorch C++.

We use torch.jit.trace with a fixed example input (size = grid in the
checkpoint).  Since the model is fully convolutional + global pool,
the traced module also accepts other (R, C) sizes at inference time —
but for safety we save the canonical grid so the C side knows what to
expect.

Alongside the .ts we write a small `.meta.json` carrying:
  - target_scale (predictions × target_scale → real depth units)
  - canonical grid (rows, cols)
  - channels
This is exactly the info the C side needs to use the model.

Usage:
  export_torchscript.py --checkpoint value_v2.pt --out value_v2.ts
"""

import argparse
import json
from pathlib import Path

import torch

from nn_value import ValueNet, INPUT_CHANNELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=None,
                    help="canonical grid rows (default: from checkpoint)")
    ap.add_argument("--cols", type=int, default=None,
                    help="canonical grid cols (default: from checkpoint)")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = args.rows or ckpt["grid_rows"]
    cols = args.cols or ckpt["grid_cols"]
    target_scale = float(ckpt["target_scale"])

    model = ValueNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    example = torch.zeros(1, INPUT_CHANNELS, rows, cols)
    traced = torch.jit.trace(model, example)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))

    meta = {
        "target_scale": target_scale,
        "rows":         rows,
        "cols":         cols,
        "channels":     INPUT_CHANNELS,
        "source_ckpt":  str(args.checkpoint),
        "task":         ckpt.get("task", "regression"),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Round-trip self-check.
    reload = torch.jit.load(str(out_path))
    reload.eval()
    with torch.no_grad():
        y1 = model(example).item()
        y2 = reload(example).item()
    print(f"wrote {out_path} ({out_path.stat().st_size:,} B) and {meta_path.name}")
    print(f"sanity: original={y1:.6f}  reloaded={y2:.6f}  diff={abs(y1-y2):.2e}")


if __name__ == "__main__":
    main()
