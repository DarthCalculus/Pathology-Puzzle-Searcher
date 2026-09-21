#!/usr/bin/env python3
"""
Load a harvest file from backsearch_worker (binary BSH1 or legacy CSV),
compute derived fields, and write a SQLite database for downstream
querying / training-data assembly.

Derived per-state fields:
  - max_descendant_depth: deepest depth reached anywhere in this state's subtree.
    (A leaf — pruned by shortcut/dedup/cap, or just one whose subtree wasn't
    fully explored before time-cap — has max_descendant_depth = its own depth.)
    This is the value-head training target.
  - is_leaf: 1 if no other row in this run lists this state as a parent.
  - n_descendants: count of states in this subtree (including self).

The binary format is documented in harvest_format.h.  CSV format is
the legacy comma-separated form; format is auto-detected by the first
four bytes ("BSH1" → binary).

Usage:
  harvest_load.py <harvest.bin|harvest.csv> [--db out.sqlite] [--no-db]
"""

import argparse
import gzip
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HARVEST_MAGIC = b"BSH1"

HEADER_DTYPE = np.dtype([
    ("magic",           "4S"),
    ("version",         "<u2"),
    ("grid_rows",       "u1"),
    ("grid_cols",       "u1"),
    ("exit_pos",        "<i2"),
    ("flags",           "<u2"),
    ("reserved_a",      "<u4"),
    ("started_at_unix", "<u8"),
    ("code_sha",        "20u1"),
    ("argv_blob_len",   "<u4"),
    ("reserved_b",      "16u1"),
])
assert HEADER_DTYPE.itemsize == 64, f"header dtype is {HEADER_DTYPE.itemsize} B, expected 64"

RECORD_DTYPE = np.dtype([
    ("state_id",        "<u8"),
    ("parent_id",       "<i8"),
    ("canonical_key",   "<u8"),
    ("depth",           "<i4"),
    ("forward_solve",   "<i4"),
    ("outcome",         "u1"),
    ("nblocks",         "u1"),
    ("nholes",          "u1"),
    ("player_pos",      "u1"),
    ("exit_pos",        "<i2"),
    ("_pad",            "2u1"),
    ("committed_empty", "<u8"),
    ("block_pos",       "32i1"),
    ("block_mask",      "32u1"),
    ("hole_pos",        "32i1"),
])
assert RECORD_DTYPE.itemsize == 144, f"record dtype is {RECORD_DTYPE.itemsize} B, expected 144"


def _blocks_text(pos_arr, mask_arr, n):
    return ":".join(f"{int(pos_arr[i])}.{int(mask_arr[i])}" for i in range(n))


def _holes_text(pos_arr, n):
    return ":".join(f"{int(pos_arr[i])}" for i in range(n))


def _open_maybe_gz(path):
    """Open path, transparently decompressing if gzip-magic is detected."""
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return open(path, "rb")


def load_binary(path):
    """Return (df, header_dict, argv_list)."""
    t0 = time.time()
    with _open_maybe_gz(path) as f:
        data = f.read()
    if len(data) < 64 or data[:4] != HARVEST_MAGIC:
        raise ValueError(f"not a binary harvest: magic={data[:4]!r}")
    h = np.frombuffer(data[:64], dtype=HEADER_DTYPE)[0]
    if int(h["version"]) != 1:
        raise ValueError(f"unsupported harvest version {h['version']}")
    argv_len = int(h["argv_blob_len"])
    records_start = 64 + argv_len
    if (len(data) - records_start) % RECORD_DTYPE.itemsize != 0:
        raise ValueError(
            f"record region not aligned: {len(data)-records_start} B not divisible by {RECORD_DTYPE.itemsize}"
        )
    records = np.frombuffer(data[records_start:], dtype=RECORD_DTYPE)

    n = len(records)
    nb = records["nblocks"].astype("int32")
    nh = records["nholes"].astype("int32")

    blocks_col = np.empty(n, dtype=object)
    holes_col  = np.empty(n, dtype=object)
    for i in range(n):
        blocks_col[i] = _blocks_text(records["block_pos"][i], records["block_mask"][i], int(nb[i]))
        holes_col[i]  = _holes_text (records["hole_pos"][i],                              int(nh[i]))

    df = pd.DataFrame({
        "id":                  records["state_id"].astype("int64"),
        "parent_id":           records["parent_id"].astype("int64"),
        "depth":               records["depth"].astype("int32"),
        "outcome":             np.array([chr(c) for c in records["outcome"]], dtype=object),
        "forward_solve":       records["forward_solve"].astype("int32"),
        "nblocks":             records["nblocks"].astype("int16"),
        "nholes":              records["nholes"].astype("int16"),
        "player_pos":          records["player_pos"].astype("int16"),
        "exit_pos":            records["exit_pos"].astype("int16"),
        "canonical_key":       records["canonical_key"].astype("int64", copy=False).view("int64"),
        "popcount":            np.array([bin(int(v)).count("1") for v in records["committed_empty"]], dtype="int16"),
        "committed_empty_hex": np.array([f"{int(v):016x}" for v in records["committed_empty"]], dtype=object),
        "blocks":              blocks_col,
        "holes":               holes_col,
    })

    # Parse argv blob
    argv_blob = bytes(data[64:records_start])
    argv = [p.decode("utf-8", errors="replace") for p in argv_blob.split(b"\x00") if p]

    header = {
        "magic":           bytes(h["magic"]).decode("ascii"),
        "version":         int(h["version"]),
        "grid_rows":       int(h["grid_rows"]),
        "grid_cols":       int(h["grid_cols"]),
        "exit_pos":        int(h["exit_pos"]),
        "flags":           int(h["flags"]),
        "started_at_unix": int(h["started_at_unix"]),
        "code_sha":        bytes(h["code_sha"]).hex(),
    }
    print(f"loaded {n:,} binary rows in {time.time()-t0:.1f}s "
          f"(grid {header['grid_rows']}x{header['grid_cols']}, "
          f"sha {header['code_sha'][:8]}…)", file=sys.stderr)
    return df, header, argv


def load_csv(path):
    """Legacy CSV path."""
    t0 = time.time()
    df = pd.read_csv(
        path,
        dtype={
            "id": "int64", "parent_id": "int64", "depth": "int32",
            "outcome": "string", "forward_solve": "int32",
            "nblocks": "int16", "nholes": "int16",
            "player_pos": "int16", "exit_pos": "int16", "popcount": "int16",
            "committed_empty_hex": "string",
            "blocks": "string", "holes": "string",
        },
        keep_default_na=False,
    )
    df["canonical_key"] = 0  # not present in CSV format
    print(f"loaded {len(df):,} csv rows in {time.time()-t0:.1f}s", file=sys.stderr)
    return df, {}, []


def detect_and_load(path):
    with open(path, "rb") as f:
        head = f.read(4)
    # Gzip-wrapped binary: peek inside.
    if head[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            inner = f.read(4)
        if inner == HARVEST_MAGIC:
            return load_binary(path)
        raise ValueError(f"gzip file does not contain a BSH1 harvest (magic={inner!r})")
    if head == HARVEST_MAGIC:
        return load_binary(path)
    return load_csv(path)


def load_and_derive(path):
    """Convenience: load, derive subtree stats, return (df, header, argv)."""
    df, header, argv = detect_and_load(path)
    df = compute_subtree_stats(df)
    return df, header, argv


def compute_subtree_stats(df):
    """For each state, compute max depth in its subtree and subtree size.
    Iterates rows by decreasing depth, bubbling values up to parents.

    Fast path: when state_ids are 0..N-1 (always true for fresh harvests
    written in emission order), parent_id IS the parent's row index, so
    we can skip the pandas-Series lookup.  At 12M rows the slow path
    takes ~20 min; the fast path takes ~30 s.
    """
    t0 = time.time()
    n = len(df)
    max_desc = np.array(df["depth"].values, dtype="int32", copy=True)
    n_desc = np.ones(n, dtype="int32")

    sorted_idx = df["depth"].argsort()[::-1].to_numpy()
    parent_ids = df["parent_id"].to_numpy()
    ids = df["id"].to_numpy()

    contiguous_ids = (n == 0) or (ids[0] == 0 and ids[-1] == n - 1
                                  and np.array_equal(ids, np.arange(n)))
    if contiguous_ids:
        # parent_id == parent's row index.  No lookup needed.
        for idx in sorted_idx:
            pid = parent_ids[idx]
            if pid < 0:
                continue
            if max_desc[idx] > max_desc[pid]:
                max_desc[pid] = max_desc[idx]
            n_desc[pid] += n_desc[idx]
    else:
        id_to_idx = pd.Series(df.index.values, index=ids)
        for idx in sorted_idx:
            pid = parent_ids[idx]
            if pid < 0:
                continue
            try:
                p_idx = int(id_to_idx[pid])
            except KeyError:
                continue
            if max_desc[idx] > max_desc[p_idx]:
                max_desc[p_idx] = max_desc[idx]
            n_desc[p_idx] += n_desc[idx]

    df["max_descendant_depth"] = max_desc
    df["n_descendants"] = n_desc
    has_children = df["id"].isin(set(df["parent_id"].unique())).to_numpy()
    df["is_leaf"] = (~has_children).astype("int8")
    print(f"derived stats in {time.time()-t0:.1f}s "
          f"({'fast' if contiguous_ids else 'slow'} path)", file=sys.stderr)
    return df


def write_sqlite(df, db_path):
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    df.to_sql("states", conn, if_exists="replace", index=False, chunksize=50_000)
    cur = conn.cursor()
    cur.execute("CREATE INDEX IF NOT EXISTS idx_depth ON states(depth)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_parent ON states(parent_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON states(outcome)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_maxdesc ON states(max_descendant_depth)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_canon ON states(canonical_key)")
    conn.commit()
    conn.close()
    print(f"sqlite written ({db_path}) in {time.time()-t0:.1f}s", file=sys.stderr)


def print_summary(df, header, argv):
    print()
    print("=== HARVEST SUMMARY ===")
    if header:
        print(f"Grid:                   {header['grid_rows']}x{header['grid_cols']}")
        print(f"Code SHA:               {header['code_sha'][:12]}…")
        print(f"Started:                unix={header['started_at_unix']}")
        if argv:
            print(f"Argv:                   {' '.join(argv)}")
    print(f"Total states:           {len(df):,}")
    print(f"Outcome breakdown:")
    for outcome, count in df["outcome"].value_counts().items():
        print(f"  {outcome}: {count:,}")
    print(f"Depth range:            {df['depth'].min()} .. {df['depth'].max()}")
    print(f"max_descendant_depth:   {df['max_descendant_depth'].max()}")
    accepted = df[df["outcome"] == "A"]
    print(f"Accepted states:        {len(accepted):,}")
    if len(accepted):
        print(f"  avg subtree size:     {accepted['n_descendants'].mean():.1f}")
        print(f"  max subtree size:     {accepted['n_descendants'].max():,}")
    pruned = df[df["outcome"].isin(["S", "E"])]
    if len(pruned):
        fs = pruned["forward_solve"]
        fs_valid = fs[fs >= 0]
        print(f"Pruned states with forward_solve:")
        print(f"  n={len(fs_valid):,}, mean={fs_valid.mean():.1f}, max={fs_valid.max()}")
    if "canonical_key" in df.columns and df["canonical_key"].nunique() > 1:
        n_distinct = df["canonical_key"].nunique()
        print(f"Distinct canonical keys: {n_distinct:,} ({n_distinct/len(df):.1%} of rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to harvest file (binary BSH1 or legacy CSV)")
    ap.add_argument("--db", default=None, help="output SQLite (default: <path>.sqlite)")
    ap.add_argument("--no-db", action="store_true", help="skip SQLite write, only print summary")
    args = ap.parse_args()

    df, header, argv = detect_and_load(args.path)
    df = compute_subtree_stats(df)
    print_summary(df, header, argv)

    if not args.no_db:
        db = args.db or str(Path(args.path).with_suffix(".sqlite"))
        write_sqlite(df, db)


if __name__ == "__main__":
    main()
