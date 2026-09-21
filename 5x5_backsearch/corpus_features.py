#!/usr/bin/env python3
"""
Feature-extractor: turn a row from the corpus `states` table into a
fixed-shape numpy tensor for ML training.

The tensor has shape (CHANNELS, R, C) where R,C are the grid dimensions
(from the run that produced the state).  Channels:

  0  player        one-hot at player_pos
  1  exit          one-hot at exit_pos
  2  block-U       1 where a block has direction-bit U (= 1) set
  3  block-R       1 where a block has direction-bit R (= 2) set
  4  block-D       1 where a block has direction-bit D (= 4) set
  5  block-L       1 where a block has direction-bit L (= 8) set
  6  hole          1 where an active hole sits
  7  committed     1 where committed_empty has the cell set (known not-wall)
  8  unknown       1 where the cell is neither committed nor an exit
                   (= cells the search has not yet ruled out as walls)

The total CHANNELS = 9 is fixed; bumping it is a model/format change.
"""

import sqlite3
import sys
from typing import Iterator, Optional, Tuple

import numpy as np


CHANNELS = 9

CH_PLAYER       = 0
CH_EXIT         = 1
CH_BLOCK_U      = 2
CH_BLOCK_R      = 3
CH_BLOCK_D      = 4
CH_BLOCK_L      = 5
CH_HOLE         = 6
CH_COMMITTED    = 7
CH_UNKNOWN      = 8

DIR_BITS = {1: CH_BLOCK_U, 2: CH_BLOCK_R, 4: CH_BLOCK_D, 8: CH_BLOCK_L}


def _parse_blocks(blocks_text: str):
    """Yield (pos, mask) pairs from a 'pos.mask:pos.mask:…' string."""
    if not blocks_text:
        return
    for tok in blocks_text.split(":"):
        if not tok:
            continue
        pos_str, mask_str = tok.split(".", 1)
        yield int(pos_str), int(mask_str)


def _parse_holes(holes_text: str):
    if not holes_text:
        return
    for tok in holes_text.split(":"):
        if not tok:
            continue
        yield int(tok)


def state_to_tensor(row: dict, grid_rows: int, grid_cols: int) -> np.ndarray:
    """Build a (CHANNELS, R, C) float32 tensor from a corpus row.

    `row` is any object supporting `row[key]` access (sqlite3.Row, dict,
    pandas Series).  Required keys: player_pos, exit_pos, blocks, holes,
    committed_empty (int).
    """
    t = np.zeros((CHANNELS, grid_rows, grid_cols), dtype=np.float32)

    def rc(p):
        return p // grid_cols, p % grid_cols

    pp = int(row["player_pos"])
    if 0 <= pp < grid_rows * grid_cols:
        r, c = rc(pp)
        t[CH_PLAYER, r, c] = 1.0

    ep = int(row["exit_pos"])
    if 0 <= ep < grid_rows * grid_cols:
        r, c = rc(ep)
        t[CH_EXIT, r, c] = 1.0

    for pos, mask in _parse_blocks(row["blocks"]):
        if pos < 0 or pos >= grid_rows * grid_cols:
            continue
        r, c = rc(pos)
        for bit, ch in DIR_BITS.items():
            if mask & bit:
                t[ch, r, c] = 1.0

    for pos in _parse_holes(row["holes"]):
        if 0 <= pos < grid_rows * grid_cols:
            r, c = rc(pos)
            t[CH_HOLE, r, c] = 1.0

    committed = int(row["committed_empty"])
    for p in range(grid_rows * grid_cols):
        if (committed >> p) & 1:
            r, c = rc(p)
            t[CH_COMMITTED, r, c] = 1.0

    # Unknown = neither committed nor exit-cell (the search hasn't ruled it out as wall yet).
    # We deliberately keep exits in CH_EXIT only, not also as committed/unknown.
    exit_bit = 1 << ep if 0 <= ep < 64 else 0
    not_committed = (~committed) & ((1 << (grid_rows * grid_cols)) - 1) & ~exit_bit
    for p in range(grid_rows * grid_cols):
        if (not_committed >> p) & 1:
            r, c = rc(p)
            t[CH_UNKNOWN, r, c] = 1.0

    return t


def pretty_print_tensor(t: np.ndarray) -> str:
    """Render the tensor channels back as a grid for eyeball verification.

    Symbols: @ player, $ exit, # walls (unknown), . committed, [URDL] = block
    (showing one mask bit), * hole.  Multi-bit masks render as a single
    letter from the highest set bit; the channel arrays have the precise info.
    """
    _, R, C = t.shape
    out = []
    for r in range(R):
        row = []
        for c in range(C):
            if t[CH_PLAYER, r, c]:
                row.append("@")
            elif t[CH_EXIT, r, c]:
                row.append("$")
            elif t[CH_HOLE, r, c]:
                row.append("*")
            elif any(t[ch, r, c] for ch in (CH_BLOCK_U, CH_BLOCK_R, CH_BLOCK_D, CH_BLOCK_L)):
                for ch, lbl in [(CH_BLOCK_L, "L"), (CH_BLOCK_D, "D"),
                                (CH_BLOCK_R, "R"), (CH_BLOCK_U, "U")]:
                    if t[ch, r, c]:
                        row.append(lbl); break
            elif t[CH_COMMITTED, r, c]:
                row.append(".")
            elif t[CH_UNKNOWN, r, c]:
                row.append("#")
            else:
                row.append("?")
        out.append("".join(row))
    return "\n".join(out)


def iter_corpus(
    db_path: str,
    where: Optional[str] = None,
    params: Optional[tuple] = None,
    chunk_size: int = 1000,
) -> Iterator[Tuple[np.ndarray, dict]]:
    """Stream rows from the corpus, yielding (tensor, meta).

    `meta` is a dict with at least: run_id, state_id, depth, outcome,
    forward_solve, max_descendant_depth, canonical_key, grid_rows,
    grid_cols.  Use `meta['max_descendant_depth']` as the value-head
    target.

    `where` is an optional SQL fragment (no `WHERE` keyword) appended to
    the SELECT.  `params` matches `where`'s `?` placeholders.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT s.run_id, s.state_id, s.canonical_key, s.depth, s.outcome,
               s.forward_solve, s.max_descendant_depth, s.player_pos,
               s.exit_pos, s.committed_empty, s.blocks, s.holes,
               r.grid_rows, r.grid_cols
        FROM states s
        JOIN runs r USING (run_id)
    """
    if where:
        sql += " WHERE " + where
    cur = conn.execute(sql, params or ())

    while True:
        batch = cur.fetchmany(chunk_size)
        if not batch:
            break
        for row in batch:
            row_d = dict(row)
            t = state_to_tensor(row_d, row_d["grid_rows"], row_d["grid_cols"])
            yield t, row_d
    conn.close()


# --- Optional PyTorch Dataset adapter (only used if torch is importable) -----

class _CorpusDataset:
    """Map-style PyTorch Dataset wrapping the corpus.  Imports torch
    lazily on first __getitem__, so this module is usable without torch.

    Use with:
        from torch.utils.data import DataLoader
        ds = CorpusDataset('corpus/corpus.sqlite',
                           where='outcome=\"A\" AND max_descendant_depth > 30')
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4)
    """

    def __init__(self, db_path, where=None, params=None, target="max_descendant_depth"):
        self.db_path = db_path
        self.where = where
        self.params = params or ()
        self.target = target
        self._row_ids = self._fetch_keys()

    def _fetch_keys(self):
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT run_id, state_id FROM states"
        if self.where:
            sql += " WHERE " + self.where
        ids = conn.execute(sql, self.params).fetchall()
        conn.close()
        return ids

    def __len__(self):
        return len(self._row_ids)

    def __getitem__(self, idx):
        run_id, state_id = self._row_ids[idx]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT s.player_pos, s.exit_pos, s.committed_empty, s.blocks,
                      s.holes, s.depth, s.forward_solve, s.max_descendant_depth,
                      r.grid_rows, r.grid_cols
               FROM states s JOIN runs r USING (run_id)
               WHERE s.run_id=? AND s.state_id=?""",
            (run_id, state_id),
        ).fetchone()
        conn.close()
        row = dict(row)
        tensor = state_to_tensor(row, row["grid_rows"], row["grid_cols"])
        target = row[self.target]
        return tensor, target


def CorpusDataset(*args, **kwargs):
    """Returns a _CorpusDataset.  Kept as a function so importing this
    module doesn't fail when torch is absent — torch.utils.data isn't
    required to instantiate the class, only to use it with a DataLoader."""
    return _CorpusDataset(*args, **kwargs)


# --- CLI for ad-hoc inspection ---

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="inspect a single corpus state as tensor + grid")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--state-id", type=int, required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(args.corpus)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT s.*, r.grid_rows, r.grid_cols FROM states s
           JOIN runs r USING (run_id) WHERE s.run_id=? AND s.state_id=?""",
        (args.run_id, args.state_id),
    ).fetchone()
    if not row:
        print(f"no state run_id={args.run_id} state_id={args.state_id}", file=sys.stderr)
        return 1
    row = dict(row)
    print(f"run_id={row['run_id']} state_id={row['state_id']} depth={row['depth']} "
          f"outcome={row['outcome']} max_desc={row['max_descendant_depth']}")
    print(f"player={row['player_pos']} exit={row['exit_pos']} "
          f"nblocks={row['nblocks']} nholes={row['nholes']}")
    print(f"blocks={row['blocks']!r}")
    print(f"holes={row['holes']!r}")
    print(f"committed_empty=0x{int(row['committed_empty']) & ((1<<64)-1):016x}")
    print()
    t = state_to_tensor(row, row["grid_rows"], row["grid_cols"])
    print("Rendered grid (each channel ORed):")
    print(pretty_print_tensor(t))
    print()
    print(f"Tensor shape: {t.shape}, sum per channel: "
          f"{[float(t[c].sum()) for c in range(t.shape[0])]}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
