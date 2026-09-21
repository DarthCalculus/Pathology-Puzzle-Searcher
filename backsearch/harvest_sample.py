#!/usr/bin/env python3
"""
Sample states from the corpus for ad-hoc analysis and training-data export.

Filters (combinable):
  --depth-band MIN MAX     accepted states with MIN <= depth <= MAX
  --near-cutoff K          shortcut-pruned states with depth - forward_solve <= K
                           (the hardest negatives for the solver surrogate)
  --top-subtree N          accepted states with deepest max_descendant_depth (limit N)
  --exit N                 restrict to exit cell N
  --grid RxC               restrict to a grid size
  --run-id N               restrict to a specific run

Plus generic:
  --where EXPR             arbitrary SQL fragment (use with care)
  --limit N                cap output rows
  --out FORMAT             csv (default) | jsonl | parquet (needs pyarrow)
  --output FILE            destination (default: stdout)

Examples:
  # Deepest 20 accepted states across the whole corpus
  harvest_sample.py --top-subtree 20

  # Pruned states one step away from being accepted on 6x6 — surrogate-solver hard negatives
  harvest_sample.py --near-cutoff 0 --grid 6x6 --limit 5000 --out jsonl --output hard_negatives.jsonl

  # All accepted depth-40+ states on 5x5
  harvest_sample.py --depth-band 40 200 --grid 5x5
"""

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus" / "corpus.sqlite"


def build_query(args):
    """Return (sql, params).  Combines filters into a single SELECT."""
    conds = []
    params = []

    if args.depth_band:
        conds.append("s.outcome = 'A' AND s.depth BETWEEN ? AND ?")
        params.extend(args.depth_band)

    if args.near_cutoff is not None:
        # Shortcut prunes when forward_solve <= depth - 2.  K measures how
        # far past the cutoff: K=0 → forward_solve == depth-2 (just pruned),
        # K=1 → forward_solve == depth-3, etc.  Hard negatives for the solver
        # surrogate are concentrated at low K.
        conds.append("s.outcome = 'S' AND s.forward_solve >= 0 "
                     "AND (s.depth - s.forward_solve - 2) <= ?")
        params.append(args.near_cutoff)

    if args.exit is not None:
        conds.append("s.exit_pos = ?")
        params.append(args.exit)

    if args.grid is not None:
        r, c = args.grid
        conds.append("r.grid_rows = ? AND r.grid_cols = ?")
        params.extend([r, c])

    if args.run_id is not None:
        conds.append("s.run_id = ?")
        params.append(args.run_id)

    if args.where:
        conds.append("(" + args.where + ")")

    where_clause = (" WHERE " + " AND ".join(conds)) if conds else ""

    order_clause = ""
    if args.top_subtree is not None:
        order_clause = " ORDER BY s.max_descendant_depth DESC, s.depth DESC"

    limit_clause = ""
    if args.top_subtree is not None:
        limit_clause = f" LIMIT {int(args.top_subtree)}"
    elif args.limit is not None:
        limit_clause = f" LIMIT {int(args.limit)}"

    sql = (
        "SELECT s.run_id, s.state_id, s.canonical_key, s.depth, s.outcome, "
        "       s.forward_solve, s.max_descendant_depth, s.n_descendants, "
        "       s.nblocks, s.nholes, s.player_pos, s.exit_pos, "
        "       s.committed_empty, s.blocks, s.holes, "
        "       r.grid_rows, r.grid_cols "
        "FROM states s JOIN runs r USING (run_id)"
        + where_clause + order_clause + limit_clause
    )
    return sql, params


def write_csv(rows, columns, out):
    w = csv.writer(out)
    w.writerow(columns)
    for row in rows:
        w.writerow([row[c] for c in columns])


def write_jsonl(rows, columns, out):
    for row in rows:
        out.write(json.dumps({c: row[c] for c in columns}) + "\n")


def write_parquet(rows, columns, path):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("error: parquet output requires `pip install pyarrow`", file=sys.stderr)
        sys.exit(2)
    data = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            data[c].append(row[c])
    pq.write_table(pa.table(data), path)


def _parse_grid(s):
    r, c = s.lower().split("x")
    return (int(r), int(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--depth-band", nargs=2, type=int, metavar=("MIN", "MAX"))
    ap.add_argument("--near-cutoff", type=int, metavar="K",
                    help="shortcut-pruned states with depth-forward_solve <= K")
    ap.add_argument("--top-subtree", type=int, metavar="N",
                    help="top-N rows by max_descendant_depth")
    ap.add_argument("--exit", type=int)
    ap.add_argument("--grid", type=_parse_grid)
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--where", help="arbitrary extra SQL fragment (no WHERE keyword)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", choices=["csv", "jsonl", "parquet"], default="csv")
    ap.add_argument("--output", help="destination file (default: stdout for csv/jsonl)")
    args = ap.parse_args()

    sql, params = build_query(args)
    conn = sqlite3.connect(args.corpus)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        print(f"(no rows matched)", file=sys.stderr)
        return 0

    columns = list(rows[0].keys())

    if args.out == "parquet":
        if not args.output:
            print("error: --output PATH required for parquet", file=sys.stderr)
            return 1
        write_parquet(rows, columns, args.output)
        print(f"wrote {len(rows):,} rows → {args.output}", file=sys.stderr)
    else:
        out = open(args.output, "w") if args.output else sys.stdout
        try:
            if args.out == "csv":
                write_csv(rows, columns, out)
            else:
                write_jsonl(rows, columns, out)
        finally:
            if out is not sys.stdout:
                out.close()
        print(f"wrote {len(rows):,} rows → {args.output or '<stdout>'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
