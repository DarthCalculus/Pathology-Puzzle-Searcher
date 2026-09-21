#!/usr/bin/env python3
"""
Quick-look plots over the corpus.  Matplotlib is imported lazily and
optional; if absent, the module still imports and the CLI prints text
summaries instead of opening figures.

Usage:
  corpus_viz.py [--corpus PATH] [--run-id N] [--out PNG_FILE | --show]
                {depth-hist | outcome-by-depth | subtree-cdf | summary}

Commands:
  depth-hist          histogram of state count by depth (split by outcome)
  outcome-by-depth    stacked bar: outcome shares at each depth
  subtree-cdf         CDF of n_descendants (log x) across accepted states
  summary             text overview of corpus contents
"""

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus" / "corpus.sqlite"


def _connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _maybe_run_filter(args):
    """Returns (where_clause, params) for a runs/states filter."""
    if args.run_id is not None:
        return " WHERE run_id = ?", (args.run_id,)
    return "", ()


def _import_mpl():
    try:
        import matplotlib
        if not (hasattr(sys, "ps1") or sys.flags.interactive):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def depth_histogram(conn, args):
    where, params = _maybe_run_filter(args)
    sql = f"SELECT depth, outcome, COUNT(*) AS n FROM states{where} GROUP BY depth, outcome ORDER BY depth, outcome"
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("(no rows)", file=sys.stderr); return

    by_outcome = {}
    depths = set()
    for r in rows:
        depths.add(int(r["depth"]))
        by_outcome.setdefault(r["outcome"], {})[int(r["depth"])] = int(r["n"])
    depths = sorted(depths)

    plt = _import_mpl()
    if plt is None:
        print("(matplotlib not available — falling back to text)")
        print(f"{'depth':>5} | " + " ".join(f"{o:>8}" for o in sorted(by_outcome)))
        for d in depths:
            line = f"{d:>5} | "
            line += " ".join(f"{by_outcome[o].get(d, 0):>8}" for o in sorted(by_outcome))
            print(line)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8
    bottoms = [0] * len(depths)
    for outcome in sorted(by_outcome):
        counts = [by_outcome[outcome].get(d, 0) for d in depths]
        ax.bar(depths, counts, width=width, bottom=bottoms, label=outcome)
        bottoms = [b + c for b, c in zip(bottoms, counts)]
    ax.set_xlabel("backward depth")
    ax.set_ylabel("state count")
    ax.set_title(f"State count by depth"
                 + (f" (run_id={args.run_id})" if args.run_id else ""))
    ax.legend()
    _finish(plt, fig, args)


def outcome_by_depth(conn, args):
    where, params = _maybe_run_filter(args)
    sql = f"SELECT depth, outcome, COUNT(*) AS n FROM states{where} GROUP BY depth, outcome ORDER BY depth"
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("(no rows)", file=sys.stderr); return

    by_outcome = {}
    depths = set()
    for r in rows:
        depths.add(int(r["depth"]))
        by_outcome.setdefault(r["outcome"], {})[int(r["depth"])] = int(r["n"])
    depths = sorted(depths)
    totals = {d: sum(by_outcome[o].get(d, 0) for o in by_outcome) for d in depths}

    plt = _import_mpl()
    if plt is None:
        print("(matplotlib unavailable; use depth-hist for text view)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8
    bottoms = [0.0] * len(depths)
    for outcome in sorted(by_outcome):
        fracs = [by_outcome[outcome].get(d, 0) / totals[d] if totals[d] else 0
                 for d in depths]
        ax.bar(depths, fracs, width=width, bottom=bottoms, label=outcome)
        bottoms = [b + f for b, f in zip(bottoms, fracs)]
    ax.set_xlabel("backward depth")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1)
    ax.set_title(f"Outcome share by depth"
                 + (f" (run_id={args.run_id})" if args.run_id else ""))
    ax.legend(loc="lower right")
    _finish(plt, fig, args)


def subtree_cdf(conn, args):
    where, params = _maybe_run_filter(args)
    extra = " AND outcome='A'" if where else " WHERE outcome='A'"
    sql = f"SELECT n_descendants FROM states{where}{extra}"
    rows = [int(r[0]) for r in conn.execute(sql, params)]
    if not rows:
        print("(no rows)", file=sys.stderr); return

    plt = _import_mpl()
    rows.sort()
    n = len(rows)

    if plt is None:
        percentiles = [50, 75, 90, 95, 99, 99.9]
        print(f"n_descendants quantiles (accepted, n={n}):")
        for p in percentiles:
            idx = min(n - 1, int(n * p / 100))
            print(f"  p{p:>4}: {rows[idx]:,}")
        print(f"  max  : {rows[-1]:,}")
        return

    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 5))
    cdf = np.arange(1, n + 1) / n
    ax.plot(rows, cdf)
    ax.set_xscale("log")
    ax.set_xlabel("n_descendants (subtree size, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Subtree-size CDF (accepted states)"
                 + (f" (run_id={args.run_id})" if args.run_id else ""))
    ax.grid(True, which="both", alpha=0.3)
    _finish(plt, fig, args)


def summary(conn, args):
    print("=== CORPUS SUMMARY ===")
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    n_states = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
    print(f"runs:   {n_runs:,}")
    print(f"states: {n_states:,}")
    try:
        n_canon = conn.execute("SELECT COUNT(*) FROM canonical_states").fetchone()[0]
        print(f"canonical_states: {n_canon:,} "
              f"({(n_states/n_canon if n_canon else 0):.2f}x dup ratio)")
    except sqlite3.OperationalError:
        print("canonical_states: (not built — run harvest_index.py)")
    print()
    print("Per-run highlights:")
    for r in conn.execute(
        "SELECT run_id, hostname, grid_rows, grid_cols, exit_pos, flags_text, "
        "       states_visited, best_depth, exit_reason FROM runs ORDER BY run_id"
    ):
        print(f"  run {r['run_id']:>3}: {r['grid_rows']}x{r['grid_cols']} "
              f"exit={r['exit_pos']:>3} flags={r['flags_text']:<30} "
              f"states={r['states_visited']:>10,} best={r['best_depth']:>3}  "
              f"({r['hostname']}, {r['exit_reason']})")


def _finish(plt, fig, args):
    if not args.out and not args.show:
        # Default: drop a PNG in /tmp so the user always gets *something*.
        args.out = f"/tmp/corpus_{args.cmd}.png"
    if args.out:
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"saved {args.out}", file=sys.stderr)
    if args.show:
        plt.show()
    plt.close(fig)


CMDS = {
    "depth-hist":       depth_histogram,
    "outcome-by-depth": outcome_by_depth,
    "subtree-cdf":      subtree_cdf,
    "summary":          summary,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--out", help="save figure to PNG/PDF file")
    ap.add_argument("--show", action="store_true", help="open figure interactively")
    ap.add_argument("cmd", choices=list(CMDS), help="which plot/summary")
    args = ap.parse_args()
    conn = _connect(args.corpus)
    CMDS[args.cmd](conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
