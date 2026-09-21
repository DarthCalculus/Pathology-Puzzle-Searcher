#!/usr/bin/env python3
"""
Rebuild the canonical_states table in the corpus SQLite from the raw
states table.

canonical_states folds duplicate state observations across runs into one
row per canonical_key, with summary stats:
  - best_max_descendant_depth: deepest depth reached anywhere downstream
    of any observation of this state.  This is the cross-run value-head
    training target.
  - tightest_forward_solve: minimum (== tightest known upper bound on
    forward solve) across all observations.  When NULL, no observation
    of this state was ever forward-solved with a non-negative answer.
  - n_observations: total rows in `states` with this canonical_key.
  - n_runs_seen: distinct run_id values that observed this key.
  - exemplar_run_id, exemplar_state_id: one concrete (run_id, state_id)
    representative, chosen as the row with the deepest
    max_descendant_depth (ties broken by smaller run_id then state_id).
    Use it to look up the blob columns when you need the actual state.

The rebuild is a full DROP+CREATE — there is no incremental path.  At
a few-million-rows scale this finishes in seconds on SQLite.

Usage:
  harvest_index.py [--corpus PATH]
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus" / "corpus.sqlite"


REBUILD_SQL = """
DROP TABLE IF EXISTS canonical_states;

CREATE TABLE canonical_states AS
WITH agg AS (
  SELECT canonical_key,
         MAX(max_descendant_depth) AS best_max_descendant_depth,
         MIN(CASE WHEN forward_solve >= 0 THEN forward_solve END) AS tightest_forward_solve,
         COUNT(*) AS n_observations,
         COUNT(DISTINCT run_id) AS n_runs_seen
  FROM states
  GROUP BY canonical_key
),
ex AS (
  SELECT canonical_key, run_id, state_id,
         ROW_NUMBER() OVER (
             PARTITION BY canonical_key
             ORDER BY max_descendant_depth DESC, run_id ASC, state_id ASC
         ) AS rn
  FROM states
)
SELECT agg.canonical_key,
       agg.best_max_descendant_depth,
       agg.tightest_forward_solve,
       agg.n_observations,
       agg.n_runs_seen,
       ex.run_id  AS exemplar_run_id,
       ex.state_id AS exemplar_state_id
FROM agg JOIN ex
  ON agg.canonical_key = ex.canonical_key
 AND ex.rn = 1;

CREATE UNIQUE INDEX IF NOT EXISTS uq_canon_states_key
    ON canonical_states(canonical_key);
CREATE INDEX IF NOT EXISTS idx_canon_states_depth
    ON canonical_states(best_max_descendant_depth);
CREATE INDEX IF NOT EXISTS idx_canon_states_fs
    ON canonical_states(tightest_forward_solve);
"""


def rebuild(corpus_path):
    conn = sqlite3.connect(corpus_path)
    t0 = time.time()
    conn.executescript(REBUILD_SQL)
    conn.commit()
    elapsed = time.time() - t0

    cur = conn.cursor()
    n_canon = cur.execute("SELECT COUNT(*) FROM canonical_states").fetchone()[0]
    n_states = cur.execute("SELECT COUNT(*) FROM states").fetchone()[0]
    dup_ratio = (n_states / n_canon) if n_canon else 1.0
    print(f"  rebuilt canonical_states in {elapsed:.1f}s: "
          f"{n_canon:,} unique keys / {n_states:,} observations "
          f"(avg {dup_ratio:.2f}× redundancy)", file=sys.stderr)
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help=f"corpus SQLite path (default: {DEFAULT_CORPUS})")
    args = ap.parse_args()
    if not Path(args.corpus).exists():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        return 1
    rebuild(args.corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
