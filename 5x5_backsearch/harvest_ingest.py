#!/usr/bin/env python3
"""
Ingest a backsearch_worker --harvest binary log into the central corpus
SQLite database.

The corpus has two tables:
  runs        — one row per harvest file (run metadata, summary stats)
  states      — one row per visited state, foreign-keyed to runs(run_id)

Idempotent on (hostname, started_at): re-ingesting the same harvest is a
no-op unless --force is passed.  --force also handles the "I tweaked the
loader and want to re-derive" workflow.

Usage:
  harvest_ingest.py <harvest.bin[.gz]> [--corpus PATH]
                                       [--hostname HOST]
                                       [--exit-reason REASON]
                                       [--force]
"""

import argparse
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from harvest_load import (
    load_and_derive,
    HEADER_DTYPE,  # noqa: F401  (re-export-ish; not used here but documents source-of-truth)
)
from harvest_index import rebuild as rebuild_canonical_index

DEFAULT_CORPUS = Path(__file__).resolve().parent / "corpus" / "corpus.sqlite"

# Keep in sync with harvest_format.h
FLAG_ALLOW_EXIT_TRANSIT = 1 << 0
FLAG_TWO_TABLES         = 1 << 1
FLAG_HOLELESS           = 1 << 2


SCHEMA_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      INTEGER NOT NULL,
    ended_at        INTEGER,
    hostname        TEXT NOT NULL,
    code_sha        TEXT,
    argv            TEXT,
    grid_rows       INTEGER NOT NULL,
    grid_cols       INTEGER NOT NULL,
    exit_pos        INTEGER,         -- -1 if multi-exit run
    flags_text      TEXT,
    states_visited  INTEGER,
    best_depth      INTEGER,
    exit_reason     TEXT             -- exhausted / time-cap / dedup-overflow / unknown
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_runs_host_started
    ON runs(hostname, started_at);

CREATE TABLE IF NOT EXISTS states (
    run_id                  INTEGER NOT NULL,
    state_id                INTEGER NOT NULL,
    parent_id               INTEGER NOT NULL,
    canonical_key           INTEGER NOT NULL,
    depth                   INTEGER NOT NULL,
    outcome                 TEXT NOT NULL,
    forward_solve           INTEGER,
    nblocks                 INTEGER,
    nholes                  INTEGER,
    player_pos              INTEGER,
    exit_pos                INTEGER,
    committed_empty         INTEGER,
    blocks                  TEXT,
    holes                   TEXT,
    max_descendant_depth    INTEGER,
    n_descendants           INTEGER,
    PRIMARY KEY (run_id, state_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

# Secondary indexes are created AFTER bulk insert — building them on
# 10M+ rows incrementally during insert is ~10x slower than building
# them once at the end.
STATES_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_states_canon       ON states(canonical_key);
CREATE INDEX IF NOT EXISTS idx_states_run_depth   ON states(run_id, depth);
CREATE INDEX IF NOT EXISTS idx_states_run_parent  ON states(run_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_states_outcome     ON states(outcome);
CREATE INDEX IF NOT EXISTS idx_states_maxdesc     ON states(max_descendant_depth);
"""


def init_schema(conn):
    conn.executescript(SCHEMA_TABLES_SQL)
    conn.commit()


def create_states_indexes(conn):
    conn.executescript(STATES_INDEXES_SQL)
    conn.commit()


def drop_states_indexes(conn):
    """Drop secondary indexes (not the primary key) so bulk insert is fast."""
    cur = conn.cursor()
    for ix in ["idx_states_canon", "idx_states_run_depth",
               "idx_states_run_parent", "idx_states_outcome",
               "idx_states_maxdesc"]:
        cur.execute(f"DROP INDEX IF EXISTS {ix}")
    conn.commit()


def flags_to_text(flags):
    parts = []
    if flags & FLAG_ALLOW_EXIT_TRANSIT: parts.append("allow-exit-transit")
    if flags & FLAG_TWO_TABLES:         parts.append("two-tables")
    if flags & FLAG_HOLELESS:           parts.append("holeless")
    return ",".join(parts) if parts else "(none)"


def has_block_on_exit(blocks_text, exit_pos):
    if not blocks_text:
        return False
    for tok in blocks_text.split(":"):
        if not tok:
            continue
        try:
            pos_str = tok.split(".", 1)[0]
            if int(pos_str) == exit_pos:
                return True
        except ValueError:
            continue
    return False


def derive_best_depth(df):
    """Best depth across accepted states whose blocks don't sit on the exit
    (matches the worker's own 'best_depth' definition)."""
    accepted = df[df["outcome"] == "A"]
    if accepted.empty:
        return 0
    blocks = accepted["blocks"].to_numpy()
    exits  = accepted["exit_pos"].to_numpy()
    depths = accepted["depth"].to_numpy()
    best = 0
    for i in range(len(accepted)):
        if has_block_on_exit(blocks[i], int(exits[i])):
            continue
        if depths[i] > best:
            best = int(depths[i])
    return best


def parse_argv_blob(argv_list):
    return " ".join(argv_list)


def find_existing_run(conn, hostname, started_at):
    cur = conn.execute(
        "SELECT run_id FROM runs WHERE hostname=? AND started_at=?",
        (hostname, started_at),
    )
    row = cur.fetchone()
    return row[0] if row else None


def insert_run(conn, header, argv, hostname, df, exit_reason):
    grid_rows = header["grid_rows"]
    grid_cols = header["grid_cols"]
    exit_pos  = header["exit_pos"]
    started   = header["started_at_unix"]
    code_sha  = header["code_sha"]
    flags_txt = flags_to_text(header["flags"])
    argv_text = parse_argv_blob(argv)
    states_n  = len(df)
    best      = derive_best_depth(df)
    ended_at  = int(time.time())  # ingest time as a proxy if not otherwise known

    cur = conn.execute(
        """INSERT INTO runs
              (started_at, ended_at, hostname, code_sha, argv,
               grid_rows, grid_cols, exit_pos, flags_text,
               states_visited, best_depth, exit_reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (started, ended_at, hostname, code_sha, argv_text,
         grid_rows, grid_cols, exit_pos, flags_txt,
         states_n, best, exit_reason),
    )
    return cur.lastrowid, best, states_n


def bulk_insert_states(conn, run_id, df):
    df_ins = df.assign(run_id=run_id)
    cols = [
        "run_id", "id", "parent_id", "canonical_key", "depth", "outcome",
        "forward_solve", "nblocks", "nholes", "player_pos", "exit_pos",
        "committed_empty_hex", "blocks", "holes",
        "max_descendant_depth", "n_descendants",
    ]
    # SQLite has no native u64; store committed_empty as INTEGER via int().
    committed_int = [int(h, 16) for h in df_ins["committed_empty_hex"]]
    rows = list(zip(
        [run_id] * len(df_ins),
        df_ins["id"].astype(int).tolist(),
        df_ins["parent_id"].astype(int).tolist(),
        df_ins["canonical_key"].astype(int).tolist(),
        df_ins["depth"].astype(int).tolist(),
        df_ins["outcome"].tolist(),
        df_ins["forward_solve"].astype(int).tolist(),
        df_ins["nblocks"].astype(int).tolist(),
        df_ins["nholes"].astype(int).tolist(),
        df_ins["player_pos"].astype(int).tolist(),
        df_ins["exit_pos"].astype(int).tolist(),
        committed_int,
        df_ins["blocks"].tolist(),
        df_ins["holes"].tolist(),
        df_ins["max_descendant_depth"].astype(int).tolist(),
        df_ins["n_descendants"].astype(int).tolist(),
    ))
    conn.executemany(
        """INSERT INTO states
              (run_id, state_id, parent_id, canonical_key, depth, outcome,
               forward_solve, nblocks, nholes, player_pos, exit_pos,
               committed_empty, blocks, holes,
               max_descendant_depth, n_descendants)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )


def delete_run(conn, run_id):
    conn.execute("DELETE FROM states WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))


def ingest(path, corpus_path, hostname, exit_reason, force, skip_index=False):
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(corpus_path)
    conn.execute("PRAGMA foreign_keys = ON")
    # Bulk-insert performance tunings.  Safe at session granularity since
    # we only have one writer.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    init_schema(conn)
    drop_states_indexes(conn)

    t0 = time.time()
    df, header, argv = load_and_derive(path)
    print(f"  load+derive: {time.time()-t0:.1f}s ({len(df):,} states)", file=sys.stderr)

    existing = find_existing_run(conn, hostname, int(header["started_at_unix"]))
    if existing is not None and not force:
        print(f"  already ingested as run_id={existing} (use --force to re-ingest)",
              file=sys.stderr)
        conn.close()
        return existing
    if existing is not None and force:
        print(f"  --force: deleting existing run_id={existing}", file=sys.stderr)
        delete_run(conn, existing)

    t1 = time.time()
    run_id, best, n = insert_run(conn, header, argv, hostname, df, exit_reason)
    bulk_insert_states(conn, run_id, df)
    conn.commit()
    print(f"  inserted run_id={run_id}: {n:,} states, best_depth={best}, "
          f"insert {time.time()-t1:.1f}s", file=sys.stderr)

    t2 = time.time()
    create_states_indexes(conn)
    print(f"  states indexes built in {time.time()-t2:.1f}s", file=sys.stderr)
    conn.close()

    if not skip_index:
        rebuild_canonical_index(corpus_path)

    return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to harvest binary (.bin or .bin.gz)")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                    help=f"corpus SQLite path (default: {DEFAULT_CORPUS})")
    ap.add_argument("--hostname", default=socket.gethostname(),
                    help="hostname recorded for this run (default: local hostname)")
    ap.add_argument("--exit-reason", default="unknown",
                    choices=["unknown", "exhausted", "time-cap", "dedup-overflow"],
                    help="run exit reason (parsed from worker stdout, set manually if known)")
    ap.add_argument("--force", action="store_true",
                    help="re-ingest even if (hostname, started_at) already present "
                         "(deletes the matching run and replaces it)")
    ap.add_argument("--skip-index", action="store_true",
                    help="skip the canonical_states rebuild step "
                         "(run harvest_index.py manually later)")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 1

    run_id = ingest(args.path, args.corpus, args.hostname, args.exit_reason,
                    args.force, args.skip_index)
    print(f"corpus: {args.corpus}")
    print(f"run_id: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
