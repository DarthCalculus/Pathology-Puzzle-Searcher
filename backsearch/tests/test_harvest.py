"""
End-to-end tests for the harvest pipeline.

These tests assume `backsearch_worker` has been built in the repo dir
(see the README).  Each test runs a small, fast harvest into a tmp_path
fixture, post-processes through harvest_load / harvest_ingest, and
asserts the derived corpus state matches expectations.
"""

import io
import os
import subprocess
import struct
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO, "backsearch_worker")


# -------- fixtures --------------------------------------------------

@pytest.fixture(scope="module")
def worker_binary():
    if not os.path.exists(WORKER):
        pytest.skip(f"worker binary not found at {WORKER}; build it first")
    return WORKER


@pytest.fixture
def tiny_harvest(tmp_path, worker_binary):
    """Run a bounded-depth harvest and return (harvest_path, worker_best, worker_stdout)."""
    out = tmp_path / "tiny.bin.gz"
    proc = subprocess.run(
        [worker_binary,
         "--grid", "5x5",
         "--exit", "12",
         "--allow-exit-transit",
         "--two-tables",
         "--time", "0",
         "--max-depth", "4",
         "--harvest", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    # Worker prints e.g. "best depth:     N  (verify N OK)"
    best = None
    for line in proc.stdout.splitlines():
        if "best depth:" in line:
            best = int(line.split(":")[1].split()[0])
            break
    assert best is not None
    return str(out), best, proc.stdout


# -------- end-to-end ingest ----------------------------------------

def test_ingest_matches_worker_best(tiny_harvest, tmp_path):
    """After ingest, the deepest accepted state in the corpus equals
    the worker's reported best_depth."""
    import harvest_ingest

    path, worker_best, _ = tiny_harvest
    corpus = tmp_path / "corpus.sqlite"
    run_id = harvest_ingest.ingest(path, str(corpus), "test-host",
                                   "unknown", force=False)

    import sqlite3
    conn = sqlite3.connect(corpus)
    cur = conn.execute(
        "SELECT best_depth FROM runs WHERE run_id=?", (run_id,))
    assert cur.fetchone()[0] == worker_best


def test_ingest_idempotent(tiny_harvest, tmp_path):
    """Re-ingesting the same harvest without --force is a no-op."""
    import harvest_ingest
    import sqlite3

    path, _, _ = tiny_harvest
    corpus = tmp_path / "corpus.sqlite"

    run_id1 = harvest_ingest.ingest(path, str(corpus), "host", "unknown", False)
    run_id2 = harvest_ingest.ingest(path, str(corpus), "host", "unknown", False)
    assert run_id1 == run_id2

    n_runs = sqlite3.connect(corpus).execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert n_runs == 1


def test_canonical_dedup_across_hosts(tiny_harvest, tmp_path):
    """Same harvest ingested under two hostnames → canonical_states row count
    unchanged, n_observations doubles per key, n_runs_seen=2 for every key
    observed in both runs (= every key, since the harvests are identical)."""
    import harvest_ingest
    import sqlite3

    path, _, _ = tiny_harvest
    corpus = tmp_path / "corpus.sqlite"

    harvest_ingest.ingest(path, str(corpus), "host_a", "unknown", False)
    after_a = sqlite3.connect(corpus).execute(
        "SELECT COUNT(*) FROM canonical_states").fetchone()[0]
    obs_sum_a = sqlite3.connect(corpus).execute(
        "SELECT SUM(n_observations) FROM canonical_states").fetchone()[0]

    harvest_ingest.ingest(path, str(corpus), "host_b", "unknown", False)
    after_b = sqlite3.connect(corpus).execute(
        "SELECT COUNT(*) FROM canonical_states").fetchone()[0]
    obs_sum_b = sqlite3.connect(corpus).execute(
        "SELECT SUM(n_observations) FROM canonical_states").fetchone()[0]
    runs_distribution = sqlite3.connect(corpus).execute(
        "SELECT n_runs_seen, COUNT(*) FROM canonical_states GROUP BY n_runs_seen"
    ).fetchall()

    assert after_a == after_b, "canonical_states count must be unchanged across runs"
    assert obs_sum_b == 2 * obs_sum_a, "total observations should double"
    # Every key was seen in both runs (identical harvests).
    assert dict(runs_distribution) == {2: after_b}


# -------- binary format -------------------------------------------

def test_binary_record_round_trip(tmp_path):
    """Write a header + a single record from Python via the documented
    layout, read it back via harvest_load.load_binary, assert fields match."""
    import harvest_load

    hdr_buf = bytearray(64)
    struct.pack_into("<4sHBBhHIQ20sI16s", hdr_buf, 0,
                     b"BSH1", 1,            # magic, version
                     7, 7,                  # grid_rows, grid_cols
                     -1,                    # exit_pos
                     0,                     # flags
                     0,                     # reserved_a
                     1700000000,            # started_at_unix
                     b"\x00" * 20,          # code_sha
                     0,                     # argv_blob_len
                     b"\x00" * 16)          # reserved_b

    rec = bytearray(144)
    # state_id=42, parent_id=-1, canonical_key=0xDEAD..., depth=5, fs=-1,
    # outcome='A', nblocks=2, nholes=1, player_pos=7, exit_pos=12,
    # committed_empty=0xfade, block_pos[0..1]=[3, 11], block_mask[0..1]=[1, 4]
    # hole_pos[0]=2
    struct.pack_into("<QqQiiBBBBh2sQ", rec, 0,
                     42, -1, 0xDEADBEEFCAFEBABE, 5, -1,
                     ord("A"), 2, 1, 7, 12, b"\x00\x00",
                     0xfade)
    rec[48] = 3   # block_pos[0]
    rec[49] = 11  # block_pos[1]
    rec[80] = 1   # block_mask[0]
    rec[81] = 4   # block_mask[1]
    rec[112] = 2  # hole_pos[0]

    p = tmp_path / "synth.bin"
    p.write_bytes(bytes(hdr_buf) + bytes(rec))

    df, header, argv = harvest_load.load_binary(str(p))
    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["id"]) == 42
    assert int(row["parent_id"]) == -1
    assert int(row["depth"]) == 5
    assert row["outcome"] == "A"
    assert int(row["nblocks"]) == 2
    assert int(row["nholes"]) == 1
    assert int(row["player_pos"]) == 7
    assert int(row["exit_pos"]) == 12
    assert row["blocks"] == "3.1:11.4"
    assert row["holes"] == "2"
    # canonical_key is stored as signed int64, but the underlying u64 is
    # 0xDEADBEEFCAFEBABE; reinterpret to confirm.
    raw_u64 = np.array([int(row["canonical_key"])], dtype="int64").view("uint64")[0]
    assert int(raw_u64) == 0xDEADBEEFCAFEBABE


# -------- feature tensor ------------------------------------------

def test_state_to_tensor_shape_and_marginals():
    """Build a hand-crafted state, assert tensor channel sums match."""
    import corpus_features as cf

    row = {
        "player_pos":      0,
        "exit_pos":        12,
        "blocks":          "5.1:6.10:7.4",   # 5: U-bit; 6: U|L (1|8=9 actually, but 10=R|D); 7: D
        "holes":           "1:2",
        "committed_empty": (1 << 0) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 12),
    }
    t = cf.state_to_tensor(row, 5, 5)
    assert t.shape == (9, 5, 5)
    assert t[cf.CH_PLAYER].sum() == 1.0
    assert t[cf.CH_EXIT].sum() == 1.0
    # block masks: 5 → mask=1 → U bit; 6 → mask=10 → R|D bits (mask 10 = 0b1010);
    #              7 → mask=4 → D bit.
    # Wait: parse: "5.1" → pos=5, mask=1 → U.  "6.10" → mask=10 → bits 2 (R=2) + 8 (L=8)? Nope.
    # mask=10 binary = 1010 = bits 1 (R=2) + 3 (L=8) = R+L.  Hmm.  Actually our DIR_BITS:
    # 1=U, 2=R, 4=D, 8=L.  mask 10 = 8|2 = L|R.
    # "7.4" → mask=4 → D bit.
    # So: U bit set once (block 5), R bit set once (block 6), L bit set once (block 6),
    # D bit set once (block 7).
    assert t[cf.CH_BLOCK_U].sum() == 1.0
    assert t[cf.CH_BLOCK_R].sum() == 1.0
    assert t[cf.CH_BLOCK_D].sum() == 1.0
    assert t[cf.CH_BLOCK_L].sum() == 1.0
    assert t[cf.CH_HOLE].sum() == 2.0
    # committed bits at 0, 5, 6, 7, 12 → 5 cells.
    assert t[cf.CH_COMMITTED].sum() == 5.0
    # Unknown = NOT committed AND NOT exit.  Exit (12) is already committed,
    # so unknown = 25 - 5 committed = 20 cells.
    assert t[cf.CH_UNKNOWN].sum() == 20.0
