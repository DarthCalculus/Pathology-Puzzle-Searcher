#!/usr/bin/env python3
"""Stand-in for the C worker (backsearch_worker_nt) that speaks the exact line
protocol of DESIGN.md section 2, for testing volunteer.py without burning CPU.

It never spins: the "search" is a sequence of short sleeps.

Behaviour
  --version                 print "SRC_HASH\t<hex>" and exit 0
  --split-after S           split after S seconds (0 = never)
  --status-every MS         STATUS line cadence
  --seed-path S, --exit E, --grid G, --time T, --two-tables, extra flags: accepted
  SIGINT / SIGTERM          split immediately (REMAINING + LEVEL + SUMMARY)

Environment knobs
  FAKE_SRC_HASH             hash to claim (default a fixed fake value)
  FAKE_MIN_S / FAKE_MAX_S   run duration range in seconds (default 1 / 4);
                            deeper seeds (more tokens) run shorter so that a
                            split tree terminates
  FAKE_DIE                  if set, exit 1 without SUMMARY after 0.5 s (to test
                            the "run without SUMMARY" path)
  FAKE_NO_LEVEL             if set, emit no LEVEL line
"""
import json
import os
import random
import signal
import sys
import time

DEFAULT_HASH = "fa4e" + "0" * 60
TOKENS = [d + n for d in "URDL" for n in "123"]


def out(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def parse_args(argv):
    a = {"grid": "5x5", "exit": 0, "seed": "", "split_after": 0.0, "status_every": 250,
         "version": False, "extra": []}
    i = 0
    while i < len(argv):
        t = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if t == "--version":
            a["version"] = True
        elif t == "--grid" and nxt is not None:
            a["grid"] = nxt; i += 1
        elif t == "--exit" and nxt is not None:
            a["exit"] = int(nxt); i += 1
        elif t == "--seed-path" and nxt is not None:
            a["seed"] = nxt; i += 1
        elif t == "--split-after" and nxt is not None:
            a["split_after"] = float(nxt); i += 1
        elif t == "--status-every" and nxt is not None:
            a["status_every"] = int(nxt); i += 1
        elif t in ("--time", "--num-holes", "--num-blocks") and nxt is not None:
            a["extra"].append((t, nxt)); i += 1
        else:
            a["extra"].append((t, None))
        i += 1
    return a


def grid_dims(g):
    try:
        w, h = g.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 5, 5


def fake_board(w, h, rng, exit_cell, unknown_frac=0.0):
    """A plausible-looking level: floor, a few walls/holes/blocks, one exit at
    the exit cell (on the border, numbered clockwise-ish is irrelevant here), one
    player. '?' cells stand for uncommitted cells in the cursor board."""
    cells = ["0"] * (w * h)
    for _ in range(rng.randint(1, 3)):
        cells[rng.randrange(w * h)] = "1"
    for _ in range(rng.randint(0, 2)):
        cells[rng.randrange(w * h)] = "5"
    for _ in range(rng.randint(1, 3)):
        cells[rng.randrange(w * h)] = rng.choice("2789ABCDEFGHIJ")
    if unknown_frac:
        for k in range(w * h):
            if rng.random() < unknown_frac:
                cells[k] = "?"
    # exit on the top border at column exit_cell % w
    cells[exit_cell % w] = "3"
    p = rng.randrange(w * h)
    while cells[p] in ("3",):
        p = rng.randrange(w * h)
    cells[p] = "4"
    return "/".join("".join(cells[r * w:(r + 1) * w]) for r in range(h))


def main():
    args = parse_args(sys.argv[1:])
    src_hash = os.environ.get("FAKE_SRC_HASH", DEFAULT_HASH)
    if args["version"]:
        out("SRC_HASH\t" + src_hash)
        return 0

    out("SRC_HASH\t" + src_hash)
    out("fake worker: this line is free text and must be ignored by the client")

    w, h = grid_dims(args["grid"])
    seed = args["seed"]
    seed_tokens = [t for t in seed.split(",") if t]
    rng = random.Random()
    depth0 = len(seed_tokens)

    min_s = float(os.environ.get("FAKE_MIN_S", "1"))
    max_s = float(os.environ.get("FAKE_MAX_S", "4"))
    extra_depth = max(0, depth0 - 8)
    duration = rng.uniform(min_s, max_s) * (0.6 ** extra_depth)
    if os.environ.get("FAKE_DIE"):
        time.sleep(0.5)
        out("fake worker: dying without SUMMARY on purpose")
        return 1

    interrupted = {"flag": False}

    def on_sig(signum, frame):
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    start = time.time()
    last_status = -1.0
    split_after = args["split_after"]
    every = max(0.05, args["status_every"] / 1000.0)
    best = depth0
    depth = depth0
    states = 0
    split = False
    try:
        while True:
            time.sleep(min(every, 0.1))
            now = time.time() - start
            states += rng.randint(50, 500)
            if rng.random() < 0.6:
                depth = min(depth + 1, depth0 + 40)
            win_best = depth if rng.random() < 0.5 else best
            if depth > best:
                best = depth
            if interrupted["flag"]:
                split = True
                break
            if split_after > 0 and now >= split_after:
                split = True
                break
            if now >= duration:
                break
            if now - last_status >= every:
                last_status = now
                status = {
                    "depth": depth,
                    "cur": fake_board(w, h, rng, args["exit"], unknown_frac=0.3),
                    "best": best,
                    "win_best": win_best,
                    "win": fake_board(w, h, rng, args["exit"]),
                }
                out("STATUS\t" + json.dumps(status, separators=(",", ":")))
    except BrokenPipeError:
        return 1

    if split:
        # cursor first, then the stack top-down: 1..4 disjoint extensions of the seed.
        n_lines = rng.randint(1, 4)
        seen = set()
        while len(seen) < n_lines:
            ext = ",".join(rng.choice(TOKENS) for _ in range(rng.randint(1, 3)))
            path = (seed + "," + ext) if seed else ext
            seen.add(path)
        for p in sorted(seen):
            out("REMAINING\t" + p)

    if not os.environ.get("FAKE_NO_LEVEL"):
        out("LEVEL\t%d\t%s" % (best, fake_board(w, h, rng, args["exit"])))

    summary = {
        "status": "split" if split else "exhausted",
        "seed": seed,
        "exit": args["exit"],
        "states": states,
        "accepted": states // 3,
        "valid": states // 7,
        "best": best,
        "evict_shallow": 0,
        "evict_recent": 0,
        "elapsed": round(time.time() - start, 3),
        "solver_calls": states // 11,
    }
    out("SUMMARY\t" + json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(1)
