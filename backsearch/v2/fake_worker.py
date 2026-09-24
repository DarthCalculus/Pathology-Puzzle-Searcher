#!/usr/bin/env python3
"""Stand-in for the C worker (backsearch_worker_nt) that speaks the protocol-3 line
protocol (v2/PROTOCOL3.md section 2), for testing volunteer.py without burning CPU.

It never spins: the "search" is a sequence of short sleeps.

Behaviour
  --version                 SRC_HASH, GIT_SHA, PROTOCOL 3, LIMITS, KNOBS (tab-separated) and exit 0
  --split-after S           split after S seconds (0 = never)
  --status-every MS         STATUS line cadence
  --seed-path S             the job's seed (omitted = the exit root "")
  --exit E, --grid RxC, --time T, --two-tables, --allow-exit-transit, --num-holes N,
  --num-blocks N, --min-walls N: accepted; the search flags are echoed in SUMMARY flags
  SIGINT / SIGTERM          split now: CURSOR + FRONTIER, REMAINING, UNRESOLVED, LEVEL, SUMMARY
                            (the handler is installed before the first output line, like the real worker)
  exit codes                exhausted/split 0, bad_seed 3, path_overflow 4, error 5

Randomness is seeded from FAKE_SEED and the seed path, so a given job behaves the
same in every run (reproducible tests).

Environment knobs
  FAKE_SRC_HASH             hash to claim (default a fixed fake value)
  FAKE_SEED                 RNG seed salt (default 1)
  FAKE_MIN_S / FAKE_MAX_S   run duration range in seconds (default 1 / 4); deeper seeds
                            (more tokens than 8) run shorter so that a split tree terminates
  FAKE_DIE                  exit 1 without SUMMARY after 0.5 s (a crash)
  FAKE_STATUS               bad_seed | path_overflow | error: end with that status and its exit code
  FAKE_NO_LEVEL             emit no LEVEL line
  FAKE_REMAINING_N          on a split, print exactly N REMAINING lines (default 1..4)
  FAKE_REMAINING_TOKS       extra tokens per REMAINING line (long seeds; default 0)
  FAKE_REMAINING_SELF       on a split, print REMAINING == the seed (interrupted before expanding)
  FAKE_BAD_REMAINING        one malformed REMAINING line among the valid ones
  FAKE_UNRESOLVED           print this many UNRESOLVED candidates (deeper than the best)
  FAKE_SIGINT_DELAY_S       wait this long after SIGINT before answering (slow SIGINT)
  FAKE_SIGINT_IGNORE        ignore SIGINT/SIGTERM entirely
  FAKE_HANG_AFTER_S         after this many seconds stop printing and ignore the split time (a hang)
  FAKE_FLAGS_GRID           claim this grid in SUMMARY flags (tests the client's flags check)
  FAKE_ONLY_EXIT            apply FAKE_DIE/FAKE_STATUS/FAKE_HANG_AFTER_S/FAKE_BAD_REMAINING only on this exit
"""
import hashlib
import json
import os
import random
import signal
import sys
import time

DEFAULT_HASH = "fa4e" + "0" * 60
TOKENS = [d + n for d in "URDL" for n in "123"]
PATH_TOK_MAX = 1024
EXIT_CODES = {"exhausted": 0, "split": 0, "bad_seed": 3, "path_overflow": 4, "error": 5}

interrupted = {"flag": False, "at": 0.0}


def on_sig(signum, frame):
    if os.environ.get("FAKE_SIGINT_IGNORE"):
        return
    if not interrupted["flag"]:
        interrupted["flag"] = True
        interrupted["at"] = time.time()


# like backsearch.c: the handler exists before the first line is printed (test_client e flake)
signal.signal(signal.SIGINT, on_sig)
signal.signal(signal.SIGTERM, on_sig)


def out(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def parse_args(argv):
    a = {"grid": "5x5", "exit": 0, "seed": "", "split_after": 0.0, "status_every": 250,
         "version": False, "transit": 0, "block_on_exit": 0, "holes": 3, "blocks": 32, "walls": 0}
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
        elif t == "--num-holes" and nxt is not None:
            a["holes"] = int(nxt); i += 1
        elif t == "--num-blocks" and nxt is not None:
            a["blocks"] = int(nxt); i += 1
        elif t == "--min-walls" and nxt is not None:
            a["walls"] = int(nxt); i += 1
        elif t == "--time" and nxt is not None:
            i += 1
        elif t == "--allow-exit-transit":
            a["transit"] = 1
        elif t == "--allow-block-on-exit":
            a["block_on_exit"] = 1
        i += 1
    return a


def grid_dims(g):
    """'RxC' -> (rows, cols), as the server's parseGrid."""
    try:
        r, c = g.lower().split("x")
        return int(r), int(c)
    except Exception:
        return 5, 5


def fake_board(rows, cols, rng, exit_cell, unknown_frac=0.0):
    """A plausible-looking level: floor, a few walls/holes/blocks, the exit at the true cell
    index exit_cell (row exit_cell // cols), one player. '?' = uncommitted (cursor board)."""
    n = rows * cols
    cells = ["0"] * n
    for _ in range(rng.randint(1, 3)):
        cells[rng.randrange(n)] = "1"
    for _ in range(rng.randint(0, 2)):
        cells[rng.randrange(n)] = "5"
    for _ in range(rng.randint(1, 3)):
        cells[rng.randrange(n)] = rng.choice("2789ABCDEFGHIJ")
    if unknown_frac:
        for k in range(n):
            if rng.random() < unknown_frac:
                cells[k] = "?"
    cells[exit_cell % n] = "3"
    p = rng.randrange(n)
    while cells[p] == "3":
        p = rng.randrange(n)
    cells[p] = "4"
    return "/".join("".join(cells[r * cols:(r + 1) * cols]) for r in range(rows))


def ext_tokens(k, width):
    """The k-th distinct extension of `width` tokens (base 12)."""
    toks = []
    for _ in range(width):
        toks.append(TOKENS[k % 12])
        k //= 12
    return toks


def main():
    args = parse_args(sys.argv[1:])
    src_hash = os.environ.get("FAKE_SRC_HASH", DEFAULT_HASH)
    if args["version"]:
        out("SRC_HASH\t" + src_hash)
        out("GIT_SHA\t")
        out("PROTOCOL\t3")
        out("LIMITS\t" + json.dumps({"path_tok_max": PATH_TOK_MAX, "max_ncells": 64, "max_blocks": 32, "state_bits": 128},
                                     separators=(",", ":")))
        out("KNOBS\t{}")
        return 0

    out("SRC_HASH\t" + src_hash)
    out("fake worker: this line is free text and must be ignored by the client")

    rows, cols = grid_dims(args["grid"])
    seed = args["seed"]
    seed_tokens = [t for t in seed.split(",") if t]
    salt = os.environ.get("FAKE_SEED", "1")
    rng = random.Random(int(hashlib.sha256(("%s|%s|%d" % (salt, seed, args["exit"])).encode()).hexdigest()[:12], 16))
    depth0 = len(seed_tokens)
    only_exit = os.environ.get("FAKE_ONLY_EXIT")
    targeted = only_exit is None or only_exit == "" or int(only_exit) == args["exit"]

    min_s = float(os.environ.get("FAKE_MIN_S", "1"))
    max_s = float(os.environ.get("FAKE_MAX_S", "4"))
    extra_depth = max(0, depth0 - 8)
    duration = rng.uniform(min_s, max_s) * (0.6 ** extra_depth)
    if targeted and os.environ.get("FAKE_DIE"):
        time.sleep(0.5)
        out("fake worker: dying without SUMMARY on purpose")
        return 1
    status_override = os.environ.get("FAKE_STATUS") if targeted else None
    if status_override:
        time.sleep(0.3)
        sys.stderr.write("fake worker: ending with status %s on purpose\n" % status_override)
        out("SUMMARY\t" + json.dumps({"status": status_override, "seed": seed, "exit": args["exit"], "states": 0,
                                        "protocol": 3, "error": "fake %s" % status_override}, separators=(",", ":")))
        return EXIT_CODES.get(status_override, 5)
    hang_after = float(os.environ.get("FAKE_HANG_AFTER_S", "0") or 0) if targeted else 0.0
    sigint_delay = float(os.environ.get("FAKE_SIGINT_DELAY_S", "0") or 0)

    start = time.time()
    t_cpu0 = time.process_time()
    last_status = -1.0
    split_after = args["split_after"]
    every = max(0.05, args["status_every"] / 1000.0)
    best = depth0
    depth = depth0
    states = 1                     # a real run always checks its seed
    split = False
    try:
        while True:
            time.sleep(min(every, 0.1))
            now = time.time() - start
            if hang_after and now >= hang_after:
                while True:        # a hung solve: silent, deaf to the split time
                    time.sleep(0.2)
                    if interrupted["flag"]:
                        break
                split = True
                break
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
                    "cur": fake_board(rows, cols, rng, args["exit"], unknown_frac=0.3),
                    "best": best,
                    "win_best": win_best,
                    "win": fake_board(rows, cols, rng, args["exit"]),
                    "unknown": 0,
                }
                out("STATUS\t" + json.dumps(status, separators=(",", ":")))
        if split and interrupted["flag"] and sigint_delay:
            time.sleep(sigint_delay)

        if split:
            if interrupted["flag"]:
                cur = seed or ""
                out("CURSOR\t%d\t%s\t%d" % (args["exit"], cur, depth0))
                out("FRONTIER\t%s" % (cur,))
            if os.environ.get("FAKE_REMAINING_SELF"):
                out("REMAINING\t" + seed)
            else:
                n_env = os.environ.get("FAKE_REMAINING_N")
                pad = int(os.environ.get("FAKE_REMAINING_TOKS", "0") or 0)
                lines = []
                if n_env:
                    n = int(n_env)
                    width = 1
                    while 12 ** width < n:
                        width += 1
                    for k in range(n):
                        toks = ext_tokens(k, width) + [TOKENS[(k + j) % 12] for j in range(pad)]
                        lines.append(",".join(seed_tokens + toks))
                else:
                    # cursor first, then the stack top-down: 1..4 distinct extensions; the last
                    # one is 3 tokens deep, so an absorption probe always finishes it
                    n_lines = rng.randint(1, 4)
                    seen = set()
                    while len(seen) < n_lines - 1:
                        ext = [rng.choice(TOKENS) for _ in range(rng.randint(1, 2))]
                        seen.add(",".join(seed_tokens + ext + [TOKENS[j % 12] for j in range(pad)]))
                    lines = sorted(seen)
                    tail = ",".join(seed_tokens + [rng.choice(TOKENS) for _ in range(3)] + [TOKENS[j % 12] for j in range(pad)])
                    while tail in seen:
                        tail = ",".join(seed_tokens + [rng.choice(TOKENS) for _ in range(3)])
                    lines.append(tail)
                if targeted and os.environ.get("FAKE_BAD_REMAINING") and lines:
                    lines.insert(len(lines) // 2, (seed + "," if seed else "") + "X9")
                for p in lines:
                    out("REMAINING\t" + p)

        n_unres = int(os.environ.get("FAKE_UNRESOLVED", "0") or 0)
        for k in range(n_unres):
            # deeper than the best, and (like the real worker) depth == the path's token count
            width = best + 1 + k - depth0
            path = ",".join(seed_tokens + ext_tokens(k, width))
            out("UNRESOLVED\t%d\t%s\t%s\t%s" % (depth0 + width, fake_board(rows, cols, rng, args["exit"]), path,
                                                ("pq", "probe", "big")[k % 3]))

        if not os.environ.get("FAKE_NO_LEVEL"):
            out("LEVEL\t%d\t%s" % (best, fake_board(rows, cols, rng, args["exit"])))

        flags = {"grid": os.environ.get("FAKE_FLAGS_GRID") or args["grid"], "exit": args["exit"],
                 "transit": args["transit"], "block_on_exit": args["block_on_exit"], "max_holes": args["holes"],
                 "max_blocks": args["blocks"], "min_walls": args["walls"], "bulk_walk": 1}
        summary = {
            "status": "split" if split else "exhausted",
            "seed": seed,
            "exit": args["exit"],
            "states": states,
            "accepted": states // 3,
            "valid": states // 7,
            "best": best,
            "verify": best,
            "evict_shallow": 0,
            "evict_recent": 0,
            "elapsed": round(time.time() - start, 3),
            "solver_calls": states // 11,
            "src_hash": src_hash,
            "protocol": 3,
            "cpu_s": round(time.process_time() - t_cpu0, 3),
            "unknown": n_unres, "unknown_pq": (n_unres + 2) // 3, "unknown_probe": (n_unres + 1) // 3,
            "unknown_big": n_unres // 3, "unresolved": n_unres,
            "flags": flags,
        }
        out("SUMMARY\t" + json.dumps(summary, separators=(",", ":")))
    except BrokenPipeError:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(1)
