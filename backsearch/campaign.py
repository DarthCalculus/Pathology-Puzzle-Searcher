#!/usr/bin/env python3
"""
campaign.py — resumable, parallel exhaustive search of one or more exits.

The DFS tree of an exit is split at depth K into independent sub-jobs using the
worker's `--list-layer K` (dedup'd layer, one `--seed-path` per node).  Each
job is a normal `backsearch_worker --seed-path P --time 0` run and is therefore
exhaustive on its own; the union of all jobs plus the (depth < K) ancestors is
the whole tree.  Completed jobs are appended to done.tsv so the campaign can be
stopped and resumed at any time, and the binary can be swapped between jobs
(any build that passes the README equivalence checks gives identical answers).

Usage:
  python3 campaign.py --out results/e0 --exits 0 --layer 8 --workers 3
  python3 campaign.py --out results/e0 --status          # progress summary only
  python3 campaign.py --out results/e0 --exits 0 --layer 8 --workers 3 \
        --extra "--num-holes 2"                         # capped variant

Everything after --extra is passed verbatim to every worker call (and to the
layer listing, so the caps shape the layer too).  --allow-exit-transit is always
added (real Pathology rules) unless --no-transit is given.

Concurrency can be changed live: echo N > DIR/workers (0 pauses after the
current jobs finish).

Range jobs and checkpoints (2026-09-21).  A job's path may also be a DFS-order
range "FROM..UNTIL" (either side empty): the worker runs `--from FROM --until
UNTIL`.  Ctrl-C (or --split-after seconds) interrupts the running workers,
which print a CURSOR line; the driver records the job as `continued` and
appends the continuation job "CURSOR..UNTIL" to jobs.tsv, so the next run
picks up exactly where it stopped.  Nothing is ever recomputed.
"""
import argparse, json, os, random, re, signal, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def list_layer(worker, exits, K, grid, extra):
    jobs = []
    for e in exits:
        r = sh([worker, "--grid", grid, "--two-tables", "--exit", str(e), "--list-layer", str(K)] + extra)
        if r.returncode != 0:
            sys.exit(f"--list-layer failed for exit {e}:\n{r.stderr}")
        for line in r.stdout.splitlines():
            if line.startswith("LAYER\t"):
                _, ex, path, depth, nb, nh = line.split("\t")
                jobs.append((int(ex), path, int(depth), int(nb), int(nh)))
    return jobs

RE_HDR   = re.compile(r"^--- Exit (\d+) \((.*?)\) ---", re.M)
RE_CUR   = re.compile(r"^CURSOR\t(\d+)\t([^\t]*)\t(\d+)", re.M)
STOP = {"flag": False}
RUNNING = {}   # pid -> Popen of a worker in flight (to forward SIGINT)
def on_sigint(sig, frame):
    STOP["flag"] = True
    for p in list(RUNNING.values()):
        try: p.send_signal(signal.SIGINT)
        except Exception: pass
    print("\n  ** stop requested: waiting for the running workers to checkpoint (CURSOR) ...", flush=True)
def job_args(path):
    """--seed-path P, or --from/--until for a range job 'FROM..UNTIL'."""
    if ".." not in path: return ["--seed-path", path]
    fr, un = path.split("..", 1); out = []
    if fr: out += ["--from", fr]
    if un: out += ["--until", un]
    return out
RE_EL    = re.compile(r"elapsed:\s+([\d.]+) s")
RE_ST    = re.compile(r"states checked: (\d+) \(shortcut (\d+), dedup (\d+)")
RE_SC    = re.compile(r"solver calls:\s+(\d+)")
RE_EV    = re.compile(r"evicts (\d+) / (\d+)")
RE_BD    = re.compile(r"best depth:\s+(\d+)(?:\s+\(verify (\d+) (\w+)\))?")

def run_job(worker, grid, exit_pos, path, extra, timeout=None):
    cmd = [worker, "--grid", grid, "--two-tables", "--exit", str(exit_pos)] + job_args(path) + ["--time", "0"] + extra
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    RUNNING[p.pid] = p
    try:
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.send_signal(signal.SIGINT)          # checkpoint instead of kill: the worker prints its CURSOR
            out, err = p.communicate()
    finally:
        RUNNING.pop(p.pid, None)
    wall = time.time() - t0
    class R: pass
    r = R(); r.returncode = p.returncode
    m = RE_HDR.search(out)
    status = m.group(2) if m else f"rc={r.returncode}"
    cm = RE_CUR.search(out)
    if cm and status != "exhausted": status = "continued"   # the continuation job carries the CURSOR
    def g(rx, k=1, d=-1):
        mm = rx.search(out); return int(float(mm.group(k))) if mm and mm.group(k) else d
    rec = dict(exit=exit_pos, path=path, status=status,
               states=g(RE_ST, 1), shortcut=g(RE_ST, 2), dedup=g(RE_ST, 3),
               calls=g(RE_SC), ev_s=g(RE_EV, 1, 0), ev_r=g(RE_EV, 2, 0),
               best=g(RE_BD, 1, 0), verify=(RE_BD.search(out).group(3) if RE_BD.search(out) and RE_BD.search(out).group(3) else "-"),
               elapsed=round(float(RE_EL.search(out).group(1)), 3) if RE_EL.search(out) else round(wall, 3),
               wall=round(wall, 3))
    rec["cursor"] = cm.group(2) if cm else None
    return rec, out, err

def list_children(worker, grid, exit_pos, path, extra, k):
    """The dedup'd depth+k layer under seed path `path`, as complete seed paths."""
    r = sh([worker, "--grid", grid, "--two-tables", "--exit", str(exit_pos), "--seed-path", path,
            "--list-layer", str(k)] + extra)
    kids = []
    for line in r.stdout.splitlines():
        if line.startswith("LAYER\t"):
            _, ex, p, depth, nb, nh = line.split("\t")
            kids.append((int(ex), p, int(depth), int(nb), int(nh)))
    return kids

COLS = ["exit", "path", "status", "states", "shortcut", "dedup", "calls", "ev_s", "ev_r", "best", "verify", "elapsed", "wall"]

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != len(COLS) or parts[0] == "exit": continue
                rec = dict(zip(COLS, parts))
                done[(int(rec["exit"]), rec["path"])] = rec
    return done

def fmt_dur(s):
    s = int(s); return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"

def status(out, jobs):
    done = load_done(os.path.join(out, "done.tsv"))
    total = len(jobs) if jobs else len(done)
    ok = [r for r in done.values() if r["status"] == "exhausted"]
    split = [r for r in done.values() if r["status"] in ("split", "continued")]
    bad = [r for r in done.values() if r["status"] not in ("exhausted", "split", "continued")]
    states = sum(int(r["states"]) for r in done.values())
    el = sum(float(r["elapsed"]) for r in done.values())
    best = max((int(r["best"]) for r in done.values()), default=0)
    by_exit = {}
    for r in done.values():
        d = by_exit.setdefault(int(r["exit"]), dict(n=0, best=0, el=0.0))
        d["n"] += 1; d["best"] = max(d["best"], int(r["best"])); d["el"] += float(r["elapsed"])
    print(f"[{out}] jobs done {len(done)}/{total}  exhausted {len(ok)}  split/continued {len(split)}  NOT exhausted {len(bad)}  "
          f"states {states:,}  cpu {fmt_dur(el)}  best depth {best}")
    for e in sorted(by_exit):
        d = by_exit[e]
        tot_e = sum(1 for j in jobs if j[0] == e) if jobs else d["n"]
        print(f"   exit {e}: {d['n']}/{tot_e} jobs, best {d['best']}, cpu {fmt_dur(d['el'])}")
    if bad:
        print("   non-exhausted jobs:"); [print("     ", r["exit"], r["path"], r["status"]) for r in bad[:10]]
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exits", default="")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--worker", default=os.path.join(HERE, "backsearch_worker"))
    ap.add_argument("--grid", default="5x5")
    ap.add_argument("--extra", default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--shuffle", action="store_true", help="random job order (better ETA by count)")
    ap.add_argument("--save-depth", type=int, default=100, help="keep full worker output for jobs reaching this depth")
    ap.add_argument("--split-after", type=float, default=0, help="seconds; a job still running after this is checkpointed (SIGINT -> CURSOR) and continued as a range job from the cursor (default 0 = never)")
    ap.add_argument("--split-k", type=int, default=3)
    ap.add_argument("--no-transit", action="store_true", help="omit --allow-exit-transit (NOT the Pathology rule set; restricted class only)")
    a = ap.parse_args()
    extra = a.extra.split()
    # The worker flags that define the class are part of the campaign's identity:
    # persist them on creation and reload them on resume, so a restart without
    # --extra cannot silently search a different class than done.tsv records.
    cfg_json = os.path.join(a.out, "config.json")
    if os.path.exists(cfg_json):
        saved = json.load(open(cfg_json))
        norm = lambda xs: [x for x in xs if x != "--allow-exit-transit"]   # transit is added automatically, so compare without it
        if a.extra and norm(extra) != norm(saved["extra"]):
            sys.exit(f"--extra {extra} differs from this campaign's saved extra {saved['extra']}; refusing")
        extra = saved["extra"]; a.grid = saved.get("grid", a.grid); a.no_transit = saved.get("no_transit", a.no_transit)
    # Pathology lets blocks be pushed over the exit; the record levels rely on it.
    # A level may not START with a block on the exit, so --allow-block-on-exit is never passed.
    if not a.no_transit and "--allow-exit-transit" not in extra:
        extra = ["--allow-exit-transit"] + extra
    if "--allow-block-on-exit" in extra: sys.exit("--allow-block-on-exit is not a legal Pathology setup; refusing")
    os.makedirs(a.out, exist_ok=True)
    jobs_file = os.path.join(a.out, "jobs.tsv")

    if os.path.exists(jobs_file):
        jobs = []
        with open(jobs_file) as f:
            for line in f:
                ex, path, depth, nb, nh = line.rstrip("\n").split("\t")
                jobs.append((int(ex), path, int(depth), int(nb), int(nh)))
    else:
        if a.status: sys.exit("no jobs.tsv yet")
        exits = [int(x) for x in a.exits.split(",") if x != ""]
        if not exits: sys.exit("--exits required for a new campaign")
        jobs = list_layer(a.worker, exits, a.layer, a.grid, extra)
        with open(jobs_file, "w") as f:
            for j in jobs: f.write("\t".join(map(str, j)) + "\n")
        with open(os.path.join(a.out, "config.txt"), "w") as f:
            f.write(f"grid={a.grid} exits={exits} layer={a.layer} extra={extra} worker={a.worker}\n")
        json.dump({"grid": a.grid, "exits": exits, "layer": a.layer, "extra": extra, "no_transit": a.no_transit},
                  open(cfg_json, "w"))
        print(f"layer K={a.layer}: {len(jobs)} jobs over exits {exits}")

    done = status(a.out, jobs)
    if a.status: return
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    if a.shuffle: random.Random(1).shuffle(todo)
    # Children of split jobs (deeper seeds than the layer) go FIRST, so the
    # heavy subtrees a split exposes are finished promptly instead of piling
    # up at the end, where they would make the job count look nearly done
    # while most of the work remained.
    layer_depth = min(j[2] for j in jobs)
    todo.sort(key=lambda j: 0 if j[2] > layer_depth else 1)
    if not todo: print("nothing to do"); return
    print(f"running {len(todo)} jobs on {a.workers} workers with {os.path.basename(a.worker)}  extra={extra}")

    lock = threading.Lock()
    import collections
    jq = collections.deque(todo)        # popleft() to take; appendleft() for split children
    done_f = open(os.path.join(a.out, "done.tsv"), "a")
    if os.path.getsize(os.path.join(a.out, "done.tsv")) == 0:
        done_f.write("\t".join(COLS) + "\n"); done_f.flush()
    best_seen = max((int(r["best"]) for r in done.values()), default=0)
    t_start = time.time(); n_done = [len(done)]; cpu = [sum(float(r["elapsed"]) for r in done.values())]

    def work(j):
        ex, path = j[0], j[1]
        rec, out, err = run_job(a.worker, a.grid, ex, path, extra,
                                timeout=(a.split_after if a.split_after > 0 else None))
        nonlocal best_seen
        if rec["status"] == "continued":
            # checkpointed: the rest of this job is the range [cursor, until)
            until = path.split("..", 1)[1] if ".." in path else ""
            cont = (ex, f"{rec['cursor']}..{until}", j[2], j[3], j[4])
            with lock:
                with open(jobs_file, "a") as f: f.write("\t".join(map(str, cont)) + "\n")
                jobs.append(cont)
                if not STOP["flag"]: jq.appendleft(cont)   # keep going now, or leave it for the next run
                done_f.write("\t".join(str(rec[c]) for c in COLS) + "\n"); done_f.flush()
                n_done[0] += 1; cpu[0] += rec["elapsed"]
                if rec["best"] > best_seen: best_seen = rec["best"]
                if rec["best"] >= a.save_depth:
                    tag = f"best{rec['best']:03d}_e{ex}_{path.replace(',', '').replace('..', '__')[:80]}"
                    with open(os.path.join(a.out, tag + ".txt"), "w") as f: f.write(" ".join([a.worker] + extra) + "\n" + out)
                print(f"  >> checkpointed exit {ex} {path} after {rec['wall']}s (best {rec['best']}, {rec['states']:,} states); "
                      f"continues as {cont[1][:60]}{'...' if len(cont[1]) > 60 else ''}", flush=True)
            return
        with lock:
            done_f.write("\t".join(str(rec[c]) for c in COLS) + "\n"); done_f.flush()
            n_done[0] += 1; cpu[0] += rec["elapsed"]
            if rec["best"] >= a.save_depth or rec["best"] > best_seen:
                tag = f"best{rec['best']:03d}_e{ex}_{path.replace(',', '').replace('..', '__')[:80]}"
                with open(os.path.join(a.out, tag + ".txt"), "w") as f: f.write(" ".join([a.worker] + extra) + "\n" + out)
            if rec["best"] > best_seen:
                best_seen = rec["best"]
                print(f"  ** new campaign best: depth {rec['best']} (exit {ex}, {path})", flush=True)
            if rec["status"] != "exhausted":
                print(f"  !! job not exhausted: exit {ex} {path} -> {rec['status']}\n{err[-500:]}", flush=True)
            el = time.time() - t_start
            frac = n_done[0] / len(jobs)
            eta = (el / max(1, n_done[0] - (len(jobs) - len(todo)))) * (len(jobs) - n_done[0])
            print(f"  [{n_done[0]}/{len(jobs)} {100*frac:5.1f}%] exit {ex} {path}: {rec['status']} "
                  f"states {rec['states']:,} best {rec['best']} {rec['elapsed']}s | cpu {fmt_dur(cpu[0])} "
                  f"wall {fmt_dur(el)} eta~{fmt_dur(eta)} best {best_seen}", flush=True)

    # Concurrency is re-read from DIR/workers before every job, so it can be
    # changed while the campaign runs:  echo 3 > DIR/workers
    wfile = os.path.join(a.out, "workers")
    if not os.path.exists(wfile):
        with open(wfile, "w") as f: f.write(str(a.workers) + "\n")
    def desired():
        try: return max(0, int(open(wfile).read().strip() or 0))
        except Exception: return a.workers
    active = [0]
    def runner():
        while True:
            # Take a slot and a job in ONE step, under the lock. Popping first
            # and then waiting for a slot let a thread sit on a job for hours:
            # the thread that had just finished re-checked immediately and won
            # every freed slot, while the sleepers never got one (starvation).
            j = None
            with lock:
                if STOP["flag"] and active[0] == 0: return
                if jq and active[0] < desired() and not STOP["flag"]:
                    j = jq.popleft(); active[0] += 1
                elif not jq and active[0] == 0:
                    return
            if j is None:
                time.sleep(1)
                continue
            try: work(j)
            finally:
                with lock: active[0] -= 1
    signal.signal(signal.SIGINT, on_sigint); signal.signal(signal.SIGTERM, on_sigint)
    threads = [threading.Thread(target=runner, daemon=True) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    done_f.close()
    if STOP["flag"]: print("  ** stopped; re-run the same command to continue from the checkpoints", flush=True)
    status(a.out, jobs)

if __name__ == "__main__":
    main()
