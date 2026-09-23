#!/usr/bin/env python3
"""Pathology collective search v2: volunteer client (DESIGN.md section 4).

    python3 volunteer.py --name "Your name" [--workers N] [--server URL] [--port 8765]

Python 3.9+, standard library only. Runs on macOS, Linux and WSL.

The client leases subtree jobs from the server, runs one worker process per
slot, parses the worker's tab-separated protocol lines (SRC_HASH, REMAINING,
LEVEL, STATUS, SUMMARY), reports results, and serves a small local GUI on
127.0.0.1:PORT. Ctrl-C = Stop (workers get SIGINT, print their split, the
splits are reported, then the client exits).
"""
import argparse
import collections
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BACKSEARCH_DIR = os.path.dirname(HERE)
DEFAULT_SERVER = "https://pathology.georgespahn.com"
TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".pathology_volunteer.json")
UI_FILE = os.path.join(HERE, "volunteer_ui.html")
DIRTY_FILES = ["backsearch.c", "sokoban_bfs.c", "sokoban_bfs.h", os.path.join("v2", "volunteer.py")]

SEED_RE = re.compile(r"^(?:[URDL][123])(?:,[URDL][123])*$")
HEX_RE = re.compile(r"^[0-9a-fA-F]{8,128}$")
HEARTBEAT_S = 30.0
STATUS_EVERY_MS = 250
LEASE_CAP = 200              # ceiling of one lease request (7.6)
MAX_NODES = 5000             # nodes per tree report (7.5)
STOP_GRACE_S = 90.0          # after SIGINT, how long a worker may take to print SUMMARY
WAKE_HEARTBEAT_RETRY_S = 120.0
MAX_LINE = 20000
BOARD_CAP = 200


def log(msg, ring=None):
    line = "[volunteer %s] %s" % (time.strftime("%H:%M:%S"), msg)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    if ring is not None:
        ring.append(line)


class ApiError(Exception):
    def __init__(self, status, body, transient):
        Exception.__init__(self, "HTTP %s: %s" % (status, body))
        self.status = status
        self.body = body
        self.transient = transient


class Api:
    """Tiny JSON-over-HTTP client with connection-state tracking."""

    def __init__(self, server):
        self.server = server.rstrip("/")
        self.connected = False
        self.last_ok = 0.0
        self.last_error = ""

    def post(self, path, body, timeout=20.0):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.server + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "pathology-volunteer/2"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(4 * 1024 * 1024)
                status = r.status
        except urllib.error.HTTPError as e:
            raw = e.read(65536)
            status = e.code
            try:
                obj = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                obj = {"error": raw.decode("utf-8", "replace")[:200]}
            transient = status >= 500 or status == 429
            if transient:
                self.connected = False
                self.last_error = "HTTP %d" % status
            else:
                # the server is reachable; the request itself was refused
                self.connected = True
                self.last_ok = time.time()
            raise ApiError(status, obj, transient)
        except Exception as e:  # network errors, timeouts, bad URL
            self.connected = False
            self.last_error = str(e)[:200]
            raise ApiError(0, {"error": str(e)[:200]}, True)
        self.connected = True
        self.last_ok = time.time()
        self.last_error = ""
        try:
            return json.loads(raw.decode("utf-8", "replace") or "{}")
        except Exception:
            raise ApiError(status, {"error": "bad json from server"}, True)


class Slot:
    """One worker slot: a thread that runs worker processes one job at a time."""

    def __init__(self, idx):
        self.idx = idx
        self.thread = None
        self.proc = None
        self.job = None
        self.state = "idle"           # idle | running | paused | finishing | retiring | dead
        self.retire = False
        self.started_at = 0.0
        self.stop_sent_at = 0.0
        self.killed_by_client = False
        self.cur = None               # {"depth", "code"} node being expanded ('?' = unknown)
        self.best = None              # best accepted state this job
        self.win_hist = collections.deque()  # (t, depth, code) from STATUS win_* fields
        self.last_line_at = 0.0
        self.consecutive_failures = 0
        self.cur_seed = None          # seed of the run in progress (a node of the local tree)
        self.stack = []               # local LIFO stack of open seeds for this window
        self.nodes_done = 0
        self.window_end = 0.0
        self.phase = None             # window | absorb | None
        self.run_started_at = 0.0
        self.saw_output = False       # first line (SRC_HASH) read: the worker is past startup
        self.stop_pending = False     # Stop requested before the worker printed anything

    def win_last_second(self, now):
        while self.win_hist and now - self.win_hist[0][0] > 1.0:
            self.win_hist.popleft()
        best = None
        for _, d, code in self.win_hist:
            if best is None or d > best["depth"]:
                best = {"depth": d, "code": code}
        return best

    def snapshot(self, now):
        j = self.job
        return {
            "idx": self.idx,
            "state": self.state,
            "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
            "job_id": j["id"] if j else None,
            "exit": j["exit"] if j else None,
            "seed": j["seed"] if j else None,
            "elapsed": round(now - self.started_at, 1) if self.job and self.started_at else 0,
            "cur_seed": self.cur_seed if self.job else None,
            "phase": self.phase,
            "window_left": max(0, round(self.window_end - now)) if self.job and self.window_end else None,
            "stack_size": len(self.stack) if self.job else 0,
            "nodes_done": self.nodes_done if self.job else 0,
            "cur": self.cur,
            "best": self.best,
            "win": self.win_last_second(now),
        }


class Volunteer:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)   # queue / stop / retire changes
        self.pending = []                            # reports waiting for the next batch
        self.send_event = threading.Event()
        self.sender_may_exit = False
        self.last_send_at = 0.0
        self.last_lease_at = 0.0
        self.job_durations = collections.deque(maxlen=20)
        self.ring = collections.deque(maxlen=60)
        self.api = Api(args.server)
        self.token = None
        self.name = None
        self.campaign = {}
        self.src_hash = None
        self.workers = max(1, args.workers)
        self.slots = []
        self.queue = collections.deque()      # leased, not yet started jobs
        self.paused = False
        self.stopping = False
        self.stop_reason = ""
        self.exit_code = 0
        self.outbox = args.outbox
        self.stats = {"jobs_done": 0, "done": 0, "split": 0, "open_root": 0, "nodes_done": 0, "nodes_open": 0,
                      "runs": 0, "cpu_seconds": 0.0, "reports_sent": 0, "batches_sent": 0,
                      "reports_failed": 0, "runs_without_summary": 0}
        self.best_per_exit = {}
        self.last_heartbeat = 0.0
        self.last_heartbeat_ok = 0.0
        self.last_tick = time.time()
        self.leases_from_server = None
        self.ui_last_poll = 0.0
        self.ui_close_at = 0.0
        self.no_jobs_until = 0.0
        self.last_no_jobs_log = 0.0
        self.sigint_count = 0

    def log(self, msg):
        log(msg, self.ring)

    # ------------------------------------------------------------ startup
    def check_dirty(self):
        if os.environ.get("VOLUNTEER_ALLOW_DIRTY") == "1":
            self.log("VOLUNTEER_ALLOW_DIRTY=1: skipping the clean-checkout check (development only)")
            return
        try:
            out = subprocess.run(["git", "-C", BACKSEARCH_DIR, "status", "--porcelain", "--"] + DIRTY_FILES,
                                 capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log("warning: cannot run git to verify the checkout (%s); continuing" % e)
            return
        if out.returncode != 0:
            self.log("warning: git status failed (%s); continuing" % out.stderr.strip()[:200])
            return
        if out.stdout.strip():
            sys.stderr.write(
                "\nRefusing to run: these files differ from the committed version:\n%s\n"
                "Run `git checkout -- backsearch.c sokoban_bfs.c sokoban_bfs.h v2/volunteer.py` in the\n"
                "backsearch directory (and rebuild), or set VOLUNTEER_ALLOW_DIRTY=1 for development.\n" % out.stdout)
            sys.exit(2)

    def worker_argv_base(self):
        w = self.args.worker
        cands = [w, os.path.join(BACKSEARCH_DIR, w)] if not os.path.isabs(w) else [w]
        path = None
        for c in cands:
            if os.path.isfile(c):
                path = os.path.abspath(c)
                break
        if path is None:
            sys.stderr.write("Worker binary not found: %s\nBuild it in %s with build_pgo.sh (see VOLUNTEERS.md).\n"
                             % (w, BACKSEARCH_DIR))
            sys.exit(2)
        if path.endswith(".py"):
            return [sys.executable, path]
        return [path]

    def read_version(self):
        argv = self.worker_argv_base() + ["--version"]
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=BACKSEARCH_DIR)
        except (OSError, subprocess.TimeoutExpired) as e:
            sys.stderr.write("Cannot run the worker (%s): %s\n" % (" ".join(argv), e))
            sys.exit(2)
        for line in out.stdout.splitlines():
            parts = line.rstrip("\r").split("\t")
            if parts[0] == "SRC_HASH" and len(parts) >= 2 and HEX_RE.match(parts[1].strip()):
                return parts[1].strip().lower()
        sys.stderr.write("The worker did not print an SRC_HASH line for --version (exit %s). It is too old;\n"
                         "update the checkout and rebuild: git pull && ./build_pgo.sh\n" % out.returncode)
        sys.exit(2)

    def load_token_file(self):
        try:
            with open(TOKEN_FILE) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def save_token_file(self, d):
        tmp = TOKEN_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, TOKEN_FILE)

    def apply_campaign(self, camp):
        if isinstance(camp, dict) and camp:
            with self.lock:
                self.campaign.update(camp)

    def register(self):
        store = self.load_token_file()
        entry = store.get(self.api.server) if not self.args.reregister else None
        if isinstance(entry, dict) and entry.get("token"):
            self.token = entry["token"]
            self.name = entry.get("name") or self.args.name or "?"
            if self.args.name and self.args.name != self.name:
                self.log("note: already registered as %r for %s; the name is fixed after registration "
                         "(use --reregister to start over)" % (self.name, self.api.server))
            # first heartbeat: validates the token and fetches the campaign parameters
            try:
                res = self.api.post("/api/v2/heartbeat", {"token": self.token, "workers": self.workers,
                                                          "paused": False, "boards": []})
                self.apply_campaign(res.get("campaign"))
                if res.get("revoked"):
                    sys.stderr.write("This client's token was revoked by the server. Stopping.\n")
                    sys.exit(2)
                self.log("reusing registration %r (token %s...)" % (self.name, self.token[:6]))
                if self.campaign:
                    return
                self.log("heartbeat carried no campaign parameters; registering again to fetch them")
            except ApiError as e:
                if e.status in (401, 403, 404):
                    self.log("stored token is no longer valid (%s); registering again" % e)
                else:
                    self.log("cannot reach the server yet (%s); will retry" % e)
                    while not self.stopping:
                        time.sleep(10)
                        try:
                            res = self.api.post("/api/v2/heartbeat", {"token": self.token, "workers": self.workers,
                                                                      "paused": False, "boards": []})
                            self.apply_campaign(res.get("campaign"))
                            if self.campaign:
                                return
                            break
                        except ApiError as e2:
                            if e2.status in (401, 403, 404):
                                break
                            self.log("still cannot reach the server (%s)" % e2)
                    if self.stopping:
                        sys.exit(0)
        name = self.args.name or (entry or {}).get("name") if isinstance(entry, dict) else self.args.name
        if not name:
            sys.stderr.write("No registration found for %s. Pass --name \"Your name\".\n" % self.api.server)
            sys.exit(2)
        name = name.strip()[:40]
        while not self.stopping:
            try:
                res = self.api.post("/api/v2/register", {"name": name, "workers": self.workers})
            except ApiError as e:
                if e.status == 503 and isinstance(e.body, dict) and "queue_position" in e.body:
                    self.log("the campaign is full; queue position %s. Retrying in 60 s."
                             % e.body.get("queue_position"))
                elif e.status == 400:
                    sys.stderr.write("Registration refused: %s\n" % e.body)
                    sys.exit(2)
                else:
                    self.log("registration failed (%s); retrying in 30 s" % e)
                time.sleep(60 if e.status == 503 else 30)
                continue
            if not res.get("token"):
                sys.stderr.write("Registration returned no token: %s\n" % json.dumps(res)[:300])
                sys.exit(2)
            self.token = res["token"]
            self.name = name
            self.apply_campaign(res.get("campaign"))
            store[self.api.server] = {"token": self.token, "name": name, "registered_at": time.time()}
            self.save_token_file(store)
            self.log("registered as %r (token stored in %s)" % (name, TOKEN_FILE))
            return
        sys.exit(0)

    def check_hash(self):
        hashes = [str(h).lower() for h in (self.campaign.get("hashes") or [])]
        if not hashes:
            self.log("warning: the campaign lists no whitelisted hashes; reports may be refused")
            return
        if self.src_hash not in hashes:
            sys.stderr.write(
                "\nThe worker's source hash %s is not accepted by the campaign.\n"
                "Accepted: %s\nUpdate and rebuild in the backsearch directory:\n"
                "    git pull && ./build_pgo.sh\nthen start volunteer.py again.\n"
                % (self.src_hash, ", ".join(hashes)))
            sys.exit(2)

    # ------------------------------------------------------------ campaign params
    def param(self, key, default):
        v = self.campaign.get(key)
        return default if v is None else v

    def lease_s(self):
        try:
            return float(self.param("lease_s", 3600))
        except (TypeError, ValueError):
            return 3600.0

    def worker_argv(self, job, seed, split_after):
        extra = self.param("extra", [])
        if isinstance(extra, str):
            extra = shlex.split(extra)
        extra = [str(x) for x in extra if str(x).strip()]
        return self.worker_argv_base() + [
            "--grid", str(self.param("grid", "5x5")), "--two-tables",
            "--exit", str(job["exit"]), "--seed-path", seed,
            "--time", "0", "--split-after", "%.1f" % split_after,
            "--status-every", str(STATUS_EVERY_MS)] + extra

    def fparam(self, key, default):
        try:
            return float(self.param(key, default))
        except (TypeError, ValueError):
            return float(default)

    def mean_job_s(self):
        """Recent mean wall time of a window (job), used to size lease requests."""
        with self.lock:
            if self.job_durations:
                return max(0.05, sum(self.job_durations) / len(self.job_durations))
        return max(1.0, self.fparam("split_after_s", 1800))

    # ------------------------------------------------------------ leasing (7.6)
    def lease_want(self):
        """How many more jobs to ask for: lease_ahead_s of work per idle worker at the
        recent mean job duration, floor 1 per idle worker, minus what is already queued."""
        with self.lock:
            idle = sum(1 for s in self.slots if s.job is None and not s.retire
                       and s.thread is not None and s.thread.is_alive())
            queued = len(self.queue)
        if idle == 0:
            return 0
        per_worker = max(1, int(math.ceil(self.fparam("lease_ahead_s", 300) / self.mean_job_s())))
        cap = int(self.fparam("lease_cap", LEASE_CAP))
        return max(0, min(cap, idle * per_worker) - queued)

    def leaser_loop(self):
        while True:
            with self.cond:
                self.cond.wait(0.5)
                if self.stopping:
                    return
            if self.paused or time.time() < self.no_jobs_until:
                continue
            want = self.lease_want()
            if want <= 0 or time.time() - self.last_lease_at < self.fparam("batch_interval_s", 10):
                continue
            self.last_lease_at = time.time()
            try:
                res = self.api.post("/api/v2/lease", {"token": self.token, "n": want})
            except ApiError as e:
                if e.status == 403 and isinstance(e.body, dict) and e.body.get("error") == "revoked":
                    self.request_stop("token revoked by the server")
                    return
                self.log("lease failed (%s); retrying in %d s" % (e, int(self.fparam("batch_interval_s", 10))))
                continue
            jobs = res.get("jobs") if isinstance(res, dict) else None
            try:
                resp_sa = float(res.get("split_after_s")) if isinstance(res, dict) and res.get("split_after_s") is not None else None
            except (TypeError, ValueError):
                resp_sa = None
            good = []
            for j in jobs or []:
                if not isinstance(j, dict):
                    continue
                try:
                    jid = int(j["id"])
                    ex = int(j["exit"])
                    seed = str(j["seed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not SEED_RE.match(seed) or len(seed) > 4000:
                    self.log("ignoring job %s with an unparsable seed" % jid)
                    continue
                try:
                    sa = float(j["split_after_s"]) if j.get("split_after_s") is not None else resp_sa
                except (TypeError, ValueError):
                    sa = resp_sa
                if sa is None:
                    sa = self.fparam("split_after_s", 1800)
                good.append({"id": jid, "exit": ex, "seed": seed, "split_after_s": max(1.0, sa),
                             "leased_at": time.time()})
            with self.cond:
                if good:
                    self.queue.extend(good)
                    self.cond.notify_all()
                else:
                    wait = max(5.0, self.fparam("batch_interval_s", 10))
                    self.no_jobs_until = time.time() + wait
                    if time.time() - self.last_no_jobs_log > 60:
                        self.last_no_jobs_log = time.time()
                        self.log("no open jobs available right now; asking again every %.0f s" % wait)

    def take_job(self, slot):
        """Block until a leased job is available (or stop/retire/pause)."""
        with self.cond:
            while True:
                if self.stopping or slot.retire or self.paused:
                    return None
                if self.queue:
                    return self.queue.popleft()
                self.cond.wait(0.5)

    def slot_loop(self, slot):
        while True:
            with self.lock:
                if self.stopping or slot.retire:
                    slot.state = "dead"
                    return
            if self.paused:
                slot.state = "idle"
                time.sleep(0.25)
                continue
            slot.state = "idle"
            job = self.take_job(slot)
            if job is None:
                continue
            ok = self.run_window(slot, job)
            wait = 0.0
            with self.lock:
                if ok or slot.killed_by_client:
                    slot.consecutive_failures = 0
                else:
                    slot.consecutive_failures += 1
                    if slot.consecutive_failures >= 3:
                        wait = min(600.0, 30.0 * 2 ** (slot.consecutive_failures - 3))
            if wait:
                self.log("worker %d failed %d times in a row; backing off %.0f s"
                         % (slot.idx, slot.consecutive_failures, wait))
                for _ in range(int(wait * 4)):   # never sleep while holding the lock
                    if self.stopping or slot.retire:
                        break
                    time.sleep(0.25)

    # ------------------------------------------------------------ window scheduler (7.2, 7.3)
    def run_window(self, slot, job):
        """Work job J and its local subtree for one window, absorb trivial children,
        then queue ONE tree report. Returns True when a report was produced."""
        start = time.time()
        deadline = start + job["split_after_s"]
        nodes = []            # in insertion order: J first, then children (parent precedes child)
        index = {}
        root = {"seed": job["seed"], "parent": None, "status": "open"}
        nodes.append(root)
        index[job["seed"]] = root
        stack = [job["seed"]]
        run_hash = None
        with self.lock:
            slot.job = job
            slot.started_at = start
            slot.window_end = deadline
            slot.stack = stack
            slot.nodes_done = 0
            slot.phase = "window"
            slot.killed_by_client = False
        self.log("worker %d: job %d exit %d window %.0f s seed %s"
                 % (slot.idx, job["id"], job["exit"], job["split_after_s"], job["seed"]))
        while stack and not self.stopping:
            now = time.time()
            if now >= deadline:
                break
            seed = stack.pop()
            res = self.run_worker(slot, job, seed, max(1.0, deadline - now))
            node = index[seed]
            if res is None:
                if slot.killed_by_client:
                    break
                if seed == job["seed"]:
                    with self.lock:
                        slot.job = None
                        slot.phase = None
                    return False          # J itself produced nothing: job untouched
                node["status"] = "open"   # child lost without SUMMARY: stays open in the tree
                continue
            summary, remaining, level, h = res
            run_hash = run_hash or h
            self._account(summary)
            if summary["status"] == "exhausted":
                node["status"] = "done"
                node["summary"] = summary
                if level:
                    node["level"] = level
                    self._note_level(job, level)
                with self.lock:
                    slot.nodes_done += 1
            else:
                new = [r for r in remaining if r not in index]
                if len(new) != len(remaining):
                    self.log("!!! worker %d: seed %s listed %d REMAINING seed(s) already present in the tree; dropped"
                             % (slot.idx, seed, len(remaining) - len(new)))
                if not new:
                    node["status"] = "open"     # a split must have children: hand the node back open
                    continue
                node["status"] = "split"
                node["summary"] = summary
                if level:                     # the best level of the explored part belongs to this node, not to its children
                    node["level"] = level
                    self._note_level(job, level)
                for r in new:
                    child = {"seed": r, "parent": seed, "status": "open"}
                    nodes.append(child)
                    index[r] = child
                # REMAINING is cursor first: push in reverse so the cursor is the next pop
                stack.extend(reversed(new))
        if slot.killed_by_client:
            with self.lock:
                slot.job = None
                slot.phase = None
            self.log("worker %d: job %d abandoned as instructed (no report)" % (slot.idx, job["id"]))
            return False
        # absorption pass: probe every still-open local seed briefly
        open_seeds = [n["seed"] for n in nodes if n["status"] == "open"]
        if open_seeds and not self.stopping:
            probe = max(0.5, self.fparam("absorb_probe_s", 10))
            absorb_end = time.time() + self.fparam("absorb_total_s", 120)
            with self.lock:
                slot.phase = "absorb"
                slot.window_end = absorb_end
            for seed in reversed(open_seeds):     # most recent (deepest) first
                if self.stopping or time.time() >= absorb_end:
                    break
                res = self.run_worker(slot, job, seed, probe)
                if res is None:
                    if slot.killed_by_client:
                        break
                    continue
                summary, remaining, level, h = res
                run_hash = run_hash or h
                self._account(summary)
                if summary["status"] == "exhausted":
                    node = index[seed]
                    node["status"] = "done"
                    node["summary"] = summary
                    if level:
                        node["level"] = level
                        self._note_level(job, level)
                    with self.lock:
                        slot.nodes_done += 1
                else:
                    # a probe that splits is reported open (its partial coverage is dropped), but a
                    # level it found is real and must not be lost: the server accepts a level on an open node
                    if level:
                        index[seed]["level"] = level
                        self._note_level(job, level)
            if slot.killed_by_client:
                with self.lock:
                    slot.job = None
                    slot.phase = None
                return False
        # a split node must have at least one child in the list (it always has: REMAINING ≥ 1)
        report = {"token": self.token, "job_id": job["id"], "src_hash": run_hash or self.src_hash,
                  "nodes": [self._clean_node(n) for n in nodes[:MAX_NODES]]}
        if len(nodes) > MAX_NODES:
            self.log("!!! worker %d: job %d tree has %d nodes, above the %d cap; the rest stay open on the server"
                     % (slot.idx, job["id"], len(nodes), MAX_NODES))
        counts = collections.Counter(n["status"] for n in nodes)
        dur = time.time() - start
        with self.lock:
            self.job_durations.append(dur)
            self.stats["jobs_done"] += 1
            self.stats["nodes_done"] += counts.get("done", 0)
            self.stats["nodes_open"] += counts.get("open", 0)
            self.stats[root["status"] if root["status"] in ("done", "split") else "open_root"] += 1
            slot.job = None
            slot.phase = None
        self.log("worker %d: job %d %s in %.0f s: %d node(s), %d done, %d split, %d open"
                 % (slot.idx, job["id"], root["status"], dur, len(nodes), counts.get("done", 0),
                    counts.get("split", 0), counts.get("open", 0)))
        self.queue_report(report)
        return True

    @staticmethod
    def _clean_node(n):
        out = {"seed": n["seed"], "parent": n["parent"], "status": n["status"]}
        if n["status"] != "open" and "summary" in n:
            out["summary"] = n["summary"]
        if n.get("level"):                # open nodes carry a level too (a probe that split), never a summary
            out["level"] = n["level"]
        return out

    def _account(self, summary):
        try:
            el = float(summary.get("elapsed") or 0)
        except (TypeError, ValueError):
            el = 0.0
        with self.lock:
            self.stats["cpu_seconds"] += max(0.0, el)
            self.stats["runs"] += 1

    def _note_level(self, job, level):
        with self.lock:
            ex = str(job["exit"])
            cur = self.best_per_exit.get(ex)
            if cur is None or level["depth"] > cur["depth"]:
                self.best_per_exit[ex] = {"depth": level["depth"], "code": level["code"], "job_id": job["id"]}

    def run_worker(self, slot, job, seed, split_after):
        """Run one worker process on `seed`. Returns (summary, remaining, level, hash)
        or None when the run produced no usable SUMMARY."""
        argv = self.worker_argv(job, seed, split_after)
        stderr = None if self.args.verbose_workers else subprocess.DEVNULL
        with self.lock:
            slot.cur_seed = seed
            slot.cur = slot.best = None
            slot.win_hist.clear()
            slot.run_started_at = time.time()
            slot.stop_sent_at = 0.0
            slot.saw_output = False
            slot.stop_pending = False
            slot.state = "running"
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr, stdin=subprocess.DEVNULL,
                                    cwd=BACKSEARCH_DIR, text=True, encoding="utf-8", errors="replace",
                                    bufsize=1, start_new_session=True)
        except OSError as e:
            self.log("cannot start worker %d for job %d: %s" % (slot.idx, job["id"], e))
            return None
        with self.lock:
            slot.proc = proc
            if self.paused:
                self._signal(slot, signal.SIGSTOP)
                slot.state = "paused"
            elif self.stopping:
                slot.stop_pending = True   # SIGINT once the worker has printed its first line

        run_hash = None
        remaining = []
        level = None
        summary = None
        bad_remaining = 0
        self_remaining = False
        seed_toks = seed.split(",")
        try:
            for line in proc.stdout:
                if not slot.saw_output:
                    with self.lock:
                        slot.saw_output = True
                        if slot.stop_pending:
                            self._send_stop(slot)
                if len(line) > MAX_LINE:
                    continue
                parts = line.rstrip("\r\n").split("\t")
                tag = parts[0]
                if tag == "STATUS" and len(parts) >= 2:
                    self._on_status(slot, parts[1])
                elif tag == "SRC_HASH" and len(parts) >= 2:
                    h = parts[1].strip().lower()
                    if HEX_RE.match(h):
                        run_hash = h
                elif tag == "REMAINING" and len(parts) >= 2:
                    p = parts[1].strip()
                    toks = p.split(",")
                    if p == seed:
                        self_remaining = True   # the cursor is the root itself: nothing was expanded
                    elif SEED_RE.match(p) and len(toks) > len(seed_toks) and toks[:len(seed_toks)] == seed_toks \
                            and p not in remaining:
                        remaining.append(p)
                    else:
                        bad_remaining += 1
                elif tag == "LEVEL" and len(parts) >= 3:
                    try:
                        d = int(parts[1])
                    except ValueError:
                        continue
                    code = parts[2].strip()
                    if 0 < len(code) <= 2000:
                        level = {"depth": d, "code": code}
                elif tag == "SUMMARY" and len(parts) >= 2:
                    try:
                        s = json.loads(parts[1])
                    except ValueError:
                        s = None
                    if isinstance(s, dict) and s.get("status") in ("exhausted", "split"):
                        summary = s
        except (OSError, ValueError) as e:
            self.log("worker %d: error reading output: %s" % (slot.idx, e))
        rc = proc.wait()
        with self.lock:
            slot.proc = None
            slot.state = "idle"
            killed = slot.killed_by_client
        if summary is None:
            with self.lock:
                self.stats["runs_without_summary"] += 1
            if not killed:
                self.log("!!! worker %d exited with code %s WITHOUT a SUMMARY line (job %d, seed %s). "
                         "The node is left untouched. Check the worker binary (--verbose-workers shows its stderr)."
                         % (slot.idx, rc, job["id"], seed))
            return None
        if summary.get("status") == "split" and self_remaining:
            # Interrupted before expanding its root: the whole subtree is still pending and a
            # tree listing the seed itself as a child is invalid. Treat the run as no result.
            self.log("worker %d: seed %s was interrupted before expanding its root; not reported (node stays untouched)"
                     % (slot.idx, seed))
            return None
        if summary.get("status") == "split" and not remaining:
            self.log("!!! worker %d: seed %s reported a split but printed no valid REMAINING line "
                     "(%d invalid). Protocol violation; the node is left untouched." % (slot.idx, seed, bad_remaining))
            return None
        if bad_remaining:
            self.log("warning: worker %d printed %d REMAINING lines that do not extend the seed; they were dropped"
                     % (slot.idx, bad_remaining))
        return summary, remaining, level, run_hash

    def _on_status(self, slot, text):
        try:
            s = json.loads(text)
        except ValueError:
            return
        if not isinstance(s, dict):
            return
        now = time.time()

        def board(depth_key, code_key):
            try:
                d = int(s.get(depth_key))
            except (TypeError, ValueError):
                return None
            code = s.get(code_key)
            if not isinstance(code, str) or not (0 < len(code) <= 2000):
                return None
            return {"depth": d, "code": code}

        cur = board("depth", "cur")
        win = board("win_best", "win")
        with self.lock:
            slot.last_line_at = now
            if cur:
                slot.cur = cur
            if win:
                slot.win_hist.append((now, win["depth"], win["code"]))
                if slot.best is None or win["depth"] > slot.best["depth"]:
                    slot.best = win
            elif "best" in s and slot.best is None:
                try:
                    slot.best = {"depth": int(s["best"]), "code": None}
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------ batched reports / outbox (7.6)
    def queue_report(self, report):
        with self.lock:
            self.pending.append(report)

    def sender_loop(self):
        while True:
            self.send_event.wait(0.5)
            self.send_event.clear()
            with self.lock:
                due = self.pending and (self.stopping or
                                        time.time() - self.last_send_at >= self.fparam("batch_interval_s", 10))
                if self.stopping and not self.pending and self.sender_may_exit:
                    return
            if due:
                self.send_pending()

    def send_pending(self):
        with self.lock:
            batch = self.pending
            self.pending = []
            self.last_send_at = time.time()
        if batch:
            self.send_batch(batch)

    def send_batch(self, reports):
        """POST one batch; per-report outcomes decide ok / dup / rejected. A transport
        failure puts the whole batch in the outbox as one file. Returns True on delivery."""
        for r in reports:
            r["token"] = self.token
        try:
            res = self.api.post("/api/v2/reports", {"token": self.token, "reports": reports}, timeout=60)
        except ApiError as e:
            if e.transient or e.status == 401:
                path = self.outbox_write(reports, attempts=1)
                self.log("batch of %d report(s) queued in outbox (%s): %s" % (len(reports), os.path.basename(path), e))
                return False
            if e.status == 409 and isinstance(e.body, dict) and e.body.get("error") == "unknown_hash":
                self.request_stop("the server no longer accepts this worker's hash; update and rebuild (git pull && ./build_pgo.sh)")
            for r in reports:
                self.reject(r, e.status, e.body)
            return False
        outcomes = None
        if isinstance(res, dict):
            for key in ("results", "outcomes", "reports"):
                if isinstance(res.get(key), list):
                    outcomes = res[key]
                    break
        ok = dup = bad = 0
        for i, r in enumerate(reports):
            o = outcomes[i] if outcomes is not None and i < len(outcomes) else {}
            if not isinstance(o, dict):
                o = {}
            if o.get("dup"):
                dup += 1
            elif o.get("error") or (isinstance(o.get("status"), int) and o["status"] >= 400) or o.get("ok") is False:
                bad += 1
                st = o.get("status") if isinstance(o.get("status"), int) else 400
                self.reject(r, st, o)
                if o.get("error") == "unknown_hash":
                    self.request_stop("the server no longer accepts this worker's hash; update and rebuild (git pull && ./build_pgo.sh)")
            else:
                ok += 1
        with self.lock:
            self.stats["reports_sent"] += ok + dup
            self.stats["batches_sent"] += 1
        self.log("batch delivered: %d report(s) ok%s%s" % (ok, ", %d dup" % dup if dup else "", ", %d REJECTED" % bad if bad else ""))
        return True

    def reject(self, report, status, body):
        with self.lock:
            self.stats["reports_failed"] += 1
        path = os.path.join(self.outbox, "rejected", "%d-%s-%s.json" % (int(time.time()), report.get("job_id"), uuid.uuid4().hex[:8]))
        try:
            with open(path, "w") as f:
                json.dump({"report": report, "error": {"status": status, "body": body}}, f)
        except OSError:
            pass
        self.log("!!! server REJECTED the report for job %s (HTTP %s %s); kept in %s"
                 % (report.get("job_id"), status, json.dumps(body)[:200], path))

    def outbox_write(self, reports, attempts, path=None):
        entry = {"reports": reports, "attempts": attempts,
                 "next_try": time.time() + min(300.0, 2.0 ** attempts)}
        if path is None:
            path = os.path.join(self.outbox, "%d-batch%d-%s.json" % (int(time.time()), len(reports), uuid.uuid4().hex[:8]))
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f)
        os.replace(tmp, path)
        return path

    def outbox_files(self):
        try:
            names = sorted(n for n in os.listdir(self.outbox) if n.endswith(".json"))
        except OSError:
            return []
        return [os.path.join(self.outbox, n) for n in names]

    def flush_outbox(self, force=False):
        now = time.time()
        for path in self.outbox_files():
            try:
                with open(path) as f:
                    entry = json.load(f)
                reports = entry["reports"] if "reports" in entry else [entry["report"]]
                assert isinstance(reports, list) and reports
            except Exception:
                self.log("outbox: unreadable file %s moved to rejected/" % os.path.basename(path))
                try:
                    os.replace(path, os.path.join(self.outbox, "rejected", os.path.basename(path)))
                except OSError:
                    pass
                continue
            if not force and entry.get("next_try", 0) > now:
                continue
            for r in reports:
                r["token"] = self.token
            try:
                os.remove(path)
            except OSError:
                pass
            if not self.send_batch(reports):
                # send_batch re-queued it as a fresh file on transport failure; keep the attempt count
                newest = self.outbox_files()
                if newest:
                    try:
                        with open(newest[-1]) as f:
                            e2 = json.load(f)
                        self.outbox_write(e2["reports"], entry.get("attempts", 0) + 1, path=newest[-1])
                    except Exception:
                        pass
                return  # server unreachable: try the rest later

    # ------------------------------------------------------------ signals / controls
    def _signal(self, slot, sig):
        p = slot.proc
        if p is None or p.poll() is not None:
            return
        try:
            os.kill(p.pid, sig)
        except OSError:
            pass

    def _send_stop(self, slot):
        """SIGINT a worker (caller holds the lock). A worker that has not printed its
        first line yet may not have installed its handler: defer until it does."""
        if slot.proc is None:
            return
        if not slot.saw_output:
            slot.stop_pending = True
            return
        slot.stop_pending = False
        self._signal(slot, signal.SIGCONT)
        self._signal(slot, signal.SIGINT)
        slot.state = "finishing"
        slot.stop_sent_at = time.time()

    def pause(self):
        with self.lock:
            if self.paused or self.stopping:
                return
            self.paused = True
            for s in self.slots:
                if s.proc and s.state == "running":
                    self._signal(s, signal.SIGSTOP)
                    s.state = "paused"
        self.log("paused (workers stopped with SIGSTOP; their memory stays allocated)")
        self.heartbeat()

    def resume(self):
        with self.lock:
            if not self.paused:
                return
            self.paused = False
            for s in self.slots:
                if s.proc and s.state == "paused":
                    self._signal(s, signal.SIGCONT)
                    s.state = "running"
        self.log("resumed")
        self.heartbeat()

    def request_stop(self, reason="stop requested"):
        with self.lock:
            if self.stopping:
                return
            self.stopping = True
            self.stop_reason = reason
            self.paused = False
            for s in self.slots:
                if s.proc:
                    self._send_stop(s)
            queued = [j["id"] for j in self.queue]
            self.queue.clear()
            self.cond.notify_all()
        self.send_event.set()
        self.log("stopping: %s. Waiting for workers to print their SUMMARY..." % reason)
        if queued:
            self.log("leased jobs never started (their leases expire on the server): %s" % queued)

    def force_quit(self):
        self.log("second Ctrl-C: killing workers and exiting; running jobs are left to their leases")
        with self.lock:
            for s in self.slots:
                if s.proc:
                    self._signal(s, signal.SIGKILL)
        os._exit(1)

    def set_workers(self, n):
        try:
            n = int(n)
        except (TypeError, ValueError):
            return self.workers
        wmax = self.campaign.get("workers_max")
        try:
            wmax = int(wmax) if wmax is not None else 64
        except (TypeError, ValueError):
            wmax = 64
        n = max(1, min(wmax, n))
        with self.lock:
            self.workers = n
            self._reconcile_slots()
        self.log("workers set to %d" % n)
        return n

    def _reconcile_slots(self):
        while len(self.slots) < self.workers:
            self.slots.append(Slot(len(self.slots)))
        for s in self.slots:
            s.retire = s.idx >= self.workers
            if not s.retire and (s.thread is None or not s.thread.is_alive()) and not self.stopping:
                s.state = "idle"
                s.thread = threading.Thread(target=self.slot_loop, args=(s,), name="worker-%d" % s.idx, daemon=True)
                s.thread.start()
            elif s.retire and s.job is not None:
                s.state = "retiring"

    # ------------------------------------------------------------ heartbeat / wake
    def boards(self):
        out = []
        with self.lock:
            for s in self.slots:
                if s.job is None or s.retire and s.job is None:
                    continue
                cur = s.cur or {}
                b = {"depth": cur.get("depth", 0), "cur": (cur.get("code") or "")[:BOARD_CAP],
                     "best": (s.best or {}).get("depth", 0)}
                out.append(b)
        return out[:self.workers]

    def heartbeat(self):
        body = {"token": self.token, "workers": self.workers, "paused": self.paused, "boards": self.boards()}
        try:
            res = self.api.post("/api/v2/heartbeat", body, timeout=20)
        except ApiError as e:
            self.log("heartbeat failed: %s" % e)
            if e.status in (401, 403):
                self.request_stop("the server does not recognise this client any more (HTTP %d)" % e.status)
            return None
        with self.lock:
            self.last_heartbeat_ok = time.time()
        if not isinstance(res, dict):
            return None
        self.apply_campaign(res.get("campaign"))
        hashes = [str(h).lower() for h in (self.campaign.get("hashes") or [])]
        if hashes and self.src_hash not in hashes and not self.stopping:
            self.request_stop("the campaign's accepted worker hashes changed; update and rebuild (git pull && ./build_pgo.sh)")
        if res.get("revoked"):
            self.request_stop("this client was revoked by the server")
        leases = res.get("leases")
        if isinstance(leases, list):
            with self.lock:
                self.leases_from_server = set(int(x) for x in leases if isinstance(x, (int, str)) and str(x).isdigit())
                running = [s.job["id"] for s in self.slots if s.job]
                lost = [j for j in running if j not in self.leases_from_server]
            if lost:
                self.log("warning: the server no longer lists these running jobs as leased to us: %s" % lost)
        return res

    def handle_wake(self, gap):
        self.log("clock jumped %.0f s (sleep?). Re-validating leases before the workers continue." % gap)
        with self.lock:
            held = []
            for s in self.slots:
                if s.proc and s.state == "running":
                    self._signal(s, signal.SIGSTOP)
                    held.append(s)
        deadline = time.time() + WAKE_HEARTBEAT_RETRY_S
        res = None
        while res is None and time.time() < deadline and not self.stopping:
            res = self.heartbeat()
            if res is None:
                time.sleep(5)
        with self.lock:
            valid = self.leases_from_server if res is not None else None
            if valid is None:
                self.log("warning: could not re-validate leases after wake; letting workers continue")
            else:
                for s in self.slots:
                    if s.job and s.job["id"] not in valid and s.proc:
                        self.log("job %d lost its lease while asleep; killing worker %d" % (s.job["id"], s.idx))
                        s.killed_by_client = True
                        self._signal(s, signal.SIGKILL)
                dropped = [j["id"] for j in self.queue if j["id"] not in valid]
                if dropped:
                    self.log("dropping queued jobs whose lease expired: %s" % dropped)
                    self.queue = collections.deque(j for j in self.queue if j["id"] in valid)
            for s in held:
                if s.proc and not self.paused and s.state == "running":
                    self._signal(s, signal.SIGCONT)

    # ------------------------------------------------------------ state for the UI
    def state(self):
        now = time.time()
        with self.lock:
            self.ui_last_poll = now
            slots = [s.snapshot(now) for s in self.slots if not (s.retire and s.job is None and s.state in ("dead", "idle"))]
            return {
                "name": self.name,
                "workers": self.workers,
                "paused": self.paused,
                "stopping": self.stopping,
                "stop_reason": self.stop_reason,
                "server": {"url": self.api.server, "connected": self.api.connected,
                           "last_ok": self.api.last_ok, "last_error": self.api.last_error,
                           "last_heartbeat_ok": self.last_heartbeat_ok},
                "campaign": {k: self.campaign.get(k) for k in ("title", "grid", "extra", "split_after_s", "lease_s", "workers_max",
                                                             "batch_interval_s", "lease_ahead_s", "absorb_total_s", "absorb_probe_s",
                                                             "heartbeat_s", "lease_cap")},
                "mean_job_s": round(self.mean_job_s(), 1),
                "src_hash": self.src_hash,
                "outbox": len(self.outbox_files()),
                "pending": len(self.pending),
                "queued": len(self.queue),
                "stats": dict(self.stats, cpu_hours=round(self.stats["cpu_seconds"] / 3600.0, 3)),
                "best_per_exit": self.best_per_exit,
                "slots": slots,
                "log": list(self.ring)[-25:],
                "time": now,
            }

    # ------------------------------------------------------------ main loop
    def install_signals(self):
        # Installed before registration so that Ctrl-C works from the first second,
        # even when the process inherited SIGINT=ignored (background job of a script).
        signal.signal(signal.SIGINT, self._on_sigint)
        signal.signal(signal.SIGTERM, lambda *_: self.request_stop("SIGTERM"))

    def run(self):
        os.makedirs(os.path.join(self.outbox, "rejected"), exist_ok=True)
        if not self.args.no_ui:
            self.start_ui()
        with self.lock:
            self._reconcile_slots()
        threading.Thread(target=self.leaser_loop, name="leaser", daemon=True).start()
        sender = threading.Thread(target=self.sender_loop, name="sender", daemon=True)
        sender.start()
        self.heartbeat()
        self.last_heartbeat = time.time()
        self.flush_outbox()
        next_outbox = time.time() + 15
        self.last_tick = time.time()
        while True:
            time.sleep(0.5)
            now = time.time()
            gap = now - self.last_tick
            self.last_tick = now
            if gap > 2 * self.lease_s() and not self.stopping:
                self.handle_wake(gap)
            if now - self.last_heartbeat >= self.fparam("heartbeat_s", HEARTBEAT_S):
                self.last_heartbeat = now
                self.heartbeat()
            if now >= next_outbox:
                next_outbox = now + 15
                self.flush_outbox()
            with self.lock:
                for s in self.slots:
                    if s.stop_pending and s.proc and now - s.run_started_at > 5.0:
                        s.saw_output = True      # silent for 5 s: assume it is up and signal it
                        self._send_stop(s)
                    if s.state == "finishing" and s.proc and s.stop_sent_at and now - s.stop_sent_at > STOP_GRACE_S:
                        self.log("!!! worker %d did not print SUMMARY within %.0f s of SIGINT; killing it. "
                                 "Job %s is left untouched." % (s.idx, STOP_GRACE_S, s.job and s.job["id"]))
                        s.killed_by_client = True
                        self._signal(s, signal.SIGKILL)
                        s.stop_sent_at = 0.0
                if self.ui_close_at and now - self.ui_close_at > 3.0:
                    if self.ui_last_poll < self.ui_close_at:
                        self.ui_close_at = 0.0
                        threading.Thread(target=self.request_stop, args=("browser window closed",), daemon=True).start()
                    else:
                        self.ui_close_at = 0.0
            if self.stopping:
                alive = [s for s in self.slots if s.thread and s.thread.is_alive()]
                if not alive:
                    break
        # every worker has finished: send the final batch now, then drain the outbox
        with self.lock:
            self.sender_may_exit = True
        self.send_event.set()
        sender.join(90)
        self.send_pending()
        self.flush_outbox(force=True)
        try:
            self.api.post("/api/v2/heartbeat", {"token": self.token, "workers": self.workers, "paused": False,
                                                "boards": []}, timeout=10)
        except ApiError:
            pass
        left = len(self.outbox_files())
        self.log("stopped. jobs done %d (%d exhausted, %d split), %d nodes done locally, %d batch(es), CPU %.2f h%s"
                 % (self.stats["jobs_done"], self.stats["done"], self.stats["split"], self.stats["nodes_done"],
                    self.stats["batches_sent"], self.stats["cpu_seconds"] / 3600.0,
                    "; %d report(s) still in the outbox, run again to deliver them" % left if left else ""))
        return self.exit_code

    def _on_sigint(self, *_):
        self.sigint_count += 1
        if self.sigint_count >= 2:
            self.force_quit()
        if not self.slots:
            # still starting up (registration / version check): nothing to wind down
            sys.stderr.write("\ninterrupted during startup\n")
            os._exit(1)
        threading.Thread(target=self.request_stop, args=("Ctrl-C",), daemon=True).start()

    # ------------------------------------------------------------ local GUI
    def start_ui(self):
        vol = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, data, ctype="application/json"):
                if isinstance(data, (dict, list)):
                    data = json.dumps(data).encode()
                elif isinstance(data, str):
                    data = data.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    try:
                        with open(UI_FILE, "rb") as f:
                            self._send(200, f.read(), "text/html")
                    except OSError:
                        self._send(500, "volunteer_ui.html not found next to volunteer.py", "text/plain")
                elif path == "/state":
                    self._send(200, vol.state())
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self):
                path = self.path.split("?")[0]
                # Same-origin pages can set this header; a foreign page cannot without CORS
                # preflight (which we never answer), so it blocks drive-by POSTs.
                if self.headers.get("X-Volunteer") != "1":
                    return self._send(403, {"error": "forbidden"})
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 65536)) if n else b""
                try:
                    body = json.loads(raw.decode() or "{}")
                except ValueError:
                    body = {}
                if path == "/pause":
                    threading.Thread(target=vol.pause, daemon=True).start()
                    return self._send(200, {"ok": True})
                if path == "/resume":
                    threading.Thread(target=vol.resume, daemon=True).start()
                    return self._send(200, {"ok": True})
                if path == "/stop":
                    threading.Thread(target=vol.request_stop, args=("Stop pressed in the UI",), daemon=True).start()
                    return self._send(200, {"ok": True})
                if path == "/workers":
                    return self._send(200, {"ok": True, "workers": vol.set_workers(body.get("workers"))})
                if path == "/close":
                    if not vol.args.no_stop_on_close:
                        with vol.lock:
                            vol.ui_close_at = time.time()
                    return self._send(200, {"ok": True})
                self._send(404, {"error": "not_found"})

        try:
            srv = ThreadingHTTPServer(("127.0.0.1", self.args.port), Handler)
        except OSError as e:
            sys.stderr.write("Cannot listen on 127.0.0.1:%d (%s). Use --port or --no-ui.\n" % (self.args.port, e))
            sys.exit(2)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, name="ui", daemon=True).start()
        url = "http://127.0.0.1:%d/" % self.args.port
        self.log("local GUI at %s" % url)
        if not self.args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Pathology collective search v2 volunteer client")
    ap.add_argument("--name", help="your display name (fixed at first registration)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--server", default=os.environ.get("VOLUNTEER_SERVER", DEFAULT_SERVER))
    ap.add_argument("--port", type=int, default=8765, help="local GUI port (127.0.0.1 only)")
    ap.add_argument("--worker", default=os.path.join(".", "backsearch_worker_nt"),
                    help="worker binary (default ./backsearch_worker_nt in the backsearch directory)")
    ap.add_argument("--outbox", default=os.path.join(HERE, "volunteer_outbox"))
    ap.add_argument("--no-ui", action="store_true", help="do not serve the local GUI")
    ap.add_argument("--no-browser", action="store_true", help="serve the GUI but do not open a browser")
    ap.add_argument("--no-stop-on-close", action="store_true", help="closing the browser tab does not stop the client")
    ap.add_argument("--reregister", action="store_true", help="ignore the stored token and register anew")
    ap.add_argument("--verbose-workers", action="store_true", help="let workers write their stderr to the terminal")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")

    vol = Volunteer(args)
    vol.install_signals()
    vol.check_dirty()
    vol.src_hash = vol.read_version()
    vol.log("worker %s hash %s" % (" ".join(vol.worker_argv_base()), vol.src_hash))
    vol.register()
    vol.check_hash()
    vol.log("campaign: grid %s, extra %s, split after %s s, lease %s s, %d worker(s)"
            % (vol.param("grid", "?"), vol.param("extra", ""), vol.param("split_after_s", "?"),
               vol.param("lease_s", "?"), vol.workers))
    sys.exit(vol.run())


if __name__ == "__main__":
    main()
