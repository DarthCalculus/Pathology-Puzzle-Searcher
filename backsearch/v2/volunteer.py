#!/usr/bin/env python3
"""Pathology collective search v2: volunteer client, protocol 3.

    python3 volunteer.py --name "Your name" [--workers N] [--server URL] [--port 8765] [--keep-going]

Python 3.9+, standard library only. Runs on macOS, Linux and WSL.

The contract is v2/PROTOCOL3.md section 3 (it wins over DESIGN.md section 4
where they differ). The client leases subtree jobs from the server, runs one
worker process per slot, parses the worker's tab-separated protocol lines
(SRC_HASH, STATUS, REMAINING, UNRESOLVED, LEVEL, SUMMARY), reports results in
byte-capped batches, and serves a small local GUI on 127.0.0.1:PORT.

  Ctrl-C (once), SIGTERM, SIGHUP, or Stop in the panel: every worker gets SIGINT,
      prints its exact remaining work, the splits are reported, then the client
      exits. A second Ctrl-C kills the workers and exits at once (reports that
      were not sent yet stay on disk in the outbox).
  Campaign complete: the client stops leasing, lets running workers finish,
      delivers every report, prints a closing summary and exits 0.
      --keep-going waits for the next campaign and joins it instead.

Exit codes: 0 = stopped or campaign complete; 1 = interrupted during startup
or second Ctrl-C; 2 = refused to run (client too old, worker not accepted,
campaign flags not allowed, second instance on the same outbox, ...).
"""
import argparse
import collections
import json
import math
import os
import platform
import queue
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import fcntl
except ImportError:          # native Windows: no advisory locks (WSL has them)
    fcntl = None

CLIENT_VERSION = "3.1.0"
PROTOCOL = 3

HERE = os.path.dirname(os.path.abspath(__file__))
BACKSEARCH_DIR = os.path.dirname(HERE)
DEFAULT_SERVER = "https://pathology.georgespahn.com"
TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".pathology_volunteer.json")
UI_FILE = os.path.join(HERE, "volunteer_ui.html")
UI_SECRET_PLACEHOLDER = "__VOLUNTEER_SECRET__"
DIRTY_FILES = ["backsearch.c", "sokoban_bfs.c", "sokoban_bfs.h", os.path.join("v2", "volunteer.py")]

SEED_RE = re.compile(r"^(?:[URDL][123])(?:,[URDL][123])*$")
HEX_RE = re.compile(r"^[0-9a-fA-F]{8,128}$")
CODE_RE = re.compile(r"^[\x21-\x7e]{1,4096}$")      # a level code: printable, no blanks (the server parses it)
CAND_CAUSES = ("pq", "probe", "big")                # PROTOCOL3 2.2; the server refuses a report with any other
STATUS_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
HEARTBEAT_S = 30.0
STATUS_EVERY_MS = 250
LEASE_CAP = 200              # ceiling of one lease request (7.6)
MAX_NODES = 5000             # nodes per tree report (server NODES_MAX)
REPORT_BYTES_MAX = 3500000   # one report, serialized (the server's report body cap is 4 MB)
BATCH_BYTES_MAX = 1000000    # one POST /reports body (PROTOCOL3 section 3); a single larger report goes alone
BATCH_COUNT_MAX = 500        # reports per batch (server BATCH_MAX 1000)
STOP_GRACE_S = 90.0          # after SIGINT, how long a worker may take to print SUMMARY
WATCHDOG_SLACK_S = 60.0      # watchdog: silent for 3 x status interval + this after the split time = hung
WAKE_HEARTBEAT_RETRY_S = 120.0
MAIN_TICK_S = 0.5            # the main loop's supervision tick
CLOCK_GAP_S = 5.0            # a tick this late means the client did not run (sleep, frozen, starved):
                             # the watchdog and the stop grace count only time the client was awake
MAX_LINE = 20000
BOARD_CAP = 200
UNRESOLVED_MAX = 1000        # the worker prints at most 1000 candidates per run (server CAND_MAX_PER_NODE)
CANDS_PER_REPORT_MAX = 20000 # server CAND_MAX_PER_REPORT
PATH_TOK_MAX_DEFAULT = 1024
OUTBOX_FLUSH_S = 15.0
LOG_QUEUE_MAX = 5000
MIN_WINDOW_S_DEFAULT = 1.0   # floor of a window and of a run's split time; a campaign may set min_window_s (tests, N18)
HOLD_PER_WORKER = 2          # M108: jobs running + queued here <= 2 x workers
LEASE_EARLY_GAP_S = 2.0      # an idle worker with nothing queued may lease this soon after the last request
DEPTH_HIST_N = 16            # window durations kept per job depth (expected work of a job, M108)
SEC_BOARD_S = 1.0            # the "last second" board

# The only worker flags a campaign may add (PROTOCOL3 section 3). Anything else makes the
# client refuse to run: the worker has flags that open files by path, and the campaign
# definition comes from the server.
EXTRA_ALLOWED = {"--allow-exit-transit": False, "--num-holes": True, "--num-blocks": True, "--min-walls": True}
EXTRA_REQUIRED = ("--allow-exit-transit",)
INT_ARG_RE = re.compile(r"^[0-9]{1,4}$")
GRID_RE = re.compile(r"^([0-9]{1,2})x([0-9]{1,2})$")

# ------------------------------------------------------------------ logging (M54)
# log() never blocks and never raises: lines go to a ring (for the panel) and to a bounded
# queue that one writer thread drains to stderr. A stalled terminal (Ctrl-S, a console
# selection on WSL) therefore cannot freeze the scheduler, and a dead tty cannot kill a thread.
_LOGQ = queue.Queue(maxsize=LOG_QUEUE_MAX)
_LOG_STATE = {"thread": None, "dropped": 0}
_LOG_LOCK = threading.Lock()


def _log_writer():
    while True:
        line = _LOGQ.get()
        try:
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
        except (OSError, ValueError, AttributeError):
            pass
        finally:
            _LOGQ.task_done()


def log(msg, ring=None):
    line = "[volunteer %s] %s" % (time.strftime("%H:%M:%S"), msg)
    if ring is not None:
        ring.append(line)
    if _LOG_STATE["thread"] is None:
        with _LOG_LOCK:
            if _LOG_STATE["thread"] is None:
                t = threading.Thread(target=_log_writer, name="log", daemon=True)
                t.start()
                _LOG_STATE["thread"] = t
    try:
        _LOGQ.put_nowait(line)
    except queue.Full:
        _LOG_STATE["dropped"] += 1


def log_flush(timeout=3.0):
    """Wait (bounded) until the writer has printed everything queued so far."""
    end = time.time() + timeout
    while _LOGQ.unfinished_tasks and time.time() < end:
        time.sleep(0.02)


def emit(text):
    """Print a block to stderr from the main thread after the queued log lines (never raises)."""
    log_flush()
    try:
        sys.stderr.write(text if text.endswith("\n") else text + "\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


# ------------------------------------------------------------------ pure helpers
def seed_tokens(seed):
    return [t for t in seed.split(",") if t] if seed else []


def valid_seed(seed, tok_max):
    """A job seed: "" (the exit root, DESIGN section 2) or tokens like U1,R3."""
    if seed == "":
        return True
    return bool(SEED_RE.match(seed)) and seed.count(",") + 1 <= tok_max


def extends(path, seed_toks, tok_max):
    """REMAINING / UNRESOLVED path check: a well-formed path, at most tok_max tokens,
    starting with the node's seed tokens. Returns the token list or None."""
    if not SEED_RE.match(path):
        return None
    toks = path.split(",")
    if len(toks) > tok_max or toks[:len(seed_toks)] != seed_toks:
        return None
    return toks


def parse_unresolved(parts, seed, tok_max):
    """An `UNRESOLVED<TAB>depth<TAB>code<TAB>path<TAB>cause` line (split on tabs) of a run on
    `seed` -> {depth, code, path, cause}, or None when it is not exactly what the server accepts:
    depth >= 1 and equal to the path's token count (one token per move), a path that is the seed
    or extends it, cause pq | probe | big (PROTOCOL3 2.2). A bad line voids the run."""
    if len(parts) < 5:
        return None
    try:
        d = int(parts[1])
    except ValueError:
        return None
    code, path, cause = parts[2].strip(), parts[3].strip(), parts[4].strip()
    seed_toks = seed_tokens(seed)
    ptoks = seed_toks if path == seed else (extends(path, seed_toks, tok_max) if path else None)
    if d >= 1 and CODE_RE.match(code) and ptoks is not None and len(ptoks) == d and cause in CAND_CAUSES:
        return {"depth": d, "code": code, "path": path, "cause": cause}
    return None


def nonneg_float(v):
    """A finite number >= 0 from a server field, else None."""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f < 1e9 else None


def pos_int(v):
    """An integer >= 1 from a server field, else None."""
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return v


def fmt_s(x):
    """Seconds for a log line: '1800', '2.5', '0.25'."""
    return "%.0f" % x if x >= 10 else ("%.2f" % x).rstrip("0").rstrip(".")


def seed_depth(seed):
    return seed.count(",") + 1 if seed else 0


def absorb_order(seeds):
    """M18: absorption probes go deepest first (most tokens). REMAINING is printed cursor first, then the
    stack from the top (deep) to the bottom (shallow): the deep entries are the cheap ones, and a probe
    spent on a shallow (large) one usually splits. The sort is stable, so the cursor stays first among
    equals; the caller's node list (parent before child) is not reordered."""
    return sorted(seeds, key=seed_depth, reverse=True)


def queue_key(job):
    """M108: the local queue runs the job the server would lease first: largest est_s first (when the
    grant carries it), then shallowest, then the order the jobs were granted in (jobs.js leaseOrder)."""
    est = job.get("est_s")
    return (est is None, -(est or 0.0), job.get("depth", 0), job.get("seq", 0))


def parse_extra(extra):
    """Validate the campaign's extra worker flags against the allowlist.
    Returns (list, None) or (None, message)."""
    if extra is None:
        extra = []
    if isinstance(extra, str):
        try:
            extra = shlex.split(extra)
        except ValueError as e:
            return None, "the campaign's worker flags cannot be parsed (%s)" % e
    if not isinstance(extra, list):
        return None, "the campaign's worker flags are not a list"
    out, seen, i = [], set(), 0
    items = [str(x).strip() for x in extra if str(x).strip()]
    while i < len(items):
        f = items[i]
        if f not in EXTRA_ALLOWED:
            return None, "worker flag %r is not allowed" % f[:60]
        if f in seen:
            return None, "worker flag %s is given twice" % f
        seen.add(f)
        out.append(f)
        if EXTRA_ALLOWED[f]:
            if i + 1 >= len(items) or not INT_ARG_RE.match(items[i + 1]):
                return None, "worker flag %s needs a non-negative integer value" % f
            out.append(items[i + 1])
            i += 1
        i += 1
    for f in EXTRA_REQUIRED:
        if f not in seen:
            return None, "the campaign's worker flags lack %s (every campaign searches with exit transit)" % f
    return out, None


def extra_value(extra, flag):
    try:
        i = extra.index(flag)
        return int(extra[i + 1])
    except (ValueError, IndexError):
        return None


def worker_env(environ, allow_dirty):
    """M48: inherited BS_* debug variables change what a worker reports (BS_DUMP_SEED makes
    every run 'exhausted' with 0 states) or fill the disk (BS_TRACE_*). They reach the worker
    only in development (VOLUNTEER_ALLOW_DIRTY=1)."""
    if allow_dirty:
        return dict(environ)
    return {k: v for k, v in environ.items() if not k.startswith("BS_")}


def clean_node(n):
    """The wire form of a tree node (DESIGN 7.5 plus PROTOCOL3 'unresolved')."""
    out = {"seed": n["seed"], "parent": n["parent"], "status": n["status"]}
    if n["status"] != "open" and "summary" in n:
        out["summary"] = n["summary"]
    if n.get("level"):                # open nodes carry a level too (a probe that split), never a summary
        out["level"] = n["level"]
    if n["status"] != "open" and n.get("unresolved"):
        out["unresolved"] = n["unresolved"]
    return out


def _jlen(obj):
    return len(json.dumps(obj, separators=(",", ":")))


def fit_tree(nodes, max_nodes, max_bytes, max_cands=CANDS_PER_REPORT_MAX):
    """PROTOCOL3 section 3 fold-back (C3/M15/N12): never cut a split's child list. While the
    tree exceeds the node, byte or candidate cap, demote the deepest split node other than the
    job itself to 'open': drop its summary, its unresolved candidates and all its descendants,
    keep its level (the server accepts a level on an open node; the node is searched again
    anyway). The deepest split node has no split descendant, so every remaining split still
    lists all of its children. When no split node is left but the bytes or candidates still do
    not fit, a finished ('done') node other than the job is demoted to 'open' the same way,
    largest first: it is simply searched again, which is exact too.
    Returns the fitted node list (insertion order kept; the input is not modified) or None
    when even the job plus its own children do not fit.
    `nodes` are dicts {seed, parent, status[, summary, level, unresolved]} with nodes[0] = job."""
    work = [dict(n) for n in nodes]
    size = {n["seed"]: _jlen(clean_node(n)) + 1 for n in work}
    ncand = {n["seed"]: len(n.get("unresolved") or []) if n["status"] != "open" else 0 for n in work}
    total_bytes = sum(size.values()) + 2
    total_cands = sum(ncand.values())
    kids = collections.defaultdict(list)
    for n in work[1:]:
        kids[n["parent"]].append(n["seed"])
    alive = set(size)
    pos = {n["seed"]: i for i, n in enumerate(work)}
    by_seed = {n["seed"]: n for n in work}

    def depth(s):
        return len(seed_tokens(s))

    while len(alive) > max_nodes or total_bytes > max_bytes or total_cands > max_cands:
        splits = [n for n in work[1:] if n["seed"] in alive and n["status"] == "split"]
        if splits:
            victim = max(splits, key=lambda n: (depth(n["seed"]), pos[n["seed"]]))
        elif len(alive) <= max_nodes:
            done = [n for n in work[1:] if n["seed"] in alive and n["status"] == "done"]
            if not done:
                return None
            victim = max(done, key=lambda n: (size[n["seed"]], pos[n["seed"]]))
        else:
            return None
        stack = list(kids.get(victim["seed"], []))
        while stack:                       # remove every descendant
            s = stack.pop()
            if s in alive:
                alive.discard(s)
                total_bytes -= size[s]
                total_cands -= ncand[s]
                stack.extend(kids.get(s, []))
        victim["status"] = "open"
        victim.pop("summary", None)
        victim.pop("unresolved", None)
        new = _jlen(clean_node(victim)) + 1
        total_bytes += new - size[victim["seed"]]
        total_cands -= ncand[victim["seed"]]
        size[victim["seed"]] = new
        ncand[victim["seed"]] = 0
    return [by_seed[n["seed"]] for n in work if n["seed"] in alive]


# ------------------------------------------------------------------ HTTP
class ApiError(Exception):
    def __init__(self, status, body, transient):
        Exception.__init__(self, "HTTP %s: %s" % (status, json.dumps(body)[:300] if isinstance(body, (dict, list)) else body))
        self.status = status
        self.body = body if isinstance(body, dict) else {"error": str(body)[:200]}
        self.transient = transient

    @property
    def error(self):
        return self.body.get("error")


def classify(e):
    """How the client reacts to a refused request (PROTOCOL3 section 3)."""
    err = e.error
    if e.status == 426 or err == "client_too_old":
        return "too_old"
    if e.status == 410 or err == "campaign_closed":
        return "closed"
    if e.transient:
        return "transient"
    if e.status == 413:
        return "too_large"
    if e.status == 403 and (err == "revoked" or e.body.get("revoked")):
        return "revoked"
    if e.status in (401, 403) or (e.status == 400 and err == "bad_token"):
        return "reregister"            # unknown_token, not_admitted, ...
    if e.status == 404 and err == "no_campaign":
        return "no_campaign"
    return "refused"


def server_message(e, default):
    msg = e.body.get("message") if isinstance(e.body, dict) else None
    rel = e.body.get("release") if isinstance(e.body, dict) else None
    out = str(msg)[:1000] if msg else default
    if rel and str(rel) not in out:
        out += " (release %s)" % str(rel)[:100]
    return out


class Api:
    """Tiny JSON-over-HTTP client with connection-state tracking. Every POST body carries
    client_version and protocol (PROTOCOL3 section 3, M25/N16)."""

    UA = "pathology-volunteer/%s (protocol %d)" % (CLIENT_VERSION, PROTOCOL)

    def __init__(self, server):
        self.server = server.rstrip("/")
        self.connected = False
        self.last_ok = 0.0
        self.last_error = ""

    @staticmethod
    def encode(body):
        body = dict(body)
        body.setdefault("client_version", CLIENT_VERSION)
        body.setdefault("protocol", PROTOCOL)
        return json.dumps(body, separators=(",", ":")).encode()

    def post(self, path, body, timeout=20.0, data=None):
        if data is None:
            data = self.encode(body)
        req = urllib.request.Request(self.server + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": self.UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(8 * 1024 * 1024)
                status = r.status
        except urllib.error.HTTPError as e:
            try:
                raw = e.read(65536)
            except Exception:
                raw = b""
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

    def get(self, path, timeout=15.0):
        req = urllib.request.Request(self.server + path, method="GET", headers={"User-Agent": self.UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read(8 * 1024 * 1024)
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, {"error": "HTTP %d" % e.code}, e.code >= 500)
        except Exception as e:
            raise ApiError(0, {"error": str(e)[:200]}, True)
        try:
            return json.loads(raw.decode("utf-8", "replace") or "{}")
        except Exception:
            raise ApiError(200, {"error": "bad json from server"}, True)


def open_candidates(e):
    """Unresolved candidates deeper than an exit's best (PROTOCOL3 section 1), from whichever field
    the server sends in an exit entry; None when it sends none."""
    if not isinstance(e, dict):
        return None
    for k in ("candidates_open_deeper", "open_candidates", "unresolved_open"):
        v = e.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
    c = e.get("candidates")
    if isinstance(c, dict):
        for k in ("open_deeper", "open"):
            v = c.get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                return v
    return None


def campaign_state_of(res):
    """'running' | 'complete' | 'closed' from a heartbeat/lease/register response."""
    if not isinstance(res, dict):
        return None
    st = res.get("campaign_state")
    if st in ("running", "complete", "closed"):
        return st
    camp = res.get("campaign")
    if isinstance(camp, dict):
        cs = camp.get("campaign_state") or camp.get("status")
        if cs in ("complete", "closed"):
            return cs
        if cs in ("open", "running"):
            return "running"
    return None


# ------------------------------------------------------------------ worker slots
class RunResult:
    """Outcome of one worker process.
    kind 'ok'   : usable (summary, remaining, level, unresolved)
         'void' : the output is not usable (reason); `failure` = report it for the job
         'none' : nothing to use and not the job's fault (killed by us, interrupted before
                  expanding its root because we stopped it, the binary would not start)"""
    __slots__ = ("kind", "summary", "remaining", "level", "unresolved", "reason", "detail", "failure", "hash")

    def __init__(self, kind, reason=None, detail="", failure=False, summary=None, remaining=None, level=None,
                 unresolved=None, run_hash=None):
        self.kind = kind
        self.reason = reason
        self.detail = detail
        self.failure = failure
        self.summary = summary
        self.remaining = remaining or []
        self.level = level
        self.unresolved = unresolved or []
        self.hash = run_hash


class Slot:
    """One worker slot: a thread that runs worker processes one job at a time."""

    def __init__(self, idx):
        self.idx = idx
        self.thread = None
        self.proc = None
        self.job = None
        self.state = "idle"           # idle | running | paused | finishing | dead (retirement is `retire`)
        self.frozen = False           # M56: the worker process is SIGSTOPped by us (Pause, Ctrl-Z)
        self.retire = False
        self.started_at = 0.0
        self.stop_sent_at = 0.0
        self.killed_by_client = False
        self.drop_requested = False
        self.watchdog_fired = False
        self.watchdog_why = ""        # what the watchdog saw (the failure detail)
        self.grace_killed = False     # SIGKILLed after Stop's grace: this run's node stays open
        self.cur = None               # {"depth", "code"} node being expanded ('?' = unknown)
        self.best = None              # best accepted state this job
        self.win_hist = collections.deque()  # (t, depth, code) from STATUS win_* fields
        self.last_line_at = 0.0       # last line of any kind from the worker (watchdog)
        self.split_at = 0.0           # when the current run should have split at the latest
        self.split_after_s = 0.0
        self.consecutive_failures = 0
        self.cur_seed = None          # seed of the run in progress (a node of the local tree)
        self.stack = []               # local LIFO stack of open seeds for this window
        self.nodes_done = 0
        self.window_end = 0.0
        self.phase = None             # window | absorb | None
        self.run_started_at = 0.0
        self.saw_output = False       # first line (SRC_HASH) read: the worker is past startup
        self.stop_pending = False     # Stop requested before the worker printed anything
        self.stderr_tail = collections.deque(maxlen=12)

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
            "frozen": self.frozen,
            "retiring": bool(self.retire and j is not None),
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


def _drain_stderr(pipe, tail):
    try:
        for line in pipe:
            line = line.rstrip()
            if line:
                tail.append(line[:300])
    except (OSError, ValueError):
        pass


class Volunteer:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)   # queue / stop / retire changes
        self.ring = collections.deque(maxlen=60)
        self.api = Api(args.server)
        self.token = None
        self.name = None
        self.src_hash = None
        self.worker_base = None          # argv prefix of the pinned private copy of the worker (M23)
        self.worker_src = None           # the binary the volunteer built
        self.worker_info = {}            # --version: PROTOCOL, LIMITS, KNOBS, GIT_SHA
        self.path_tok_max = PATH_TOK_MAX_DEFAULT
        self.workers = max(1, args.workers)
        self.outbox = os.path.abspath(args.outbox)
        self.allow_dirty = os.environ.get("VOLUNTEER_ALLOW_DIRTY") == "1"
        self.env = worker_env(os.environ, self.allow_dirty)
        self.lock_fd = None
        self.exit_code = 0
        self.fatal_msg = None
        self.user_stop = False
        self.phase_name = "startup"       # startup | running | waiting | exiting
        self.sigint_count = 0
        self.ui_started = False
        self.ui_last_poll = 0.0
        self.ui_close_at = 0.0
        self.ui_secret = secrets.token_hex(16)
        self.session_start = time.time()
        self.completed_campaign_id = None
        self.completion = None           # closing summary shown by the panel {"campaign_id", "lines", "at"}
        self.waiting_next = False
        self.ever_ran = False            # a campaign ran in this process (Ctrl-C then means 'stop waiting')
        self.pre_registered = False      # wait_for_campaign already registered
        self.reregistered = False
        self.outbox_meta = {}            # path -> {"jobs": [...], "created_at": t, "campaign_id": id}
        self.exit_pref = args.exit       # None = any exit
        self.workers_changed_at = 0.0
        self.workers_prev = self.workers
        self.pause_requested_at = 0.0
        self.resume_requested_at = 0.0
        self.paused = False
        self.job_durations = collections.deque(maxlen=20)
        self.hour_best = collections.deque()         # monotonic deque of (t, depth, code, exit): rolling 1 h max
        self.sec_levels = collections.deque()        # (t, depth, code, exit) of LEVEL lines: the "last second" board
        self.depth_hist = {}                         # job depth -> recent window durations (expected work, M108)
        self.grant_depths = collections.deque(maxlen=20)  # depths of recent grants (lease sizing)
        self.lease_seq = 0
        self._reset_campaign_state()

    def _reset_campaign_state(self):
        """Everything that belongs to one campaign (a re-registration starts from scratch)."""
        with self.lock:
            self.campaign = {}
            self.extra = []
            self.slots = []
            self.queue = collections.deque()      # leased, not yet started jobs
            self.pending = []                     # reports waiting for the next batch
            self.send_event = threading.Event()
            self.sender_may_exit = False
            self.last_send_at = 0.0
            self.last_lease_at = 0.0
            self.stopping = False                 # winding down: workers get SIGINT
            self.draining = False                 # campaign complete: no new leases, running work finishes
            self.end_reason = None                # stop | complete | closed | reregister | fatal
            self.no_send = False                  # the server refuses this token: keep reports on disk
            self.stop_reason = ""
            self.heartbeat_now = False
            self.campaign_state = "running"
            self.campaign_result = None
            self.stats = {"jobs_done": 0, "done": 0, "split": 0, "open_root": 0, "nodes_done": 0, "nodes_open": 0,
                          "runs": 0, "cpu_seconds": 0.0, "reports_sent": 0, "batches_sent": 0,
                          "reports_failed": 0, "runs_without_summary": 0, "failures_reported": 0,
                          "runs_void": 0, "unresolved": 0, "grants_deduped": 0, "released_too_big": 0,
                          "folded_back": 0, "watchdog": 0, "batches_413": 0}
            self.best_per_exit = {}
            self.last_heartbeat = 0.0
            self.last_heartbeat_ok = 0.0
            self.last_tick = time.time()
            self.leases_from_server = None
            self.no_jobs_until = 0.0
            self.last_no_jobs_log = 0.0
            self.exit_applied = self.exit_pref is None
            self.exit_pending = False             # M58: an exit preference no lease response has answered yet
            self.exit_fallback = False
            self.me = None
            self.exits = None
            self.campaign_status = None
            self.campaign_status_at = 0.0         # last fetch attempt (throttle)
            self.campaign_status_ok_at = 0.0      # M60: last successful fetch (the age shown)
            self.window_now = None                # M108: the server's current window {split_after_s, absorb_total_s, mode, at}
            self.job_total_s = 0.0
            self.longest_job_s = 0.0
            self.hour_best.clear()
            self.sec_levels.clear()
            self.job_durations.clear()
            self.depth_hist = {}
            self.grant_depths.clear()
            self.last_reconcile = 0.0

    def log(self, msg):
        log(msg, self.ring)

    # ------------------------------------------------------------ startup
    def check_dirty(self):
        if self.allow_dirty:
            self.log("VOLUNTEER_ALLOW_DIRTY=1: skipping the clean-checkout check; BS_* variables reach the workers "
                     "(development only)")
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
            emit("\nRefusing to run: these files differ from the committed version:\n%s\n"
                 "Run `git checkout -- backsearch.c sokoban_bfs.c sokoban_bfs.h v2/volunteer.py` in the\n"
                 "backsearch directory (and rebuild), or set VOLUNTEER_ALLOW_DIRTY=1 for development." % out.stdout)
            sys.exit(2)

    def rebuild_cmd(self):
        """M13: the exact command that rebuilds the binary this client runs."""
        w = self.args.worker
        p = w if os.path.isabs(w) else os.path.join(BACKSEARCH_DIR, w)
        p = os.path.abspath(p)
        rel = os.path.relpath(p, BACKSEARCH_DIR)
        target = rel if not rel.startswith("..") else p
        return "./build_pgo.sh -o %s --no-torch" % shlex.quote(target)

    def update_advice(self):
        """The same two commands the collective page gives; the campaign's release tag (if it
        names one) as an alternative, since `git checkout TAG` fails where the tag does not exist."""
        adv = "git pull && %s" % self.rebuild_cmd()
        rel = self.campaign.get("release") or self.campaign.get("release_tag")
        if rel:
            adv += "\n    (or the campaign's release: git fetch && git checkout %s && %s)" % (
                shlex.quote(str(rel)[:80]), self.rebuild_cmd())
        return adv

    def acquire_outbox_lock(self):
        """M17: one client process per outbox. The token is keyed by server and outbox, so two
        instances never share a token (a shared token makes each release the other's jobs)."""
        try:
            os.makedirs(self.outbox, exist_ok=True)
            for sub in ("rejected", "closed"):
                os.makedirs(os.path.join(self.outbox, sub), exist_ok=True)
        except OSError as e:
            emit("Cannot create the outbox directory %s: %s" % (self.outbox, e))
            sys.exit(2)
        if fcntl is None:
            self.log("warning: no file locking on this platform; do not run two clients with the same --outbox")
            return
        path = os.path.join(self.outbox, ".lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            other = ""
            try:
                other = os.pread(fd, 64, 0).decode("ascii", "replace").strip()
            except (OSError, AttributeError):
                pass
            os.close(fd)
            emit("\nAnother volunteer client%s is already running with the outbox\n    %s\n"
                 "To run a second client on this machine, give it its own outbox and panel port, e.g.\n"
                 "    python3 v2/volunteer.py --outbox %s --port %d ...\n"
                 "(each outbox gets its own registration, so the two clients never share a token)."
                 % (" (pid %s)" % other if other else "", self.outbox, shlex.quote(self.outbox + "2"), self.args.port + 1))
            sys.exit(2)
        try:
            os.ftruncate(fd, 0)
            os.pwrite(fd, ("%d\n" % os.getpid()).encode(), 0)
        except (OSError, AttributeError):
            pass
        self.lock_fd = fd

    def resolve_worker(self):
        w = self.args.worker
        cands = [w, os.path.join(BACKSEARCH_DIR, w)] if not os.path.isabs(w) else [w]
        for c in cands:
            if os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def pin_worker(self):
        """M23: run a private copy of the worker, pinned at startup, so a rebuild while the
        client runs can never mix binaries (or hashes) inside one report. Resolved once, so no
        per-run lookup can fail (M50/M64)."""
        src = self.resolve_worker()
        if src is None:
            emit("Worker binary not found: %s\nBuild it in %s with:\n    %s"
                 % (self.args.worker, BACKSEARCH_DIR, self.rebuild_cmd()))
            sys.exit(2)
        bindir = os.path.join(self.outbox, ".bin")
        dst = os.path.join(bindir, os.path.basename(src))
        try:
            os.makedirs(bindir, exist_ok=True)
            tmp = "%s.tmp%d" % (dst, os.getpid())
            shutil.copyfile(src, tmp)
            os.chmod(tmp, 0o700)
            os.replace(tmp, dst)          # a new inode: never rewrites an image that is still running
        except OSError as e:
            emit("Cannot copy the worker binary to %s: %s" % (bindir, e))
            sys.exit(2)
        self.worker_src = src
        self.worker_base = [sys.executable, dst] if dst.endswith(".py") else [dst]

    def read_version(self):
        argv = self.worker_base + ["--version"]
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=BACKSEARCH_DIR, env=self.env)
        except (OSError, subprocess.TimeoutExpired) as e:
            emit("Cannot run the worker (%s): %s" % (" ".join(argv), e))
            sys.exit(2)
        info = {}
        for line in out.stdout.splitlines():
            parts = line.rstrip("\r").split("\t", 1)
            if len(parts) == 2:
                info[parts[0]] = parts[1].strip()
        h = info.get("SRC_HASH", "")
        if not HEX_RE.match(h):
            if "SRC_HASH" in info:
                emit("The worker %s was built without a source hash (a plain cc build). Rebuild it in the\n"
                     "backsearch directory with:\n    %s" % (self.worker_src, self.rebuild_cmd()))
            else:
                emit("The worker did not print an SRC_HASH line for --version (exit %s). It is too old;\n"
                     "update the checkout and rebuild:\n    git pull && %s" % (out.returncode, self.rebuild_cmd()))
            sys.exit(2)
        try:
            proto = int(info.get("PROTOCOL", "0"))
        except ValueError:
            proto = 0
        if proto < PROTOCOL:
            if not self.allow_dirty:
                emit("The worker %s speaks protocol %d; this client needs protocol %d.\n"
                     "Update the checkout and rebuild:\n    git pull && %s"
                     % (self.worker_src, proto, PROTOCOL, self.rebuild_cmd()))
                sys.exit(2)
            self.log("!!! VOLUNTEER_ALLOW_DIRTY=1: running a protocol-%d worker with a protocol-%d client "
                     "(development only)" % (proto, PROTOCOL))
        limits = {}
        try:
            limits = json.loads(info.get("LIMITS", "{}"))
        except ValueError:
            pass
        ptm = limits.get("path_tok_max") if isinstance(limits, dict) else None
        if isinstance(ptm, int) and ptm > 0:
            self.path_tok_max = min(ptm, 4096)
        elif proto < PROTOCOL:
            self.path_tok_max = 250
        info["PROTOCOL"] = proto
        self.worker_info = info
        self.src_hash = h.lower()
        return self.src_hash

    def pidfile(self):
        return os.path.join(self.outbox, ".workers.pid")

    def cleanup_orphans(self):
        """M14: workers left behind by a client that died (e.g. SIGSTOPped while paused, then
        the terminal was closed) are killed at startup. Only processes whose command line still
        runs this outbox's pinned binary are touched."""
        try:
            with open(self.pidfile()) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        pids = d.get("workers") if isinstance(d, dict) else None
        binp = str(d.get("bin") or "") if isinstance(d, dict) else ""
        if not isinstance(pids, list) or not binp:
            return
        killed = []
        for pid in pids:
            if not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            try:
                cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                                     timeout=5).stdout
            except (OSError, subprocess.TimeoutExpired):
                continue
            if binp not in cmd:
                continue
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            killed.append(pid)
        if killed:
            self.log("killed %d worker(s) left behind by an earlier client run: %s" % (len(killed), killed))
        self._write_pidfile()

    def _write_pidfile(self):
        """Caller holds the lock (or is single-threaded at startup)."""
        if not self.worker_base:
            return
        data = {"client_pid": os.getpid(), "bin": self.worker_base[-1],
                "workers": [s.proc.pid for s in self.slots if s.proc is not None]}
        try:
            tmp = self.pidfile() + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.pidfile())
        except OSError:
            pass

    def kill_all_workers(self):
        """Last resort on any non-graceful exit: no worker may outlive the client."""
        with self.lock:
            for s in self.slots:
                p = s.proc
                if p is not None and p.poll() is None:
                    for sig in (signal.SIGCONT, signal.SIGKILL):
                        try:
                            os.kill(p.pid, sig)
                        except OSError:
                            pass

    # ------------------------------------------------------------ token store (M17)
    def token_key(self):
        return "%s|%s" % (self.api.server, os.path.realpath(self.outbox))

    def _token_store_lock(self):
        if fcntl is None:
            return None
        try:
            fd = os.open(TOKEN_FILE + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fd
        except OSError:
            return None

    def _token_store_unlock(self, fd):
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except OSError:
                pass

    def load_token_file(self):
        try:
            with open(TOKEN_FILE) as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def update_token_file(self, fn):
        """Read-modify-write the store under a lock (other clients on this machine share it)."""
        fd = self._token_store_lock()
        try:
            d = self.load_token_file()
            fn(d)
            tmp = TOKEN_FILE + ".tmp%d" % os.getpid()
            with open(tmp, "w") as f:
                json.dump(d, f, indent=1)
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, TOKEN_FILE)
        except OSError as e:
            self.log("warning: cannot write %s: %s" % (TOKEN_FILE, e))
        finally:
            self._token_store_unlock(fd)

    def drop_token(self):
        key = self.token_key()
        self.token = None
        self.update_token_file(lambda d: d.pop(key, None))

    def store_token(self, name):
        key = self.token_key()
        entry = {"token": self.token, "name": name, "registered_at": time.time(),
                 "campaign_id": self.campaign.get("id"), "outbox": self.outbox}
        self.update_token_file(lambda d: d.__setitem__(key, entry))

    # ------------------------------------------------------------ campaign definition
    def apply_campaign(self, camp, initial=False):
        """Store the campaign parameters. The definition (grid, extra) is validated every
        time it arrives (M85): a change the allowlist refuses stops the client."""
        if not (isinstance(camp, dict) and camp):
            return True
        merged = dict(self.campaign)
        merged.update(camp)
        extra, err = parse_extra(merged.get("extra"))
        grid = str(merged.get("grid") or "")
        if err is None and not GRID_RE.match(grid):
            err = "the campaign grid %r is not ROWSxCOLS" % grid[:20]
        if err is not None:
            msg = ("Refusing to run: %s.\nAllowed worker flags: --allow-exit-transit, --num-holes N, --num-blocks N, "
                   "--min-walls N. This protects your machine from a misconfigured or compromised server; "
                   "the campaign owner has to fix the campaign definition." % err)
            if initial:
                self.fatal_now(2, msg)
            else:
                self.fatal_stop(2, msg)
            return False
        with self.lock:
            changed = self.campaign and (self.extra != extra or self.campaign.get("grid") != grid)
            self.campaign = merged
            self.extra = extra
        if changed:
            self.log("!!! the campaign definition changed: grid %s, flags %s (new runs use it)" % (grid, " ".join(extra)))
        self.validate_exit_pref()
        return True

    def validate_exit_pref(self):
        """M53: an exit preference outside the campaign would make every lease fail."""
        allowed = self.campaign_exits()
        with self.lock:
            pref = self.exit_pref
            if pref is not None and allowed and pref not in allowed:
                self.exit_pref = None
                self.exit_applied = True
                bad = True
            else:
                bad = False
        if bad:
            self.log("warning: exit %d is not part of this campaign (exits %s); leasing jobs of any exit"
                     % (pref, ", ".join(str(a) for a in allowed)))

    def campaign_exits(self):
        ex = self.campaign.get("exits")
        out = []
        if isinstance(ex, list):
            for a in ex:
                try:
                    out.append(int(a))
                except (TypeError, ValueError):
                    pass
        return out

    def check_hash(self):
        hashes = [str(h).lower() for h in (self.campaign.get("hashes") or [])]
        if not hashes:
            self.log("warning: the campaign lists no whitelisted hashes; reports may be refused")
            return True
        if self.src_hash in hashes:
            return True
        self.fatal_now(2, self.hash_refusal(hashes))
        return False

    def hash_refusal(self, hashes):
        """M114: the build may also be newer than what the campaign accepts yet."""
        retired = [str(h).lower() for h in (self.campaign.get("retired_hashes") or [])]
        if self.src_hash in retired:
            why = "Your build is older than what the campaign accepts. Update and rebuild in the backsearch directory:"
        else:
            why = ("Either your build is older than the campaign's (update and rebuild as below), or it is newer\n"
                   "than what the campaign accepts yet (if you pulled very recently: wait until the owner accepts\n"
                   "the new build, or check out the campaign's release). To update, in the backsearch directory:")
        return ("The worker's source hash %s is not accepted by the campaign.\nAccepted: %s\n%s\n    %s\n"
                "then start volunteer.py again." % (self.src_hash, ", ".join(hashes), why, self.update_advice()))

    # ------------------------------------------------------------ campaign life cycle
    def fatal_now(self, code, msg):
        """A refusal before the workers start (main thread)."""
        self.exit_code = code
        self.fatal_msg = msg
        with self.lock:
            self.end_reason = "fatal"
            self.stopping = True
        self.completion = {"campaign_id": self.campaign.get("id"), "lines": msg.splitlines(), "at": time.time(),
                           "fatal": True}

    def fatal_stop(self, code, msg, no_send=False):
        """A refusal while running (any thread): wind down, then exit with `code`.
        no_send: the server refuses this client's requests, keep reports on disk."""
        with self.lock:
            first = self.fatal_msg is None
            self.exit_code = code
            if first:
                self.fatal_msg = msg
            self.end_reason = "fatal"
            if no_send:
                self.no_send = True
        if first:
            self.log("!!! " + msg.splitlines()[0])
            self.completion = {"campaign_id": self.campaign.get("id"), "lines": msg.splitlines(), "at": time.time(),
                               "fatal": True}
        self.request_stop(msg.splitlines()[0])

    def end_campaign(self, kind, msg):
        """The campaign ended under us: 'closed' (410) or 'reregister' (the token is unknown).
        Wind down; run() returns and the main loop registers again."""
        with self.lock:
            if self.end_reason in ("fatal", "closed", "reregister"):
                return
            self.end_reason = kind
            self.no_send = True
            if kind == "closed":
                self.campaign_state = "closed"
        self.log(msg)
        self.request_stop(msg)

    def campaign_gone(self, e):
        """HTTP 410 on a request: the owner closed the campaign, or it is complete and a newer
        campaign is open (body campaign_state 'complete'). Either way this token is done."""
        if isinstance(e.body, dict) and e.body.get("campaign_state") == "complete":
            with self.lock:
                self.campaign_state = "complete"
                if self.end_reason in (None, "complete", "stop"):
                    self.end_reason = "complete"
                self.no_send = True               # its jobs are all finished: nothing left to deliver
            self.log("the campaign is complete and a newer campaign is open (HTTP 410)")
            self.request_stop("the campaign is complete")
        else:
            self.end_campaign("closed", "the campaign was closed by its owner (HTTP 410); registering again")

    def on_campaign_state(self, res):
        st = campaign_state_of(res)
        if st is None:
            return
        with self.lock:
            prev = self.campaign_state
            self.campaign_state = st
            if isinstance(res.get("result"), (dict, list)):
                self.campaign_result = res["result"]
            elif isinstance(res.get("campaign_result"), (dict, list)):
                self.campaign_result = res["campaign_result"]
        if st == "complete" and prev != "complete":
            self.begin_drain()
        elif st == "closed" and prev != "closed":
            self.end_campaign("closed", "the campaign was closed by its owner; registering again")

    def begin_drain(self):
        """Campaign complete (new feature): stop asking for work, let running workers finish,
        deliver everything, then run() returns 'complete'."""
        with self.cond:
            if self.draining or self.stopping:
                return
            self.draining = True
            if self.end_reason is None:
                self.end_reason = "complete"
            queued = [j["id"] for j in self.queue]
            self.queue.clear()
            self.cond.notify_all()
            running = [s.job["id"] for s in self.slots if s.job]
        self.send_event.set()
        self.log("the campaign is complete: no new jobs; delivering the last reports%s"
                 % (" (letting %d running job(s) finish)" % len(running) if running else ""))
        if queued:
            self.log("dropped %d queued job(s) that the finished campaign no longer needs" % len(queued))

    # ------------------------------------------------------------ registration
    def _sleep(self, seconds):
        end = time.time() + seconds
        while time.time() < end and not self.user_stop:
            time.sleep(0.25)

    def base_body(self):
        return {"token": self.token, "workers": self.workers, "paused": False, "boards": []}

    def register(self):
        """Returns 'ok', 'no_campaign' or 'stopped' (fatal refusals set exit_code and fatal_msg).
        A stored token is probed with a heartbeat first: 410 (campaign closed), 401/403/404
        (unknown token) and any other refusal drop it and register again (M16); 426 = this
        client is too old (M25)."""
        store = self.load_token_file()
        entry = store.get(self.token_key())
        legacy = store.get(self.api.server)             # pre-protocol-3 entry (keyed by server only): its name only
        known_name = None
        for e in (entry, legacy):
            if isinstance(e, dict) and e.get("name"):
                known_name = e["name"]
                break
        if self.args.reregister and not self.reregistered:
            self.reregistered = True
            entry = None
        if self.token is None and isinstance(entry, dict) and entry.get("token"):
            self.token = entry["token"]
            self.name = entry.get("name") or self.args.name or "?"
            if self.args.name and self.args.name != self.name:
                self.log("note: already registered as %r for %s; the name is fixed after registration "
                         "(use --reregister to start over)" % (self.name, self.api.server))
        while not self.user_stop:
            if self.token:
                try:
                    res = self.api.post("/api/v2/heartbeat", self.base_body())
                except ApiError as e:
                    kind = classify(e)
                    if kind == "too_old":
                        self.fatal_now(2, "This client (version %s) is too old for the campaign.\n%s"
                                       % (CLIENT_VERSION, server_message(e, "Update the checkout: " + self.update_advice())))
                        return "stopped"
                    if kind == "revoked":
                        self.fatal_now(2, "This client's token was revoked by the server. Stopping.")
                        return "stopped"
                    if kind == "transient":
                        self.log("cannot reach the server yet (%s); will retry" % e)
                        self._sleep(10)
                        continue
                    self.log("the stored registration is no longer valid (%s); registering again" % e)
                    if kind == "closed":
                        self.retire_outbox("the campaign of the stored registration was closed")
                    self.drop_token()
                    continue
                if res.get("revoked"):
                    self.fatal_now(2, "This client's token was revoked by the server. Stopping.")
                    return "stopped"
                if not self.apply_campaign(res.get("campaign"), initial=True):
                    return "stopped"
                st = campaign_state_of(res)
                if st in ("complete", "closed"):
                    self.finish_stale_campaign(res, st)
                    continue
                if self.campaign:
                    self.log("reusing registration %r for campaign %s" % (self.name, self.campaign.get("id", "?")))
                    return "ok"
                self.log("heartbeat carried no campaign parameters; registering again to fetch them")
                self.drop_token()
                continue
            name = self.args.name or (self.name if self.name and self.name != "?" else None) or known_name
            if not name:
                self.fatal_now(2, "No registration found for %s. Pass --name \"Your name\"." % self.api.server)
                return "stopped"
            name = name.strip()[:40]
            try:
                res = self.api.post("/api/v2/register", {"name": name, "workers": self.workers})
            except ApiError as e:
                kind = classify(e)
                if kind == "no_campaign":
                    return "no_campaign"
                if kind == "too_old":
                    self.fatal_now(2, "This client (version %s) is too old for the campaign.\n%s"
                                   % (CLIENT_VERSION, server_message(e, "Update the checkout: " + self.update_advice())))
                    return "stopped"
                if kind == "revoked":
                    self.fatal_now(2, "This name or client was revoked by the server. Stopping.")
                    return "stopped"
                if e.status == 503 and "queue_position" in e.body:
                    self.log("the campaign is full; queue position %s. Retrying in 60 s." % e.body.get("queue_position"))
                    self._sleep(60)
                    continue
                if e.status == 400:
                    self.fatal_now(2, "Registration refused: %s" % json.dumps(e.body)[:300])
                    return "stopped"
                self.log("registration failed (%s); retrying in 30 s" % e)
                self._sleep(30)
                continue
            if not res.get("token"):
                self.fatal_now(2, "Registration returned no token: %s" % json.dumps(res)[:300])
                return "stopped"
            st = campaign_state_of(res)
            camp = res.get("campaign") if isinstance(res.get("campaign"), dict) else {}
            if st in ("complete", "closed") or (self.completed_campaign_id is not None
                                                and camp.get("id") == self.completed_campaign_id):
                return "no_campaign"
            self.token = res["token"]
            self.name = name
            if not self.apply_campaign(camp, initial=True):
                return "stopped"
            self.store_token(name)
            self.log("registered as %r for campaign %s (token stored in %s)" % (name, self.campaign.get("id", "?"), TOKEN_FILE))
            return "ok"
        return "stopped"

    def finish_stale_campaign(self, res, st):
        """The stored registration's campaign ended while this client was not running: deliver
        its outbox (complete) or set it aside (closed), show the result, then register anew."""
        cid = (res.get("campaign") or {}).get("id") if isinstance(res.get("campaign"), dict) else None
        if st == "complete":
            with self.lock:
                if isinstance(res.get("me"), dict):
                    self.me = res["me"]
                if isinstance(res.get("exits"), dict):
                    self.exits = res["exits"]
                r = res.get("result") or res.get("campaign_result")
                if isinstance(r, (dict, list)):
                    self.campaign_result = r
            self.scan_outbox()
            if self.outbox_meta:
                self.log("delivering %d outbox file(s) of the finished campaign" % len(self.outbox_meta))
                self.flush_outbox(force=True)
            if cid is None or cid != self.completed_campaign_id:
                lines = self.closing_summary(session=False)
                self.completion = {"campaign_id": cid, "lines": lines, "at": time.time()}
                emit("\n".join(["", "=" * 72] + lines + ["=" * 72, ""]))
            self.completed_campaign_id = cid
        else:
            self.retire_outbox("the campaign was closed")
        if not self.outbox_files():
            self.drop_token()
        else:
            self.log("%d report file(s) could not be delivered yet; keeping the registration to retry them "
                     "on the next start" % len(self.outbox_files()))
            self.token = None
        with self.lock:
            self.campaign = {}

    def wait_for_campaign(self):
        """--keep-going: poll until a new campaign is open; returns False on Ctrl-C."""
        self.waiting_next = True
        self.phase_name = "waiting"
        poll = max(1.0, float(self.args.campaign_poll_s))
        self.log("waiting for the next campaign (checking every %.0f s; Ctrl-C to quit)" % poll)
        try:
            while not self.user_stop:
                self._sleep(poll)
                if self.user_stop:
                    return False
                r = self.register()
                if r == "ok":
                    return True
                if r == "stopped":
                    return False
            return False
        finally:
            self.waiting_next = False

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
        with self.lock:
            extra = list(self.extra)
            grid = str(self.param("grid", "5x5"))
            nodes = self.split_after_nodes()
        argv = list(self.worker_base) + ["--grid", grid, "--two-tables", "--exit", str(job["exit"])]
        if seed != "":
            argv += ["--seed-path", seed]         # M41: the exit root "" runs without --seed-path
        # never 0 (= never split): sub-second windows (N18) keep three decimals
        argv += ["--time", "0", "--split-after", "%.3f" % max(0.001, split_after)]
        if nodes:
            argv += ["--split-after-nodes", str(nodes)]   # deterministic splits (test campaigns, N18)
        return argv + ["--status-every", str(STATUS_EVERY_MS)] + extra

    def min_window_s(self):
        """N18: the floor of a window and of every run's split time: 1 s unless the campaign sets
        min_window_s (test campaigns use sub-second windows)."""
        v = nonneg_float(self.campaign.get("min_window_s"))
        return MIN_WINDOW_S_DEFAULT if v is None else min(3600.0, max(0.01, v))

    def split_after_nodes(self):
        """N18: a campaign's split_after_nodes N adds --split-after-nodes N to every run (the worker then
        also splits deterministically after N expansions; test campaigns)."""
        return pos_int(self.campaign.get("split_after_nodes"))

    # ------------------------------------------------------------ windows (M108)
    def note_window(self, sa, absorb, mode, source):
        """The server's current window, from every lease response (and a heartbeat that carries one).
        A job's window is chosen when the window starts, from the latest of these, never frozen at
        lease time: a job leased while the pool was deep must not run a full window in the endgame."""
        sa = nonneg_float(sa)
        if not sa:
            return
        w = {"split_after_s": sa, "absorb_total_s": nonneg_float(absorb),
             "mode": mode if isinstance(mode, str) and mode else None, "at": time.time(), "source": source}
        with self.lock:
            prev = self.window_now
            self.window_now = w
        # logged when the mode changes (an endgame window varies with the pool at every lease)
        if w["mode"] and prev is not None and prev.get("mode") != w["mode"]:
            self.log("the server's window is now %s s (%s); new windows use it" % (fmt_s(sa), w["mode"]))

    def window_for(self, job):
        """(split_after_s, absorb_total_s, mode) for a window of `job` that starts now."""
        with self.lock:
            w = dict(self.window_now) if self.window_now else None
        sa = job.get("split_after_fixed")             # a fingerprint re-run keeps its own (longer) window
        if sa is None:
            sa = w["split_after_s"] if w else job.get("split_after_s")
        if sa is None:
            sa = self.fparam("split_after_s", 1800)
        absorb = job.get("absorb_fixed")              # a per-job absorption budget from the grant
        if absorb is None and w is not None and w.get("absorb_total_s") is not None:
            absorb = w["absorb_total_s"]
        if absorb is None:
            absorb = job.get("absorb_total_s")
        if absorb is None:
            absorb = self.fparam("absorb_total_s", 120)
        mode = w.get("mode") if w else job.get("window_mode")
        return max(self.min_window_s(), float(sa)), max(0.0, float(absorb)), mode

    def expected_s(self, depth, full):
        """Expected wall time of a window of a job at `depth` (caller holds the lock): the mean of the
        recent windows at that depth, else at the nearest shallower sampled depth (a deeper subtree is
        rarely larger), else `full` (a job nothing is known about counts as a whole window)."""
        h = self.depth_hist.get(depth)
        if h and len(h) >= 2:
            return sum(h) / len(h)
        known = [d for d, x in self.depth_hist.items() if d < depth and len(x) >= 2]
        if known:
            x = self.depth_hist[max(known)]
            return min(full, sum(x) / len(x))
        return full

    def fparam(self, key, default):
        try:
            return float(self.param(key, default))
        except (TypeError, ValueError):
            return float(default)

    def limits(self):
        """Report caps: the client's defaults, lowered by any the server announces."""
        nodes, rbytes = MAX_NODES, REPORT_BYTES_MAX
        lim = self.campaign.get("limits")
        if isinstance(lim, dict):
            for k, v in lim.items():
                if not isinstance(v, int) or v <= 0:
                    continue
                if k.lower() in ("nodes_max", "max_nodes"):
                    nodes = min(nodes, v)
                elif k.lower() in ("report_body_max", "report_bytes_max"):
                    rbytes = min(rbytes, int(v * 0.85))
        return nodes, rbytes

    def mean_job_s(self):
        """Recent mean wall time of a window (job), for the panel."""
        with self.lock:
            if self.job_durations:
                return max(0.05, sum(self.job_durations) / len(self.job_durations))
        return max(0.05, self.fparam("split_after_s", 1800))

    def held_ids(self):
        """Every job this client still owns (caller holds the lock): running, queued, and
        finished ones whose report is pending or in the outbox (M21, PROTOCOL3 section 3).
        Outbox reports count only while younger than lease_s, so a report the server keeps
        failing cannot hold its job forever."""
        ids = []
        seen = set()

        def add(i):
            if isinstance(i, int) and i not in seen:
                seen.add(i)
                ids.append(i)
        for s in self.slots:
            if s.job:
                add(s.job["id"])
        for j in self.queue:
            add(j["id"])
        for r in self.pending:
            add(r.get("job_id"))
        horizon = time.time() - self.lease_s()
        cid = self.campaign.get("id")
        for meta in self.outbox_meta.values():
            if meta.get("created_at", 0) >= horizon and (cid is None or meta.get("campaign_id") in (None, cid)):
                for i in meta.get("jobs", []):
                    add(i)
        return ids

    # ------------------------------------------------------------ leasing (7.6)
    def lease_want(self):
        """M108: how many jobs to ask for now, and whether an idle worker waits with nothing queued
        (then the request may come before batch_interval_s is over).
        - At most HOLD_PER_WORKER x workers jobs are here at once (running + queued): a big batch of
          leased jobs waits behind the local queue while other clients could run it.
        - By expected work, not by count: every worker should have its next job here when its current
          window has less than L = min(lease_ahead_s, window) left. A running window's remaining time is
          min(time to its split, max(expected - elapsed, elapsed)); a queued job counts its expected
          work (expected_s: a job of a depth nothing is known about counts as a whole window, so shallow
          jobs count as large and deep trivial ones as small). An idle worker always gets one."""
        now = time.time()
        with self.lock:
            active = [s for s in self.slots if not s.retire and s.thread is not None and s.thread.is_alive()]
            if not active:
                return 0, False
            running = sum(1 for s in self.slots if s.job is not None)
            queued = list(self.queue)
            w = self.window_now
            sa = w["split_after_s"] if w else self.fparam("split_after_s", 1800)
            ab = w["absorb_total_s"] if w and w.get("absorb_total_s") is not None else self.fparam("absorb_total_s", 120)
            sa = max(self.min_window_s(), sa)
            full = sa + max(0.0, ab)
            horizon = min(max(1.0, self.fparam("lease_ahead_s", 300)), sa)
            need, idle = 0.0, 0
            for s in active:
                if s.job is None:
                    idle += 1
                    need += horizon
                    continue
                e = self.expected_s(s.job.get("depth", seed_depth(s.job["seed"])), full)
                el = max(0.0, now - s.started_at) if s.started_at else 0.0
                to_end = s.window_end - now if s.window_end else full
                rem = max(0.0, min(to_end, max(e - el, el)))
                need += max(0.0, horizon - rem)
            need -= sum(self.expected_s(j.get("depth", 0), full) for j in queued)
            room = HOLD_PER_WORKER * len(active) - running - len(queued)
            # a new grant is expected to look like the recent ones (their depths, valued with what is known now)
            e_new = (sum(self.expected_s(d, full) for d in self.grant_depths) / len(self.grant_depths)
                     if self.grant_depths else full)
        if room <= 0:
            return 0, False
        floor = max(0, idle - len(queued))           # every idle worker gets a job
        n = max(floor, int(math.ceil(need / max(0.05, e_new))) if need > 0 else 0)
        cap = int(self.fparam("lease_cap", LEASE_CAP))
        return max(0, min(n, room, cap)), idle > 0 and not queued

    def leaser_loop(self):
        while True:
            with self.cond:
                self.cond.wait(0.5)
                if self.stopping or self.draining:
                    return
            try:
                self._lease_once()
            except Exception:
                self.log("!!! leaser: internal error (continuing): %s" % traceback.format_exc().strip().splitlines()[-1])
                time.sleep(2)

    def _lease_once(self):
        if self.paused or time.time() < self.no_jobs_until:
            return
        want, early = self.lease_want()
        if want <= 0:
            return
        since = time.time() - self.last_lease_at
        interval = self.fparam("batch_interval_s", 10)
        # one request per batch_interval_s; sooner (LEASE_EARLY_GAP_S) only for an idle worker with
        # nothing queued, so a short queue does not idle workers in a phase of trivial jobs (M108)
        if since < interval and not (early and since >= min(interval, LEASE_EARLY_GAP_S)):
            return
        self.last_lease_at = time.time()
        body = {"token": self.token, "n": want}
        with self.lock:
            pref = self.exit_pref
        if pref is not None:
            body["exit"] = pref
        try:
            res = self.api.post("/api/v2/lease", body)
        except ApiError as e:
            kind = classify(e)
            if kind == "too_old":
                self.fatal_stop(2, "This client (version %s) is too old for the campaign.\n%s"
                                % (CLIENT_VERSION, server_message(e, "Update: " + self.update_advice())), no_send=True)
            elif kind == "revoked":
                self.fatal_stop(2, "This client's token was revoked by the server.", no_send=True)
            elif kind == "closed":
                self.campaign_gone(e)
            elif kind == "reregister":
                self.end_campaign("reregister", "the server does not know this client's token any more (%s); "
                                  "registering again" % e)
            elif e.status == 400 and pref is not None and "exit" in str(e.error or ""):
                with self.lock:
                    if self.exit_pref == pref:
                        self.exit_pref = None
                        self.exit_applied = True
                self.log("the server refused exit %d (%s); leasing jobs of any exit" % (pref, e.error))
                self.last_lease_at = 0.0
            else:
                self.log("lease failed (%s); retrying in %d s" % (e, int(self.fparam("batch_interval_s", 10))))
            return
        if not isinstance(res, dict):
            return
        self.on_campaign_state(res)
        if self.draining or self.stopping:
            jobs = res.get("jobs") or []
            if jobs:
                self.log("the campaign ended while a lease was in flight; %d granted job(s) are not started" % len(jobs))
            return
        jobs = res.get("jobs")
        resp_sa = nonneg_float(res.get("split_after_s"))
        # the absorption budget of this lease (jobs.js windowFor, M109): 0 in the endgame, so that
        # small windows hand their children back to idle clients; absent = the campaign's absorb_total_s
        resp_abs = nonneg_float(res.get("absorb_total_s"))
        mode = res.get("window_mode") if isinstance(res.get("window_mode"), str) else None
        self.note_window(resp_sa, resp_abs, mode, "lease")
        good, bad_seed = [], []
        for j in jobs or []:
            if not isinstance(j, dict):
                continue
            try:
                jid = int(j["id"])
                ex = int(j["exit"])
                seed = str(j["seed"] if j["seed"] is not None else "")
            except (KeyError, TypeError, ValueError):
                continue
            if len(seed) > 4 * self.path_tok_max or not valid_seed(seed, self.path_tok_max):
                bad_seed.append({"id": jid, "exit": ex, "seed": seed})
                continue
            sa_job = nonneg_float(j.get("split_after_s"))   # only a fingerprint re-run carries its own window
            depth = j.get("depth") if isinstance(j.get("depth"), int) and not isinstance(j.get("depth"), bool) else seed_depth(seed)
            with self.lock:
                self.lease_seq += 1
                seq = self.lease_seq
            good.append({"id": jid, "exit": ex, "seed": seed, "depth": depth, "est_s": nonneg_float(j.get("est_s")),
                         "split_after_fixed": sa_job if sa_job else None,
                         "split_after_s": resp_sa, "absorb_total_s": resp_abs,
                         "absorb_fixed": nonneg_float(j.get("absorb_total_s")), "window_mode": mode,
                         "dup": bool(j.get("dup")), "stolen": bool(j.get("stolen")), "leased_at": time.time(), "seq": seq})
        for j in bad_seed:
            self.log("!!! job %d has a seed this client cannot run (%d chars); reporting it as a failure"
                     % (j["id"], len(j["seed"])))
            self.queue_failure(j, "bad_seed", "client: unparsable or overlong seed", None)
        fallback = bool(res.get("exit_fallback"))
        dups = []
        with self.cond:
            held = set(self.held_ids())           # M19: never queue a job this client already owns
            fresh = []
            for g in good:
                if g["id"] in held:
                    dups.append(g["id"])
                    continue
                held.add(g["id"])
                fresh.append(g)
            if dups:
                self.stats["grants_deduped"] += len(dups)
            if pref is not None and self.exit_pref == pref:
                # M58: the first lease response after a change settles it: applied (a job of that exit
                # came), or no open job there (jobs of other exits came: the server's fallback or stolen
                # jobs); with no job at all the preference is simply tried again at the next lease
                self.exit_pending = False
                if any(j["exit"] == pref for j in good):
                    self.exit_applied = True
                    self.exit_fallback = False
                elif good:
                    if not self.exit_fallback:
                        self.log("the server had no open jobs for exit %d; it handed out jobs of other exits%s"
                                 % (pref, "" if fallback else " (taken from other clients' queues)"))
                    self.exit_fallback = True
            if fresh:
                self.grant_depths.extend(g["depth"] for g in fresh)
                self.queue.extend(fresh)
                self.cond.notify_all()
            elif not good:
                wait = max(5.0, self.fparam("batch_interval_s", 10))
                self.no_jobs_until = time.time() + wait
                if time.time() - self.last_no_jobs_log > 60:
                    self.last_no_jobs_log = time.time()
                    self.log("no open jobs available right now; asking again every %.0f s" % wait)
        if dups:
            self.log("skipping duplicate grant(s) of job(s) this client already holds: %s" % dups[:20])

    def take_job(self, slot):
        """Block until a leased job is available (or stop/retire/pause/drain). The job is
        assigned to the slot under the same lock as the pop, so it never leaves `holding`."""
        with self.cond:
            while True:
                if self.stopping or slot.retire or self.paused or self.draining:
                    return None
                if self.queue:
                    job = min(self.queue, key=queue_key)     # M108: the server's lease order, not FIFO
                    self.queue.remove(job)
                    slot.job = job
                    slot.drop_requested = False
                    return job
                self.cond.wait(0.5)

    def slot_loop(self, slot):
        while True:
            try:
                with self.lock:
                    if self.stopping or slot.retire or self.draining:
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
                outcome = self.run_window(slot, job)
                wait = 0.0
                with self.lock:
                    if outcome != "fail" or slot.killed_by_client:
                        slot.consecutive_failures = 0
                    else:
                        slot.consecutive_failures += 1
                        if slot.consecutive_failures >= 3:
                            wait = min(600.0, 30.0 * 2 ** (slot.consecutive_failures - 3))
                if wait:
                    self.log("worker %d failed %d times in a row; backing off %.0f s"
                             % (slot.idx, slot.consecutive_failures, wait))
                    self._slot_sleep(slot, wait)
            except BaseException:
                # M50: a slot never dies silently holding a job: the job leaves `holding` and is released
                tb = traceback.format_exc().strip().splitlines()
                self.log("!!! worker %d: internal error: %s; its job is handed back" % (slot.idx, tb[-1] if tb else "?"))
                with self.lock:
                    p = slot.proc
                    if p is not None and p.poll() is None:
                        for sig in (signal.SIGCONT, signal.SIGKILL):
                            try:
                                os.kill(p.pid, sig)
                            except OSError:
                                pass
                    slot.proc = None
                    slot.job = None
                    slot.phase = None
                    slot.state = "idle"
                    slot.consecutive_failures += 1
                    self._write_pidfile()
                self._slot_sleep(slot, 5.0)

    def _slot_sleep(self, slot, wait):
        for _ in range(int(wait * 4)):   # never sleep while holding the lock
            if self.stopping or slot.retire or self.draining:
                break
            time.sleep(0.25)

    # ------------------------------------------------------------ window scheduler (7.2, 7.3)
    def run_window(self, slot, job):
        """Work job J and its local subtree for one window, absorb trivial children, then queue
        ONE tree report. Returns 'ok' (a report was queued), 'fail' (J's run was void and a
        failure was reported), 'none' (nothing for J; the job is left untouched) or 'abandoned'."""
        start = time.time()
        window_s, absorb_total, mode = self.window_for(job)     # M108: chosen now, not at lease time
        min_w = self.min_window_s()
        deadline = start + window_s
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
        self.log("worker %d: job %d exit %d window %s s%s seed %s"
                 % (slot.idx, job["id"], job["exit"], fmt_s(window_s),
                    " (%s)" % mode if mode not in (None, "full") else "",
                    job["seed"] or "(exit root)"))
        first = True
        while stack and not self.stopping:
            now = time.time()
            if now >= deadline and not first:
                break
            seed = stack.pop()
            res = self.run_worker(slot, job, seed, max(min_w, deadline - now))
            node = index[seed]
            if res.kind != "ok":
                if slot.killed_by_client or slot.drop_requested:
                    break
                if first:
                    if res.kind == "void" and res.failure:
                        self.queue_failure(job, res.reason, res.detail, res.hash)   # before the slot lets go
                    with self.lock:
                        slot.job = None
                        slot.phase = None
                    return "fail" if res.kind == "void" else "none"   # J produced nothing: job untouched
                node["status"] = "open"    # a child's run is void: the child stays open in the tree
                first = False
                continue
            first = False
            run_hash = run_hash or res.hash
            self._account(res.summary)
            if res.summary["status"] == "exhausted":
                node["status"] = "done"
                node["summary"] = res.summary
                if res.unresolved:
                    node["unresolved"] = res.unresolved
                if res.level:
                    node["level"] = res.level
                    self._note_level(job, res.level)
                with self.lock:
                    slot.nodes_done += 1
            else:
                new = [r for r in res.remaining if r not in index]
                if len(new) != len(res.remaining):
                    self.log("worker %d: seed %s listed %d REMAINING seed(s) already present in the tree; "
                             "their subtrees are covered there" % (slot.idx, seed, len(res.remaining) - len(new)))
                if not new:
                    node["status"] = "open"     # a split must have children: hand the node back open
                    continue
                node["status"] = "split"
                node["summary"] = res.summary
                if res.unresolved:
                    node["unresolved"] = res.unresolved
                if res.level:                 # the best level of the explored part belongs to this node, not to its children
                    node["level"] = res.level
                    self._note_level(job, res.level)
                for r in new:
                    child = {"seed": r, "parent": seed, "status": "open"}
                    nodes.append(child)
                    index[r] = child
                # REMAINING is cursor first: push in reverse so the cursor is the next pop
                stack.extend(reversed(new))
        if slot.killed_by_client or slot.drop_requested:
            with self.lock:
                slot.job = None
                slot.phase = None
            self.log("worker %d: job %d abandoned as instructed (no report)" % (slot.idx, job["id"]))
            return "abandoned"
        if root["status"] == "open":
            # J's run was interrupted by a stop before it produced anything usable
            with self.lock:
                slot.job = None
                slot.phase = None
            return "none"
        # absorption pass (DESIGN 7.3, review M18): probe the still-open local seeds briefly, deepest first
        # (the cheap ones: REMAINING lists the cursor and the top of the stack first). A probe that
        # finishes makes its node done. A probe that splits is kept as a split node with its summary and
        # its REMAINING children, so none of its probe time is lost; the pass then goes on with that
        # probe's own children only (deeper, the ones near its cursor are trivial): every seed still to
        # come is no deeper than the one that split, i.e. larger. A probe that cannot even expand its
        # root in the probe time ends the pass the same way. The budget is absorb_total_s.
        todo = absorb_order([n["seed"] for n in nodes if n["status"] == "open"])
        absorbed = probe_splits = 0
        if todo and not self.stopping and absorb_total > 0:
            probe = max(min(0.5, min_w), self.fparam("absorb_probe_s", 10))
            absorb_end = time.time() + absorb_total
            with self.lock:
                slot.phase = "absorb"
                slot.window_end = absorb_end
                slot.stack = todo
            while todo:
                left = absorb_end - time.time()
                if self.stopping or left < min(0.5, min_w) or slot.drop_requested:
                    break
                seed = todo.pop(0)
                # spend up to absorb_total_s (PROTOCOL3 section 3): the last probe gets what is left
                res = self.run_worker(slot, job, seed, min(probe, left), probe_run=True)
                node = index[seed]
                if res.kind != "ok":
                    if slot.killed_by_client:
                        break
                    if res.reason == "not_expanded":
                        break               # its root alone outlasts a probe: the rest is no deeper
                    continue                # a void probe: the node stays open (searched again anyway)
                run_hash = run_hash or res.hash
                self._account(res.summary)
                if res.level:               # a probe's level is real whatever the probe's outcome
                    node["level"] = res.level
                    self._note_level(job, res.level)
                if res.summary["status"] == "exhausted":
                    node["status"] = "done"
                    node["summary"] = res.summary
                    if res.unresolved:
                        node["unresolved"] = res.unresolved
                    absorbed += 1
                    with self.lock:
                        slot.nodes_done += 1
                    continue
                new = [r for r in res.remaining if r not in index]
                if not new:
                    break                   # no usable split: the node stays open (with its level)
                node["status"] = "split"
                node["summary"] = res.summary
                if res.unresolved:
                    node["unresolved"] = res.unresolved
                for r in new:
                    child = {"seed": r, "parent": seed, "status": "open"}
                    nodes.append(child)
                    index[r] = child
                probe_splits += 1
                todo = absorb_order(new)    # only the split probe's own (deeper) children from here on
                with self.lock:
                    slot.stack = todo
            if absorbed or probe_splits:
                self.log("worker %d: job %d absorption: %d node(s) finished, %d probe(s) split and kept"
                         % (slot.idx, job["id"], absorbed, probe_splits))
            if slot.killed_by_client or slot.drop_requested:
                with self.lock:
                    slot.job = None
                    slot.phase = None
                self.log("worker %d: job %d abandoned as instructed (no report)" % (slot.idx, job["id"]))
                return "abandoned"
        # fold-back (never truncate): demote the deepest split nodes until the report fits
        max_nodes, max_bytes = self.limits()
        fitted = fit_tree(nodes, max_nodes, max_bytes - 400)
        if fitted is None:
            # PROTOCOL3 section 3: return the job untouched and log loudly. It goes back as a
            # failure report ('too_big'): the server reopens it and counts it, so a job that can
            # never fit one report is quarantined and listed in the audit instead of cycling unseen.
            n_kids = sum(1 for n in nodes if n["parent"] == job["seed"])
            detail = ("its own REMAINING has %d children (%d nodes in all); one report takes %d nodes / %d bytes"
                      % (n_kids, len(nodes), max_nodes, max_bytes))
            self.log("!!! worker %d: job %d: %s. Nothing of this window is reported; the job is handed back untouched"
                     % (slot.idx, job["id"], detail))
            with self.lock:
                self.stats["released_too_big"] += 1
            self.queue_failure(job, "too_big", detail, run_hash)      # before the slot lets go (holding)
            with self.lock:
                slot.job = None
                slot.phase = None
            return "fail"
        if len(fitted) != len(nodes):
            with self.lock:
                self.stats["folded_back"] += 1
            self.log("worker %d: job %d tree folded back from %d to %d nodes to fit one report"
                     % (slot.idx, job["id"], len(nodes), len(fitted)))
        report = {"job_id": job["id"], "src_hash": run_hash or self.src_hash,
                  "client_version": CLIENT_VERSION, "protocol": PROTOCOL,
                  "nodes": [clean_node(n) for n in fitted]}
        counts = collections.Counter(n["status"] for n in fitted)
        n_unres = sum(len(n.get("unresolved") or []) for n in fitted if n["status"] != "open")
        dur = time.time() - start
        with self.lock:
            self.job_durations.append(dur)
            d = job.get("depth", seed_depth(job["seed"]))
            if d not in self.depth_hist and len(self.depth_hist) >= 512:
                self.depth_hist.pop(next(iter(self.depth_hist)))
            self.depth_hist.setdefault(d, collections.deque(maxlen=DEPTH_HIST_N)).append(dur)
            self.job_total_s += dur
            self.longest_job_s = max(self.longest_job_s, dur)
            self.stats["jobs_done"] += 1
            self.stats["nodes_done"] += counts.get("done", 0)
            self.stats["nodes_open"] += counts.get("open", 0)
            self.stats["unresolved"] += n_unres
            self.stats[root["status"] if root["status"] in ("done", "split") else "open_root"] += 1
            self.pending.append(report)     # queued before the slot lets go: the job never leaves `holding`
            slot.job = None
            slot.phase = None
        self.log("worker %d: job %d %s in %.0f s: %d node(s), %d done, %d split, %d open%s"
                 % (slot.idx, job["id"], root["status"], dur, len(fitted), counts.get("done", 0),
                    counts.get("split", 0), counts.get("open", 0),
                    ", %d unresolved candidate(s)" % n_unres if n_unres else ""))
        if self.draining or self.stopping:
            self.send_event.set()
        return "ok"

    def _account(self, summary):
        """CPU for the panel and the closing summary: the worker's own cpu_s (protocol 3),
        wall time for older workers."""
        el = summary.get("cpu_s")
        if not isinstance(el, (int, float)):
            el = summary.get("elapsed")
        try:
            el = float(el or 0)
        except (TypeError, ValueError):
            el = 0.0
        with self.lock:
            self.stats["cpu_seconds"] += max(0.0, el)
            self.stats["runs"] += 1

    def _note_level(self, job, level):
        """A run's LEVEL line. M66: it also feeds the last-hour and last-second boards (a run shorter
        than one STATUS interval, or a level found after the last STATUS, never reaches them otherwise)."""
        now = time.time()
        with self.lock:
            ex = str(job["exit"])
            cur = self.best_per_exit.get(ex)
            if cur is None or level["depth"] > cur["depth"]:
                self.best_per_exit[ex] = {"depth": level["depth"], "code": level["code"], "job_id": job["id"],
                                          "exit": job["exit"], "at": now}
            self._hour_best_add(now, level["depth"], level["code"], job["exit"])
            self.sec_levels.append((now, level["depth"], level["code"], job["exit"]))
            while self.sec_levels and (now - self.sec_levels[0][0] > SEC_BOARD_S or len(self.sec_levels) > 256):
                self.sec_levels.popleft()

    def queue_failure(self, job, reason, detail, run_hash):
        """PROTOCOL3 section 3 failure report {"job": id, "failed": reason}: the server counts
        failures per job and quarantines poison jobs (M20/N15)."""
        el = {"job": job["id"], "job_id": job["id"], "failed": str(reason)[:40], "src_hash": run_hash or self.src_hash,
              "detail": str(detail or "")[:300], "client_version": CLIENT_VERSION, "protocol": PROTOCOL}
        with self.lock:
            self.pending.append(el)
            self.stats["failures_reported"] += 1
        self.log("!!! job %d (seed %s): run failed (%s%s); reporting the failure to the server"
                 % (job["id"], job["seed"] or "(exit root)", reason, ": " + detail[:120] if detail else ""))

    def flags_mismatch(self, flags, job):
        """The worker's effective search definition (SUMMARY flags) must be the campaign's."""
        if not isinstance(flags, dict):
            return None
        want = {"grid": str(self.param("grid", "")), "exit": job["exit"], "transit": 1, "block_on_exit": 0, "bulk_walk": 1,
                "min_walls": 0}
        for flag, key in (("--num-holes", "max_holes"), ("--num-blocks", "max_blocks"), ("--min-walls", "min_walls")):
            v = extra_value(self.extra, flag)
            if v is not None:
                want[key] = v
        bad = []
        for k, v in want.items():
            if k in flags and flags[k] != v and str(flags[k]) != str(v):
                bad.append("%s=%s (campaign %s)" % (k, flags[k], v))
        return ", ".join(bad) if bad else None

    def run_worker(self, slot, job, seed, split_after, probe_run=False):
        """Run one worker process on `seed` and judge its output (PROTOCOL3 sections 2.2/2.3/3)."""
        if slot.drop_requested:
            return RunResult("none", "dropped")
        argv = self.worker_argv(job, seed, split_after)
        capture = not self.args.verbose_workers
        now = time.time()
        with self.lock:
            slot.cur_seed = seed
            slot.cur = slot.best = None
            slot.win_hist.clear()
            slot.run_started_at = now
            slot.last_line_at = now
            slot.split_at = now + split_after
            slot.split_after_s = split_after
            slot.stop_sent_at = 0.0
            slot.saw_output = False
            slot.stop_pending = False
            slot.watchdog_fired = False
            slot.watchdog_why = ""
            slot.grace_killed = False
            slot.frozen = False
            slot.stderr_tail.clear()
            slot.state = "running"
        pg = {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}
        try:
            # Own process group, same session (M14): Ctrl-C in the terminal reaches only the
            # client, and when the client dies for any reason a stopped (paused) worker's group
            # is orphaned, so the kernel sends it SIGHUP + SIGCONT and it exits.
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE if capture else None,
                                    stdin=subprocess.DEVNULL, cwd=BACKSEARCH_DIR, text=True, encoding="utf-8",
                                    errors="replace", bufsize=1, env=self.env, **pg)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            self.log("!!! cannot start worker %d for job %d: %s" % (slot.idx, job["id"], e))
            with self.lock:
                slot.state = "idle"
            return RunResult("void", "start_failed", str(e)[:200], failure=False)
        with self.lock:
            slot.proc = proc
            self._write_pidfile()
            if self.paused:
                self._signal(slot, signal.SIGSTOP)
                slot.frozen = True
                slot.state = "paused"
            elif self.stopping:
                slot.stop_pending = True   # SIGINT once the worker has printed its first line
        err_thread = None
        if capture:
            err_thread = threading.Thread(target=_drain_stderr, args=(proc.stderr, slot.stderr_tail), daemon=True)
            err_thread.start()

        run_hash = None
        hash_mismatch = False
        remaining, seen = [], set()
        unresolved = []
        level = None
        summary = None
        summary_bad = False
        bad = []              # reasons that void the run
        self_remaining = False
        dup_remaining = 0
        seed_toks = seed_tokens(seed)
        try:
            for line in proc.stdout:
                if not slot.saw_output:
                    with self.lock:
                        slot.saw_output = True
                        if slot.stop_pending:
                            self._send_stop(slot)
                slot.last_line_at = time.time()
                if len(line) > MAX_LINE:
                    tag = line[:16].split("\t", 1)[0]
                    if tag in ("REMAINING", "UNRESOLVED", "LEVEL", "SUMMARY"):
                        bad.append("overlong %s line" % tag)
                    continue
                parts = line.rstrip("\r\n").split("\t")
                tag = parts[0]
                if tag == "STATUS" and len(parts) >= 2:
                    self._on_status(slot, parts[1])
                elif tag == "SRC_HASH" and len(parts) >= 2:
                    h = parts[1].strip().lower()
                    if HEX_RE.match(h):
                        run_hash = h
                        if h != self.src_hash:
                            hash_mismatch = True
                elif tag == "REMAINING":
                    p = parts[1].strip() if len(parts) >= 2 else ""
                    if p == seed:
                        self_remaining = True   # the cursor is the root itself: nothing was expanded
                        continue
                    toks = extends(p, seed_toks, self.path_tok_max) if p else None
                    if toks is None or len(toks) <= len(seed_toks):
                        bad.append("REMAINING %r is malformed, overlong or does not extend the seed" % p[:60])
                    elif p in seen:
                        dup_remaining += 1      # the same subtree twice: harmless, listed once
                    else:
                        seen.add(p)
                        remaining.append(p)
                elif tag == "UNRESOLVED":
                    cand = parse_unresolved(parts, seed, self.path_tok_max)
                    if cand is None or len(unresolved) >= UNRESOLVED_MAX:
                        bad.append("bad UNRESOLVED line %r" % line[:60] if cand is None else "more than %d UNRESOLVED lines" % UNRESOLVED_MAX)
                    else:
                        unresolved.append(cand)
                elif tag == "LEVEL":
                    d = None
                    if len(parts) >= 3:
                        try:
                            d = int(parts[1])
                        except ValueError:
                            d = None
                    code = parts[2].strip() if len(parts) >= 3 else ""
                    if d is not None and d >= 0 and 0 < len(code) <= 2000:
                        level = {"depth": d, "code": code}
                    else:
                        bad.append("bad LEVEL line %r" % line[:60])
                elif tag == "SUMMARY" and len(parts) >= 2:
                    try:
                        s = json.loads(parts[1])
                    except ValueError:
                        s = None
                    if isinstance(s, dict) and isinstance(s.get("status"), str):
                        summary = s
                    else:
                        summary_bad = True
        except (OSError, ValueError) as e:
            self.log("worker %d: error reading output: %s" % (slot.idx, e))
        rc = proc.wait()
        if err_thread is not None:
            err_thread.join(2.0)
        with self.lock:
            slot.proc = None
            slot.frozen = False
            slot.state = "idle"
            self._write_pidfile()
            # 'client': killed on purpose (drop, lease lost), nothing of this window is reported;
            # 'grace': Stop's grace ran out, only this run is lost (its node stays open)
            killed = "client" if slot.killed_by_client else "grace" if slot.grace_killed else None
            watchdog = slot.watchdog_fired
            stopped = slot.stop_sent_at > 0 and not watchdog   # we sent SIGINT (Stop)
            tail = " | ".join(list(slot.stderr_tail)[-3:])
        return self._judge(slot, job, seed, probe_run, rc, killed, watchdog, stopped, tail, run_hash, hash_mismatch,
                           summary, summary_bad, bad, self_remaining, dup_remaining, remaining, unresolved, level)

    def _judge(self, slot, job, seed, probe_run, rc, killed, watchdog, stopped, tail, run_hash, hash_mismatch,
               summary, summary_bad, bad, self_remaining, dup_remaining, remaining, unresolved, level):
        where = "worker %d (job %d, seed %s)" % (slot.idx, job["id"], seed or "(exit root)")

        def void(reason, detail, failure=True):
            with self.lock:
                self.stats["runs_void"] += 1
            self.log("%s%s: run void (%s): %s" % ("" if probe_run else "!!! ", where, reason, detail[:200]))
            return RunResult("void", reason, "rc %s; %s%s" % (rc, detail, "; stderr: " + tail if tail else ""),
                             failure=failure, run_hash=run_hash)

        if killed:
            return RunResult("none", "killed" if killed == "client" else "stop_grace")
        if watchdog:
            with self.lock:
                self.stats["watchdog"] += 1
                why = slot.watchdog_why or "no output after the split time"
            return void("hang", why)
        if hash_mismatch:
            self.fatal_stop(2, "The worker binary changed while the client was running (SRC_HASH %s, expected %s). "
                            "Restart the client." % (run_hash, self.src_hash))
            return RunResult("none", "hash_changed")
        if summary is None:
            with self.lock:
                self.stats["runs_without_summary"] += 1
            if stopped:
                return RunResult("none", "stopped")    # our SIGINT (Stop) ended it before it answered: not the job's fault
            reason = "crash" if rc != 0 else "no_summary"
            return void(reason, "exited with code %s WITHOUT a %sSUMMARY line" % (rc, "valid " if summary_bad else ""))
        status = summary.get("status")
        if status not in ("exhausted", "split"):
            reason = status if STATUS_NAME_RE.match(status) else "bad_status"
            return void(reason, "worker status %s" % status[:40])
        if rc != 0:
            if stopped and rc < 0:
                return RunResult("none", "stopped")
            return void("crash", "status %s but exit code %s" % (status, rc))
        if run_hash is None:
            return void("bad_output", "no SRC_HASH line")
        if bad:
            reason = ("bad_remaining" if any("REMAINING" in b for b in bad) else
                      "bad_unresolved" if any("UNRESOLVED" in b for b in bad) else "bad_output")
            return void(reason, "%d bad line(s): %s" % (len(bad), "; ".join(bad[:3]))[:250])
        # What the server requires of a protocol-3 SUMMARY (it refuses the whole report otherwise, and
        # a refused report comes back every time the job is run): the search definition, and the
        # number of candidates, which must be exactly the UNRESOLVED lines forwarded for this node.
        if self.worker_info.get("PROTOCOL", 0) >= PROTOCOL:
            if not isinstance(summary.get("flags"), dict) or not isinstance(summary.get("unresolved"), int):
                return void("bad_summary", "a protocol-%d SUMMARY without flags or unresolved" % PROTOCOL)
        n_sum = summary.get("unresolved")
        if isinstance(n_sum, int) and n_sum != len(unresolved):
            return void("bad_unresolved", "SUMMARY counts %d unresolved candidate(s) but %d UNRESOLVED line(s) arrived"
                        % (n_sum, len(unresolved)))
        mism = self.flags_mismatch(summary.get("flags"), job)
        if mism:
            self.fatal_stop(2, "The worker's effective search definition differs from the campaign's: %s. "
                            "Rebuild the worker (%s) and restart." % (mism, self.rebuild_cmd()))
            return RunResult("none", "bad_flags")
        if status == "exhausted":
            if remaining or self_remaining:
                return void("bad_remaining", "REMAINING lines in an exhausted run")
            states = summary.get("states")
            if not isinstance(states, int) or states < 1:
                return void("bad_summary", "an exhausted run that checked %r states" % states)
        else:
            if self_remaining:
                # Interrupted before expanding its root: the whole subtree is still pending and a
                # tree listing the seed itself as a child is invalid. No result; a failure only when
                # the worker split on its own timer without progress (a window too short for its root).
                if stopped or probe_run:
                    if not probe_run:
                        self.log("%s was interrupted before expanding its root; not reported (node stays untouched)" % where)
                    return RunResult("none", "not_expanded")
                return void("no_progress", "split before expanding its root (REMAINING = the seed)")
            if not remaining:
                return void("bad_remaining", "split without any REMAINING line")
        if dup_remaining:
            self.log("%s: %d duplicate REMAINING line(s) listed once" % (where, dup_remaining))
        clean = {k: v for k, v in summary.items() if k != "src_hash"}   # the report carries the hash once
        return RunResult("ok", summary=clean, remaining=remaining, level=level, unresolved=unresolved, run_hash=run_hash)

    def _hour_best_add(self, now, depth, code, exit_):
        """Rolling one-hour maximum (caller holds the lock): drop older entries that the
        new one dominates, so the leftmost entry is always the current hour's best."""
        hb = self.hour_best
        while hb and hb[-1][1] <= depth:
            hb.pop()
        hb.append((now, depth, code, exit_))
        while hb and now - hb[0][0] > 3600:
            hb.popleft()

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
                self._hour_best_add(now, win["depth"], win["code"], slot.job["exit"] if slot.job else None)
            elif "best" in s and slot.best is None:
                try:
                    slot.best = {"depth": int(s["best"]), "code": None}
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------ reports: batches, outbox (7.6, M37, M52)
    def queue_report(self, report):
        with self.lock:
            self.pending.append(report)

    def sender_loop(self):
        next_flush = time.time() + OUTBOX_FLUSH_S
        was_connected = True
        while True:
            self.send_event.wait(0.5)
            self.send_event.clear()
            try:
                with self.lock:
                    winding = self.stopping or self.draining
                    due = self.pending and (winding or time.time() - self.last_send_at >= self.fparam("batch_interval_s", 10))
                    if winding and not self.pending and self.sender_may_exit:
                        return
                if self.no_send:
                    if self.sender_may_exit:
                        return
                    continue                  # nothing can be delivered under this token any more
                if due:
                    self.send_pending()
                now = time.time()
                reconnected = not was_connected and self.api.connected
                was_connected = self.api.connected
                if reconnected or now >= next_flush:
                    # M21: once the server answers again, retry the whole outbox at once (ignore backoff)
                    next_flush = now + OUTBOX_FLUSH_S
                    self.flush_outbox(force=reconnected)
            except Exception:
                self.log("!!! sender: internal error (continuing): %s" % traceback.format_exc().strip().splitlines()[-1])
                time.sleep(1)

    def chunk_reports(self, reports):
        """Batches of at most BATCH_BYTES_MAX serialized bytes and BATCH_COUNT_MAX reports;
        a report larger than the byte cap on its own travels alone."""
        chunks, cur, cur_bytes = [], [], 0
        for r in reports:
            n = _jlen(r) + 1
            if cur and (cur_bytes + n > BATCH_BYTES_MAX or len(cur) >= BATCH_COUNT_MAX):
                chunks.append(cur)
                cur, cur_bytes = [], 0
            cur.append(r)
            cur_bytes += n
        if cur:
            chunks.append(cur)
        return chunks

    def send_pending(self):
        """Write-ahead (M52): a batch goes to an outbox file before it is POSTed, and the file
        is deleted only once the server has answered for every report in it."""
        with self.lock:
            batch = self.pending
            self.pending = []
            self.last_send_at = time.time()
        if not batch:
            return True
        chunks = self.chunk_reports(batch)
        for i, chunk in enumerate(chunks):
            path = self.outbox_write(chunk, attempts=0, next_try=0.0)
            r = self.deliver(chunk, path)      # path None (disk full): straight from memory, and
            if r == "done":                    # deliver() puts back in `pending` what it could not send
                continue
            with self.lock:
                for later in chunks[i + 1:]:
                    if self.outbox_write(later, attempts=0, next_try=time.time() + 2.0) is None:
                        self.pending.extend(later)
            return False
        return True

    def post_batch(self, reports):
        """POST one batch. Returns (kind, info): 'ok' | 'retry' | 'too_large' | 'closed' |
        'reregister' | 'revoked' | 'too_old' | 'refused'."""
        body = {"token": self.token, "reports": reports}
        try:
            res = self.api.post("/api/v2/reports", body, timeout=60)
        except ApiError as e:
            kind = classify(e)
            if kind == "transient":
                return "retry", e
            if kind == "no_campaign":
                kind = "closed"
            return kind, e
        outcomes = None
        if isinstance(res, dict):
            for key in ("results", "outcomes", "reports"):
                if isinstance(res.get(key), list):
                    outcomes = res[key]
                    break
        self.on_campaign_state(res)
        ok = dup = bad = fails = 0
        for i, r in enumerate(reports):
            o = outcomes[i] if outcomes is not None and i < len(outcomes) else {}
            if not isinstance(o, dict):
                o = {}
            code = o.get("code") if isinstance(o.get("code"), int) else o.get("status") if isinstance(o.get("status"), int) else None
            if o.get("dup"):
                dup += 1
            elif o.get("error") or (code is not None and code >= 400) or o.get("ok") is False:
                bad += 1
                self.reject(r, code or 400, o)
                if o.get("error") == "unknown_hash":
                    self.fatal_stop(2, "The server no longer accepts this worker's hash. Update and rebuild:\n    %s"
                                    % self.update_advice())
            else:
                ok += 1
                if "failed" in r:
                    fails += 1
        with self.lock:
            self.stats["reports_sent"] += ok + dup
            self.stats["batches_sent"] += 1
        self.log("batch delivered: %d report(s) ok%s%s%s" % (ok, " (%d of them failure reports)" % fails if fails else "",
                                                            ", %d dup" % dup if dup else "",
                                                            ", %d REJECTED" % bad if bad else ""))
        return "ok", None

    def deliver(self, reports, path):
        """Deliver one batch (from its outbox file when `path` is set). Returns 'done' when
        the server answered for every report (the file is gone), 'retry' (kept for later) or
        'halt' (the campaign or the token ended; nothing more can be sent now). Without a file
        (the disk refused it), what is kept for later goes back to `pending`, exactly once."""
        cid = self.campaign.get("id")
        meta = self.outbox_meta.get(path) if path else None
        if meta and cid is not None and meta.get("campaign_id") not in (None, cid):
            self.retire_file(path, "it belongs to campaign %s" % meta.get("campaign_id"))
            return "done"
        if not self.token:
            if not path:
                with self.lock:
                    self.pending[0:0] = reports
            return "retry"
        kind, info = self.post_batch(reports)
        if kind == "ok":
            self.outbox_remove(path)
            return "done"
        if kind == "too_large":                       # M37: halve on 413, down to one report
            with self.lock:
                self.stats["batches_413"] += 1
            if len(reports) == 1:
                self.reject(reports[0], 413, info.body if info else {})
                self.outbox_remove(path)
                return "done"
            half = len(reports) // 2
            parts = [reports[:half], reports[half:]]
            self.log("the server refused a batch of %d reports as too large (413); sending it in halves" % len(reports))
            paths = [self.outbox_write(p, attempts=0, next_try=0.0) if path else None for p in parts]
            self.outbox_remove(path)
            result = "done"
            for p, pp in zip(parts, paths):
                if result == "done":
                    result = self.deliver(p, pp)      # a part without a file requeues itself on retry
                elif pp is None:
                    with self.lock:
                        self.pending.extend(p)        # not tried: kept in memory (a part with a file stays on disk)
            return result
        if kind == "retry":
            if path:
                attempts = (meta or {}).get("attempts", 0) + 1
                self.outbox_write(reports, attempts=attempts, next_try=time.time() + min(300.0, 2.0 ** attempts),
                                  path=path)
                self.log("batch of %d report(s) kept in the outbox (%s): %s" % (len(reports), os.path.basename(path), info))
            else:
                with self.lock:
                    self.pending[0:0] = reports    # keep it in memory for the next try
            return "retry"
        if kind == "closed":
            if path:
                self.retire_file(path, "the campaign was closed")
            else:
                self.retire_reports(reports, "the campaign was closed")
            self.campaign_gone(info)
            return "halt"
        if kind == "reregister":
            if not path:
                self.keep_on_disk(reports)
            self.end_campaign("reregister", "the server does not know this client's token any more (%s); "
                              "registering again" % info)
            return "halt"
        if kind in ("revoked", "too_old"):
            if not path:
                self.keep_on_disk(reports)
            if kind == "revoked":
                self.fatal_stop(2, "This client's token was revoked by the server.", no_send=True)
            else:
                self.fatal_stop(2, "This client (version %s) is too old for the campaign.\n%s"
                                % (CLIENT_VERSION, server_message(info, "Update: " + self.update_advice())), no_send=True)
            return "halt"
        # refused as a whole (400 and similar): every report is filed for the owner
        for r in reports:
            self.reject(r, info.status if info else 400, info.body if info else {})
        self.outbox_remove(path)
        return "done"

    def keep_on_disk(self, reports):
        """Reports that cannot be sent under this token: to the outbox, else back to memory."""
        if self.outbox_write(reports, attempts=1, next_try=time.time() + 5) is None:
            with self.lock:
                self.pending[0:0] = reports

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

    def outbox_write(self, reports, attempts, next_try=None, path=None):
        """Atomic write of one batch file; returns its path or None when the disk refuses."""
        if next_try is None:
            next_try = time.time() + min(300.0, 2.0 ** attempts)
        prev = self.outbox_meta.get(path) if path else None
        entry = {"reports": reports, "attempts": attempts, "next_try": next_try,
                 "created_at": prev["created_at"] if prev else time.time(),
                 "campaign_id": prev.get("campaign_id") if prev else self.campaign.get("id"),
                 "client_version": CLIENT_VERSION}
        if path is None:
            path = os.path.join(self.outbox, "%d-batch%d-%s.json" % (int(time.time() * 1000), len(reports), uuid.uuid4().hex[:8]))
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(entry, f, separators=(",", ":"))
            os.replace(tmp, path)
        except OSError as e:
            self.log("!!! cannot write the outbox file %s: %s" % (path, e))
            try:
                os.remove(tmp)
            except OSError:
                pass
            return None
        with self.lock:
            self.outbox_meta[path] = {"jobs": [r.get("job_id") for r in reports if isinstance(r.get("job_id"), int)],
                                      "created_at": entry["created_at"], "campaign_id": entry["campaign_id"],
                                      "attempts": attempts, "next_try": next_try}
        return path

    def outbox_remove(self, path):
        if not path:
            return
        try:
            os.remove(path)
        except OSError:
            pass
        with self.lock:
            self.outbox_meta.pop(path, None)

    def retire_file(self, path, why):
        """Reports that can never be delivered (their campaign ended) are kept in closed/."""
        dst = os.path.join(self.outbox, "closed", os.path.basename(path))
        try:
            os.replace(path, dst)
        except OSError:
            pass
        with self.lock:
            self.outbox_meta.pop(path, None)
        self.log("outbox file %s moved to closed/ (%s)" % (os.path.basename(path), why))

    def retire_reports(self, reports, why):
        path = os.path.join(self.outbox, "closed", "%d-batch%d-%s.json" % (int(time.time() * 1000), len(reports), uuid.uuid4().hex[:8]))
        try:
            with open(path, "w") as f:
                json.dump({"reports": reports, "why": why}, f)
        except OSError:
            pass

    def retire_outbox(self, why):
        self.scan_outbox()
        for p in list(self.outbox_meta):
            self.retire_file(p, why)
        with self.lock:
            left, self.pending = self.pending, []
        if left:
            self.retire_reports(left, why)

    def outbox_files(self):
        try:
            names = sorted(n for n in os.listdir(self.outbox) if n.endswith(".json") and not n.startswith("."))
        except OSError:
            return []
        return [os.path.join(self.outbox, n) for n in names]

    def scan_outbox(self):
        """Index the outbox files (job ids for `holding`, campaign, attempts) at startup."""
        meta = {}
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
            meta[path] = {"jobs": [r.get("job_id") for r in reports if isinstance(r, dict) and isinstance(r.get("job_id"), int)],
                          "created_at": entry.get("created_at") or os.path.getmtime(path),
                          "campaign_id": entry.get("campaign_id"), "attempts": entry.get("attempts", 0),
                          "next_try": entry.get("next_try", 0)}
        with self.lock:
            self.outbox_meta = meta

    def flush_outbox(self, force=False):
        """Retry outbox files (oldest first). Only one thread flushes at a time: the sender
        while running, the main thread before it starts and after it has exited."""
        now = time.time()
        for path in self.outbox_files():
            if self.no_send:
                return
            meta = self.outbox_meta.get(path)
            if meta is None:
                continue                      # written by nobody we know (scan_outbox indexes at start)
            if not force and meta.get("next_try", 0) > now:
                continue
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
                with self.lock:
                    self.outbox_meta.pop(path, None)
                continue
            if self.deliver(reports, path) != "done":
                return  # server unreachable (or the campaign ended): try the rest later

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
        if slot.frozen:                      # a paused worker must run to print anything
            self._signal(slot, signal.SIGCONT)
            slot.frozen = False
            if slot.state == "paused":
                slot.state = "running"
        if not slot.saw_output:
            slot.stop_pending = True
            return
        slot.stop_pending = False
        self._signal(slot, signal.SIGCONT)
        self._signal(slot, signal.SIGINT)
        slot.state = "finishing"
        slot.stop_sent_at = time.time()

    def set_exit(self, value):
        """Exit preference for the next lease request (None = any). Returns (ok, message)."""
        if value is None or value == "":
            ex = None
        else:
            try:
                ex = int(value)
            except (TypeError, ValueError):
                return False, "exit must be a number or null"
            allowed = self.campaign_exits()
            if allowed and ex not in allowed:
                return False, "exit %d is not part of this campaign (%s)" % (ex, ", ".join(str(a) for a in allowed))
            if ex < 0:
                return False, "exit must be >= 0"
        with self.lock:
            self.exit_pref = ex
            self.exit_applied = ex is None
            self.exit_pending = ex is not None       # until the next lease response settles it (M58)
            self.exit_fallback = False
        self.log("exit preference: %s (applies to the next lease request)" % ("any" if ex is None else ex))
        return True, None

    def pause(self):
        """M56: every live worker except one handing back its work (after Stop or the watchdog) is
        frozen, retiring ones included; `frozen` records it (never the state string)."""
        with self.lock:
            self.resume_requested_at = 0.0           # M57: a click is answered either way
            if self.paused or self.stopping:
                self.pause_requested_at = 0.0
                return
            self.paused = True
            for s in self.slots:
                if s.proc and not s.frozen and s.state != "finishing":
                    self._signal(s, signal.SIGSTOP)
                    s.frozen = True
                    if s.state == "running":
                        s.state = "paused"
            self.pause_requested_at = 0.0
        self.log("paused (workers stopped with SIGSTOP; their memory stays allocated)")
        self.heartbeat_now = True

    def resume(self):
        with self.lock:
            self.pause_requested_at = 0.0
            if not self.paused:
                self.resume_requested_at = 0.0
                return
            self.paused = False
            now = time.time()
            for s in self.slots:
                if s.proc and s.frozen:
                    self._signal(s, signal.SIGCONT)
                    s.frozen = False
                    if s.state == "paused":
                        s.state = "running"
                    # the watchdog counts silence and overrun only while running: a worker whose split
                    # time passed during the pause splits at once, and gets the usual slack to do it
                    s.last_line_at = now
                    s.split_at = max(s.split_at, now)
            self.resume_requested_at = 0.0
        self.log("resumed")
        self.heartbeat_now = True

    def _on_sigtstp(self, *_):
        """Ctrl-Z (M61): the workers run in their own process group, so the terminal's SIGTSTP reaches
        only the client. Freeze them with it (a stopped client would otherwise leave them running
        unsupervised), stop, and thaw exactly those when the shell continues us (fg / SIGCONT)."""
        froze = []
        with self.lock:
            for s in self.slots:
                if s.proc and not s.frozen:
                    self._signal(s, signal.SIGSTOP)
                    s.frozen = True
                    froze.append((s, s.proc))
        try:
            sys.stderr.write("\n[volunteer] Ctrl-Z: client and workers stopped; `fg` continues them\n")
        except (OSError, ValueError):
            pass
        os.kill(os.getpid(), signal.SIGSTOP)
        with self.lock:                        # continued
            for s, p in froze:
                if s.proc is p and s.frozen:
                    self._signal(s, signal.SIGCONT)
                    s.frozen = False
        self.log("continued after Ctrl-Z (%d worker(s) resumed)" % len(froze))

    def request_stop(self, reason="stop requested", user=False):
        with self.lock:
            if user:
                self.user_stop = True
                if self.end_reason in (None, "complete") and not self.draining:
                    self.end_reason = "stop"
            if self.stopping:
                return
            self.stopping = True
            if self.end_reason is None:
                self.end_reason = "stop"
            self.stop_reason = reason
            self.paused = False
            for s in self.slots:
                if s.proc:
                    self._send_stop(s)
            queued = [j["id"] for j in self.queue]
            self.queue.clear()
            self.cond.notify_all()
            self.heartbeat_now = True       # release the never-started queue at once (M22)
        self.send_event.set()
        self.log("stopping: %s. Waiting for workers to print their SUMMARY..." % reason)
        if queued:
            self.log("leased jobs never started (released with the next heartbeat): %s" % queued[:50])

    def force_quit(self):
        self.log("second Ctrl-C: killing workers and exiting; running jobs are left to their leases")
        self.kill_all_workers()
        with self.lock:
            left, self.pending = self.pending, []
        if left:                               # M52: finished reports stay on disk for the next start
            for chunk in self.chunk_reports(left):
                self.outbox_write(chunk, attempts=0, next_try=0.0)
        log_flush(1.0)
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
            if n != self.workers:
                self.workers_prev = self.workers
                self.workers_changed_at = time.time()
            self.workers = n
            self._reconcile_slots()
        self.log("workers set to %d" % n)
        return n

    def _reconcile_slots(self):
        """Caller holds the lock. Also revives a slot whose thread died (M50 backstop)."""
        while len(self.slots) < self.workers:
            self.slots.append(Slot(len(self.slots)))
        for s in self.slots:
            s.retire = s.idx >= self.workers
            if not s.retire and (s.thread is None or not s.thread.is_alive()) and not self.stopping and not self.draining:
                if s.thread is not None and s.job is not None:
                    self.log("!!! worker %d's thread had died; its job %d is handed back" % (s.idx, s.job["id"]))
                s.job = None
                s.proc = None
                s.phase = None
                s.state = "idle"
                s.thread = threading.Thread(target=self.slot_loop, args=(s,), name="worker-%d" % s.idx, daemon=True)
                s.thread.start()
            # M56: retirement is `s.retire` (shown by the panel); the process state is never overwritten,
            # so Pause, Resume, the watchdog and the stop grace keep working on a retiring worker

    # ------------------------------------------------------------ heartbeat / wake
    def boards(self):
        out = []
        with self.lock:
            for s in self.slots:
                if s.job is None:
                    continue
                cur = s.cur or {}
                b = {"depth": cur.get("depth", 0), "cur": (cur.get("code") or "")[:BOARD_CAP],
                     "best": (s.best or {}).get("depth", 0)}
                out.append(b)
        return out[:self.workers]

    def heartbeat(self):
        with self.lock:
            running = [s.job["id"] for s in self.slots if s.job]
            holding = self.held_ids()
        # `holding` lets the server renew exactly what we hold (including finished jobs whose reports
        # are on their way), hand back a job whose lease lapsed while we slept if nobody took it, and
        # tell us to drop one that someone else now holds. `running` lets the server prefer our
        # still-queued jobs (no progress lost) when it has to hand work to an idle client.
        body = {"token": self.token, "workers": self.workers, "paused": self.paused, "boards": self.boards(),
                "holding": holding, "running": running}
        try:
            res = self.api.post("/api/v2/heartbeat", body, timeout=20)
        except ApiError as e:
            kind = classify(e)
            if kind == "too_old":
                self.fatal_stop(2, "This client (version %s) is too old for the campaign.\n%s"
                                % (CLIENT_VERSION, server_message(e, "Update: " + self.update_advice())), no_send=True)
            elif kind == "revoked":
                self.fatal_stop(2, "This client was revoked by the server (HTTP %d)." % e.status, no_send=True)
            elif kind == "closed":
                self.campaign_gone(e)
            elif kind == "reregister":
                self.end_campaign("reregister", "the server does not recognise this client any more (%s); "
                                  "registering again" % e)
            else:
                self.log("heartbeat failed: %s" % e)
            return None
        with self.lock:
            self.last_heartbeat_ok = time.time()
        if not isinstance(res, dict):
            return None
        if res.get("revoked"):
            self.fatal_stop(2, "This client was revoked by the server.", no_send=True)
            return res
        if not self.apply_campaign(res.get("campaign")):
            return res
        # the server's current window, when the heartbeat carries it (M108): windows starting now use it
        win = res.get("window")
        if isinstance(win, dict):
            self.note_window(win.get("split_after_s"), win.get("absorb_total_s"), win.get("mode") or win.get("window_mode"),
                             "heartbeat")
        elif res.get("split_after_s_now") is not None:
            self.note_window(res.get("split_after_s_now"), res.get("absorb_total_s_now"), res.get("window_mode_now"), "heartbeat")
        hashes = [str(h).lower() for h in (self.campaign.get("hashes") or [])]
        if hashes and self.src_hash not in hashes and not self.stopping:
            self.fatal_stop(2, "The campaign's accepted worker hashes changed.\n" + self.hash_refusal(hashes))
        with self.lock:
            if isinstance(res.get("me"), dict):
                self.me = res["me"]
            ex = res.get("exits")
            if isinstance(ex, dict):
                self.exits = ex
            elif isinstance(ex, list):   # older servers sent a list
                self.exits = {str(e.get("exit")): e for e in ex if isinstance(e, dict)}
        self.on_campaign_state(res)
        leases = res.get("leases")
        if isinstance(leases, list):
            with self.lock:
                self.leases_from_server = set(int(x) for x in leases if isinstance(x, (int, str)) and str(x).isdigit())
                running = [s.job["id"] for s in self.slots if s.job]
                lost = [j for j in running if j not in self.leases_from_server]
            if lost and not self.draining:
                self.log("warning: the server no longer lists these running jobs as leased to us: %s" % lost)
        drop = res.get("drop")
        if isinstance(drop, list) and drop:
            drop = set(int(x) for x in drop if isinstance(x, (int, str)) and str(x).isdigit())
            with self.lock:
                for s in self.slots:
                    if s.job and s.job["id"] in drop:
                        s.drop_requested = True     # also a slot between two runs (absorption)
                        if s.proc:
                            self.log("job %d is held by someone else now (or finished); stopping worker %d without a report"
                                     % (s.job["id"], s.idx))
                            s.killed_by_client = True
                            self._signal(s, signal.SIGCONT)
                            self._signal(s, signal.SIGKILL)
                gone = [j["id"] for j in self.queue if j["id"] in drop]
                if gone:
                    self.log("dropping queued jobs now held by others: %s" % gone)
                    self.queue = collections.deque(j for j in self.queue if j["id"] not in drop)
        reclaimed = res.get("reclaimed")
        if isinstance(reclaimed, list) and reclaimed:
            self.log("server handed back %d job(s) whose lease had lapsed while we were silent" % len(reclaimed))
        return res

    def final_heartbeat(self):
        """PROTOCOL3 section 3: the last heartbeat releases everything at once."""
        if not self.token or self.end_reason in ("closed", "reregister"):
            return
        body = {"token": self.token, "workers": self.workers, "paused": False, "boards": [],
                "holding": [], "running": [], "stopping": True}
        try:
            res = self.api.post("/api/v2/heartbeat", body, timeout=10)
        except ApiError:
            return
        # the freshest numbers for the closing summary (they include the reports just delivered)
        if isinstance(res, dict):
            with self.lock:
                if isinstance(res.get("me"), dict):
                    self.me = res["me"]
                if isinstance(res.get("exits"), dict):
                    self.exits = res["exits"]
                r = res.get("result") or res.get("campaign_result")
                if isinstance(r, (dict, list)):
                    self.campaign_result = r

    def fetch_campaign_status(self):
        try:
            st = self.api.get("/api/v2/status")
        except ApiError as e:
            self.log("campaign status unavailable: %s" % e)
            return
        if isinstance(st, dict):
            with self.lock:
                self.campaign_status = st
                self.campaign_status_ok_at = time.time()     # M60: the age shown counts from here

    def pending_flags(self, now):
        """Human-readable 'working...' phases for controls that are not instantaneous (8.1)."""
        p = {"workers": None, "exit": None, "pause": None, "stop": None}
        retiring = [s for s in self.slots if s.retire and s.job is not None]
        fresh = [s for s in self.slots if not s.retire and s.job is None and s.thread and s.thread.is_alive()]
        if retiring:
            p["workers"] = "%d worker%s finishing %s current job before retiring…" % (
                len(retiring), "" if len(retiring) == 1 else "s", "its" if len(retiring) == 1 else "their")
        elif fresh and now - self.workers_changed_at < 60 and self.workers > self.workers_prev and not self.paused:
            p["workers"] = "%d new worker%s waiting for a job (next lease within %d s)…" % (
                len(fresh), "" if len(fresh) == 1 else "s", int(self.fparam("batch_interval_s", 10)))
        if self.exit_pref is not None and self.exit_pending and not self.stopping and not self.draining:
            p["exit"] = "current jobs finish first; new leases use exit %d" % self.exit_pref
        # M56/M57: workers still to freeze = live, not frozen, not handing back their work
        running = [s for s in self.slots if s.proc and not s.frozen and s.state != "finishing"]
        if self.pause_requested_at and (not self.paused or running) and not self.stopping:
            p["pause"] = "freezing %d worker%s…" % (len(running), "" if len(running) == 1 else "s")
        elif self.resume_requested_at and self.paused and not self.stopping:
            p["pause"] = "resuming…"
        if self.stopping or self.draining:
            alive = [s for s in self.slots if s.proc]
            if alive:
                p["stop"] = "waiting for %d worker%s to %s…" % (len(alive), "" if len(alive) == 1 else "s",
                                                                  "print their remaining work" if self.stopping else "finish")
            elif self.pending or self.outbox_meta:
                p["stop"] = "sending the last report…"
            else:
                p["stop"] = "exiting…"
        elif self.waiting_next:
            p["stop"] = "waiting for the next campaign…"
        return p

    def shift_baselines(self, lost, now):
        """The main loop did not run for `lost` seconds: a system sleep (the laptop lid), a frozen
        client (Ctrl-Z, a debugger) or a starved machine. The watchdog and the stop grace must count
        only time the client was awake to read the workers' output, so a sleep that spans a split
        time is not a hang: the silence baseline moves to now, and the split time, the overrun limit
        and the stop grace move by the lost time (a Linux worker's own split timer, CLOCK_MONOTONIC,
        also stops during suspend and splits that much later)."""
        n = 0
        with self.lock:
            for s in self.slots:
                if not s.proc:
                    continue
                n += 1
                s.last_line_at = max(s.last_line_at, now)
                s.split_at += lost
                s.run_started_at += lost
                if s.stop_sent_at:
                    s.stop_sent_at += lost
        if n and lost >= 30:
            self.log("the client did not run for %.0f s (sleep?); the watchdog and the stop grace of %d running "
                     "worker(s) count from now" % (lost, n))

    def handle_wake(self, gap):
        self.log("clock jumped %.0f s (sleep?). Re-validating leases before the workers continue." % gap)
        with self.lock:
            held = []
            for s in self.slots:
                if s.proc and not s.frozen and s.state == "running":
                    self._signal(s, signal.SIGSTOP)
                    s.frozen = True
                    held.append((s, s.proc))
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
            now = time.time()
            for s, p in held:
                s.last_line_at = now
                s.split_at = max(s.split_at, now)     # the sleep is not a hang (watchdog baseline)
                if s.proc is not p or not s.frozen:
                    continue                          # finished, killed or stopped meanwhile
                if self.paused:
                    s.state = "paused"                # paused meanwhile: stays frozen until Resume (M56)
                else:
                    self._signal(s, signal.SIGCONT)
                    s.frozen = False

    # ------------------------------------------------------------ closing summary (new feature)
    def closing_summary(self, session=True):
        camp = self.campaign or {}
        cid, title = camp.get("id"), camp.get("title")
        lines = ["Campaign %s%s is complete." % ("#%s " % cid if cid is not None else "",
                                                  "(%s)" % title if title else "")]
        me = self.me or {}
        if me:
            cpu_h = float(me.get("cpu_s") or 0) / 3600.0
            rank = " (rank %s of %s)" % (me["rank"], me.get("contributors", "?")) if me.get("rank") else ""
            lines.append("Your contribution as %s: %s job(s), %.2f CPU-hours%s." % (
                me.get("name") or self.name or "?", me.get("jobs_done", 0), cpu_h, rank))
            b = me.get("best")
            if isinstance(b, dict) and b.get("moves") is not None:
                lines.append("Your deepest level: %s moves (exit %s)." % (b.get("moves"), b.get("exit")))
        if session:
            lines.append("This client, this run: %d window(s), %d node(s) finished, %.2f CPU-hours."
                         % (self.stats["jobs_done"], self.stats["nodes_done"], self.stats["cpu_seconds"] / 3600.0))
        rows = self.result_rows()
        if rows:
            lines.append("Result per exit:")
            lines.extend("  " + r for r in rows)
        else:
            lines.append("The server did not send the per-exit result; see the campaign page.")
        lines.append("Thank you for contributing to the search!")
        return lines

    def result_rows(self):
        rows = []
        res = self.campaign_result
        entries = []
        if isinstance(res, dict):
            for k in sorted(res, key=lambda x: (len(str(x)), str(x))):
                v = res[k]
                if isinstance(v, dict):
                    entries.append((str(v.get("exit", k)), v))
        elif isinstance(res, list):
            entries = [(str(v.get("exit")), v) for v in res if isinstance(v, dict)]
        if entries:
            for ex, v in entries:
                best = v.get("moves", v.get("best"))
                if isinstance(best, dict):
                    best = best.get("moves")
                tags = []
                for k in ("exact", "clean", "published"):
                    if k in v:
                        tags.append(("%s" % k) if v[k] else ("not %s" % k))
                if v.get("open_candidates"):
                    tags.append("%s open candidate(s)" % v["open_candidates"])
                rows.append("exit %s: longest level %s moves%s" % (ex, best if best is not None else "?",
                                                                    " (%s)" % ", ".join(tags) if tags else ""))
            return rows
        ex = self.exits or {}
        for k in sorted(ex, key=lambda x: int(x) if str(x).lstrip("-").isdigit() else 0):
            e = ex[k] if isinstance(ex[k], dict) else {}
            b = e.get("best") if isinstance(e.get("best"), dict) else None
            tags = []
            # null = the server has not audited the exit yet (a cache warming up)
            if e.get("exact") is True:
                tags.append("exact")
            elif e.get("exact") is False and e.get("clean") is True:
                tags.append("NOT exact yet: unresolved candidates are being checked")
            elif e.get("clean") is False:
                tags.append("not complete yet")
            else:
                tags.append("not exact yet")
            if e.get("roots") is not None:
                cov = e.get("roots_covered")
                tags.append("%s/%s roots covered" % ("?" if cov is None else cov, e.get("roots")))
            nc = open_candidates(e)
            if nc:
                tags.append("%d unresolved candidate(s) deeper than the best" % nc)
            rows.append("exit %s: longest level %s moves%s%s" % (
                k, b.get("moves") if b else "?", " by %s" % b.get("by") if b and b.get("by") else "",
                " (%s)" % "; ".join(tags) if tags else ""))
        return rows

    # ------------------------------------------------------------ state for the UI
    def state(self):
        now = time.time()
        with self.lock:
            self.ui_last_poll = now
            slots = [s.snapshot(now) for s in self.slots if not (s.retire and s.job is None and s.state in ("dead", "idle"))]
            best_second = None
            for sn in slots:
                w = sn.get("win")
                if w and (best_second is None or w["depth"] > best_second["depth"]):
                    best_second = dict(w, exit=sn.get("exit"))
            while self.sec_levels and now - self.sec_levels[0][0] > SEC_BOARD_S:
                self.sec_levels.popleft()
            for t, d, code, ex in self.sec_levels:        # M66: LEVEL lines of runs that just ended
                if best_second is None or d > best_second["depth"]:
                    best_second = {"depth": d, "code": code, "exit": ex}
            while self.hour_best and now - self.hour_best[0][0] > 3600:
                self.hour_best.popleft()
            hb = self.hour_best
            best_session = None
            for b in self.best_per_exit.values():
                if best_session is None or b["depth"] > best_session["depth"]:
                    best_session = dict(b)
            n_active = sum(1 for x in self.slots if not x.retire)
            status_age = round(now - self.campaign_status_ok_at) if self.campaign_status_ok_at else None
            return {
                "name": self.name,
                "client_version": CLIENT_VERSION,
                "protocol": PROTOCOL,
                "workers": self.workers,
                "paused": self.paused,
                "stopping": self.stopping,
                "draining": self.draining,
                "stop_reason": self.stop_reason,
                "end_reason": self.end_reason,
                "campaign_state": self.campaign_state,
                "completion": self.completion,
                "waiting_next": self.waiting_next,
                "server": {"url": self.api.server, "connected": self.api.connected,
                           "last_ok": self.api.last_ok, "last_error": self.api.last_error,
                           "last_heartbeat_ok": self.last_heartbeat_ok},
                "campaign": {k: self.campaign.get(k) for k in ("id", "title", "grid", "extra", "split_after_s",
                                                             "ramp_split_after_s", "lease_s", "workers_max", "paused_max_s",
                                                             "batch_interval_s", "lease_ahead_s", "absorb_total_s",
                                                             "absorb_probe_s", "heartbeat_s", "lease_cap", "min_window_s",
                                                             "split_after_nodes", "release")},
                "phase": self.phase_name,
                "keep_going": bool(getattr(self.args, "keep_going", False)),
                "stop_on_close": not getattr(self.args, "no_stop_on_close", False),
                "window_now": self.window_now,
                "hold_cap": HOLD_PER_WORKER * n_active,
                "mean_job_s": round(self.mean_job_s(), 1),
                "src_hash": self.src_hash,
                "outbox": len(self.outbox_meta),
                "pending": len(self.pending),
                "queued": len(self.queue),
                "stats": dict(self.stats, cpu_hours=round(self.stats["cpu_seconds"] / 3600.0, 3)),
                "best_per_exit": self.best_per_exit,
                "slots": slots,
                "log": list(self.ring)[-25:],
                "time": now,
                # section 8
                "exit_pref": self.exit_pref,
                "exit_applied": self.exit_applied,
                "exit_pending": self.exit_pending,
                "exit_fallback": self.exit_fallback,
                "campaign_exits": self.campaign.get("exits"),
                "pending_flags": self.pending_flags(now),
                "me": self.me,
                "exits": self.exits,
                "campaign_status": self.campaign_status,
                # M60: counted from the last successful fetch; stale after 90 s
                "campaign_status_age": status_age if self.campaign_status else None,
                "campaign_status_stale": bool(self.campaign_status and (status_age is None or status_age > 90)),
                "best_second": best_second,
                "best_session": best_session,
                "best_hour": ({"depth": hb[0][1], "code": hb[0][2], "exit": hb[0][3], "at": hb[0][0]} if hb else None),
                "session": {
                    "uptime_s": round(now - self.session_start),
                    "jobs": self.stats["jobs_done"],
                    "mean_job_s": round(self.job_total_s / self.stats["jobs_done"], 1) if self.stats["jobs_done"] else None,
                    "longest_job_s": round(self.longest_job_s, 1),
                    "absorbed": max(0, self.stats["nodes_done"] - self.stats["done"]),
                    "splits": self.stats["split"],
                    "open_handed_back": self.stats["nodes_open"],
                    "runs": self.stats["runs"],
                },
                "worker_path": " ".join(self.worker_base or []),
                "worker_source": self.worker_src,
            }

    # ------------------------------------------------------------ main loop
    def install_signals(self):
        # Installed before registration so that Ctrl-C works from the first second,
        # even when the process inherited SIGINT=ignored (background job of a script).
        signal.signal(signal.SIGINT, self._on_sigint)
        signal.signal(signal.SIGTERM, lambda *_: self._async_stop("SIGTERM"))
        if hasattr(signal, "SIGHUP"):     # M14: closing the terminal = Stop, never a silent death
            signal.signal(signal.SIGHUP, lambda *_: self._async_stop("SIGHUP (terminal closed)"))
        if hasattr(signal, "SIGTSTP"):    # M61: Ctrl-Z stops the workers with the client
            signal.signal(signal.SIGTSTP, self._on_sigtstp)

    def _async_stop(self, reason):
        if self.phase_name in ("startup", "waiting"):
            self.user_stop = True
            if self.phase_name == "startup":
                log_flush(0.5)
                os._exit(1)
            return
        threading.Thread(target=self.request_stop, args=(reason, True), daemon=True).start()

    def _on_sigint(self, *_):
        self.sigint_count += 1
        if self.sigint_count >= 2 and self.phase_name in ("running", "exiting"):
            self.force_quit()
        if self.phase_name == "startup":
            # still starting up (registration / version check): nothing to wind down
            try:
                sys.stderr.write("\ninterrupted during startup\n")
            except OSError:
                pass
            os._exit(1)
        if self.phase_name == "waiting":
            self.user_stop = True
            self.log("Ctrl-C: no longer waiting for the next campaign")
            return
        threading.Thread(target=self.request_stop, args=("Ctrl-C", True), daemon=True).start()

    def main_loop(self):
        """Register, run the campaign, and follow the campaign life cycle."""
        while True:
            self.phase_name = "waiting" if self.ever_ran else "startup"
            if self.pre_registered:
                self.pre_registered = False
                r = "ok"
            else:
                r = self.register()
            if r == "stopped":
                return self.finish_fatal()
            if r == "no_campaign":
                if not self.args.keep_going:
                    emit("\nNo campaign is running on %s right now%s. The client exits; start it again when the\n"
                         "next campaign is announced, or run it with --keep-going to wait for it."
                         % (self.api.server, " (the last one is complete)" if self.completed_campaign_id is not None else ""))
                    return 0
                if not self.wait_for_campaign():
                    emit("Stopped while waiting for the next campaign.")
                    return 0
            if not self.check_hash():
                return self.finish_fatal()
            self.log("campaign %s: grid %s, flags %s, split after %s s, lease %s s, %d worker(s)"
                     % (self.campaign.get("id", "?"), self.param("grid", "?"), " ".join(self.extra),
                        self.param("split_after_s", "?"), self.param("lease_s", "?"), self.workers))
            self.phase_name = "running"
            self.ever_ran = True
            self.sigint_count = 0
            self.completion = None    # an earlier campaign's closing summary (--keep-going) is not this one's
            end = self.run()
            if end == "complete":
                cid = self.campaign.get("id")
                lines = self.closing_summary()
                self.completion = {"campaign_id": cid, "lines": lines, "at": time.time()}
                emit("\n".join(["", "=" * 72] + lines + ["=" * 72, ""]))
                self.completed_campaign_id = cid
                if self.no_send:                  # 410: the token is gone with its campaign
                    self.retire_outbox("the campaign is complete and a newer one is open")
                if self.no_send or not self.outbox_files():
                    self.drop_token()
                self.wait_for_panel()
                if self.user_stop or not self.args.keep_going:
                    return 0
                self._reset_campaign_state()
                self.token = None
                if not self.wait_for_campaign():
                    emit("Stopped while waiting for the next campaign.")
                    return 0
                self.pre_registered = True
                continue
            if end in ("closed", "reregister"):
                if end == "closed":
                    self.retire_outbox("the campaign was closed")
                    self.drop_token()
                else:
                    self.token = None
                    self.update_token_file(lambda d: d.pop(self.token_key(), None))
                if self.user_stop:
                    return 0
                self._reset_campaign_state()
                self.log("registering again (%s)" % ("the campaign was closed" if end == "closed" else "unknown token"))
                continue
            if end == "fatal":
                return self.finish_fatal()
            return self.exit_code

    def finish_fatal(self):
        if self.fatal_msg:
            emit("\n" + self.fatal_msg)
        self.wait_for_panel()
        return self.exit_code or 2 if self.fatal_msg else self.exit_code

    def wait_for_panel(self):
        """Tell the panel (it polls every 250 ms) before the client goes away."""
        if not self.ui_started or time.time() - self.ui_last_poll > 5:
            return
        t0 = time.time()
        while time.time() - t0 < 3.0 and self.ui_last_poll < t0 + 0.3:
            time.sleep(0.1)

    def run(self):
        """One campaign. Returns the end reason: stop | complete | closed | reregister | fatal."""
        if not self.args.no_ui and not self.ui_started:
            self.start_ui()
        self.scan_outbox()
        with self.lock:
            self._reconcile_slots()
        leaser = threading.Thread(target=self.leaser_loop, name="leaser", daemon=True)
        leaser.start()
        self.heartbeat()
        self.last_heartbeat = time.time()
        if not (self.stopping or self.end_reason in ("closed", "reregister")):
            self.flush_outbox(force=True)     # the sender takes over the outbox from here
        sender = threading.Thread(target=self.sender_loop, name="sender", daemon=True)
        sender.start()
        self.last_tick = time.time()
        slack = 3 * STATUS_EVERY_MS / 1000.0 + float(self.args.watchdog_slack_s)
        grace = float(self.args.stop_grace_s)
        while True:
            time.sleep(MAIN_TICK_S)
            try:
                now = time.time()
                gap = now - self.last_tick
                self.last_tick = now
                if gap > CLOCK_GAP_S:
                    # before any watchdog check: the silence and the overrun it measures happened
                    # while the client could not look (or the whole machine was asleep)
                    self.shift_baselines(gap - MAIN_TICK_S, now)
                if gap > 2 * self.lease_s() and not self.stopping:
                    self.handle_wake(gap)
                token_ok = self.end_reason not in ("closed", "reregister")
                if token_ok and (self.heartbeat_now or now - self.last_heartbeat >= self.fparam("heartbeat_s", HEARTBEAT_S)):
                    self.heartbeat_now = False
                    self.last_heartbeat = now
                    self.heartbeat()
                if not self.args.no_ui and now - self.ui_last_poll < 60 and now - self.campaign_status_at >= 30:
                    self.campaign_status_at = now
                    threading.Thread(target=self.fetch_campaign_status, daemon=True).start()
                self._supervise(now, slack, grace)
                if now - self.last_reconcile > 5 and not (self.stopping or self.draining):
                    self.last_reconcile = now
                    with self.lock:
                        self._reconcile_slots()
                if self.stopping or self.draining:
                    alive = [s for s in self.slots if s.thread and s.thread.is_alive()]
                    if not alive:
                        break
            except Exception:
                self.log("!!! main loop: internal error (continuing): %s" % traceback.format_exc().strip().splitlines()[-1])
        # every worker has finished: send the final batch now, then drain the outbox
        self.phase_name = "exiting" if self.end_reason in ("stop", "fatal") else self.phase_name
        with self.lock:
            self.sender_may_exit = True
        self.send_event.set()
        sender.join(90)
        leaser.join(5)
        self.wind_down_reports()
        self.final_heartbeat()
        left = len(self.outbox_files())
        self.log("%s. jobs done %d (%d exhausted, %d split), %d nodes done locally, %d batch(es), CPU %.2f h%s"
                 % ({"complete": "campaign complete", "closed": "campaign closed", "reregister": "token no longer valid",
                     "fatal": "stopped (refused)"}.get(self.end_reason, "stopped"),
                    self.stats["jobs_done"], self.stats["done"], self.stats["split"], self.stats["nodes_done"],
                    self.stats["batches_sent"], self.stats["cpu_seconds"] / 3600.0,
                    "; %d report file(s) still in the outbox, run again to deliver them" % left if left else ""))
        return self.end_reason or "stop"

    def wind_down_reports(self):
        reason = self.end_reason
        if reason == "closed":
            self.retire_outbox("the campaign was closed")
            return
        if self.no_send:
            with self.lock:
                left, self.pending = self.pending, []
            for chunk in self.chunk_reports(left):
                self.outbox_write(chunk, attempts=0, next_try=0.0)
            return
        self.send_pending()
        self.flush_outbox(force=True)
        if reason == "complete":
            # deliver everything before saying goodbye (bounded; Ctrl-C ends the wait)
            end = time.time() + float(self.args.complete_flush_s)
            while self.outbox_files() and time.time() < end and not self.user_stop and self.end_reason == "complete":
                self._sleep(5)
                self.flush_outbox(force=True)

    def _supervise(self, now, slack, grace):
        """Deferred stops, the per-run watchdog (H1) and the stop grace (main thread, 2 Hz)."""
        with self.lock:
            for s in self.slots:
                if s.stop_pending and s.proc and now - s.run_started_at > 5.0:
                    s.saw_output = True      # silent for 5 s: assume it is up and signal it
                    self._send_stop(s)
                if s.proc and s.state == "running" and not s.frozen and not s.watchdog_fired and not self.paused:
                    quiet_since = max(s.last_line_at, s.split_at)
                    overrun = now > s.split_at + max(600.0, s.split_after_s)
                    if now - quiet_since > slack or overrun:
                        why = ("no output for %.0f s after its split time" % (now - quiet_since) if not overrun
                               else "still running %.0f s after its split time" % (now - s.split_at))
                        self.log("!!! worker %d (job %s): %s; sending SIGINT, then SIGKILL after %.0f s. "
                                 "The run is void (hang)." % (s.idx, s.job and s.job["id"], why, grace))
                        s.watchdog_fired = True
                        s.watchdog_why = "watchdog: " + why
                        self._signal(s, signal.SIGCONT)
                        self._signal(s, signal.SIGINT)
                        s.state = "finishing"
                        s.stop_sent_at = now
                if s.state == "finishing" and s.proc and s.stop_sent_at and now - s.stop_sent_at > grace:
                    if s.watchdog_fired:
                        self.log("!!! worker %d did not exit within %.0f s of SIGINT; killing it" % (s.idx, grace))
                    else:
                        # only this run is lost: its node goes back open, the rest of the window is
                        # reported (J's own run has nothing to report: the job is left untouched)
                        own = s.job is not None and s.cur_seed == s.job["seed"]
                        self.log("!!! worker %d did not print SUMMARY within %.0f s of SIGINT; killing it. %s"
                                 % (s.idx, grace, "Job %s is left untouched." % (s.job and s.job["id"]) if own or s.job is None
                                    else "Node %s of job %d stays open; the rest of the window is reported."
                                    % (s.cur_seed or "(exit root)", s.job["id"])))
                        s.grace_killed = True
                    self._signal(s, signal.SIGCONT)
                    self._signal(s, signal.SIGKILL)
                    s.stop_sent_at = 0.0
            if self.ui_close_at and now - self.ui_close_at > 3.0:
                if self.ui_last_poll < self.ui_close_at:
                    self.ui_close_at = 0.0
                    threading.Thread(target=self.request_stop, args=("browser window closed", True), daemon=True).start()
                else:
                    self.ui_close_at = 0.0

    # ------------------------------------------------------------ local GUI
    def start_ui(self):
        vol = self
        port = self.args.port
        hosts = {"127.0.0.1:%d" % port, "localhost:%d" % port}
        origins = {"http://127.0.0.1:%d" % port, "http://localhost:%d" % port}

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
                self.send_header("X-Frame-Options", "DENY")          # M26: no clickjacking of the buttons
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)

            def _host_ok(self):
                # M26: a DNS-rebinding page is same-origin with us but still sends its own Host name
                return self.headers.get("Host") in hosts

            def do_GET(self):
                try:
                    if not self._host_ok():
                        return self._send(403, {"error": "forbidden host"})
                    path = self.path.split("?")[0]
                    if path in ("/", "/index.html"):
                        try:
                            with open(UI_FILE, "rb") as f:
                                html = f.read().replace(UI_SECRET_PLACEHOLDER.encode(), vol.ui_secret.encode())
                            self._send(200, html, "text/html")
                        except OSError:
                            self._send(500, "volunteer_ui.html not found next to volunteer.py", "text/plain")
                    elif path == "/state":
                        self._send(200, vol.state())
                    else:
                        self._send(404, {"error": "not_found"})
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    try:
                        self._send(500, {"error": "internal"})
                    except Exception:
                        pass

            def do_POST(self):
                try:
                    self._post()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception:
                    try:
                        self._send(500, {"error": "internal"})
                    except Exception:
                        pass

            def _post(self):
                path = self.path.split("?")[0]
                if not self._host_ok():
                    return self._send(403, {"error": "forbidden host"})
                origin = self.headers.get("Origin")
                if origin is not None and origin not in origins:
                    return self._send(403, {"error": "forbidden origin"})
                # the per-session secret is in the page this client served; a foreign page never sees it
                if self.headers.get("X-Volunteer") != vol.ui_secret:
                    return self._send(403, {"error": "forbidden"})
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 65536)) if n else b""
                try:
                    body = json.loads(raw.decode() or "{}")
                except ValueError:
                    body = {}
                if path == "/pause":
                    with vol.lock:
                        vol.pause_requested_at = time.time()
                    threading.Thread(target=vol.pause, daemon=True).start()
                    return self._send(200, {"ok": True, "pending": True})
                if path == "/resume":
                    with vol.lock:
                        vol.resume_requested_at = time.time()
                    threading.Thread(target=vol.resume, daemon=True).start()
                    return self._send(200, {"ok": True, "pending": True})
                if path == "/exit":
                    ok, msg = vol.set_exit(body.get("exit") if isinstance(body, dict) else None)
                    return self._send(200 if ok else 400, {"ok": ok, "error": msg, "exit": vol.exit_pref})
                if path == "/stop":
                    if vol.phase_name == "waiting":
                        vol.user_stop = True
                    else:
                        threading.Thread(target=vol.request_stop, args=("Stop pressed in the UI", True), daemon=True).start()
                    return self._send(200, {"ok": True})
                if path == "/workers":
                    n = vol.set_workers(body.get("workers") if isinstance(body, dict) else None)
                    return self._send(200, {"ok": True, "workers": n, "pending": True})
                if path == "/close":
                    if not vol.args.no_stop_on_close:
                        with vol.lock:
                            vol.ui_close_at = time.time()
                    return self._send(200, {"ok": True})
                self._send(404, {"error": "not_found"})

        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as e:
            emit("Cannot listen on 127.0.0.1:%d (%s). Use --port or --no-ui." % (port, e))
            sys.exit(2)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, name="ui", daemon=True).start()
        self.ui_started = True
        url = "http://127.0.0.1:%d/" % port
        self.log("local GUI at %s" % url)
        if not self.args.no_browser:
            # M65: never block on a console browser (headless Linux over SSH picks lynx/w3m)
            is_linux = sys.platform.startswith("linux")
            wsl = "microsoft" in platform.release().lower()
            if is_linux and not wsl and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                self.log("no graphical display: open %s in a browser (or pass --no-ui)" % url)
            else:
                threading.Thread(target=self._open_browser, args=(url,), name="browser", daemon=True).start()

    @staticmethod
    def _open_browser(url):
        try:
            webbrowser.open(url)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="Pathology collective search volunteer client (protocol %d)" % PROTOCOL)
    ap.add_argument("--name", help="your display name (fixed at first registration)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="search processes, one core and about 0.5 GB of memory each (default: half the logical cores)")
    ap.add_argument("--exit", type=int, default=None, help="prefer jobs of this exit (default: any)")
    ap.add_argument("--server", default=os.environ.get("VOLUNTEER_SERVER", DEFAULT_SERVER))
    ap.add_argument("--port", type=int, default=8765, help="local GUI port (127.0.0.1 only)")
    ap.add_argument("--worker", default=os.path.join(".", "backsearch_worker_nt"),
                    help="worker binary (default ./backsearch_worker_nt in the backsearch directory)")
    ap.add_argument("--outbox", default=os.path.join(HERE, "volunteer_outbox"),
                    help="undelivered reports and this client's lock; one client per outbox")
    ap.add_argument("--keep-going", action="store_true",
                    help="when the campaign is complete (or none is running), wait for the next one and join it")
    ap.add_argument("--no-ui", action="store_true", help="do not serve the local GUI")
    ap.add_argument("--no-browser", action="store_true", help="serve the GUI but do not open a browser")
    ap.add_argument("--no-stop-on-close", action="store_true", help="closing the browser tab does not stop the client")
    ap.add_argument("--reregister", action="store_true", help="ignore the stored token and register anew")
    ap.add_argument("--verbose-workers", action="store_true", help="let workers write their stderr to the terminal")
    # test and tuning knobs (not for volunteers)
    ap.add_argument("--watchdog-slack-s", type=float, default=WATCHDOG_SLACK_S, help=argparse.SUPPRESS)
    ap.add_argument("--stop-grace-s", type=float, default=STOP_GRACE_S, help=argparse.SUPPRESS)
    ap.add_argument("--campaign-poll-s", type=float, default=60.0, help=argparse.SUPPRESS)
    ap.add_argument("--complete-flush-s", type=float, default=120.0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")

    vol = Volunteer(args)
    vol.install_signals()
    code = 1
    try:
        vol.acquire_outbox_lock()
        vol.check_dirty()
        vol.cleanup_orphans()
        vol.pin_worker()
        vol.src_hash = vol.read_version()
        vol.log("worker %s (pinned copy of %s) hash %s, protocol %s"
                % (" ".join(vol.worker_base), vol.worker_src, vol.src_hash, vol.worker_info.get("PROTOCOL")))
        code = vol.main_loop()
    finally:
        vol.kill_all_workers()
        log_flush(2.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
