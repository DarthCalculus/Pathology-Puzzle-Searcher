#!/usr/bin/env python3
"""Minimal in-memory stand-in for the v2 server (DESIGN.md section 3), stdlib
only, for testing volunteer.py. Not a reference implementation of the gates:
it implements the coverage tree (open/leased/done/split), lease expiry,
paused-lease extension, the hash whitelist and the token-prefix check on
REMAINING, which is what the client tests need.

  python3 fake_server.py --port 8899 --roots 8 --lease-s 6 --split-after-s 3 \
      --hashes fa4e000...

Routes: /api/v2/register, /heartbeat, /lease, /report, /status, /audit (GET).
"""
import argparse
import json
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEED_RE = re.compile(r"^(?:[URDL][123])(?:,[URDL][123])*$")
TOKENS = [d + n for d in "URDL" for n in "123"]


class Store:
    def __init__(self, opts):
        self.lock = threading.Lock()
        self.opts = opts
        self.campaign = {
            "id": 1, "title": "fake campaign", "grid": opts.grid,
            "extra": opts.extra.split() if opts.extra else [],
            "exits": [int(x) for x in opts.exits.split(",")],
            "hashes": [h for h in opts.hashes.split(",") if h],
            "max_clients": opts.max_clients, "workers_max": 32,
            "job_target_s": opts.split_after_s, "split_after_s": opts.split_after_s,
            "lease_s": opts.lease_s, "paused_max_s": opts.paused_max_s,
        }
        self.clients = {}
        for spec in (opts.preregister or []):   # NAME:TOKEN, survives a restart in tests
            name, _, token = spec.partition(":")
            self.clients[token] = {"token": token, "name": name, "workers": 1, "created_at": time.time(),
                                   "last_seen": time.time(), "paused": False, "boards": [], "revoked": False}
        self.jobs = {}
        self.reports = []
        self.next_id = 1
        import random
        rng = random.Random(opts.seed)
        for k in range(opts.roots):
            ex = self.campaign["exits"][k % len(self.campaign["exits"])]
            seed = ",".join(rng.choice(TOKENS) for _ in range(opts.layer))
            self.add_job(ex, seed, None)

    def add_job(self, exit_, seed, parent_id):
        jid = self.next_id
        self.next_id += 1
        self.jobs[jid] = {
            "id": jid, "exit": exit_, "seed": seed, "parent_id": parent_id,
            "depth": len(seed.split(",")) if seed else 0, "status": "open",
            "client_token": None, "lease_until": None, "leased_at": None,
            "done_at": None, "src_hash": None, "states": None, "best": None,
            "level": None, "elapsed_s": None, "children": [],
        }
        if parent_id is not None:
            self.jobs[parent_id]["children"].append(jid)
        return jid

    def sweep(self):
        now = time.time()
        for j in self.jobs.values():
            if j["status"] == "leased" and j["lease_until"] is not None and j["lease_until"] < now:
                c = self.clients.get(j["client_token"])
                if c and c["paused"] and j["leased_at"] and now - j["leased_at"] < self.campaign["paused_max_s"]:
                    continue
                self.log("lease expired: job %d (%s)" % (j["id"], j["seed"]))
                j["status"] = "open"
                j["client_token"] = None
                j["lease_until"] = None

    def log(self, msg):
        sys.stderr.write("[fake_server %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()

    # ---- routes ----
    def register(self, body):
        name = str(body.get("name", ""))[:40]
        if not name:
            return 400, {"error": "name"}
        workers = max(1, min(32, int(body.get("workers", 1))))
        active = [c for c in self.clients.values() if time.time() - c["last_seen"] < 180 and not c["revoked"]]
        if len(active) >= self.campaign["max_clients"]:
            return 503, {"error": "full", "queue_position": 1}
        token = secrets.token_hex(16)
        self.clients[token] = {"token": token, "name": name, "workers": workers, "created_at": time.time(),
                               "last_seen": time.time(), "paused": False, "boards": [], "revoked": False}
        self.log("register %s workers=%d token=%s" % (name, workers, token[:8]))
        return 200, {"token": token, "campaign": self.campaign}

    def heartbeat(self, body):
        c = self.clients.get(body.get("token"))
        if not c:
            return 401, {"error": "unknown_token"}
        c["last_seen"] = time.time()
        c["workers"] = max(0, min(32, int(body.get("workers", c["workers"]))))
        c["paused"] = bool(body.get("paused", False))
        boards = body.get("boards", [])
        if isinstance(boards, list):
            c["boards"] = boards[:max(1, c["workers"])]
        self.sweep()
        leases = [j["id"] for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"]]
        res = {"ok": True, "leases": leases, "campaign": self.campaign}
        if c["revoked"]:
            res["revoked"] = True
        return 200, res

    def lease(self, body):
        c = self.clients.get(body.get("token"))
        if not c:
            return 401, {"error": "unknown_token"}
        if c["revoked"]:
            return 403, {"error": "revoked"}
        c["last_seen"] = time.time()
        self.sweep()
        n = max(0, min(64, int(body.get("n", 1))))
        held = sum(1 for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"])
        n = min(n, 3 * max(1, c["workers"]) - held)
        granted = []
        if n > 0:
            open_by_exit = {}
            for j in self.jobs.values():
                if j["status"] == "open":
                    open_by_exit.setdefault(j["exit"], []).append(j)
            order = sorted(open_by_exit.items(), key=lambda kv: -len(kv[1]))
            for _, lst in order:
                for j in sorted(lst, key=lambda j: j["id"]):
                    if len(granted) >= n:
                        break
                    j["status"] = "leased"
                    j["client_token"] = c["token"]
                    j["leased_at"] = time.time()
                    j["lease_until"] = time.time() + self.campaign["lease_s"]
                    granted.append({"id": j["id"], "exit": j["exit"], "seed": j["seed"]})
                if len(granted) >= n:
                    break
        self.log("lease %s n=%d -> %s" % (c["name"], n, [g["id"] for g in granted]))
        return 200, {"jobs": granted}

    def report(self, body):
        c = self.clients.get(body.get("token"))
        if not c:
            return 401, {"error": "unknown_token"}
        c["last_seen"] = time.time()
        try:
            jid = int(body.get("job_id"))
        except Exception:
            return 400, {"error": "job_id"}
        j = self.jobs.get(jid)
        if not j:
            return 404, {"error": "no_such_job"}
        src_hash = str(body.get("src_hash", ""))
        summary = body.get("summary")
        if not isinstance(summary, dict):
            return 400, {"error": "summary"}
        self.reports.append({"job_id": jid, "token": c["token"], "src_hash": src_hash,
                             "summary": summary, "received_at": time.time()})
        if j["status"] in ("done", "split"):
            self.log("dup report for job %d" % jid)
            return 200, {"dup": True}
        foreign = j["client_token"] != c["token"]
        if src_hash not in self.campaign["hashes"]:
            j["status"] = "open"; j["client_token"] = None; j["lease_until"] = None
            self.log("unknown hash %s on job %d" % (src_hash[:12], jid))
            return 409, {"error": "unknown_hash"}
        st = summary.get("status")
        if st == "exhausted":
            j["status"] = "done"
        elif st == "split":
            rem = body.get("remaining")
            if not isinstance(rem, list) or not rem:
                j["status"] = "open"; j["client_token"] = None; j["lease_until"] = None
                return 400, {"error": "remaining_empty"}
            seed_toks = j["seed"].split(",") if j["seed"] else []
            for r in rem:
                if not isinstance(r, str) or not SEED_RE.match(r):
                    j["status"] = "open"; j["client_token"] = None; j["lease_until"] = None
                    return 400, {"error": "remaining_parse", "path": str(r)[:100]}
                rt = r.split(",")
                if len(rt) <= len(seed_toks) or rt[:len(seed_toks)] != seed_toks:
                    j["status"] = "open"; j["client_token"] = None; j["lease_until"] = None
                    return 400, {"error": "remaining_not_extension", "path": r}
            j["status"] = "split"
            for r in rem:
                self.add_job(j["exit"], r, jid)
        else:
            return 400, {"error": "summary_status"}
        j["done_at"] = time.time()
        j["src_hash"] = src_hash
        j["states"] = summary.get("states")
        j["best"] = summary.get("best")
        j["elapsed_s"] = summary.get("elapsed")
        j["level"] = body.get("level")
        j["client_token"] = None
        j["lease_until"] = None
        self.log("report job %d %s by %s%s%s" % (jid, st, c["name"], " (foreign)" if foreign else "",
                                                 " children=%d" % len(j["children"]) if st == "split" else ""))
        return 200, {"ok": True}

    def status(self):
        self.sweep()
        per_exit = {}
        for j in self.jobs.values():
            d = per_exit.setdefault(j["exit"], {"open": 0, "leased": 0, "done": 0, "split": 0})
            d[j["status"]] += 1
        cpu = sum((j["elapsed_s"] or 0) for j in self.jobs.values() if j["status"] in ("done", "split"))
        return 200, {"per_exit": per_exit, "cpu_hours": cpu / 3600.0,
                     "clients": [{"name": c["name"], "workers": c["workers"], "paused": c["paused"],
                                  "boards": c["boards"], "last_seen": c["last_seen"]} for c in self.clients.values()],
                     "hashes": self.campaign["hashes"], "jobs": len(self.jobs)}

    def covered(self, jid):
        j = self.jobs[jid]
        if j["status"] == "done":
            return True
        if j["status"] == "split":
            return all(self.covered(c) for c in j["children"])
        return False

    def audit(self):
        self.sweep()
        roots = [j for j in self.jobs.values() if j["parent_id"] is None]
        uncovered = [j["id"] for j in roots if not self.covered(j["id"])]
        counts = {"open": 0, "leased": 0, "done": 0, "split": 0}
        for j in self.jobs.values():
            counts[j["status"]] += 1
        return 200, {"roots": len(roots), "uncovered_roots": uncovered, "counts": counts,
                     "unknown_hash_jobs": [r["job_id"] for r in self.reports if r["src_hash"] not in self.campaign["hashes"]],
                     "mismatches": [], "reports": len(self.reports)}


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            with store.lock:
                if path == "/api/v2/status":
                    code, obj = store.status()
                elif path == "/api/v2/audit":
                    code, obj = store.audit()
                else:
                    code, obj = 404, {"error": "not_found"}
            self._send(code, obj)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 4 * 1024 * 1024:
                return self._send(413, {"error": "too_large"})
            raw = self.rfile.read(n) if n else b""
            try:
                body = json.loads(raw.decode() or "{}")
                if not isinstance(body, dict):
                    raise ValueError
            except Exception:
                return self._send(400, {"error": "bad_json"})
            path = self.path.split("?")[0]
            with store.lock:
                if path == "/api/v2/register":
                    code, obj = store.register(body)
                elif path == "/api/v2/heartbeat":
                    code, obj = store.heartbeat(body)
                elif path == "/api/v2/lease":
                    code, obj = store.lease(body)
                elif path == "/api/v2/report":
                    code, obj = store.report(body)
                else:
                    code, obj = 404, {"error": "not_found"}
            self._send(code, obj)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--roots", type=int, default=8)
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--lease-s", type=float, default=6)
    ap.add_argument("--split-after-s", type=float, default=3)
    ap.add_argument("--paused-max-s", type=float, default=60)
    ap.add_argument("--grid", default="5x5")
    ap.add_argument("--extra", default="--allow-exit-transit --num-holes 3")
    ap.add_argument("--exits", default="0,1,2")
    ap.add_argument("--hashes", default="fa4e" + "0" * 60)
    ap.add_argument("--max-clients", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--preregister", action="append", help="NAME:TOKEN client that exists at startup")
    opts = ap.parse_args()
    store = Store(opts)
    srv = ThreadingHTTPServer(("127.0.0.1", opts.port), make_handler(store))
    srv.daemon_threads = True
    store.log("listening on http://127.0.0.1:%d with %d root jobs, lease_s=%g split_after_s=%g"
              % (opts.port, opts.roots, opts.lease_s, opts.split_after_s))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
