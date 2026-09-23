#!/usr/bin/env python3
"""Minimal in-memory stand-in for the v2 server (DESIGN.md section 3), stdlib
only, for testing volunteer.py. Not a reference implementation of the gates:
it implements the coverage tree (open/leased/done/split), lease expiry,
paused-lease extension, the hash whitelist and the token-prefix check on
REMAINING, which is what the client tests need.

  python3 fake_server.py --port 8899 --roots 8 --lease-s 6 --split-after-s 3 \
      --hashes fa4e000...

Routes: /api/v2/register, /heartbeat, /lease, /report (one-node or tree, 7.5),
/reports (batch, per-report outcomes in order, 7.6), /status, /audit (GET; carries
`stats` with lease/report/batch counters used by test_client.sh).
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
            # section 7 load rules
            "ramp_split_after_s": opts.ramp_split_after_s, "batch_interval_s": opts.batch_interval_s,
            "lease_ahead_s": opts.lease_ahead_s, "absorb_total_s": opts.absorb_total_s,
            "absorb_probe_s": opts.absorb_probe_s, "lease_cap": opts.lease_cap, "heartbeat_s": opts.heartbeat_s,
        }
        self.stats = {"lease_requests": 0, "lease_grants": 0, "report_posts": 0, "batch_posts": 0,
                      "report_bodies": 0, "tree_reports": 0, "one_node_reports": 0, "max_nodes": 0,
                      "absorbed_done": 0, "grants_of_absorbed": 0, "max_batch": 0,
                      "split_nodes_with_level": 0, "open_nodes_with_level": 0, "split_nodes": 0}
        self.inserted_done = set()   # ids of nodes that arrived already done inside a tree report
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

    def add_job(self, exit_, seed, parent_id, status="open"):
        jid = self.next_id
        self.next_id += 1
        self.jobs[jid] = {
            "id": jid, "exit": exit_, "seed": seed, "parent_id": parent_id,
            "depth": len(seed.split(",")) if seed else 0, "status": status,
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
        res = {"ok": True, "leases": leases, "campaign": self.campaign,
               "me": self.me_for(c["name"]), "exits": self.exits_summary()}
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
        n = max(0, min(self.campaign["lease_cap"], int(body.get("n", 1))))
        held = sum(1 for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"])
        n = min(n, 3 * max(1, c["workers"]) + 200 - held)
        self.stats["lease_requests"] += 1
        granted = []
        pref = body.get("exit")
        pref = int(pref) if isinstance(pref, int) or (isinstance(pref, str) and pref.isdigit()) else None
        fallback = False
        if n > 0:
            open_by_exit = {}
            for j in self.jobs.values():
                if j["status"] == "open":
                    open_by_exit.setdefault(j["exit"], []).append(j)
            order = sorted(open_by_exit.items(), key=lambda kv: -len(kv[1]))
            if pref is not None:
                if pref in open_by_exit:
                    order = [(pref, open_by_exit[pref])]   # 8: the preferred exit first and only
                else:
                    fallback = bool(open_by_exit)
            for _, lst in order:
                for j in sorted(lst, key=lambda j: j["id"]):
                    if len(granted) >= n:
                        break
                    j["status"] = "leased"
                    j["client_token"] = c["token"]
                    j["leased_at"] = time.time()
                    j["lease_until"] = time.time() + self.campaign["lease_s"]
                    granted.append({"id": j["id"], "exit": j["exit"], "seed": j["seed"]})
                    if j["id"] in self.inserted_done:
                        self.stats["grants_of_absorbed"] += 1
                if len(granted) >= n:
                    break
        self.stats["lease_grants"] += len(granted)
        # 7.4: the server sets the window
        open_jobs = sum(1 for j in self.jobs.values() if j["status"] == "open")
        active_workers = sum(cl["workers"] for cl in self.clients.values()
                             if time.time() - cl["last_seen"] < 180 and not cl["revoked"])
        sa = self.campaign["ramp_split_after_s"] if open_jobs < 3 * active_workers else self.campaign["split_after_s"]
        self.log("lease %s n=%d exit=%s -> %s (split_after_s %g%s)" % (c["name"], n, pref, [g["id"] for g in granted], sa,
                                                                        ", fallback" if fallback else ""))
        res = {"jobs": granted, "split_after_s": sa}
        if fallback:
            res["exit_fallback"] = True
        return 200, res

    def _free(self, j):
        j["status"] = "open"; j["client_token"] = None; j["lease_until"] = None

    def report(self, body, token=None):
        c = self.clients.get(token or body.get("token"))
        if not c:
            return 401, {"error": "unknown_token"}
        c["last_seen"] = time.time()
        self.stats["report_bodies"] += 1
        try:
            jid = int(body.get("job_id"))
        except Exception:
            return 400, {"error": "job_id"}
        j = self.jobs.get(jid)
        if not j:
            return 404, {"error": "no_such_job"}
        src_hash = str(body.get("src_hash", ""))
        nodes = body.get("nodes")
        if nodes is None:
            # old one-node shape: summary + remaining
            summary = body.get("summary")
            if not isinstance(summary, dict):
                return 400, {"error": "summary"}
            st = {"exhausted": "done", "split": "split"}.get(summary.get("status"))
            if st is None:
                return 400, {"error": "summary_status"}
            nodes = [{"seed": j["seed"], "parent": None, "status": st, "summary": summary, "level": body.get("level")}]
            for r in body.get("remaining") or []:
                nodes.append({"seed": r, "parent": j["seed"], "status": "open"})
        if not isinstance(nodes, list) or not nodes or len(nodes) > 5000:
            return 400, {"error": "nodes"}
        self.reports.append({"job_id": jid, "token": c["token"], "src_hash": src_hash,
                             "nodes": len(nodes), "received_at": time.time()})
        if j["status"] in ("done", "split"):
            self.log("dup report for job %d" % jid)
            return 200, {"dup": True}
        foreign = j["client_token"] != c["token"]
        if src_hash not in self.campaign["hashes"]:
            self._free(j)
            self.log("unknown hash %s on job %d" % (src_hash[:12], jid))
            return 409, {"error": "unknown_hash"}
        # validate the tree (7.5)
        seed_toks = j["seed"].split(",") if j["seed"] else []
        seen = {}
        children_of = {}
        for k, n in enumerate(nodes):
            if not isinstance(n, dict) or not isinstance(n.get("seed"), str) or not SEED_RE.match(n["seed"]):
                self._free(j); return 400, {"error": "node_seed", "index": k}
            toks = n["seed"].split(",")
            if toks[:len(seed_toks)] != seed_toks or (k > 0 and len(toks) <= len(seed_toks)):
                self._free(j); return 400, {"error": "node_not_extension", "index": k}
            if k == 0 and n["seed"] != j["seed"]:
                self._free(j); return 400, {"error": "first_node_not_job"}
            if n["seed"] in seen:
                self._free(j); return 400, {"error": "node_duplicate", "index": k}
            if k > 0 and (n.get("parent") not in seen):
                self._free(j); return 400, {"error": "node_parent", "index": k}
            st = n.get("status")
            if st not in ("done", "split", "open"):
                self._free(j); return 400, {"error": "node_status", "index": k}
            if st == "done" and not (isinstance(n.get("summary"), dict) and n["summary"].get("status") == "exhausted"):
                self._free(j); return 400, {"error": "done_without_summary", "index": k}
            if st == "open" and n.get("summary") is not None:
                self._free(j); return 400, {"error": "open_with_summary", "index": k}
            lv = n.get("level")
            if lv is not None and not (isinstance(lv, dict) and isinstance(lv.get("depth"), int)
                                       and isinstance(lv.get("code"), str) and 0 < len(lv["code"]) <= 2000):
                self._free(j); return 400, {"error": "node_level", "index": k}
            seen[n["seed"]] = k
            if k > 0:
                children_of.setdefault(n["parent"], []).append(k)
        for k, n in enumerate(nodes):
            if n["status"] == "split" and not children_of.get(n["seed"]):
                self._free(j); return 400, {"error": "split_without_children", "index": k}
        # insert in one go
        ids = {j["seed"]: jid}
        root = nodes[0]
        j["status"] = root["status"] if root["status"] != "open" else "open"
        if root["status"] == "open":
            self._free(j)
            return 400, {"error": "job_node_open"}
        self._fill(j, root, src_hash, c["name"])
        for n in nodes[1:]:
            nid = self.add_job(j["exit"], n["seed"], ids[n["parent"]], status=n["status"])
            ids[n["seed"]] = nid
            if n["status"] != "open":
                self._fill(self.jobs[nid], n, src_hash, c["name"])
            elif n.get("level"):          # an open probe node that found a level: keep it, depth as best
                self.jobs[nid]["level"] = n["level"]
                self.jobs[nid]["best"] = n["level"]["depth"]
                self.stats["open_nodes_with_level"] += 1
            if n["status"] == "done":
                self.inserted_done.add(nid)
                self.stats["absorbed_done"] += 1
        j["client_token"] = None
        j["lease_until"] = None
        for n in nodes:
            if n["status"] == "split":
                self.stats["split_nodes"] += 1
                if n.get("level"):
                    self.stats["split_nodes_with_level"] += 1
        self.stats["max_nodes"] = max(self.stats["max_nodes"], len(nodes))
        if len(nodes) > 1:
            self.stats["tree_reports"] += 1
        else:
            self.stats["one_node_reports"] += 1
        cnt = {}
        for n in nodes:
            cnt[n["status"]] = cnt.get(n["status"], 0) + 1
        self.log("report job %d %s by %s%s: %d node(s) %s" % (jid, root["status"], c["name"],
                                                            " (foreign)" if foreign else "", len(nodes), cnt))
        return 200, {"ok": True, "nodes": len(nodes)}

    def _fill(self, j, n, src_hash, reporter=None):
        sm = n.get("summary") or {}
        j["reporter"] = reporter
        j["solver_calls"] = sm.get("solver_calls")
        j["states"] = sm.get("states")
        j["done_at"] = time.time()
        j["src_hash"] = src_hash
        j["states"] = sm.get("states")
        j["best"] = sm.get("best")
        j["elapsed_s"] = sm.get("elapsed")
        j["level"] = n.get("level")

    def report_batch(self, body):
        c = self.clients.get(body.get("token"))
        if not c:
            return 401, {"error": "unknown_token"}
        lst = body.get("reports")
        if not isinstance(lst, list) or len(lst) > 1000:
            return 400, {"error": "reports"}
        self.stats["batch_posts"] += 1
        self.stats["max_batch"] = max(self.stats["max_batch"], len(lst))
        results = []
        for r in lst:
            if not isinstance(r, dict):
                results.append({"error": "bad_report", "status": 400}); continue
            code, obj = self.report(r, token=c["token"])
            if code == 200:
                results.append(obj)
            else:
                results.append({"error": obj.get("error", "error"), "status": code, "detail": obj})
        return 200, {"results": results}

    def status(self):
        self.sweep()
        per_exit = {}
        for j in self.jobs.values():
            d = per_exit.setdefault(j["exit"], {"open": 0, "leased": 0, "done": 0, "split": 0})
            d[j["status"]] += 1
        cpu = sum((j["elapsed_s"] or 0) for j in self.jobs.values() if j["status"] in ("done", "split"))
        return 200, {"per_exit": per_exit, "cpu_hours": cpu / 3600.0, "exits": self.exits_summary(),
                     "active_clients": sum(1 for c in self.clients.values() if time.time() - c["last_seen"] < 180),
                     "clients": [{"name": c["name"], "workers": c["workers"], "paused": c["paused"],
                                  "boards": c["boards"], "last_seen": c["last_seen"]} for c in self.clients.values()],
                     "hashes": self.campaign["hashes"], "jobs": len(self.jobs), "stats": self.stats}

    def me_for(self, name):
        """Section 8 `me`: totals for every client with this name."""
        mine = [j for j in self.jobs.values() if j.get("reporter") == name and j["status"] in ("done", "split")]
        contrib = {}
        for j in self.jobs.values():
            if j.get("reporter") and j["status"] in ("done", "split"):
                contrib[j["reporter"]] = contrib.get(j["reporter"], 0) + (j["elapsed_s"] or 0)
        ranked = sorted(contrib.items(), key=lambda kv: -kv[1])
        rank = next((i + 1 for i, (nm, _) in enumerate(ranked) if nm == name), None)
        best = None
        for j in mine:
            lv = j.get("level")
            if lv and (best is None or lv["depth"] > best["moves"]):
                best = {"moves": lv["depth"], "code": lv["code"], "exit": j["exit"], "at": j["done_at"]}
        first = [cl["created_at"] for cl in self.clients.values() if cl["name"] == name]
        return {"jobs_done": len(mine), "splits": sum(1 for j in mine if j["status"] == "split"),
                "nodes_done": sum(1 for j in mine if j["status"] == "done"),
                "cpu_s": sum((j["elapsed_s"] or 0) for j in mine),
                "states": sum((j.get("states") or 0) for j in mine),
                "solver_calls": sum((j.get("solver_calls") or 0) for j in mine),
                "best": best, "rank": rank, "contributors": len(contrib),
                "first_seen": min(first) if first else None}

    def exits_summary(self):
        out = {}
        for ex in self.campaign["exits"]:
            js = [j for j in self.jobs.values() if j["exit"] == ex]
            roots = [j for j in js if j["parent_id"] is None]
            d = {"roots": len(roots), "roots_covered": sum(1 for j in roots if self.covered(j["id"])),
                 "open": 0, "leased": 0, "done": 0, "split": 0,
                 "cpu_s": sum((j["elapsed_s"] or 0) for j in js if j["status"] in ("done", "split")), "best": None}
            for j in js:
                d[j["status"]] += 1
                lv = j.get("level")
                if lv and (d["best"] is None or lv["depth"] > d["best"]["moves"]):
                    d["best"] = {"moves": lv["depth"], "code": lv["code"], "by": j.get("reporter"), "at": j["done_at"]}
            out[str(ex)] = d
        return out

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
                     "mismatches": [], "reports": len(self.reports), "stats": self.stats}


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
                    store.stats["report_posts"] += 1
                    code, obj = store.report(body)
                elif path == "/api/v2/reports":
                    code, obj = store.report_batch(body)
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
    ap.add_argument("--ramp-split-after-s", type=float, default=None, help="default: same as --split-after-s")
    ap.add_argument("--batch-interval-s", type=float, default=10)
    ap.add_argument("--lease-ahead-s", type=float, default=300)
    ap.add_argument("--absorb-total-s", type=float, default=120)
    ap.add_argument("--absorb-probe-s", type=float, default=10)
    ap.add_argument("--lease-cap", type=int, default=200)
    ap.add_argument("--heartbeat-s", type=float, default=30)
    ap.add_argument("--grid", default="5x5")
    ap.add_argument("--extra", default="--allow-exit-transit --num-holes 3")
    ap.add_argument("--exits", default="0,1,2")
    ap.add_argument("--hashes", default="fa4e" + "0" * 60)
    ap.add_argument("--max-clients", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--preregister", action="append", help="NAME:TOKEN client that exists at startup")
    opts = ap.parse_args()
    if opts.ramp_split_after_s is None:
        opts.ramp_split_after_s = opts.split_after_s
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
