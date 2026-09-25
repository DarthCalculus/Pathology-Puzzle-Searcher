#!/usr/bin/env python3
"""In-memory stand-in for the v2 server (jobs.js / v2routes.js) speaking protocol 3
(v2/PROTOCOL3.md section 4), stdlib only, for testing volunteer.py.

Close to jobs.js where the client is fragile (M93): 403 for unknown/revoked tokens,
410 campaign_closed, 426 client_too_old below min_client_version/min_protocol,
heartbeat renew / release-by-omission (with RELEASE_GRACE_S) / reclaim / drop,
deepest-first leasing with the exit preference, work stealing (never a job granted
within the holder's last --steal-protect-s or listed as running in its last
heartbeat, never to the client already holding it), tree-report validation
(parents, split children, summary.seed/exit, flags, unresolved candidates, states),
failure reports with fail_count and quarantine (3 failures from >= 2 clients, or 5) and no
regrant to the failing client within --fail-regrant-s, lease `reclaimed` (an open job the client
still listed at its last heartbeat goes back to it, never as a second copy in `jobs`),
campaign_state running/complete/closed, a body cap answered with 413, and several
campaigns (admin route to close one and open the next). Level codes are stored and returned
with rows joined by '\n', like jobs.js (M55).

  python3 fake_server.py --port 19131 --roots 8 --lease-s 6 --split-after-s 3 --hashes fa4e000...

Routes: POST /api/v2/register, /heartbeat, /lease, /report (one tree), /reports (batch,
per-report outcomes in order), /_admin (tests: close | complete | open | revoke | set | forget_clients);
GET /api/v2/status, /api/v2/audit[?campaign=ID] (carry `stats` counters used by test_client.sh).
"""
import argparse
import json
import random
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEED_RE = re.compile(r"^(?:[URDL][123])(?:,[URDL][123])*$")
CODE_RE = re.compile(r"^[0-9A-Za-z?]+(?:/[0-9A-Za-z?]+)*$")
TOKENS = [d + n for d in "URDL" for n in "123"]
NODES_MAX = 5000
CAND_MAX_PER_NODE = 1000
CAND_MAX_PER_REPORT = 20000
FAIL_REASON_RE = re.compile(r"^[a-z][a-z0-9_:.-]{0,39}$")
HEX_RE = re.compile(r"^[0-9a-f]{8,128}$")
BATCH_MAX = 1000
LEASE_HELD_EXTRA = 200


def vtuple(v):
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except (TypeError, ValueError):
        return (0,)


def toks_of(seed):
    return seed.split(",") if seed else []


class Store:
    def __init__(self, opts):
        self.lock = threading.Lock()
        self.opts = opts
        self.stats = {"lease_requests": 0, "lease_grants": 0, "report_posts": 0, "batch_posts": 0,
                      "report_bodies": 0, "tree_reports": 0, "one_node_reports": 0, "max_nodes": 0,
                      "absorbed_done": 0, "grants_of_absorbed": 0, "max_batch": 0, "max_batch_bytes": 0,
                      "split_nodes_with_level": 0, "open_nodes_with_level": 0, "split_nodes": 0,
                      "failures": 0, "quarantined": 0, "candidates": 0, "stopping_heartbeats": 0,
                      "http_413": 0, "http_426": 0, "http_410": 0, "http_403": 0, "registers": 0,
                      "dup_grants_sent": 0, "steals": 0, "released": 0, "drops": 0, "reclaimed": 0,
                      "rejected_reports": 0, "max_lease_n": 0, "max_held": 0}
        self.failure_reasons = {}
        self.last_heartbeat = {}          # name -> {holding, running, stopping, at}
        self.client_versions = set()
        self.candidates = []
        self.reports = []
        self.report_log = []              # first reports: [(depth, status), ...] per report (absorption tests)
        self.inserted_done = set()        # ids of nodes that arrived already done inside a tree report
        self.clients = {}
        self.campaigns = {}
        self.jobs = {}
        self.next_id = 1
        self.next_campaign = 1
        self.rng = random.Random(opts.seed)
        if not opts.no_campaign:
            self.open_campaign({})
        for spec in (opts.preregister or []):   # NAME:TOKEN client that exists at startup
            name, _, token = spec.partition(":")
            cid = max(self.campaigns) if self.campaigns else None
            self.clients[token] = self.new_client(token, name, 1, cid)

    def log(self, msg):
        sys.stderr.write("[fake_server %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()

    # ---- campaigns ----
    def open_campaign(self, spec):
        o = self.opts
        cid = self.next_campaign
        self.next_campaign += 1
        extra = spec.get("extra", o.extra)
        camp = {
            "id": cid, "title": spec.get("title", "fake campaign %d" % cid), "grid": spec.get("grid", o.grid),
            "extra": extra.split() if isinstance(extra, str) else list(extra),
            "exits": [int(x) for x in str(spec.get("exits", o.exits)).split(",")],
            "hashes": [h for h in str(spec.get("hashes", o.hashes)).split(",") if h],
            "max_clients": o.max_clients, "workers_max": 32, "status": "open",
            "job_target_s": o.split_after_s, "split_after_s": float(spec.get("split_after_s", o.split_after_s)),
            "lease_s": float(spec.get("lease_s", o.lease_s)), "paused_max_s": o.paused_max_s,
            "ramp_split_after_s": float(spec.get("split_after_s", o.ramp_split_after_s)),
            "batch_interval_s": o.batch_interval_s, "lease_ahead_s": o.lease_ahead_s,
            "absorb_total_s": o.absorb_total_s, "absorb_probe_s": o.absorb_probe_s,
            "lease_cap": o.lease_cap, "heartbeat_s": o.heartbeat_s,
            "min_window_s": spec.get("min_window_s", o.min_window_s),
            "split_after_nodes": spec.get("split_after_nodes", o.split_after_nodes),
            "min_client_version": spec.get("min_client_version", o.min_client_version),
            "min_protocol": int(spec.get("min_protocol", o.min_protocol)),
            "release": "v%s" % spec.get("min_client_version", o.min_client_version),
            "created_at": time.time(),
        }
        self.campaigns[cid] = camp
        roots = int(spec.get("roots", o.roots))
        layer = int(spec.get("layer", o.layer))
        for k in range(roots):
            ex = camp["exits"][k % len(camp["exits"])]
            seed = ",".join(self.rng.choice(TOKENS) for _ in range(layer))
            self.add_job(cid, ex, seed, None)
        if spec.get("empty_root", o.empty_root):
            self.add_job(cid, camp["exits"][0], "", None)
        self.log("campaign %d open: %d root(s), exits %s, extra %s" % (cid, roots, camp["exits"], camp["extra"]))
        return camp

    def current(self):
        """The newest campaign (whatever its status): what the status and audit routes show."""
        return self.campaigns[max(self.campaigns)] if self.campaigns else None

    def active(self):
        opened = [c for c in self.campaigns.values() if c["status"] == "open"]
        return max(opened, key=lambda c: c["id"]) if opened else None

    def public(self, c):
        keys = ("id", "title", "grid", "extra", "exits", "hashes", "max_clients", "workers_max", "job_target_s",
                "split_after_s", "ramp_split_after_s", "lease_s", "paused_max_s", "batch_interval_s", "lease_ahead_s",
                "absorb_total_s", "absorb_probe_s", "lease_cap", "heartbeat_s", "status", "min_client_version",
                "min_protocol", "release", "min_window_s", "split_after_nodes")
        return {k: c[k] for k in keys if c.get(k) is not None}

    def state_of(self, c):
        return {"open": "running", "complete": "complete", "closed": "closed"}[c["status"]]

    def update_complete(self, c):
        """PROTOCOL3 section 4: complete when nothing is open or leased, nothing is quarantined,
        and every root is covered."""
        if c["status"] != "open":
            return
        js = [j for j in self.jobs.values() if j["campaign_id"] == c["id"]]
        if any(j["status"] in ("open", "leased", "quarantined") for j in js):
            return
        roots = [j for j in js if j["parent_id"] is None]
        if roots and all(self.covered(j["id"]) for j in roots):
            c["status"] = "complete"
            self.log("campaign %d is complete" % c["id"])

    def add_job(self, cid, exit_, seed, parent_id, status="open"):
        jid = self.next_id
        self.next_id += 1
        self.jobs[jid] = {
            "id": jid, "campaign_id": cid, "exit": exit_, "seed": seed, "parent_id": parent_id,
            "depth": len(toks_of(seed)), "status": status,
            "client_token": None, "lease_until": None, "leased_at": None,
            "done_at": None, "src_hash": None, "states": None, "best": None,
            "level": None, "elapsed_s": None, "cpu_s": None, "children": [], "fail_count": 0,
            "fail_tokens": set(), "fail_reasons": [], "fail_at": {}, "created_at": time.time(),
        }
        if parent_id is not None:
            self.jobs[parent_id]["children"].append(jid)
        return jid

    def new_client(self, token, name, workers, cid):
        return {"token": token, "name": name, "workers": workers, "created_at": time.time(), "campaign_id": cid,
                "last_seen": time.time(), "paused": False, "boards": [], "revoked": False, "running": set()}

    def sweep(self):
        now = time.time()
        for j in self.jobs.values():
            if j["status"] == "leased" and j["lease_until"] is not None and j["lease_until"] < now:
                c = self.clients.get(j["client_token"])
                camp = self.campaigns[j["campaign_id"]]
                if c and c["paused"] and j["leased_at"] and now - j["leased_at"] < camp["paused_max_s"]:
                    continue
                self.log("lease expired: job %d (%s)" % (j["id"], j["seed"]))
                self._free(j)

    def _free(self, j):
        j["status"] = "open"
        j["client_token"] = None
        j["lease_until"] = None
        j["leased_at"] = None

    def _lease(self, j, c, camp, t):
        j["status"] = "leased"
        j["client_token"] = c["token"]
        j["leased_at"] = t
        j["lease_until"] = t + camp["lease_s"]

    # ---- gates ----
    def version_gate(self, body, camp=None, route=None):
        camp = camp or self.active() or self.current()
        min_v = camp["min_client_version"] if camp else self.opts.min_client_version
        min_p = camp["min_protocol"] if camp else self.opts.min_protocol
        cv, pv = body.get("client_version"), body.get("protocol")
        if cv is not None:
            self.client_versions.add(str(cv)[:20])
        if cv is None or vtuple(cv) < vtuple(min_v) or not isinstance(pv, int) or pv < min_p:
            rel = "v%s" % min_v
            obj = {"error": "client_too_old", "release": rel, "min_client_version": min_v, "min_protocol": min_p,
                   "message": "This volunteer client (%s) is too old for the running campaign. Update it: "
                              "git fetch && git checkout %s && ./build_pgo.sh -o backsearch_worker_nt --no-torch"
                              % (cv or "no version", rel)}
            if cv is None and pv is None and route in ("register", "heartbeat", "lease"):
                # like jobs.js: a pre-protocol-3 client gets a status it already treats as fatal
                return (400 if route == "register" else 403), obj
            self.stats["http_426"] += 1
            return 426, obj
        return None

    def require(self, token):
        c = self.clients.get(token) if isinstance(token, str) else None
        if not c:
            self.stats["http_403"] += 1
            return None, None, (403, {"error": "unknown_token"})
        if c["revoked"]:
            self.stats["http_403"] += 1
            return None, None, (403, {"error": "revoked", "revoked": True})
        camp = self.campaigns.get(c["campaign_id"])
        if not camp or camp["status"] == "closed":
            self.stats["http_410"] += 1
            return None, None, (410, {"error": "campaign_closed", "campaign_state": "closed"})
        if camp["status"] == "complete":
            # like jobs.js: a complete campaign's tokens get 200 until a newer campaign opens
            nxt = self.active()
            if nxt and nxt["id"] != camp["id"]:
                self.stats["http_410"] += 1
                return None, None, (410, {"error": "campaign_closed", "campaign_state": "complete", "next_campaign": nxt["id"]})
        return c, camp, None

    # ---- routes ----
    def register(self, body):
        g = self.version_gate(body, route="register")
        if g:
            return g
        self.stats["registers"] += 1
        name = str(body.get("name", ""))[:40]
        if not name:
            return 400, {"error": "name must be 1..40 characters"}
        camp = self.active()
        if not camp:
            last = self.current()
            obj = {"error": "no_campaign", "message": "No campaign is running right now."}
            if last:
                obj["last_campaign"] = {"id": last["id"], "title": last["title"], "campaign_state": self.state_of(last)}
            return 404, obj
        workers = max(1, min(32, int(body.get("workers", 1))))
        active = [c for c in self.clients.values() if c["campaign_id"] == camp["id"]
                  and time.time() - c["last_seen"] < 180 and not c["revoked"]]
        if len(active) >= camp["max_clients"]:
            return 503, {"error": "full", "queue_position": 1}
        token = secrets.token_hex(16)
        self.clients[token] = self.new_client(token, name, workers, camp["id"])
        self.log("register %s workers=%d token=%s campaign=%d" % (name, workers, token[:8], camp["id"]))
        return 200, {"ok": True, "token": token, "campaign": self.public(camp), "campaign_state": self.state_of(camp)}

    def heartbeat(self, body):
        g = self.version_gate(body, route="heartbeat")
        if g:
            return g
        c, camp, err = self.require(body.get("token"))
        if err:
            return err
        t = time.time()
        c["last_seen"] = t
        c["workers"] = max(1, min(32, int(body.get("workers", c["workers"]))))
        c["paused"] = bool(body.get("paused", False))
        boards = body.get("boards", [])
        if isinstance(boards, list):
            c["boards"] = boards[:max(1, c["workers"])]
        if isinstance(body.get("running"), list):
            c["running"] = set(x for x in body["running"] if isinstance(x, int))
        if isinstance(body.get("holding"), list):
            c["holding"] = set(x for x in body["holding"] if isinstance(x, int))
        stopping = body.get("stopping") is True
        holding_list = body.get("holding") if isinstance(body.get("holding"), list) else None
        self.last_heartbeat[c["name"]] = {"holding": holding_list, "running": body.get("running"),
                                          "stopping": stopping, "at": t, "token": c["token"][:8]}
        if stopping:
            self.stats["stopping_heartbeats"] += 1
        self.sweep()
        holding = set(x for x in holding_list if isinstance(x, int)) if holding_list is not None else None
        leases, released = [], []
        for j in sorted((j for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"]),
                        key=lambda j: j["id"]):
            # release by omission (jobs.js); a stopping client releases everything it does not list at once
            if holding is not None and j["id"] not in holding and (stopping or t - j["leased_at"] > self.opts.release_grace_s):
                self._free(j)
                released.append(j["id"])
                continue
            want = t + camp["lease_s"]
            if c["paused"]:
                want = min(want, j["leased_at"] + camp["lease_s"] + camp["paused_max_s"])
            if want > j["lease_until"]:
                j["lease_until"] = want
            leases.append(j["id"])
        drop, reclaimed = [], []
        if holding is not None:
            held = set(leases)
            for jid in holding_list:
                if jid in held or not isinstance(jid, int):
                    continue
                j = self.jobs.get(jid)
                if j and j["campaign_id"] == camp["id"] and j["status"] == "open" and not stopping:
                    self._lease(j, c, camp, t)
                    reclaimed.append(jid)
                    leases.append(jid)
                else:
                    drop.append(jid)
        self.stats["released"] += len(released)
        self.stats["drops"] += len(drop)
        self.stats["reclaimed"] += len(reclaimed)
        if released:
            self.log("heartbeat %s released %s" % (c["name"], released))
        self.update_complete(camp)
        res = {"ok": True, "leases": leases, "drop": drop, "reclaimed": reclaimed, "released": released,
               "campaign": self.public(camp), "campaign_state": self.state_of(camp),
               "me": self.me_for(c["name"], camp), "exits": self.exits_summary(camp)}
        if self.opts.heartbeat_window:
            # the current window (M108 fix note: jobs.js does not send this yet); clients use it for
            # windows that start after it arrives
            res["window"] = {"split_after_s": self.window_now(camp), "mode": "full"}
        if camp["status"] == "complete" and self.opts.result:
            res["result"] = {e: {"exit": int(e), "best": d["best"]["moves"] if d["best"] else None,
                                 "clean": d["roots_covered"] == d["roots"], "exact": d["roots_covered"] == d["roots"]}
                             for e, d in res["exits"].items()}
        return 200, res

    def lease(self, body):
        g = self.version_gate(body, route="lease")
        if g:
            return g
        c, camp, err = self.require(body.get("token"))
        if err:
            return err
        t = time.time()
        c["last_seen"] = t
        self.sweep()
        pref = body.get("exit")
        if pref is not None and (not isinstance(pref, int) or pref not in camp["exits"]):
            return 400, {"error": "exit is not part of the campaign"}
        n = max(0, min(camp["lease_cap"], int(body.get("n", 1))))
        held = sum(1 for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"])
        n = min(n, 3 * max(1, c["workers"]) + LEASE_HELD_EXTRA - held)
        self.stats["lease_requests"] += 1
        granted = []
        reclaimed = []
        fallback = False
        if n > 0 and camp["status"] == "open":
            # like jobs.js: what the client listed as running or holding at its last heartbeat is never
            # granted to it a second time; an open one (its lease lapsed) goes back as `reclaimed`
            mine = set(c["running"]) | set(c.get("holding") or ())
            open_by_exit = {}
            for j in self.jobs.values():
                if j["campaign_id"] != camp["id"] or j["status"] != "open":
                    continue
                if t - j["fail_at"].get(c["token"], -1e18) < self.opts.fail_regrant_s:
                    continue                  # failed here recently: neither granted nor reclaimed (jobs.js SQL)
                if j["id"] in mine:
                    self._lease(j, c, camp, t)
                    reclaimed.append(j["id"])
                    continue
                open_by_exit.setdefault(j["exit"], []).append(j)
            order = sorted(open_by_exit.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            if pref is not None:
                order.sort(key=lambda kv: kv[0] != pref)
            for ex, lst in order:
                for j in sorted(lst, key=lambda j: (-j["depth"], j["created_at"], j["id"])):
                    if len(granted) >= n:
                        break
                    self._lease(j, c, camp, t)
                    granted.append({"id": j["id"], "exit": j["exit"], "seed": j["seed"], "depth": j["depth"]})
                    if pref is not None and j["exit"] != pref:
                        fallback = True
                    if j["id"] in self.inserted_done:
                        self.stats["grants_of_absorbed"] += 1
                if len(granted) >= n:
                    break
            if not granted and n > 0 and not self.opts.no_steal:
                granted += self.steal(c, camp, n, t)
        if self.opts.dup_grants:
            # test hook (M19): repeat one job this client already holds
            # (one it held before this request, else one granted just now: the same id twice)
            mine = [j for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"]
                    and j["id"] not in [g["id"] for g in granted]]
            j = min(mine, key=lambda j: j["id"]) if mine else (self.jobs[granted[0]["id"]] if granted else None)
            if j is not None:
                granted.append({"id": j["id"], "exit": j["exit"], "seed": j["seed"], "depth": j["depth"]})
                self.stats["dup_grants_sent"] += 1
        self.stats["lease_grants"] += len(granted)
        self.stats["max_lease_n"] = max(self.stats["max_lease_n"], int(body.get("n", 1)))
        held_now = sum(1 for j in self.jobs.values() if j["status"] == "leased" and j["client_token"] == c["token"])
        self.stats["max_held"] = max(self.stats["max_held"], held_now)
        sa = self.window_now(camp)
        self.update_complete(camp)
        self.log("lease %s n=%d exit=%s -> %s (split_after_s %g%s)" % (c["name"], n, pref, [g["id"] for g in granted], sa,
                                                                        ", fallback" if fallback else ""))
        res = {"ok": True, "jobs": granted, "reclaimed": reclaimed, "split_after_s": sa, "campaign_state": self.state_of(camp)}
        if self.opts.lease_absorb_total_s is not None:
            # like jobs.js windowFor (M109): the lease's own absorption budget, 0 in the endgame
            res["absorb_total_s"] = self.opts.lease_absorb_total_s
            res["window_mode"] = "endgame" if self.opts.lease_absorb_total_s == 0 else "full"
        if fallback:
            res["exit_fallback"] = True
        return 200, res

    def window_now(self, camp):
        open_jobs = sum(1 for j in self.jobs.values() if j["campaign_id"] == camp["id"] and j["status"] == "open")
        active_workers = sum(cl["workers"] for cl in self.clients.values() if cl["campaign_id"] == camp["id"]
                             and time.time() - cl["last_seen"] < 180 and not cl["revoked"])
        return camp["ramp_split_after_s"] if open_jobs < 3 * active_workers else camp["split_after_s"]

    def steal(self, c, camp, need, t):
        """PROTOCOL3 section 4: take queued jobs from the client holding the most, never a job
        granted or stolen within the holder's last steal_protect_s, never one it listed as running."""
        cand = []
        held_by = {}
        for j in self.jobs.values():
            if j["campaign_id"] != camp["id"] or j["status"] != "leased" or j["client_token"] == c["token"]:
                continue
            holder = self.clients.get(j["client_token"])
            if holder is None or j["id"] in holder["running"] or t - (j["leased_at"] or t) < self.opts.steal_protect_s \
                    or t - j["fail_at"].get(c["token"], -1e18) < self.opts.fail_regrant_s:
                continue
            held_by[j["client_token"]] = held_by.get(j["client_token"], 0) + 1
            cand.append(j)
        cand.sort(key=lambda j: (-held_by[j["client_token"]], -(j["leased_at"] or 0), j["id"]))
        out = []
        for j in cand[:need]:
            self._lease(j, c, camp, t)
            out.append({"id": j["id"], "exit": j["exit"], "seed": j["seed"], "depth": j["depth"], "stolen": True})
        self.stats["steals"] += len(out)
        return out

    def failure(self, r, c, camp):
        jid = r.get("job", r.get("job_id"))
        if not isinstance(jid, int) or jid < 1:
            return 400, {"error": "job required"}
        if not isinstance(r.get("failed"), str) or not FAIL_REASON_RE.match(r["failed"]):
            return 400, {"error": "failed must be a short lower-case reason", "job_id": jid}
        if r.get("src_hash") is not None and not HEX_RE.match(str(r["src_hash"])):
            return 400, {"error": "src_hash must be hex", "job_id": jid}
        j = self.jobs.get(jid)
        if not j or j["campaign_id"] != camp["id"]:
            return 404, {"error": "unknown_job", "job_id": jid}
        reason = r["failed"]
        self.stats["failures"] += 1
        self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
        j["fail_count"] += 1
        j["fail_tokens"].add(c["token"])
        j["fail_at"][c["token"]] = time.time()
        j["fail_reasons"].append(reason)
        q = False
        if j["status"] in ("leased", "open") and (j["client_token"] == c["token"] or j["status"] == "open"):
            if j["fail_count"] >= 5 or (j["fail_count"] >= 3 and len(j["fail_tokens"]) >= 2):
                j["status"] = "quarantined"
                j["client_token"] = None
                j["lease_until"] = None
                self.stats["quarantined"] += 1
                q = True
            else:
                self._free(j)
        self.log("failure on job %d (%s) from %s: %s -> %s" % (jid, j["seed"], c["name"], reason, j["status"]))
        return 200, {"ok": True, "job_id": jid, "failed": True, "fail_count": j["fail_count"], "quarantined": q}

    def check_flags(self, flags, camp, job):
        if not isinstance(flags, dict):
            return None
        want = {"grid": camp["grid"], "exit": job["exit"], "transit": 1, "block_on_exit": 0, "bulk_walk": 1, "min_walls": 0}
        ex = camp["extra"]
        for flag, key in (("--num-holes", "max_holes"), ("--num-blocks", "max_blocks"), ("--min-walls", "min_walls")):
            if flag in ex:
                try:
                    want[key] = int(ex[ex.index(flag) + 1])
                except (ValueError, IndexError):
                    pass
        for k, v in want.items():
            if k in flags and flags[k] != v:
                return "flags.%s=%s disagrees with the campaign (%s)" % (k, flags[k], v)
        return None

    def report(self, body, c, camp):
        self.stats["report_bodies"] += 1
        try:
            jid = int(body.get("job_id"))
        except Exception:
            return 400, {"error": "job_id required"}
        j = self.jobs.get(jid)
        if not j or j["campaign_id"] != camp["id"]:
            return 404, {"error": "unknown_job", "job_id": jid}
        src_hash = str(body.get("src_hash", ""))
        nodes = body.get("nodes")
        if not isinstance(nodes, list) or not nodes or len(nodes) > NODES_MAX:
            return 400, {"error": "nodes must be a list of 1..%d nodes" % NODES_MAX}
        self.reports.append({"job_id": jid, "token": c["token"], "src_hash": src_hash, "nodes": len(nodes),
                             "received_at": time.time(), "campaign_id": camp["id"],
                             "client_version": body.get("client_version")})
        if j["status"] in ("done", "split"):
            self.log("dup report for job %d" % jid)
            return 200, {"ok": True, "dup": True, "job_id": jid}
        foreign = j["client_token"] != c["token"]
        mine = j["client_token"] == c["token"] or j["status"] == "open"

        def give_back():
            if mine and j["status"] == "leased":
                self._free(j)

        if src_hash not in camp["hashes"]:
            give_back()
            self.log("unknown hash %s on job %d" % (src_hash[:12], jid))
            return 409, {"error": "unknown_hash", "job_id": jid}
        seed_toks = toks_of(j["seed"])
        seen = {}
        children_of = {}
        cands = []
        for k, n in enumerate(nodes):
            def bad(msg):
                give_back()
                return 400, {"error": "node %d: %s" % (k, msg), "job_id": jid}
            if not isinstance(n, dict) or not isinstance(n.get("seed"), str):
                return bad("must be an object with a seed")
            s = n["seed"]
            if not (SEED_RE.match(s) or (s == "" and k == 0 and j["seed"] == "")):
                return bad("bad seed")
            toks = toks_of(s)
            if toks[:len(seed_toks)] != seed_toks or (k > 0 and len(toks) <= len(seed_toks)):
                return bad("seed does not extend the job seed")
            if k == 0 and s != j["seed"]:
                return bad("the first node must be the job itself")
            if s in seen:
                return bad("duplicate seed")
            st = n.get("status")
            if st not in ("done", "split", "open"):
                return bad("status must be done|split|open")
            if k == 0 and st == "open":
                return bad("the job itself must be done or split")
            if k > 0:
                p = n.get("parent")
                if p not in seen:
                    return bad("parent is neither the job nor an earlier node")
                if nodes[seen[p]]["status"] != "split":
                    return bad("parent is %s; only split nodes have children" % nodes[seen[p]]["status"])
                if toks_of(p) != toks[:len(toks_of(p))]:
                    return bad("seed does not extend its parent")
            sm = n.get("summary")
            if st == "open" and sm is not None:
                return bad("an open node has no summary")
            if st == "done" and not (isinstance(sm, dict) and sm.get("status") == "exhausted"):
                return bad("done needs a summary with status exhausted")
            if isinstance(sm, dict):
                if (st == "done") != (sm.get("status") == "exhausted"):
                    return bad("summary status does not match the node")
                if "seed" in sm and sm["seed"] != s:
                    return bad("summary.seed does not match the node")
                if "exit" in sm and sm["exit"] != j["exit"]:
                    return bad("summary.exit does not match the job")
                if st == "done" and not (isinstance(sm.get("states"), int) and sm["states"] >= 1):
                    return bad("an exhausted run checks at least its seed (states >= 1)")
                if not self.opts.lax:      # like jobs.js sanitizeSummary(strict): protocol-3 fields are required
                    if not (isinstance(sm.get("protocol"), int) and sm["protocol"] >= 3):
                        return bad("summary.protocol must be 3 (a protocol-3 worker is required)")
                    if not isinstance(sm.get("flags"), dict):
                        return bad("summary.flags is required (protocol 3)")
                    if not isinstance(sm.get("unresolved"), int):
                        return bad("summary.unresolved is required (protocol 3)")
                fm = self.check_flags(sm.get("flags"), camp, j)
                if fm:
                    return bad(fm)
            lv = n.get("level")
            if lv is not None and not (isinstance(lv, dict) and isinstance(lv.get("depth"), int)
                                       and isinstance(lv.get("code"), str) and 0 < len(lv["code"]) <= 2000):
                return bad("bad level")
            if lv is not None:
                # like jobs.js parseLevel: rows split on '/' or newlines, stored joined with '\n' (M55)
                n = dict(n, level=dict(lv, code="\n".join(r for r in re.split(r"[/\n]+", lv["code"].strip()) if r)))
                nodes[k] = n
            un = n.get("unresolved")
            if un is not None:
                if st == "open" or not isinstance(un, list) or len(un) > CAND_MAX_PER_NODE:
                    return bad("unresolved must be a list (<= %d) on a done or split node" % CAND_MAX_PER_NODE)
                for u in un:
                    if not (isinstance(u, dict) and isinstance(u.get("depth"), int) and u["depth"] >= 1
                            and isinstance(u.get("code"), str) and CODE_RE.match(u["code"])
                            and isinstance(u.get("path"), str) and SEED_RE.match(u["path"])
                            and u.get("cause") in ("pq", "probe", "big")
                            and toks_of(u["path"])[:len(toks)] == toks
                            and len(toks_of(u["path"])) == u["depth"]):
                        return bad("bad unresolved candidate")
                    cands.append(dict(u, job_id=jid, exit=j["exit"], campaign_id=camp["id"], status="open"))
            if isinstance(sm, dict) and isinstance(sm.get("unresolved"), int) and sm["unresolved"] != len(un or []):
                return bad("summary.unresolved is %d but %d candidate(s) were forwarded" % (sm["unresolved"], len(un or [])))
            if len(cands) > CAND_MAX_PER_REPORT:
                return bad("more than %d unresolved candidates in one report" % CAND_MAX_PER_REPORT)
            seen[s] = k
            if k > 0:
                children_of.setdefault(n["parent"], []).append(k)
        for k, n in enumerate(nodes):
            if n["status"] == "split" and not children_of.get(n["seed"]):
                give_back()
                return 400, {"error": "node %d: split without any child in the list" % k, "job_id": jid}
        # insert in one go
        ids = {j["seed"]: jid}
        root = nodes[0]
        j["status"] = root["status"]
        self._fill(j, root, src_hash, c["name"])
        for n in nodes[1:]:
            nid = self.add_job(camp["id"], j["exit"], n["seed"], ids[n["parent"]], status=n["status"])
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
        self.candidates.extend(cands)
        self.stats["candidates"] += len(cands)
        for n in nodes:
            if n["status"] == "split":
                self.stats["split_nodes"] += 1
                if n.get("level"):
                    self.stats["split_nodes_with_level"] += 1
        if len(self.report_log) < 200:
            self.report_log.append({"job_id": jid, "depth": len(seed_toks),
                                    "nodes": [[len(toks_of(n["seed"])), n["status"],
                                               (n.get("summary") or {}).get("elapsed")] for n in nodes]})
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
        self.update_complete(camp)
        return 200, {"ok": True, "job_id": jid, "nodes": len(nodes)}

    def _fill(self, j, n, src_hash, reporter=None):
        sm = n.get("summary") or {}
        j["reporter"] = reporter
        j["solver_calls"] = sm.get("solver_calls")
        j["done_at"] = time.time()
        j["src_hash"] = src_hash
        j["states"] = sm.get("states")
        j["best"] = sm.get("best")
        j["elapsed_s"] = sm.get("elapsed")
        j["cpu_s"] = sm.get("cpu_s")
        j["level"] = n.get("level")

    def one_report(self, body):
        g = self.version_gate(body)
        if g:
            return g
        c, camp, err = self.require(body.get("token"))
        if err:
            return err
        c["last_seen"] = time.time()
        if "failed" in body:
            return self.failure(body, c, camp)
        return self.report(body, c, camp)

    def report_batch(self, body, nbytes):
        g = self.version_gate(body)
        if g:
            return g
        c, camp, err = self.require(body.get("token"))
        if err:
            return err
        c["last_seen"] = time.time()
        lst = body.get("reports")
        if not isinstance(lst, list) or not lst:
            return 400, {"error": "reports must be a non-empty list"}
        if len(lst) > BATCH_MAX:
            return 400, {"error": "more than %d reports in one batch" % BATCH_MAX}
        self.stats["batch_posts"] += 1
        self.stats["max_batch"] = max(self.stats["max_batch"], len(lst))
        self.stats["max_batch_bytes"] = max(self.stats["max_batch_bytes"], nbytes)
        results = []
        for r in lst:
            if not isinstance(r, dict):
                results.append({"ok": False, "code": 400, "error": "report must be an object"})
                continue
            code, obj = self.failure(r, c, camp) if "failed" in r else self.report(r, c, camp)
            if code == 200:
                results.append(obj)
            else:
                self.stats["rejected_reports"] += 1
                results.append(dict(obj, ok=False, code=code))
        return 200, {"ok": True, "results": results}

    def admin(self, body):
        op = body.get("op")
        if op == "close":
            camp = self.active()
            if camp:
                camp["status"] = "closed"
                self.log("campaign %d closed by the owner" % camp["id"])
            return 200, {"ok": True, "closed": camp["id"] if camp else None}
        if op == "complete":           # tests: mark the open campaign complete now
            camp = self.active()
            if camp:
                camp["status"] = "complete"
                self.log("campaign %d marked complete by the owner (test)" % camp["id"])
            return 200, {"ok": True, "completed": camp["id"] if camp else None}
        if op == "open":
            camp = self.open_campaign(body)
            return 200, {"ok": True, "campaign": self.public(camp)}
        if op == "revoke":
            n = 0
            for cl in self.clients.values():
                if cl["name"] == body.get("name"):
                    cl["revoked"] = True
                    n += 1
            return 200, {"ok": True, "revoked": n}
        if op == "set":
            camp = self.current()
            for k, v in (body.get("params") or {}).items():
                camp[k] = v
            return 200, {"ok": True, "campaign": self.public(camp)}
        if op == "forget_clients":
            self.clients.clear()
            return 200, {"ok": True}
        return 400, {"error": "unknown op"}

    def status(self):
        self.sweep()
        camp = self.current()
        per_exit = {}
        js = [j for j in self.jobs.values() if camp and j["campaign_id"] == camp["id"]]
        for j in js:
            d = per_exit.setdefault(j["exit"], {"open": 0, "leased": 0, "done": 0, "split": 0, "quarantined": 0})
            d[j["status"]] += 1
        cpu = sum((j["elapsed_s"] or 0) for j in js if j["status"] in ("done", "split"))
        return 200, {"campaign": self.public(camp) if camp else None, "per_exit": per_exit, "cpu_hours": cpu / 3600.0,
                     "exits": self.exits_summary(camp) if camp else {},
                     "active_clients": sum(1 for c in self.clients.values() if time.time() - c["last_seen"] < 180),
                     "clients": [{"name": c["name"], "workers": c["workers"], "paused": c["paused"],
                                  "boards": c["boards"], "last_seen": c["last_seen"]} for c in self.clients.values()],
                     "hashes": camp["hashes"] if camp else [], "jobs": len(js), "stats": self.stats}

    def me_for(self, name, camp):
        """Section 8 `me`: totals for every client with this name in this campaign."""
        js = [j for j in self.jobs.values() if j["campaign_id"] == camp["id"]]
        mine = [j for j in js if j.get("reporter") == name and j["status"] in ("done", "split")]
        contrib = {}
        for j in js:
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
        return {"name": name, "jobs_done": len(mine), "splits": sum(1 for j in mine if j["status"] == "split"),
                "nodes_done": sum(1 for j in mine if j["status"] == "done"),
                "cpu_s": sum((j["elapsed_s"] or 0) for j in mine),
                "states": sum((j.get("states") or 0) for j in mine),
                "solver_calls": sum((j.get("solver_calls") or 0) for j in mine),
                "best": best, "rank": rank, "contributors": len(contrib),
                "first_seen": min(first) if first else None}

    def exits_summary(self, camp):
        out = {}
        for ex in camp["exits"]:
            js = [j for j in self.jobs.values() if j["campaign_id"] == camp["id"] and j["exit"] == ex]
            roots = [j for j in js if j["parent_id"] is None]
            covered = sum(1 for j in roots if self.covered(j["id"]))
            d = {"exit": ex, "roots": len(roots), "roots_covered": covered,
                 "clean": bool(roots) and covered == len(roots), "exact": bool(roots) and covered == len(roots),
                 "open": 0, "leased": 0, "done": 0, "split": 0, "quarantined": 0,
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

    def audit(self, cid=None):
        self.sweep()
        camp = self.campaigns.get(cid) if cid else self.current()
        if not camp:
            return 404, {"error": "no_campaign"}
        js = [j for j in self.jobs.values() if j["campaign_id"] == camp["id"]]
        roots = [j for j in js if j["parent_id"] is None]
        uncovered = [j["id"] for j in roots if not self.covered(j["id"])]
        counts = {"open": 0, "leased": 0, "done": 0, "split": 0, "quarantined": 0}
        for j in js:
            counts[j["status"]] += 1
        reps = [r for r in self.reports if r["campaign_id"] == camp["id"]]
        return 200, {"campaign_id": camp["id"], "campaign_state": self.state_of(camp), "roots": len(roots),
                     "uncovered_roots": uncovered, "counts": counts,
                     "quarantined": [{"id": j["id"], "seed": j["seed"], "reasons": j["fail_reasons"]} for j in js
                                     if j["status"] == "quarantined"],
                     "unknown_hash_jobs": [r["job_id"] for r in reps if r["src_hash"] not in camp["hashes"]],
                     "mismatches": [], "reports": len(reps), "stats": self.stats,
                     "failure_reasons": self.failure_reasons, "last_heartbeat": self.last_heartbeat,
                     "client_versions": sorted(self.client_versions),
                     "candidates": self.candidates[:50],
                     "report_log": [r for r in self.report_log if self.jobs[r["job_id"]]["campaign_id"] == camp["id"]][:100],
                     "fail_counts": {str(j["id"]): j["fail_count"] for j in js if j["fail_count"]},
                     "clean": bool(roots) and not uncovered and counts["quarantined"] == 0}


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, obj, close=False):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, qs = self.path.partition("?")
            with store.lock:
                if path == "/api/v2/status":
                    code, obj = store.status()
                elif path == "/api/v2/audit":
                    cid = None
                    for kv in qs.split("&"):
                        k, _, v = kv.partition("=")
                        if k == "campaign" and v.isdigit():
                            cid = int(v)
                    code, obj = store.audit(cid)
                else:
                    code, obj = 404, {"error": "not_found"}
            self._send(code, obj)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            path = self.path.split("?")[0]
            limit = store.opts.body_max if path in ("/api/v2/report", "/api/v2/reports") else 64 * 1024 * 1024
            if n > limit:
                # like v2routes.js: refuse on the declared length, answer 413, close the connection
                with store.lock:
                    store.stats["http_413"] += 1
                try:
                    self.rfile.read(n)            # drain, so the client reliably reads the 413 (tests)
                except OSError:
                    pass
                return self._send(413, {"error": "body too large"}, close=True)
            raw = self.rfile.read(n) if n else b""
            try:
                body = json.loads(raw.decode() or "{}")
                if not isinstance(body, dict):
                    raise ValueError
            except Exception:
                return self._send(400, {"error": "invalid JSON"})
            with store.lock:
                if path == "/api/v2/register":
                    code, obj = store.register(body)
                elif path == "/api/v2/heartbeat":
                    code, obj = store.heartbeat(body)
                elif path == "/api/v2/lease":
                    code, obj = store.lease(body)
                elif path == "/api/v2/report":
                    store.stats["report_posts"] += 1
                    code, obj = store.one_report(body)
                elif path == "/api/v2/reports":
                    code, obj = store.report_batch(body, n)
                elif path == "/api/v2/_admin":
                    code, obj = store.admin(body)
                else:
                    code, obj = 404, {"error": "not_found"}
            self._send(code, obj)

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19131)
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
    ap.add_argument("--lease-absorb-total-s", type=float, default=None,
                    help="send this absorb_total_s (and window_mode) in every lease response, like jobs.js")
    ap.add_argument("--lease-cap", type=int, default=200)
    ap.add_argument("--min-window-s", type=float, default=None, help="campaign client param min_window_s (N18)")
    ap.add_argument("--split-after-nodes", type=int, default=None, help="campaign client param split_after_nodes (N18)")
    ap.add_argument("--heartbeat-window", action="store_true",
                    help="heartbeat responses carry the current window {split_after_s, mode} (jobs.js does not yet)")
    ap.add_argument("--heartbeat-s", type=float, default=30)
    ap.add_argument("--grid", default="5x5")
    ap.add_argument("--extra", default="--allow-exit-transit --num-holes 3")
    ap.add_argument("--exits", default="0,1,2")
    ap.add_argument("--hashes", default="fa4e" + "0" * 60)
    ap.add_argument("--max-clients", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--preregister", action="append", help="NAME:TOKEN client that exists at startup")
    ap.add_argument("--release-grace-s", type=float, default=120.0)
    ap.add_argument("--steal-protect-s", type=float, default=60.0)
    ap.add_argument("--fail-regrant-s", type=float, default=3600.0,
                    help="a job is not granted again to a client that failed it within this window (jobs.js)")
    ap.add_argument("--no-steal", action="store_true")
    ap.add_argument("--min-client-version", default="3.0.0")
    ap.add_argument("--min-protocol", type=int, default=3)
    ap.add_argument("--body-max", type=int, default=4 * 1024 * 1024, help="report body cap (413 above)")
    ap.add_argument("--dup-grants", action="store_true", help="test hook: re-grant a job the client already holds")
    ap.add_argument("--empty-root", action="store_true", help="add a root job with the empty seed")
    ap.add_argument("--no-campaign", action="store_true", help="start without an open campaign")
    ap.add_argument("--result", action="store_true", help="also send a per-exit `result` object on complete (jobs.js does not)")
    ap.add_argument("--lax", action="store_true", help="accept summaries without the protocol-3 fields")
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
