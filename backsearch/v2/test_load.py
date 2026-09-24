#!/usr/bin/env python3
"""Server load test (DESIGN.md §7, review H8/M30): simulated protocol-3 clients, no real workers.

Starts nothing itself: point it at a running local server that has an open protocol-3
campaign with roots (test_load.sh seeds one). Each simulated client behaves like
volunteer.py under the load rules: one lease request and one batched report POST every
`--interval` seconds, one heartbeat per 30 s (with the held and running job ids, as the real
client sends them), reporting tree reports whose node count follows a heavy-tailed
distribution (mostly one-node reports, some 300-node splits; every finished node carries its
SUMMARY, and its level when it has a best, as the worker prints them). `--status-poll S` adds
one dashboard poller of /api/v2/status every S seconds (hunt.html polls every 3 s). Measures
server request latency and throughput; prints rows written per second.

  python3 test_load.py --server http://127.0.0.1:3199 --clients 40 --workers 32 --seconds 60 --hash <64 hex>
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--server', default='http://127.0.0.1:3199'); ap.add_argument('--clients', type=int, default=40)
ap.add_argument('--workers', type=int, default=32); ap.add_argument('--seconds', type=float, default=60)
ap.add_argument('--interval', type=float, default=10); ap.add_argument('--job-s', type=float, default=2.0,
    help='mean simulated job duration (2 s = the worst case the rules allow after absorption)')
ap.add_argument('--hash', required=True, help='a whitelisted src_hash')
ap.add_argument('--status-poll', type=float, default=0, help='poll /api/v2/status every S seconds (0 = off)')
a = ap.parse_args()

VER = {"client_version": "3.0.0", "protocol": 3}
FLAGS = {"grid": "5x5", "transit": 1, "block_on_exit": 0, "bulk_walk": 1, "max_holes": 3, "max_blocks": 32, "min_walls": 0}
LEVELS = {12: '00000/00000/00300/00000/40000', 0: '30000/00000/00500/00000/00004'}   # a valid level for the exits that have one here

def post(path, body, timeout=30):
    req = urllib.request.Request(a.server + path, data=json.dumps({**VER, **body}).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: out = json.loads(r.read().decode()); code = r.status
    except urllib.error.HTTPError as e:
        out = json.loads(e.read().decode() or '{}'); code = e.code
    return code, out, time.time() - t0

def get(path, timeout=30):
    t0 = time.time()
    try:
        with urllib.request.urlopen(a.server + path, timeout=timeout) as r: r.read(); code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    return code, time.time() - t0

lat = {'lease': [], 'reports': [], 'heartbeat': [], 'status': []}; errors = []; rows = [0]; lock = threading.Lock()
stop = time.time() + a.seconds

def node(seed, parent, status, exit_):
    n = {"seed": seed, "parent": parent, "status": status}
    if status == 'open':
        return n
    level = LEVELS.get(exit_)
    best = random.randint(5, 60) if level else 0
    n["summary"] = {"status": 'exhausted' if status == 'done' else 'split', "seed": seed, "exit": exit_,
                    "states": random.randint(1000, 10 ** 6), "accepted": 100, "valid": 10, "best": best, "verify": best if best else -1,
                    "evict_shallow": 0, "evict_recent": 0, "elapsed": a.job_s, "cpu_s": a.job_s * 0.98, "solver_calls": 100,
                    "protocol": 3, "flags": FLAGS, "unresolved": 0, "unknown": 0}
    if level:
        n["level"] = level
    return n

def tree_report(job):
    """Most windows are one node; ~5 % are a split with a few hundred children (mostly done, some open)."""
    seed, exit_ = job['seed'], job['exit']
    if random.random() < 0.95:
        return {"job_id": job['id'], "src_hash": a.hash, "nodes": [node(seed, None, 'done', exit_)]}
    nodes = [node(seed, None, 'split', exit_)]
    toks = [d + v for d in 'URDL' for v in '123']   # 12 tokens; i encoded in base 12 keeps children unique
    for i in range(random.randint(50, 300)):
        child = f"{seed},{random.choice(toks)},{toks[i % 12]},{toks[(i // 12) % 12]},{toks[(i // 144) % 12]}"
        nodes.append(node(child, seed, 'done' if random.random() < 0.9 else 'open', exit_))
    return {"job_id": job['id'], "src_hash": a.hash, "nodes": nodes}

def client(idx):
    code, reg, _ = post('/api/v2/register', {"name": f"load{idx}", "workers": a.workers})
    if code != 200: errors.append(('register', code, str(reg)[:200])); return
    token = reg['token']; held = []; last_hb = 0
    while time.time() < stop:
        t = time.time()
        if t - last_hb >= 30:
            ids = [j['id'] for j in held]
            c, o, d = post('/api/v2/heartbeat', {"token": token, "workers": a.workers, "paused": False, "boards": [],
                                                 "holding": ids, "running": ids[:a.workers]})
            with lock: lat['heartbeat'].append(d)
            if c != 200: errors.append(('heartbeat', c, str(o)[:200]))
            else:
                drop = set(o.get('drop') or [])
                if drop: held = [j for j in held if j['id'] not in drop]
            last_hb = t
        # jobs "finished" this interval: workers × interval / job_s
        finished = held[:int(a.workers * a.interval / a.job_s)]; held = held[len(finished):]
        if finished:
            reports = [tree_report(j) for j in finished]
            c, o, d = post('/api/v2/reports', {"token": token, "reports": reports})
            with lock:
                lat['reports'].append(d)
                if c == 200:
                    rows[0] += sum(len(r['nodes']) for r in reports)
                    bad = [x for x in o.get('results', []) if not x.get('ok')]
                    if bad: errors.append(('report results', len(bad), str(bad[0])[:200]))
                else: errors.append(('reports', c, str(o)[:200]))
        want = min(200, max(0, int(a.workers * 300 / a.job_s) - len(held)))
        if want:
            c, o, d = post('/api/v2/lease', {"token": token, "n": want})
            with lock: lat['lease'].append(d)
            if c == 200: held += o.get('jobs', [])
            else: errors.append(('lease', c, str(o)[:200]))
        time.sleep(max(0, a.interval - (time.time() - t)))

def poller():
    while time.time() < stop:
        t = time.time()
        c, d = get('/api/v2/status')
        with lock: lat['status'].append(d)
        if c != 200: errors.append(('status', c, ''))
        time.sleep(max(0, a.status_poll - (time.time() - t)))

threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(a.clients)]
if a.status_poll > 0: threads.append(threading.Thread(target=poller, daemon=True))
t0 = time.time()
for th in threads: th.start()
for th in threads: th.join()
el = time.time() - t0
def pct(xs, p): return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
print(f"clients {a.clients} × workers {a.workers}, {el:.0f} s, mean job {a.job_s}s")
for k, xs in lat.items():
    if xs: print(f"  {k:9s} n={len(xs):5d} rate={len(xs)/el:6.2f}/s  median={statistics.median(xs)*1000:6.1f} ms  p95={pct(xs,0.95)*1000:7.1f} ms  max={max(xs)*1000:7.1f} ms")
print(f"  rows written: {rows[0]} ({rows[0]/el:.0f}/s);  errors: {len(errors)}")
for e in errors[:5]: print('   ', e)
sys.exit(1 if errors else 0)
