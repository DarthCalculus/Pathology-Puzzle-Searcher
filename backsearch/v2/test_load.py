#!/usr/bin/env python3
"""Server load test (DESIGN.md §7): simulated clients, no real workers.

Starts nothing itself: point it at a running local server that has an open
campaign with roots (see test_chaos.sh for how to seed one, or run with
--seed to create a throwaway campaign through the tools).  Each simulated
client behaves like volunteer.py under the load rules: one lease request and
one batched report POST every `--interval` seconds, one heartbeat per 30 s,
reporting tree reports whose node count follows a heavy-tailed distribution
(mostly one-node "trivial" reports, some 300-node splits).  Measures server
request latency and throughput; prints rows written per second.

  python3 test_load.py --server http://127.0.0.1:3199 --clients 40 --workers 32 --seconds 60
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--server', default='http://127.0.0.1:3199'); ap.add_argument('--clients', type=int, default=40)
ap.add_argument('--workers', type=int, default=32); ap.add_argument('--seconds', type=float, default=60)
ap.add_argument('--interval', type=float, default=10); ap.add_argument('--job-s', type=float, default=2.0,
    help='mean simulated job duration (2 s = the worst case the rules allow after absorption)')
ap.add_argument('--hash', required=True, help='a whitelisted src_hash')
a = ap.parse_args()

def post(path, body, timeout=30):
    req = urllib.request.Request(a.server + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: out = json.loads(r.read().decode()); code = r.status
    except urllib.error.HTTPError as e:
        out = json.loads(e.read().decode() or '{}'); code = e.code
    return code, out, time.time() - t0

lat = {'lease': [], 'reports': [], 'heartbeat': []}; errors = []; rows = [0]; lock = threading.Lock()
stop = time.time() + a.seconds

def fake_summary(seed, exit_):
    return {"status": "exhausted", "seed": seed, "exit": exit_, "states": random.randint(1000, 10 ** 6), "accepted": 100, "valid": 10,
            "best": random.randint(5, 60), "verify": 0, "evict_shallow": 0, "evict_recent": 0, "elapsed": a.job_s, "solver_calls": 100, "src_hash": a.hash}

def tree_report(job):
    """Most windows are one node; ~5 % are a split with a few hundred children (mostly done, some open)."""
    seed, exit_ = job['seed'], job['exit']
    if random.random() < 0.95:
        return {"job_id": job['id'], "src_hash": a.hash, "nodes": [{"seed": seed, "parent": None, "status": "done", "summary": fake_summary(seed, exit_)}]}
    nodes = [{"seed": seed, "parent": None, "status": "split"}]
    for i in range(random.randint(50, 300)):
        toks = [d + v for d in 'URDL' for v in '123']   # 12 tokens; i encoded in base 12 keeps children unique
        child = f"{seed},{random.choice(toks)},{toks[i % 12]},{toks[(i // 12) % 12]},{toks[(i // 144) % 12]}"
        st = 'done' if random.random() < 0.9 else 'open'
        n = {"seed": child, "parent": seed, "status": st}
        if st == 'done': n["summary"] = fake_summary(child, exit_)
        nodes.append(n)
    return {"job_id": job['id'], "src_hash": a.hash, "nodes": nodes}

def client(idx):
    code, reg, _ = post('/api/v2/register', {"name": f"load{idx}", "workers": a.workers})
    if code != 200: errors.append(('register', code, reg)); return
    token = reg['token']; held = []; last_hb = 0
    while time.time() < stop:
        t = time.time()
        if t - last_hb >= 30:
            c, o, d = post('/api/v2/heartbeat', {"token": token, "workers": a.workers, "paused": False, "boards": []})
            with lock: lat['heartbeat'].append(d)
            if c != 200: errors.append(('heartbeat', c, o))
            last_hb = t
        # jobs "finished" this interval: workers × interval / job_s
        finished = held[:int(a.workers * a.interval / a.job_s)]; held = held[len(finished):]
        if finished:
            reports = [tree_report(j) for j in finished]
            c, o, d = post('/api/v2/reports', {"token": token, "reports": reports})
            with lock:
                lat['reports'].append(d)
                if c == 200: rows[0] += sum(len(r['nodes']) for r in reports)
                else: errors.append(('reports', c, str(o)[:200]))
        want = min(200, max(0, int(a.workers * 300 / a.job_s) - len(held)))
        if want:
            c, o, d = post('/api/v2/lease', {"token": token, "n": want})
            with lock: lat['lease'].append(d)
            if c == 200: held += o.get('jobs', [])
            else: errors.append(('lease', c, str(o)[:200]))
        time.sleep(max(0, a.interval - (time.time() - t)))

threads = [threading.Thread(target=client, args=(i,), daemon=True) for i in range(a.clients)]
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
