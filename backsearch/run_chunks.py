#!/usr/bin/env python3
"""run_chunks.py --chunks 12,13 --workers 4 --name "Your Name" [--plan URL|FILE]

Volunteer runner for the collective proof.  Fetches the chunk plan (default:
the tracker on pathology.georgespahn.com); each chunk is a DFS-order range of
one exit's search tree, cut into sub-ranges that run in parallel through
campaign.py with the given number of worker processes.  Ctrl-C is safe at any
time: every worker prints a checkpoint and the same command continues from it.
At the end it prints a report block to paste into the tracker.  Requires a worker built from
this checkout:  ./build_pgo.sh -o backsearch_worker_nt --no-torch
"""
import argparse, hashlib, json, os, re, shlex, subprocess, sys, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument('--chunks', required=True, help='comma-separated chunk ids')
ap.add_argument('--workers', type=int, default=2); ap.add_argument('--name', required=True)
ap.add_argument('--plan', default='https://pathology.georgespahn.com/api/chunks/plan')
ap.add_argument('--worker', default='./backsearch_worker_nt')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
argv_line = 'python3 run_chunks.py ' + shlex.join(sys.argv[1:])

def die(m): print('error: ' + m, file=sys.stderr); sys.exit(1)
if not os.access(a.worker, os.X_OK): die(f"{a.worker} not found; build it with: ./build_pgo.sh -o backsearch_worker_nt --no-torch")
sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'unknown'
dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'backsearch.c', 'sokoban_bfs.c', 'sokoban_bfs.h', 'campaign.py'], capture_output=True, text=True).stdout.strip()
if dirty and not os.environ.get('RUN_CHUNKS_ALLOW_DIRTY'): die("backsearch.c / sokoban_bfs.c / campaign.py are modified in this checkout; a proof needs an unmodified build (git stash or checkout)")
wsha = hashlib.sha256(open(a.worker, 'rb').read()).hexdigest()[:12]
plan = json.load(urllib.request.urlopen(a.plan, timeout=60)) if a.plan.startswith('http') else json.load(open(a.plan))
by_id = {c['id']: c for c in plan['chunks']}
ids = [int(x) for x in a.chunks.split(',') if x]
missing = [i for i in ids if i not in by_id]
if missing: die(f"unknown chunk ids {missing} (plan has 1..{max(by_id)})")
extra = list(plan['extra']);  transit = ['--allow-exit-transit']
if '--allow-exit-transit' not in extra: extra = transit + extra
grid = plan.get('grid', '5x5')

# --- level text -> Pathology code (port of level_to_pathology.c) ------------
MASK_TO_CHAR = {1:'7',2:'8',3:'B',4:'9',5:'J',6:'C',7:'E',8:'6',9:'A',10:'I',11:'H',12:'D',13:'G',14:'F',15:'2'}
def level_code(text):
    rows, masks = [], {}
    for line in text.split('\n'):
        m = re.match(r'^\s*([.#O$@A-Z]+)(?:\s+([A-Z])=\[([URDL]*)\])?\s*$', line)
        if m and m.group(1) and not re.match(r'^\s*[A-Z]=\[', line):
            rows.append(m.group(1))
        for bm in re.finditer(r'([A-Z])=\[([URDL]*)\]', line):
            masks[bm.group(1)] = sum({'U':1,'R':2,'D':4,'L':8}[c] for c in bm.group(2))
    out = []
    for r in rows:
        s = ''
        for ch in r:
            if ch == '.': s += '0'
            elif ch == '#': s += '1'
            elif ch == '$': s += '3'
            elif ch == '@': s += '4'
            elif ch == 'O': s += '5'
            else: s += MASK_TO_CHAR.get(masks.get(ch, 15), '2')
        out.append(s)
    return '\n'.join(out)
def last_level(best_file):
    """The grid printed right after the last 'N (time)' header (the worker's
    final summary reprints the level, so take only that first block)."""
    lines = open(best_file).read().split('\n')
    heads = [i for i, l in enumerate(lines) if re.match(r'^\d+ \(', l)]
    if not heads: return None
    block = []
    for l in lines[heads[-1] + 1:]:
        if not l.strip(): break
        block.append(l)
    return level_code('\n'.join(block))

# --- run each chunk as its own resumable campaign --------------------------
t_start = time.time(); report_chunks = []
for cid in ids:
    c = by_id[cid]; d = f"results/chunks/chunk_{cid:03d}"; os.makedirs(d, exist_ok=True)
    jf = os.path.join(d, 'jobs.tsv')
    if not os.path.exists(jf):
        # A chunk is the DFS-order range [range_from, range_until); its cut points
        # (c['jobs'], in order) split it into sub-ranges that run in parallel and
        # are each checkpointable (Ctrl-C prints a CURSOR; re-run to continue).
        cuts = [j['path'] for j in c['jobs']]
        until = c.get('range_until') or ''
        with open(jf, 'w') as f:
            for i, p0 in enumerate(cuts):
                p1 = cuts[i + 1] if i + 1 < len(cuts) else until
                f.write(f"{c['exit']}\t{p0}..{p1}\t{c['jobs'][i]['depth']}\t0\t0\n")
        json.dump({"grid": grid, "exits": [c['exit']], "layer": plan.get('layer', 8), "extra": extra, "no_transit": False}, open(os.path.join(d, 'config.json'), 'w'))
    print(f"== chunk {cid} (exit {c['exit']}, {len(c['jobs'])} jobs, est {c['est_hours']} h): running with {a.workers} workers", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, 'campaign.py', '--out', d, '--exits', str(c['exit']), '--extra', ' '.join(extra),
                    '--workers', str(a.workers), '--worker', a.worker, '--shuffle'], stdout=open(os.path.join(d, 'driver.out'), 'a'), stderr=subprocess.STDOUT)
    wall = time.time() - t0
    # gather results
    import csv
    rows = list(csv.DictReader(open(os.path.join(d, 'done.tsv')), delimiter='\t')) if os.path.exists(os.path.join(d, 'done.tsv')) else []
    # every job listed in jobs.tsv (including continuation ranges the driver
    # appended) must have ended exhausted or been continued by a later job
    listed = [l.split('\t')[1] for l in open(jf) if l.strip()]
    st = {r['path']: r['status'] for r in rows}
    unfinished = [p for p in listed if st.get(p) not in ('exhausted', 'continued')]
    done = {p for p in listed if st.get(p) == 'exhausted'}
    all_paths = set(listed)
    exhausted = not unfinished
    remaining = [p.split('..', 1) for p in unfinished]
    best = max((int(r['best']) for r in rows), default=-1)
    states = sum(int(r['states']) for r in rows); cpu = sum(float(r['elapsed']) for r in rows)
    best_files = sorted(f for f in os.listdir(d) if f.startswith(f"best{best:03d}_"))
    code = last_level(os.path.join(d, best_files[-1])) if best_files else None
    report_chunks.append(dict(id=cid, exit=c['exit'], exhausted=exhausted, jobs_done=len(done), jobs=len(c['jobs']),
                              best=best, level=code, states=states, cpu_s=round(cpu), wall_s=round(wall),
                              remaining=remaining[:200] if not exhausted else []))
    print(f"   {'EXHAUSTED' if exhausted else 'INCOMPLETE (' + str(len(unfinished)) + ' ranges left; re-run to continue)'}: best {best}, {states:,} states, {cpu/3600:.2f} CPU-h, {wall/3600:.2f} h wall", flush=True)

report = dict(name=a.name, command=argv_line, git=sha, worker_sha256=wsha, workers=a.workers, plan_created=plan.get('created'),
              started=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_start)), wall_s=round(time.time() - t_start), chunks=report_chunks)
txt = "-----BEGIN CHUNK REPORT-----\n" + json.dumps(report, indent=1) + "\n-----END CHUNK REPORT-----"
os.makedirs('results/chunks', exist_ok=True)
rf = f"results/chunks/report_{'_'.join(map(str, ids))}_{time.strftime('%Y%m%d_%H%M')}.txt"; open(rf, 'w').write(txt + '\n')
print("\nPaste everything between the markers into the tracker (also saved to " + rf + "):\n"); print(txt)
