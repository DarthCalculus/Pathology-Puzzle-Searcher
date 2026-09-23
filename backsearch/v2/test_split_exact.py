#!/usr/bin/env python3
"""Split exactness test for the v2 worker protocol (v2/DESIGN.md §5.1).

Runs the full search once with a visit trace, then re-runs it as a tree of
subtree jobs: every run uses --split-after S (or gets SIGINT after S seconds),
its REMAINING seeds become new jobs, until every job exhausts.  Asserts that
the union of the jobs' visit traces covers every node of the full run and that
the best depth agrees.  One worker process at a time; small configs only.

  python3 test_split_exact.py WORKER [--sigint] [--split 0.02] -- <worker config args>
"""
import argparse, glob, json, os, re, shutil, signal, subprocess, sys, tempfile, time

ap = argparse.ArgumentParser()
ap.add_argument('worker'); ap.add_argument('--sigint', action='store_true')
ap.add_argument('--split', type=float, default=0.02); ap.add_argument('--max-runs', type=int, default=3000)
argv = sys.argv[1:]; cfg = []
if '--' in argv: i = argv.index('--'); cfg = argv[i + 1:]; argv = argv[:i]
a = ap.parse_args(argv)
tmp = tempfile.mkdtemp(prefix='splitx_')

def canon(code):
    """Min over the 8 dihedral images of a level code (rows joined by '/'), block
    masks remapped (U=1 R=2 D=4 L=8), so mirrored twins compare equal."""
    rows = code.split('/'); R, C = len(rows), len(rows[0])
    M2C = {1:'7',2:'8',3:'B',4:'9',5:'J',6:'C',7:'E',8:'6',9:'A',10:'I',11:'H',12:'D',13:'G',14:'F',15:'2'}
    C2M = {v: k for k, v in M2C.items()}
    def tmask(m, swap, fr, fc):
        b = [m & 1, m & 2, m & 4, m & 8]
        if swap: b = [b[3], b[2], b[1], b[0]]
        if fr: b = [b[2], b[1], b[0], b[3]]
        if fc: b = [b[0], b[3], b[2], b[1]]
        return (1 if b[0] else 0) | (2 if b[1] else 0) | (4 if b[2] else 0) | (8 if b[3] else 0)
    best = None
    for swap in (0, 1):
        for fr in (0, 1):
            for fc in (0, 1):
                g = [[rows[r][c] for c in range(C)] for r in range(R)]
                if swap: g = [[g[r][c] for r in range(R)] for c in range(C)]
                if fr: g = g[::-1]
                if fc: g = [row[::-1] for row in g]
                out = '/'.join(''.join(M2C[tmask(C2M[ch], swap, fr, fc)] if ch in C2M else ch for ch in row) for row in g)
                if best is None or out < best: best = out
    return best

def run(seed, split):
    """Run one job; return (status, best, remaining, visited-set)."""
    tr = os.path.join(tmp, 'tr'); tv = os.path.join(tmp, 'tv')
    for f in glob.glob(tr + '.*') + glob.glob(tv + '.*'): os.remove(f)
    cmd = [a.worker] + cfg + ['--time', '0'] + (['--seed-path', seed] if seed else [])
    if split and not a.sigint: cmd += ['--split-after', str(split)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env={**os.environ, 'BS_TRACE_VISITS': tr, 'BS_TRACE_VALID': tv})
    if split and a.sigint:
        time.sleep(split); p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=120)
    summ = None; rem = []
    for line in out.split('\n'):
        if line.startswith('SUMMARY\t'): summ = json.loads(line[8:])
        elif line.startswith('REMAINING\t'): rem.append(line[10:])
    assert summ, f'no SUMMARY for seed {seed!r}\n{out[-500:]}'
    vis = set()
    for f in glob.glob(tr + '.*'):
        vis.update(l.rstrip('\n') for l in open(f))
    val = set()
    for f in glob.glob(tv + '.*'):
        for l in open(f):
            _, d, code = l.rstrip('\n').split('\t'); val.add((int(d), canon(code)))
    return summ, rem, vis, val

full, _, full_vis, full_val = run('', 0)
assert full['status'] == 'exhausted'
print(f"full: {len(full_vis)} nodes, {len(full_val)} distinct valid levels (canonical), best {full['best']}")

jobs = ['']; union = set(); union_val = set(); runs = 0; splits = 0; best = 0; expanded_twice = 0; jobinfo = []
while jobs:
    seed = jobs.pop()
    summ, rem, vis, val = run(seed, a.split); runs += 1
    union_val |= val
    if runs > a.max_runs: sys.exit('too many runs')
    best = max(best, summ['best'])
    if summ['status'] == 'exhausted':
        assert not rem
    elif summ['status'] == 'split':
        assert rem, 'split without REMAINING'
        assert rem[0] == seed or rem[0].startswith(seed) or seed == '', (seed, rem[0])
        for r in rem: assert r == '' or r.startswith(seed) or seed == '', (seed, r)
        jobs.extend(rem); splits += 1
    else:
        sys.exit(f'unexpected status {summ}')
    expanded_twice += len(vis & union)
    union |= vis
    jobinfo.append((seed, summ['status'], vis, rem))
missing = full_vis - union
lost = full_val - union_val
print(f"jobs: {runs} runs, {splits} splits, union {len(union)} nodes, best {best}; valid levels lost {len(lost)} (extra {len(union_val - full_val)}); nodes not re-visited {len(missing)} (informational: dedup twins), re-expanded {expanded_twice}")
shutil.rmtree(tmp)
if lost or best != full['best']:
    print('LOST LEVELS:', sorted(lost)[:5]); sys.exit('FAIL')
if missing and os.environ.get('SPLITX_STRICT'):
    print('MISSING:', sorted(missing)[:3])
    tok = lambda p: [t for t in p.split(',') if t]
    for m in sorted(missing)[:3]:
        mt = tok(m)
        cov = [(sd, st, len(v), any(r and mt[:len(tok(r))] == tok(r) for r in rem)) for sd, st, v, rem in jobinfo if mt[:len(tok(sd))] == tok(sd)]
        cov.sort(key=lambda x: -len(tok(x[0])))
        print('  node', m); 
        for sd, st, n, handed in cov[:4]: print(f'    job seed={sd!r} status={st} visited={n} handed_off_to_child={handed}')
    sys.exit('FAIL')
print('OK')
