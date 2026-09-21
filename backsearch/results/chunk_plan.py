#!/usr/bin/env python3
"""chunk_plan.py --out chunks.json [--exits 0,1,2,6,7,12] [--extra "--num-holes 3"]
                 [--chunks 50] [--layer 8] [--probes-per-node 20] [--split-depth 4]
                 [--worker ./backsearch_est]

Cut one search class into chunks of roughly equal estimated work, each a list
of --seed-path jobs for one exit.  Weights come from the worker's Knuth
estimator (--estimate-dump): every depth-K layer node gets an estimated
subtree time.  A node heavier than the chunk budget is re-estimated from its
own seed at depth K+split (recursively) and replaced by its children, so heavy
prefixes are cut fine and the light ones stay coarse.  Nodes are then packed
greedily (heaviest first) into the requested number of chunks.

The estimator is a lower bound (real campaigns ran 1.6-7x its number) and its
per-node numbers are noisy, so "equal" is approximate; the chunk file records
every node's estimate so the plan can be re-cut later.  Ancestors of the
layer (depth < K) are never records and need no chunk.
"""
import argparse, json, os, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--exits', default='0,1,2,6,7,12')
ap.add_argument('--extra', default='--num-holes 3'); ap.add_argument('--chunks', type=int, default=50)
ap.add_argument('--layer', type=int, default=8); ap.add_argument('--probes-per-node', type=int, default=20)
ap.add_argument('--split-depth', type=int, default=4); ap.add_argument('--worker', default='./backsearch_est')
ap.add_argument('--grid', default='5x5'); ap.add_argument('--max-split-rounds', type=int, default=4)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(HERE, '..'))
extra = a.extra.split()
tmp = a.out + '.dump.tsv'
def log(m): print(time.strftime('[%H:%M:%S] ') + m, flush=True)

def estimate(exit_pos, seed, depth, probes_per_node):
    """Return a list of nodes (dicts) for the depth-`depth` layer under `seed` (None = exit root)."""
    if os.path.exists(tmp): os.remove(tmp)
    cmd = [a.worker, '--grid', a.grid, '--two-tables', '--allow-exit-transit', '--exit', str(exit_pos)] + extra
    if seed: cmd += ['--seed-path', seed]
    # layer size first (cheap), then probes = nodes * probes_per_node
    lay = subprocess.run(cmd + ['--list-layer', str(depth)], capture_output=True, text=True).stdout
    n = sum(1 for l in lay.split('\n') if l.startswith('LAYER'))
    if n == 0: return []
    probes = max(1000, n * probes_per_node)
    subprocess.run(cmd + ['--estimate', str(probes), '--estimate-depth', str(depth), '--estimate-dump', tmp], capture_output=True, text=True)
    nodes = []
    for l in open(tmp):
        ex, path, d, nb, nh, est_nodes, est_s, se_s = l.rstrip('\n').split('\t')
        nodes.append(dict(exit=int(ex), path=path, depth=int(d), blocks=int(nb), holes=int(nh),
                          est_nodes=float(est_nodes), est_s=float(est_s), se_s=float(se_s)))
    return nodes

# 1. per-exit layer estimates
nodes = []
for e in [int(x) for x in a.exits.split(',')]:
    t0 = time.time(); ns = estimate(e, None, a.layer, a.probes_per_node)
    log(f"exit {e}: {len(ns)} layer-{a.layer} nodes, est {sum(n['est_s'] for n in ns)/3600:.1f} h ({time.time()-t0:.0f} s)")
    nodes += ns
total = sum(n['est_s'] for n in nodes); budget = total / a.chunks
log(f"total est {total/3600:.1f} h dedup-free; chunk budget {budget/3600:.2f} h")

# 2. split heavy nodes by re-estimating from their own seed at a deeper layer
for rnd in range(a.max_split_rounds):
    heavy = [n for n in nodes if n['est_s'] > budget]
    if not heavy: break
    log(f"split round {rnd+1}: {len(heavy)} nodes over budget (heaviest {max(n['est_s'] for n in heavy)/3600:.1f} h)")
    keep = [n for n in nodes if n['est_s'] <= budget]
    for h in sorted(heavy, key=lambda n: -n['est_s']):
        kids = estimate(h['exit'], h['path'], h['depth'] + a.split_depth, a.probes_per_node)
        ks = sum(k['est_s'] for k in kids)
        log(f"  {h['path']} (exit {h['exit']}, est {h['est_s']/3600:.1f} h) -> {len(kids)} children, est {ks/3600:.1f} h")
        if not kids: keep.append(h)   # nothing below (should not happen for a heavy node)
        else: keep += kids
    nodes = keep
    total = sum(n['est_s'] for n in nodes); budget = total / a.chunks
    log(f"  now {len(nodes)} nodes, total est {total/3600:.1f} h, budget {budget/3600:.2f} h")

# 3. greedy packing, heaviest first, into the least loaded chunk of the SAME exit
#    (chunks are per exit so one worker run has one exit); chunk count per exit
#    proportional to its share, at least 1.
by_exit = {}
for n in nodes: by_exit.setdefault(n['exit'], []).append(n)
plan = []
for e, ns in sorted(by_exit.items()):
    share = sum(n['est_s'] for n in ns) / total
    k = max(1, round(a.chunks * share))
    bins = [dict(exit=e, est_s=0.0, jobs=[]) for _ in range(k)]
    for n in sorted(ns, key=lambda n: -n['est_s']):
        b = min(bins, key=lambda b: b['est_s']); b['est_s'] += n['est_s']; b['jobs'].append(n)
    plan += bins
plan.sort(key=lambda b: (b['exit'], -b['est_s']))
out = dict(created=time.strftime('%Y-%m-%d %H:%M'), grid=a.grid, extra=extra, layer=a.layer, chunks=[])
for i, b in enumerate(plan, 1):
    out['chunks'].append(dict(id=i, exit=b['exit'], est_hours=round(b['est_s']/3600, 2), n_jobs=len(b['jobs']),
                              jobs=[dict(path=j['path'], depth=j['depth'], est_s=round(j['est_s'], 1)) for j in b['jobs']]))
json.dump(out, open(a.out, 'w'), indent=1)
if os.path.exists(tmp): os.remove(tmp)
log(f"wrote {a.out}: {len(plan)} chunks; est hours per chunk: min {min(b['est_s'] for b in plan)/3600:.2f} "
    f"median {sorted(b['est_s'] for b in plan)[len(plan)//2]/3600:.2f} max {max(b['est_s'] for b in plan)/3600:.2f}")
for b in plan: print(f"  chunk exit {b['exit']}: est {b['est_s']/3600:6.2f} h, {len(b['jobs']):6d} jobs")
