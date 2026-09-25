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

Legacy v1 tool (the v2 collective search seeds with server/tools/v2_seed.js).
Since 2026-09-24 the worker's listing and dump start each exit with a
LAYERINFO header (skipped here), and a root can lie deeper than the layer
(a bulk walk-back child); --list-layer / --estimate-depth under --seed-path
count from the seed's depth.
"""
import argparse, json, os, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--exits', default='0,1,2,6,7,12')
ap.add_argument('--extra', default='--num-holes 3'); ap.add_argument('--chunks', type=int, default=50)
ap.add_argument('--layer', type=int, default=8); ap.add_argument('--probes-per-node', type=int, default=20)
ap.add_argument('--split-depth', type=int, default=4); ap.add_argument('--worker', default='./backsearch_est')
ap.add_argument('--grid', default='5x5'); ap.add_argument('--max-split-rounds', type=int, default=4)
ap.add_argument('--cheap-split', action='store_true', help='split heavy nodes by listing their sub-layer (no probes); children share the parent estimate equally')
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(os.path.join(HERE, '..'))
extra = a.extra.split()
tmp = f"{a.out}.dump.{os.getpid()}.tsv"   # private to this planner process (a stray worker must not append to it)
def log(m): print(time.strftime('[%H:%M:%S] ') + m, flush=True)

def estimate(exit_pos, seed, depth, probes_per_node):
    """Return a list of nodes (dicts) for the depth-`depth` layer under `seed` (None = exit root)."""
    if os.path.exists(tmp): os.remove(tmp)
    cmd = [a.worker, '--grid', a.grid, '--two-tables', '--allow-exit-transit', '--exit', str(exit_pos)] + extra
    if seed: cmd += ['--seed-path', seed]
    # layer size first (cheap), then probes = nodes * probes_per_node
    lay = subprocess.run(cmd + ['--list-layer', str(depth)], capture_output=True, text=True).stdout
    n = sum(1 for l in lay.split('\n') if l.startswith('LAYER\t'))
    if n == 0: return []
    probes = max(1000, n * probes_per_node)
    subprocess.run(cmd + ['--estimate', str(probes), '--estimate-depth', str(depth), '--estimate-dump', tmp], capture_output=True, text=True)
    nodes = []
    for l in open(tmp):
        if l.startswith('LAYERINFO') or not l.strip(): continue   # the dump's per-exit header
        ex, path, d, nb, nh, est_nodes, est_s, se_s = l.rstrip('\n').split('\t')
        if int(ex) != exit_pos: continue
        nodes.append(dict(exit=int(ex), path=path, depth=int(d), blocks=int(nb), holes=int(nh),
                          est_nodes=float(est_nodes), est_s=float(est_s), se_s=float(se_s)))
    os.remove(tmp)
    if seed is None and len(nodes) != n: log(f"  WARNING exit {exit_pos}: dump has {len(nodes)} rows but the layer has {n} nodes")
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
    keep = []
    for n in nodes:
        if n['est_s'] <= budget: keep.append(n); continue
        if a.cheap_split:
            cmd = [a.worker, '--grid', a.grid, '--two-tables', '--allow-exit-transit', '--exit', str(n['exit'])] + extra + ['--seed-path', n['path'], '--list-layer', str(a.split_depth)]
            lay = [l.split('\t') for l in subprocess.run(cmd, capture_output=True, text=True).stdout.split('\n') if l.startswith('LAYER\t')]
            kids = [dict(exit=n['exit'], path=r[2], depth=int(r[3]), blocks=int(r[4]), holes=int(r[5]), est_nodes=0.0, est_s=n['est_s'] / max(1, len(lay)), se_s=0.0) for r in lay]
        else:
            kids = estimate(n['exit'], n['path'], a.split_depth, a.probes_per_node)   # sub-layer (split_depth below the node), DFS order
        ks = sum(k['est_s'] for k in kids)
        log(f"  {n['path']} (exit {n['exit']}, est {n['est_s']/3600:.1f} h) -> {len(kids)} children, est {ks/3600:.1f} h")
        keep += kids if kids else [n]     # spliced in place: the sub-layer occupies exactly the node's slot in DFS order
    nodes = keep
    total = sum(n['est_s'] for n in nodes); budget = total / a.chunks
    log(f"  now {len(nodes)} nodes, total est {total/3600:.1f} h, budget {budget/3600:.2f} h")

# 3. contiguous cutting in DFS order.  The estimator lists a layer in the DFS
#    visiting order and a heavy node's sub-layer lies exactly where the node was,
#    so `nodes` (per exit) is the whole tree in visiting order.  Cut it into runs
#    whose estimates add up to about one budget; a chunk is then the DFS-order
#    range [first node, next chunk's first node) and its nodes are the cut points
#    the runner uses to make parallel sub-jobs.
by_exit = {}
for n in nodes: by_exit.setdefault(n['exit'], []).append(n)
plan = []
for e in sorted(by_exit):
    ns = by_exit[e]           # already in DFS order: layer order, with sub-layers spliced in place
    share = sum(n['est_s'] for n in ns) / total
    k = max(1, round(a.chunks * share)); target = sum(n['est_s'] for n in ns) / k
    cur = dict(exit=e, est_s=0.0, jobs=[])
    for i, n in enumerate(ns):
        if cur['jobs'] and cur['est_s'] + n['est_s'] > target * 1.15 and len(plan) - sum(1 for b in plan if b['exit'] != e) < k - 1:
            plan.append(cur); cur = dict(exit=e, est_s=0.0, jobs=[])
        cur['est_s'] += n['est_s']; cur['jobs'].append(n)
    plan.append(cur)
out = dict(created=time.strftime('%Y-%m-%d %H:%M'), grid=a.grid, extra=extra, layer=a.layer, chunks=[])
for i, b in enumerate(plan):
    nxt = plan[i + 1] if i + 1 < len(plan) and plan[i + 1]['exit'] == b['exit'] else None
    out['chunks'].append(dict(id=i + 1, exit=b['exit'], est_hours=round(b['est_s'] / 3600, 2), n_jobs=len(b['jobs']),
                              range_from=b['jobs'][0]['path'], range_until=nxt['jobs'][0]['path'] if nxt else None,
                              jobs=[dict(path=j['path'], depth=j['depth'], est_s=round(j['est_s'], 1)) for j in b['jobs']]))
json.dump(out, open(a.out, 'w'), indent=1)
if os.path.exists(tmp): os.remove(tmp)
log(f"wrote {a.out}: {len(plan)} chunks; est hours per chunk: min {min(b['est_s'] for b in plan)/3600:.2f} "
    f"median {sorted(b['est_s'] for b in plan)[len(plan)//2]/3600:.2f} max {max(b['est_s'] for b in plan)/3600:.2f}")
for c in out['chunks']: print(f"  chunk {c['id']:3d} exit {c['exit']:2d}: est {c['est_hours']:6.2f} h, {c['n_jobs']:6d} cut points, from {c['range_from']}")
