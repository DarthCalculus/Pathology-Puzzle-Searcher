#!/usr/bin/env python3
"""Independent oracle for the backward generator (review M89).

Every other exactness test compares the worker with itself.  This one compares it with oracle_enum.c,
a separate enumerator with its own move rules and its own BFS judge (see that file), on grids small
enough to enumerate every level: 2x4, 2x5, 3x3, 3x4, 4x4 and 4x5 with 0 holes and unbounded blocks
(campaign #2's rule), 2x4, 2x5, 3x3, 3x4 with at most 2 holes, 4x4 with at most 3 holes (campaign #1's
rule), and block caps (3x4 <=2 blocks, 4x4 <=3 holes <=3 blocks); always with exit transit and no
block on the exit.

The oracle's set T (per canonical exit) is every level L of the class such that some optimal solution of
L touches every non-wall cell, pushes every block in every direction of its mask and fills every hole,
paired with its length.  Every class maximum is attained inside T (restrict any level to what one
optimal solution uses: the optimum does not change), and T is exactly what BS_TRACE_VALID lists.  For
each config and exit the test asserts:

  1. canonical exits: the worker searches exactly one exit per symmetry orbit of the grid's cells;
  2. W == T, canonical under the 8 symmetries, depth by depth: no LOST level (the worker pruned a real
     level: the completeness bug no self-comparison can see) and no EXTRA level (the worker traced a
     level the oracle rejects: wrong depth, a shortcut, a block on the exit, an illegal level);
  3. the worker's best == max(T), and the maximum is PROVEN: the oracle also enumerates the tight
     configurations with a block on the exit (E); if a class level of length n existed, every suffix
     of its optimal solution would be a T or E configuration of each length m <= n, so a depth above
     max(T) with no T and no E configuration bounds the class (STATS "proof");
  4. the same W == T in every A/B mode of the worker (--no-bulk-walk, --no-parent-table,
     --no-exit-block-prune, --no-state-canon, --no-two-tables, whichever the worker accepts), and for
     a split tree of jobs when the worker has --split-after-nodes;
  5. the oracle's two algorithms agree: --mode lazy (forward generation with lazily decided cells) and
     --mode bf (brute force over every level + a covering DP over all optimal solutions), T and E
     identical, on the configs small enough for bf;
  6. the oracle's judge agrees with the site's solver6 (PathologyRecords/solver/solver6.c, --batch) on
     every T level and on a seeded random sample of arbitrary levels (solvable or not).
The full run also runs the review's mode matrix on bigger configs (4x4 exit 0 <=3 holes, 5x5 exit 7
<=3 blocks, 6x6 exit 14 0 holes <=2 blocks; skip with --no-matrix): worker vs worker, the canonical
valid-level sets must be identical across all modes.

  python3 test_oracle.py WORKER            full run (about 5 CPU-minutes; each config well under 2 minutes)
  python3 test_oracle.py WORKER --quick    smoke test (about 20 s)
  options: --solver6 SRC|none, --only NAME (repeatable; also filters the matrix, e.g. --only matrix-6x6),
           --no-matrix, --no-modes, --no-cross, --timeout S
Exit status 0 = all passed, 1 = a failure.  One process at a time.
"""
import argparse, json, os, random, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_split_exact as TS          # noqa: E402  (canon, Worker, run_job, check_run, Fail)
from test_split_exact import Fail, canon  # noqa: E402

SOLVER6_DEFAULT = '/Users/george/PathologyRecords/solver/solver6.c'
MODES = ['--no-bulk-walk', '--no-parent-table', '--no-exit-block-prune', '--no-state-canon', '--no-two-tables']
# (name, grid, max holes, max blocks or None, cross-checks of the oracle itself)
#   ('lazy', dmax): forward lazy generation up to dmax (with block-on-exit configs) == back up to dmax
#   ('bf', cap, dmax): brute force with at most cap blocks == back with at most cap blocks
CONFIGS = [
    ('2x4h0', '2x4', 0, None, [('lazy', 20), ('bf', 12, 20)]),
    ('2x5h0', '2x5', 0, None, [('lazy', 20), ('bf', 12, 20)]),
    ('3x3h0', '3x3', 0, None, [('lazy', 24), ('bf', 3, 24)]),
    ('3x4h0', '3x4', 0, None, [('lazy', 30), ('bf', 2, 30)]),
    ('2x4h2', '2x4', 2, None, [('lazy', 20), ('bf', 12, 20)]),
    ('3x3h2', '3x3', 2, None, [('lazy', 24), ('bf', 2, 24)]),
    ('2x5h2', '2x5', 2, None, [('lazy', 24), ('bf', 3, 24)]),
    ('3x4h2', '3x4', 2, None, [('lazy', 36)]),
    ('3x4h0b2', '3x4', 0, 2, []),            # the worker's --num-blocks cap
    ('4x4h0', '4x4', 0, None, [('lazy', 16)]),   # even square grid, like 6x6 (lazy is cheap only for the corner)
    ('4x4h3', '4x4', 3, None, []),           # campaign #1's hole rule
    ('4x4h3b3', '4x4', 3, 3, []),
    ('4x5h0', '4x5', 0, None, []),           # 20 cells, 0 holes, up to 8+ blocks: campaign #2's regime in small
]
QUICK = ['2x4h0', '3x3h0', '2x4h2', '3x3h2', '4x4h0']
QUICK_CROSS = {'2x4h0', '3x3h0', '3x3h2'}


def orbit_reps(R, C):
    """One representative (the smallest index) per orbit of the cells under the grid's symmetries."""
    def images(r, c):
        out = {(r, c), (R - 1 - r, c), (r, C - 1 - c), (R - 1 - r, C - 1 - c)}
        if R == C:
            out |= {(c, r) for r, c in list(out)}
        return out
    return sorted({min(i * C + j for i, j in images(r, c)) for r in range(R) for c in range(C)})


def worker_exits(worker, grid):
    """The exits the worker iterates when --exit is not given."""
    p = subprocess.run([worker.path, '--grid', grid, '--allow-exit-transit', '--num-holes', '0', '--max-depth', '1',
                        '--time', '0'], capture_output=True, text=True, timeout=60, env=TS.clean_env())
    ex = [json.loads(l[8:])['exit'] for l in p.stdout.splitlines() if l.startswith('SUMMARY\t')]
    if not ex:
        raise Fail(f'no SUMMARY from the exit listing run on {grid}: {p.stdout[-300:]} {p.stderr[-300:]}')
    return ex


def build(tmp, solver6_src):
    exe = os.path.join(tmp, 'oracle_enum')
    p = subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-o', exe, os.path.join(HERE, 'oracle_enum.c')],
                       capture_output=True, text=True, timeout=300)
    if p.returncode:
        raise Fail(f'cannot build oracle_enum.c:\n{p.stderr[-1500:]}')
    s6 = None
    if solver6_src and solver6_src != 'none':
        if not os.path.isfile(solver6_src):
            raise Fail(f'solver6 source not found: {solver6_src} (pass --solver6 SRC, or --solver6 none to skip)')
        s6 = os.path.join(tmp, 'solver6')
        p = subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-w', '-o', s6, solver6_src], capture_output=True,
                           text=True, timeout=300)
        if p.returncode:
            raise Fail(f'cannot build solver6:\n{p.stderr[-1500:]}')
    return exe, s6


def oracle(exe, grid, exit_, holes, blocks, dmax=0, mode='back', timeout=600):
    cmd = [exe, '--grid', grid, '--exit', str(exit_), '--max-holes', str(holes), '--max-blocks',
           str(blocks if blocks is not None else 32), '--mode', mode, '--table-lg2', '23']
    if mode != 'back':
        cmd += ['--dmax', str(dmax), '--extended']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise Fail(f'oracle failed ({" ".join(cmd)}): {p.stderr[-500:]}')
    T, E, st = {}, set(), None
    for line in p.stdout.splitlines():
        f = line.split('\t')
        if f[0] == 'T':
            T[(int(f[1]), canon(f[2]))] = f[2]
        elif f[0] == 'E':
            E.add((int(f[1]), f[2]))
        elif f[0] == 'STATS':
            st = json.loads(f[1])
    if st is None:
        raise Fail(f'oracle printed no STATS ({" ".join(cmd)})')
    return T, E, st, p.stdout


def worker_set(worker, grid, exit_, holes, blocks, tmp, timeout, extra=(), split_nodes=0):
    """W for one exit: one monolithic run, or (split_nodes) the union of a split tree of jobs."""
    flags = ['--allow-exit-transit', '--num-holes', str(holes)] + (['--num-blocks', str(blocks)] if blocks is not None else [])
    jobs, W, runs, splits, best, states = [''], {}, 0, 0, 0, 0
    while jobs:
        seed = jobs.pop()
        r = TS.run_job(worker, grid, exit_, flags, seed, tmp, timeout, split_nodes=split_nodes, argv_extra=extra,
                       keep_paths=True)
        TS.check_run(worker, r, split_expected=bool(split_nodes), grid=grid, exit_=exit_, extra=flags)
        if r.summary.get('unknown', 0):
            raise Fail(f'{grid} exit {exit_} {extra}: {r.summary["unknown"]} capacity results on a tiny grid')
        runs += 1
        best = max(best, r.summary['best'])
        states += r.summary.get('states', 0)
        for d, code, path in r.levels:
            W.setdefault((d, canon(code)), (code, path, seed))
        if r.summary['status'] == 'split':
            splits += 1
            jobs.extend(r.remaining)
        if runs > 5000:
            raise Fail('split tree exceeded 5000 runs')
    return W, best, runs, splits, states


def compare(label, T, W, best, out):
    lost = sorted(k for k in T if k not in W)
    extra = sorted(k for k in W if k not in T)
    tmax = max((d for d, _ in T), default=0)
    ok = not lost and not extra and best == tmax
    if not ok:
        out(f'    {label}: LOST {len(lost)} EXTRA {len(extra)} best {best} vs oracle max {tmax}')
        for d, c in lost[:5]:
            out(f'      lost  depth {d} {T[(d, c)]}')
        for k in extra[:5]:
            code, path, seed = W[k]
            out(f'      extra depth {k[0]} {code}  path {path!r} (job seed {seed!r})')
    return ok, len(lost), len(extra)


def judge_check(exe, s6, grid, codes, out):
    """Our BFS judge vs solver6 on these codes.  Returns (n, mismatches)."""
    if not codes:
        return 0, 0
    inp = '\n'.join(codes) + '\n'
    a = subprocess.run([exe, '--grid', grid, '--judge'], input=inp, capture_output=True, text=True, timeout=600)
    mine = a.stdout.split()
    b = subprocess.run([s6, '20', '0.25', '--batch'], input=inp.replace('/', '|'), capture_output=True, text=True,
                       timeout=1200)
    theirs = []
    for line in b.stdout.splitlines():
        j = json.loads(line)
        if j.get('solvable') is True:
            theirs.append(str(j['moves']))
        elif j.get('solvable') is False:
            theirs.append('-1')
        else:
            theirs.append('?' + line)
    if len(mine) != len(codes) or len(theirs) != len(codes):
        raise Fail(f'judge cross-check: {len(codes)} codes, {len(mine)} oracle answers, {len(theirs)} solver6 answers')
    bad = [(c, m, t) for c, m, t in zip(codes, mine, theirs) if m != t]
    for c, m, t in bad[:5]:
        out(f'    JUDGE MISMATCH {c}: oracle {m}, solver6 {t}')
    return len(codes), len(bad)


def random_levels(R, C, n, rng, max_holes):
    """Arbitrary levels: one exit, one player, and per cell 40% floor, 25% wall, 25% a block of a random
    mask, 10% a hole (at most max_holes; at most 10 blocks).  Mostly not tight, often unsolvable."""
    out = []
    masks = list('789BJCE6AIHDGF2')
    for _ in range(n):
        cells, nb, nh = [], 0, 0
        for _ in range(R * C):
            x = rng.random()
            if x < 0.40: cells.append('0')
            elif x < 0.65: cells.append('1')
            elif x < 0.90 and nb < 10: cells.append(rng.choice(masks)); nb += 1
            elif nh < max_holes: cells.append('5'); nh += 1
            else: cells.append('0')
        e, p = rng.sample(range(R * C), 2)
        cells[e], cells[p] = '3', '4'
        out.append('/'.join(''.join(cells[r * C:(r + 1) * C]) for r in range(R)))
    return out


MATRIX = [   # (name, grid, exit, flags); about 5 s, 2 minutes and 15 s for all modes
    ('matrix-4x4h3e0', '4x4', 0, ['--allow-exit-transit', '--num-holes', '3']),
    ('matrix-5x5e7b3', '5x5', 7, ['--allow-exit-transit', '--num-blocks', '3']),
    ('matrix-6x6h0e14b2', '6x6', 14, ['--allow-exit-transit', '--num-holes', '0', '--num-blocks', '2']),
]


def matrix(worker, tmp, timeout, out, only=()):
    """Worker vs worker across A/B modes on bigger configs (review M89 fix (2)): identical level sets."""
    cfgs = [c for c in MATRIX if not only or any(o in c[0] for o in only)]
    modes = [m for m in MODES if worker.supports(m)]
    fails = 0
    for label, grid, exit_, flags in cfgs:
        t0 = time.time()
        ref = None
        line = []
        for m in [None] + modes:
            r = TS.run_job(worker, grid, exit_, flags, '', tmp, timeout, argv_extra=[m] if m else [])
            TS.check_run(worker, r, split_expected=False, grid=grid, exit_=exit_, extra=flags)
            if r.summary.get('unknown', 0):
                raise Fail(f'matrix {label} {m}: {r.summary["unknown"]} capacity results; the sets are not comparable')
            s = set((d, canon(c)) for d, c, _ in r.levels)
            if ref is None:
                ref, rbest = s, r.summary['best']
                line.append(f'default {len(s)} levels best {rbest} ({r.wall:.1f}s)')
                continue
            lost, extra = len(ref - s), len(s - ref)
            same = not lost and not extra and r.summary['best'] == rbest
            line.append(f'{m[5:]} {"same" if same else f"LOST {lost} EXTRA {extra} best {r.summary["best"]}"}')
            fails += not same
        out(f'  matrix {label}: ' + '; '.join(line) + f' ({time.time() - t0:.0f}s)')
    return fails


def cross_check(exe, grid, e, holes, blocks, T, E, kind, out):
    """One self-check of the oracle: lazy (forward) or bf (brute force) must equal back."""
    if kind[0] == 'lazy':
        dmax = kind[1]
        Tl, El, st, _ = oracle(exe, grid, e, holes, blocks, dmax, 'lazy')
        Tb = {k for k in T if k[0] <= dmax}
        Eb = {x for x in E if x[0] <= dmax}
        same = set(Tl) == Tb and El == Eb
        note = f'lazy(d<={dmax}) {"=" if same else "DIFFERS"} ({st["nodes"]} nodes {st["elapsed"]:.1f}s)'
    else:
        cap, dmax = kind[1], kind[2]
        cap = cap if blocks is None else min(cap, blocks)
        Tc, Ec, _, _ = oracle(exe, grid, e, holes, cap)
        Tl, El, st, _ = oracle(exe, grid, e, holes, cap, dmax, 'bf')
        Tb = {k for k in Tc if k[0] <= dmax}
        Eb = {x for x in Ec if x[0] <= dmax}
        same = set(Tl) == Tb and El == Eb
        note = f'bf(<={cap} blocks) {"=" if same else "DIFFERS"} ({st["bf_levels"]} levels {st["elapsed"]:.1f}s)'
    if not same:
        out(f'    {note}: T only-{kind[0]} {len(set(Tl) - Tb)}, only-back {len(Tb - set(Tl))}; '
            f'E only-{kind[0]} {len(El - Eb)}, only-back {len(Eb - El)}')
    return same, note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('worker')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', action='append', default=[], help='only configs whose name contains this')
    ap.add_argument('--solver6', default=SOLVER6_DEFAULT, help='solver6.c source, or "none" to skip the judge check')
    ap.add_argument('--no-matrix', action='store_true', help='skip the worker-vs-worker mode matrix')
    ap.add_argument('--no-modes', action='store_true', help='compare only the default mode with the oracle')
    ap.add_argument('--no-cross', action='store_true', help='skip the lazy/bf self-checks of the oracle')
    ap.add_argument('--timeout', type=float, default=120.0, help='per worker run')
    ap.add_argument('--random', type=int, default=2000, help='random levels per grid for the judge check')
    ap.add_argument('--judge-sample', type=int, default=3000, help='oracle levels per config for the judge check')
    a = ap.parse_args()
    t_start = time.time()
    tmp = tempfile.mkdtemp(prefix='oracle_')
    failures = []

    def out(s):
        print(s, flush=True)
    try:
        worker = TS.Worker(a.worker)
        exe, s6 = build(tmp, a.solver6)
        modes = [] if a.no_modes else [m for m in MODES if worker.supports(m)]
        if a.quick:
            modes = [m for m in modes if m in ('--no-bulk-walk', '--no-state-canon')]
        nodes = worker.supports('--split-after-nodes', '5')
        out(f'worker {worker.path} src {worker.version.get("SRC_HASH", "?")[:16]}; modes: default '
            f'{" ".join(modes)}{"; split trees via --split-after-nodes" if nodes else "; no --split-after-nodes: split trees skipped"}'
            f'; judge cross-check: {"solver6" if s6 else "SKIPPED (--solver6 none)"}')
        names = QUICK if a.quick else [c[0] for c in CONFIGS]
        cfgs = [c for c in CONFIGS if c[0] in names and (not a.only or any(o in c[0] for o in a.only))]
        judged = {}
        rng = random.Random(20260924)
        exits_checked = set()
        for name, grid, holes, blocks, cross in cfgs:
            R, C = (int(x) for x in grid.split('x'))
            if grid not in exits_checked:
                exits_checked.add(grid)
                we, reps = worker_exits(worker, grid), orbit_reps(R, C)
                if sorted(we) != reps:
                    out(f'FAIL {grid}: the worker searches exits {we}, the symmetry orbits are represented by {reps}')
                    failures.append(f'{grid}-exits')
                    continue
                out(f'{grid}: canonical exits {reps} (one per symmetry orbit)')
            for e in orbit_reps(R, C):
                t0 = time.time()
                tag = f'{name} exit {e}'
                T, E, st, _raw = oracle(exe, grid, e, holes, blocks)
                if not st['proof']:
                    out(f'FAIL {tag}: the oracle did not exhaust')
                    failures.append(tag)
                    continue
                msg = [f'{tag}: oracle {len(T)} levels, max {st["max"]}, {len(E)} block-on-exit configs '
                       f'(exhausted: {st["back_states"]} states {st["elapsed"]:.1f}s)']
                ok = True
                wstates = None
                for label, extra in [('default', [])] + [(m[5:], [m]) for m in modes]:
                    W, best, nrun, nsplit, wst = worker_set(worker, grid, e, holes, blocks, tmp, a.timeout, extra)
                    wstates = wstates or wst
                    good, nl, nx = compare(label, T, W, best, out)
                    msg.append(f'{label} {"=" if good else "DIFFERS"}')
                    ok &= good
                if nodes and wstates and wstates > 20:
                    sn = max(3, wstates // 60)
                    W, best, nrun, nsplit, _ = worker_set(worker, grid, e, holes, blocks, tmp, a.timeout, [], sn)
                    good, nl, nx = compare('split-tree', T, W, best, out)
                    if nsplit == 0:
                        out('    split-tree: never split'); good = False
                    msg.append(f'split-tree {"=" if good else "DIFFERS"} ({nrun} runs of {sn} nodes)')
                    ok &= good
                if not a.no_cross and (not a.quick or name in QUICK_CROSS):
                    for kind in cross:
                        same, note = cross_check(exe, grid, e, holes, blocks, T, E, kind, out)
                        msg.append(note)
                        ok &= same
                lv = sorted(((d, code) for (d, _), code in T.items()), reverse=True)
                n = a.judge_sample if not a.quick else 300
                pick = lv[:n // 2] + (rng.sample(lv[n // 2:], min(len(lv) - n // 2, n - n // 2)) if len(lv) > n // 2 else [])
                judged.setdefault(grid, []).extend(pick)
                out(('  ' if ok else 'FAIL ') + '; '.join(msg) + f' ({time.time() - t0:.0f}s)')
                if not ok:
                    failures.append(tag)
        if s6:
            tot = bad = 0
            for grid, lv in judged.items():
                R, C = (int(x) for x in grid.split('x'))
                codes = [c for _, c in lv]
                n1, b1 = judge_check(exe, s6, grid, codes, out)
                mine = subprocess.run([exe, '--grid', grid, '--judge'], input='\n'.join(codes) + '\n',
                                      capture_output=True, text=True, timeout=600).stdout.split()
                wrong = sum(1 for (dd, c), m in zip(lv, mine) if m != str(dd))
                rnd = random_levels(R, C, a.random if not a.quick else 300, rng, 2)
                n2, b2 = judge_check(exe, s6, grid, rnd, out)
                out(f'  judge {grid}: {n1} oracle levels (the deepest half, then random), {n2} random levels: '
                    f'{b1 + b2} disagreements with solver6; {wrong} oracle levels judged at another depth')
                tot += n1 + n2
                bad += b1 + b2 + wrong
                if b1 + b2 + wrong:
                    failures.append(f'judge-{grid}')
            out(f'  judge total: {tot} levels, {bad} disagreements')
        if not a.quick and not a.no_matrix:
            if matrix(worker, tmp, a.timeout, out, a.only):
                failures.append('matrix')
    except Fail as e:
        print(f'FAIL: {e}')
        failures.append('error')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print(f'FAILED: {" ".join(failures)} ({time.time() - t_start:.0f}s)')
        return 1
    print(f'OK ({time.time() - t_start:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
