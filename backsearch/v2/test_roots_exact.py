#!/usr/bin/env python3
"""Root-cover exactness gate (review M87 fix 4 / C2; PROTOCOL3 §2.4 "Root listing").

A campaign is seeded with the depth-K layer printed by `--list-layer K`; every root becomes a job
that starts with fresh dedup tables.  The union of the roots' subtrees must therefore contain every
valid level of the monolithic run at depth >= K (levels shallower than the layer are above every root),
and nothing the monolithic run does not find.  For each K this test:

  1. lists the layer with the campaign flags (the client's argv minus --seed-path / --split-after);
     if the worker prints a LAYERINFO header, checks it (k, grid, the exit's root count, flags) against
     the listing; every LAYER line must be well formed, at depth K, and no root may repeat;
  2. runs every root to exhaustion in the client's argv (one process at a time, BS_TRACE_VALID);
  3. compares the union with the monolithic run: LOST levels at depth >= K and EXTRA levels (a root
     found a level the monolithic run pruned) both fail, and so does a best depth that differs.
A worker may refuse a layer (PROTOCOL3's fallback: "refuse K >= 5"): a refusal is accepted for K >= 5
and fails for K <= 4.

  python3 test_roots_exact.py WORKER [--layers 2 4 6] [--timeout 120] -- --grid 4x4 --exit 0 \\
          --allow-exit-transit --num-holes 3
  python3 test_roots_exact.py WORKER --suite quick|campaign [--only NAME]
  python3 test_roots_exact.py WORKER --quick        (= --suite quick)

The lead's measurement this gate encodes: 5x5 exit 7 <=3 blocks with transit, layer 8, lost 9,566
valid levels at depth >= 8 when roots were listed with bulk walk-back off but run with it on.  The
'campaign' suite runs configs of that shape small enough for about 2 minutes each, including 0-hole
configs (campaign #2's flags).  Exit status 0 = all passed, 1 = a failure.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_split_exact as TS          # noqa: E402
from test_split_exact import Fail, canon  # noqa: E402

SEED_RE = TS.SEED_RE

# (name, config, layers)
SUITES = {
    'quick': [
        ('4x4h3e0', ['--grid', '4x4', '--exit', '0', '--allow-exit-transit', '--num-holes', '3'], [2, 4, 6]),
        ('4x4h0e5', ['--grid', '4x4', '--exit', '5', '--allow-exit-transit', '--num-holes', '0'], [4, 8]),
    ],
    'campaign': [
        ('4x4h3e1', ['--grid', '4x4', '--exit', '1', '--allow-exit-transit', '--num-holes', '3'], [4, 6, 8]),
        ('4x4h3e5', ['--grid', '4x4', '--exit', '5', '--allow-exit-transit', '--num-holes', '3'], [4, 8]),
        ('4x5h0e7', ['--grid', '4x5', '--exit', '7', '--allow-exit-transit', '--num-holes', '0'], [4, 6, 8]),
        ('4x5h0e6', ['--grid', '4x5', '--exit', '6', '--allow-exit-transit', '--num-holes', '0'], [4, 8]),
        ('5x5e7b3', ['--grid', '5x5', '--exit', '7', '--allow-exit-transit', '--num-blocks', '3'], [4, 6]),
        ('6x6h0e14b2', ['--grid', '6x6', '--exit', '14', '--allow-exit-transit', '--num-holes', '0', '--num-blocks', '2'],
         [4, 6]),
    ],
}
SUITES['all'] = SUITES['quick'] + SUITES['campaign']


def list_layer(worker, grid, exit_, flags, k, timeout):
    """Run --list-layer k.  Returns (roots, layerinfo or None, refused message or None)."""
    cmd = [worker.path, '--grid', grid, '--two-tables', '--exit', str(exit_), '--time', '0'] + flags + \
        ['--list-layer', str(k)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=TS.clean_env())
    except subprocess.TimeoutExpired:
        raise Fail(f'--list-layer {k} exceeded {timeout}s')
    roots, info, bad = [], None, []
    for line in p.stdout.splitlines():
        if line.startswith('LAYERINFO\t'):
            try:
                info = json.loads(line.split('\t', 1)[1])
            except ValueError:
                raise Fail(f'unparsable LAYERINFO: {line[:200]}')
        elif line.startswith('LAYER\t'):
            f = line.split('\t')
            # LAYER <exit> <path> <depth> ...
            if len(f) < 4 or not f[3].lstrip('-').isdigit():
                bad.append(line)
                continue
            roots.append((int(f[1]), f[2], int(f[3])))
    if p.returncode != 0 or not roots:
        msg = (p.stderr.strip() or p.stdout.strip())[-300:]
        return [], info, f'exit code {p.returncode}: {msg}'
    if bad:
        raise Fail(f'{len(bad)} malformed LAYER lines, e.g. {bad[0][:200]}')
    return roots, info, None


def check_layer(roots, info, grid, exit_, k, flags):
    """Structural checks of one listing.  Raises Fail."""
    for e, path, depth in roots:
        if e != exit_:
            raise Fail(f'LAYER line for exit {e}, asked for exit {exit_}')
        if not SEED_RE.match(path):
            raise Fail(f'malformed root path {path!r}')
        if depth != k:
            raise Fail(f'root {path!r} at depth {depth}, layer {k}')
    paths = [p for _, p, _ in roots]
    if len(set(paths)) != len(paths):
        raise Fail(f'{len(paths) - len(set(paths))} repeated roots in layer {k}')
    notes = []
    if info is not None:
        if info.get('k') != k:
            raise Fail(f'LAYERINFO k {info.get("k")} for layer {k}')
        if 'grid' in info and str(info['grid']) != grid:
            raise Fail(f'LAYERINFO grid {info["grid"]} for {grid}')
        n = (info.get('roots') or {}).get(str(exit_))
        if n is not None and n != len(roots):
            raise Fail(f'LAYERINFO says {n} roots for exit {exit_}, the listing has {len(roots)}')
        fl = info.get('flags') or {}
        if fl.get('transit') not in (None, 1):
            raise Fail(f'LAYERINFO flags transit {fl.get("transit")}')
        if fl.get('block_on_exit') not in (None, 0):
            raise Fail(f'LAYERINFO flags block_on_exit {fl.get("block_on_exit")}')
        if '--num-holes' in flags and fl.get('max_holes') not in (None, int(flags[flags.index('--num-holes') + 1])):
            raise Fail(f'LAYERINFO flags max_holes {fl.get("max_holes")}')
        if fl.get('bulk_walk') not in (None, 1):
            raise Fail(f'LAYERINFO flags bulk_walk {fl.get("bulk_walk")}: roots must be listed with the job tree\'s semantics')
        notes.append('LAYERINFO ok')
    else:
        notes.append('no LAYERINFO header')
    # a root that is a token prefix of another one would double-cover its subtree (not a loss)
    s = set(paths)
    pref = sum(1 for p in paths for i in range(1, len(p.split(','))) if ','.join(p.split(',')[:i]) in s)
    if pref:
        notes.append(f'{pref} roots nested in other roots (overlap, not a loss)')
    return notes


def run_config(worker, cfg, layers, timeout, name='', out=print, max_levels=3_000_000):
    grid, exit_, flags = TS.split_cfg(cfg)
    tmp = tempfile.mkdtemp(prefix='rootsx_')
    fails = []
    try:
        t0 = time.time()
        ref = TS.run_job(worker, grid, exit_, flags, '', tmp, timeout, max_levels=max_levels, keep_paths=True)
        TS.check_run(worker, ref, split_expected=False, grid=grid, exit_=exit_, extra=flags)
        if ref.summary['status'] != 'exhausted':
            raise Fail(f'monolithic run did not exhaust: {ref.summary}')
        full = {}
        for d, code, path in ref.levels:
            full.setdefault((d, canon(code)), path)
        fbest = ref.summary['best']
        out(f'{name + ": " if name else ""}{grid} exit {exit_} {" ".join(flags)}: monolithic {len(full)} canonical '
            f'valid levels, best {fbest}, {ref.summary.get("states")} states ({time.time() - t0:.1f}s)')
        for k in layers:
            t1 = time.time()
            roots, info, refused = list_layer(worker, grid, exit_, flags, k, timeout)
            if refused:
                if k >= 5:
                    out(f'  layer {k}: refused ({refused}) -- allowed for K >= 5')
                    continue
                out(f'  layer {k}: FAIL refused for K <= 4 ({refused})')
                fails.append(f'{name}-K{k}')
                continue
            try:
                notes = check_layer(roots, info, grid, exit_, k, flags)
            except Fail as e:
                out(f'  layer {k}: FAIL {e}')
                fails.append(f'{name}-K{k}')
                continue
            union, extras, statuses, states, unknown = {}, [], {}, 0, ref.summary.get('unknown', 0)
            for _, path, _ in roots:
                r = TS.run_job(worker, grid, exit_, flags, path, tmp, timeout, max_levels=max_levels, keep_paths=True)
                TS.check_run(worker, r, split_expected=False, grid=grid, exit_=exit_, extra=flags)
                st = r.summary['status']
                statuses[st] = statuses.get(st, 0) + 1
                states += r.summary.get('states', 0)
                unknown += r.summary.get('unknown', 0)
                for d, code, p in r.levels:
                    key = (d, canon(code))
                    if key not in full and key not in union and len(extras) < 5:
                        extras.append((key, path, p))
                    union.setdefault(key, p)
            lost = sorted((key for key in full if key[0] >= k and key not in union), reverse=True)
            above = sum(1 for key in full if key[0] < k and key not in union)
            extra_n = sum(1 for key in union if key not in full)
            ubest = max((d for d, _ in union), default=0)
            ok = not lost and not extra_n and (ubest == fbest or fbest < k) and not unknown
            out(f'  layer {k}: {len(roots)} roots ({"; ".join(notes)}), statuses {statuses}, {states} states, union '
                f'{len(union)}; lost at depth >= {k}: {len(lost)}, extra {extra_n}, best {ubest} vs {fbest} '
                f'({above} levels above the layer){f"; {unknown} capacity results (sets not comparable)" if unknown else ""} '
                f'({time.time() - t1:.0f}s){"" if ok else "  FAIL"}')
            for d, c in lost[:3]:
                out(f'    lost depth {d} {c}  monolithic path {full[(d, c)]}')
            for (d, c), root, p in extras[:3]:
                out(f'    extra depth {d} {c}  root {root!r} path {p}')
            if not ok:
                fails.append(f'{name}-K{k}')
        return fails
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('worker')
    ap.add_argument('--layers', type=int, nargs='+', default=[2, 4, 6])
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--suite', choices=sorted(SUITES))
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', action='append', default=[])
    argv = sys.argv[1:]
    cfg = []
    if '--' in argv:
        i = argv.index('--')
        cfg, argv = argv[i + 1:], argv[:i]
    a = ap.parse_args(argv)
    if a.quick:
        a.suite = 'quick'
    t0 = time.time()
    try:
        worker = TS.Worker(a.worker)
        print(f'worker {worker.path}: src {worker.version.get("SRC_HASH", "?")[:16]} protocol '
              f'{worker.version.get("PROTOCOL", "<3")}', flush=True)
        if a.suite:
            todo = [c for c in SUITES[a.suite] if not a.only or any(o in c[0] for o in a.only)]
        elif cfg:
            todo = [('', cfg, a.layers)]
        else:
            ap.error('give a worker config after --, or --suite')
        fails = []
        for name, c, layers in todo:
            try:
                fails += run_config(worker, c, layers, a.timeout, name=name,
                                    out=lambda s: print(s, flush=True))
            except Fail as e:
                print(f'FAIL {name}: {e}', flush=True)
                fails.append(name)
    except Fail as e:
        print(f'FAIL: {e}')
        return 1
    if fails:
        print(f'FAILED: {" ".join(fails)} ({time.time() - t0:.0f}s)')
        return 1
    print(f'OK ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
