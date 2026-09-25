#!/usr/bin/env python3
"""Root-cover exactness gate (review M87 fix 4 / C2; PROTOCOL3 §2.4 "Root listing").

A campaign is seeded with the root layer printed by `--list-layer K`; every root becomes a job that
starts with fresh dedup tables.  The listing cuts the JOB TREE (bulk walk-back on, as the jobs run): a
root is a node of depth >= K whose tree parent is shallower, so a bulk walk-back child can be a root
deeper than K, and every tree node of depth >= K lies below some root.  The union of the roots' subtrees
must therefore contain every valid level of the monolithic run at depth >= K, and nothing the
monolithic run does not find; the levels above the layer belong to no job, and the header's
shallow_best must be their best.  For each K this test:

  1. lists the layer with the campaign flags (the client's argv minus --seed-path / --split-after) and
     checks it as v2_seed will (server/lib/v2seed.js): a protocol-3 worker MUST print the LAYERINFO
     header, with this k and grid, its own SRC_HASH, this exit's root count, and flags that describe the
     search with bulk_walk 1 (the job tree's semantics); every LAYER line has exactly 6 fields (LAYER,
     exit, path, depth, blocks, holes) and comes after the header, every root has depth == tokens >= K
     (a root deeper than K must end in a run of walk tokens reaching back above the layer: a bulk
     child) and respects the hole / block caps, no root repeats; shallow_best must equal the
     monolithic run's best above the layer (depth < K);
  2. runs every root to exhaustion in the client's argv (--jobs N at a time, BS_TRACE_VALID);
  3. compares the union with the monolithic run: LOST levels at depth >= K and EXTRA levels (a root
     found a level the monolithic run pruned) both fail, and so does a best depth that differs
     (max of the union's best and shallow_best).
A worker may refuse a layer (PROTOCOL3's fallback: "refuse K >= 5"): a refusal (a non-zero exit code)
is accepted for K >= 5 and fails for K <= 4.  Exit code 0 with no LAYER lines is not a refusal but an
empty listing: every level at depth >= K is then lost.  An --only filter that selects nothing fails.

  python3 test_roots_exact.py WORKER [--layers 2 4 6] [--timeout 120] [--jobs N] -- --grid 4x4 --exit 0 \\
          --allow-exit-transit --num-holes 3
  python3 test_roots_exact.py WORKER --suite quick|campaign [--only NAME]
  python3 test_roots_exact.py WORKER --quick        (= --suite quick)

The lead's measurement this gate encodes: 5x5 exit 7 <=3 blocks with transit, layer 8, lost 9,566
valid levels at depth >= 8 when roots were listed with bulk walk-back off but run with it on (the
job-tree listing adds 107 roots at depth 9-10 there, the bulk children the old listing lost).  The
'campaign' suite runs configs of that shape small enough for about 2 minutes each, including 0-hole
6x6 configs (campaign #2's flags); 'deep' is the lead's layer-8 case (about 20k roots: use --jobs 2).
Exit status 0 = all passed, 1 = a failure.
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
        ('6x6h0e14d13', ['--grid', '6x6', '--exit', '14', '--allow-exit-transit', '--num-holes', '0', '--max-depth', '13'],
         [4, 6]),
    ],
    'deep': [
        ('5x5e7b3', ['--grid', '5x5', '--exit', '7', '--allow-exit-transit', '--num-blocks', '3'], [8]),
    ],
}
SUITES['all'] = SUITES['quick'] + SUITES['campaign']


def list_layer(worker, grid, exit_, flags, k, timeout):
    """Run --list-layer k.  Returns (roots, layerinfo or None, refused message or None, errors).
    roots are (exit, path, depth, blocks, holes).  Only a non-zero exit code is a refusal: exit code 0
    with no LAYER lines is an EMPTY listing (every level at depth >= k is then lost), never a refusal.
    errors: format problems v2_seed would reject (server/lib/v2seed.js parseLayerText)."""
    cmd = [worker.path, '--grid', grid, '--two-tables', '--exit', str(exit_), '--time', '0'] + flags + \
        ['--list-layer', str(k)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=TS.clean_env())
    except subprocess.TimeoutExpired:
        raise Fail(f'--list-layer {k} exceeded {timeout}s')
    if p.returncode != 0:
        msg = (p.stderr.strip() or p.stdout.strip())[-300:]
        return [], None, f'exit code {p.returncode}: {msg}', []
    roots, info, errors, before = [], None, [], 0
    for n, line in enumerate(p.stdout.splitlines(), 1):
        if line.startswith('LAYERINFO'):
            if not line.startswith('LAYERINFO\t'):
                errors.append(f'line {n}: malformed LAYERINFO header')
                continue
            if info is not None:
                errors.append(f'line {n}: a second LAYERINFO header in a one-exit listing')
                continue
            try:
                info = json.loads(line.split('\t', 1)[1])
            except ValueError:
                errors.append(f'line {n}: LAYERINFO is not valid JSON: {line[:200]}')
                continue
            if not isinstance(info, dict):
                errors.append(f'line {n}: LAYERINFO is not a JSON object')
                info = None
            continue
        if line.find('LAYER', 1) >= 0:
            errors.append(f'line {n}: a LAYER record in the middle of a line (two lines merged?): {line[:120]}')
            continue
        if line.startswith('LAYER'):
            f = line.split('\t')
            # LAYER <exit> <path> <depth> <blocks> <holes>: exactly 6 fields, integers where v2seed wants them
            if f[0] != 'LAYER' or len(f) != 6 or not all(x.isdigit() for x in (f[1], f[3], f[4], f[5])) or \
                    not SEED_RE.match(f[2]):
                errors.append(f'line {n}: malformed LAYER line (want LAYER, exit, seed path, depth, blocks, holes): '
                              f'{line[:200]}')
                continue
            if info is None:
                before += 1
            roots.append((int(f[1]), f[2], int(f[3]), int(f[4]), int(f[5])))
    if info is not None and before:          # (no header at all is reported by check_layer)
        errors.append(f'{before} LAYER lines before the LAYERINFO header')
    return roots, info, None, errors


def _cap(flags, name):
    return int(flags[flags.index(name) + 1]) if name in flags else None


def flags_mismatch(fl, grid, exit_, flags):
    """None when a flags record describes this search, as the server checks it (server/lib/jobs.js
    sanitizeFlags + flagsMismatch: grid required, transit 1, block_on_exit 0, bulk_walk 1, the hole /
    block caps, min_walls; an uncapped count must be at least the grid's cells - 2, blocks capped at 32
    by the worker), else the reason."""
    if not isinstance(fl, dict):
        return f'not an object: {fl!r}'
    R, C = (int(x) for x in grid.split('x'))
    cells = R * C
    if fl.get('grid') != grid:
        return f'grid {fl.get("grid")!r} != {grid!r} (required)'
    if 'exit' in fl and fl['exit'] != exit_:
        return f'exit {fl["exit"]!r} != {exit_}'
    for x, v in (('transit', 1), ('block_on_exit', 0), ('bulk_walk', 1)):
        if fl.get(x) != v:
            return f'{x} {fl.get(x)!r} != {v}' + (' (roots must be listed with the job tree\'s semantics)'
                                                  if x == 'bulk_walk' else '')
    for x in ('max_holes', 'max_blocks', 'min_walls'):
        if not isinstance(fl.get(x), int) or isinstance(fl.get(x), bool) or fl[x] < 0:
            return f'{x} {fl.get(x)!r} is not a non-negative integer'
    mh, mb = _cap(flags, '--num-holes'), _cap(flags, '--num-blocks')
    if (fl['max_holes'] != mh) if mh is not None else (fl['max_holes'] < cells - 2):
        return f'max_holes {fl["max_holes"]} for --num-holes {mh}'
    if (fl['max_blocks'] != mb) if mb is not None else (fl['max_blocks'] < min(32, cells - 2)):
        return f'max_blocks {fl["max_blocks"]} for --num-blocks {mb}'
    if fl['min_walls'] != (_cap(flags, '--min-walls') or 0):
        return f'min_walls {fl["min_walls"]}'
    return None


def check_layer(worker, roots, info, grid, exit_, k, flags, errors=()):
    """Structural checks of one listing (PROTOCOL3 §2.4, and what v2_seed requires of it).  A protocol-3
    worker must print the LAYERINFO header, with the listing's k, grid, the search flags (transit on,
    no block on the exit, the hole / block caps, bulk_walk 1 = the job tree's semantics), its own
    SRC_HASH and this exit's root count.  Raises Fail; returns notes."""
    p3 = worker.version.get('PROTOCOL') == '3'
    if info is None and p3:
        raise Fail('no LAYERINFO header: a protocol-3 worker must print it (PROTOCOL3 §2.4), and v2_seed refuses '
                   'a listing without one')
    if errors:
        raise Fail(f'{len(errors)} listing format errors, e.g. {errors[0]}')
    max_h, max_b = _cap(flags, '--num-holes'), _cap(flags, '--num-blocks')
    for e, path, depth, blocks, holes in roots:
        if e != exit_:
            raise Fail(f'LAYER line for exit {e}, asked for exit {exit_}')
        t = path.split(',')
        if depth < k or len(t) != depth or (info is None and depth != k):
            raise Fail(f'root {path!r} has depth {depth} and {len(t)} tokens; the layer is {k} (roots have '
                       f'depth == tokens >= K{", == K for a pre-protocol-3 lister" if info is None else ""})')
        if depth > k and not all(x.endswith('1') for x in t[k - 1:]):
            # a root deeper than K is a bulk walk-back child of a node above the layer (depth < K): its
            # last (depth - parent depth) tokens are one walk chain, which covers tokens k-1 .. depth-1
            raise Fail(f'root {path!r} is deeper than the layer {k} but its tokens from {k} on are not one walk '
                       f'chain: its parent would be at depth >= {k}, i.e. a root or below one')
        if max_h is not None and holes > max_h:
            raise Fail(f'root {path!r} has {holes} holes > --num-holes {max_h}')
        if max_b is not None and blocks > max_b:
            raise Fail(f'root {path!r} has {blocks} blocks > --num-blocks {max_b}')
    paths = [r[1] for r in roots]
    if len(set(paths)) != len(paths):
        raise Fail(f'{len(paths) - len(set(paths))} repeated roots in layer {k}')
    notes = []
    if info is None:
        notes.append('no LAYERINFO header (pre-protocol-3 worker)')
    else:
        if info.get('k') != k:
            raise Fail(f'LAYERINFO k {info.get("k")!r} for layer {k}')
        if str(info.get('grid')) != grid:
            raise Fail(f'LAYERINFO grid {info.get("grid")!r} for {grid}')
        want_hash = worker.version.get('SRC_HASH')
        if want_hash and info.get('src_hash') != want_hash:
            raise Fail(f'LAYERINFO src_hash {str(info.get("src_hash"))[:16]!r} is not the worker\'s SRC_HASH '
                       f'{want_hash[:16]}')
        rmap = info.get('roots')
        if not isinstance(rmap, dict) or str(exit_) not in rmap:
            raise Fail(f'LAYERINFO roots {rmap!r} does not give exit {exit_}\'s root count')
        if set(rmap) != {str(exit_)}:
            raise Fail(f'LAYERINFO roots {rmap!r} names exits other than {exit_}')
        if rmap[str(exit_)] != len(roots):
            raise Fail(f'LAYERINFO says {rmap[str(exit_)]} roots for exit {exit_}, the listing has {len(roots)}')
        why = flags_mismatch(info.get('flags'), grid, exit_, flags)
        if why:
            raise Fail(f'LAYERINFO flags: {why}')
        notes.append('LAYERINFO ok')
    # a root that is a token prefix of another one would double-cover its subtree (not a loss)
    s = set(paths)
    pref = sum(1 for p in paths for i in range(1, len(p.split(','))) if ','.join(p.split(',')[:i]) in s)
    if pref:
        notes.append(f'{pref} roots nested in other roots (overlap, not a loss)')
    return notes


def check_shallow_best(info, full, k, exit_):
    """LAYERINFO.shallow_best: the exit's best level above the root layer (the nodes of depth < K, which
    no job reports).  Returns (note, depth).  A header without the key is a pre-job-tree lister (note
    only); with it, the exit's value must be null exactly when the monolithic run has no level above the
    layer, else its depth must be the monolithic run's best above the layer (every tree node of depth < K
    is expanded by the listing, and a state is valid at one depth only), and a code given with it must be
    a monolithic level at that depth."""
    lo = max((d for d, _ in full if d < k), default=0)
    if info is None or 'shallow_best' not in info:
        return (None, 0)
    sbm = info.get('shallow_best')
    if not isinstance(sbm, dict) or str(exit_) not in sbm:
        raise Fail(f'LAYERINFO shallow_best {sbm!r} does not give exit {exit_}')
    sb, code = sbm[str(exit_)], None
    if sb is None:
        if lo:
            raise Fail(f'LAYERINFO shallow_best is null, the monolithic run\'s best above layer {k} is {lo}')
        return ('shallow_best null ok', 0)
    if isinstance(sb, dict):
        code, sb = sb.get('code'), sb.get('depth')
    if not isinstance(sb, int) or isinstance(sb, bool) or sb < 1:
        raise Fail(f'LAYERINFO shallow_best {sb!r} is not a depth')
    if sb != lo:
        raise Fail(f'LAYERINFO shallow_best {sb}, the monolithic run\'s best above layer {k} is {lo}')
    if code is not None and (sb, canon(code)) not in full:
        raise Fail(f'LAYERINFO shallow_best code {code!r} is not a monolithic level at depth {sb}')
    return (f'shallow_best {sb} ok', sb)


def run_roots(worker, grid, exit_, flags, roots, tmp, timeout, max_levels, jobs):
    """Run every root to exhaustion, `jobs` worker processes at a time (each thread in its own directory,
    so trace files never collide).  Yields (root path, Run) as runs finish; run_job kills a worker that
    exceeds the timeout, and a failure cancels the runs not started yet."""
    if jobs <= 1:
        for _, path, _, _, _ in roots:
            yield path, TS.run_job(worker, grid, exit_, flags, path, tmp, timeout, max_levels=max_levels, keep_paths=True)
        return
    import concurrent.futures as cf
    import threading
    local = threading.local()

    def one(path):
        if not hasattr(local, 'dir'):
            local.dir = tempfile.mkdtemp(prefix='t', dir=tmp)
        return path, TS.run_job(worker, grid, exit_, flags, path, local.dir, timeout, max_levels=max_levels,
                                keep_paths=True)
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        it = iter(roots)
        pending = set()
        try:
            for _ in range(2 * jobs):                    # a bounded window: results are consumed as they come
                r = next(it, None)
                if r is None:
                    break
                pending.add(ex.submit(one, r[1]))
            while pending:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    yield f.result()
                    r = next(it, None)
                    if r is not None:
                        pending.add(ex.submit(one, r[1]))
        finally:
            for f in pending:
                f.cancel()


def run_config(worker, cfg, layers, timeout, name='', out=print, max_levels=3_000_000, jobs=1):
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
        ref.levels = None                                # bounded memory: keep one entry per canonical level
        fbest = ref.summary['best']
        out(f'{name + ": " if name else ""}{grid} exit {exit_} {" ".join(flags)}: monolithic {len(full)} canonical '
            f'valid levels, best {fbest}, {ref.summary.get("states")} states ({time.time() - t0:.1f}s)')
        for k in layers:
            t1 = time.time()
            roots, info, refused, errors = list_layer(worker, grid, exit_, flags, k, timeout)
            if refused:
                if k >= 5:
                    out(f'  layer {k}: refused ({refused}) -- allowed for K >= 5')
                    continue
                out(f'  layer {k}: FAIL refused for K <= 4 ({refused})')
                fails.append(f'{name}-K{k}')
                continue
            problem, sb_depth = None, 0
            try:
                notes = check_layer(worker, roots, info, grid, exit_, k, flags, errors)
                sb, sb_depth = check_shallow_best(info, full, k, exit_)
                if sb:
                    notes.append(sb)
                deeper = sum(1 for r in roots if r[2] > k)
                if deeper:
                    notes.append(f'{deeper} roots deeper than {k} (bulk walk-back children)')
            except Fail as e:
                # a listing v2_seed would refuse fails the layer; the roots are still run (when they are
                # at depth K) so a cover loss is reported as well
                problem, notes = str(e), []
                out(f'  layer {k}: FAIL {e}')
                if any(r[2] < k or len(r[1].split(',')) != r[2] or r[0] != exit_ for r in roots):
                    fails.append(f'{name}-K{k}')
                    continue
            if not roots:
                notes.append('EMPTY listing (exit code 0, no LAYER lines)')
            union, extras, statuses, states, unknown = set(), [], {}, 0, ref.summary.get('unknown', 0)
            extra_n = 0
            for path, r in run_roots(worker, grid, exit_, flags, roots, tmp, timeout, max_levels, jobs):
                TS.check_run(worker, r, split_expected=False, grid=grid, exit_=exit_, extra=flags)
                st = r.summary['status']
                statuses[st] = statuses.get(st, 0) + 1
                states += r.summary.get('states', 0)
                unknown += r.summary.get('unknown', 0)
                for d, code, p in r.levels:
                    key = (d, canon(code))
                    if key in union:
                        continue
                    union.add(key)
                    if key not in full:
                        extra_n += 1
                        if len(extras) < 5:
                            extras.append((key, path, p))
            lost = sorted((key for key in full if key[0] >= k and key not in union), reverse=True)
            above = sum(1 for key in full if key[0] < k and key not in union)
            ubest = max((d for d, _ in union), default=0)
            best = max(ubest, sb_depth)                  # the exit's best: the jobs' and the header's shallow_best
            ok = not lost and not extra_n and (best == fbest or (fbest < k and sb_depth == 0 and info is None)) \
                and not unknown and problem is None
            out(f'  layer {k}: {len(roots)} roots{" (" + "; ".join(notes) + ")" if notes else ""}, statuses {statuses}, {states} states, union '
                f'{len(union)}; lost at depth >= {k}: {len(lost)}, extra {extra_n}, best {best} vs {fbest} '
                f'({above} levels above the layer){f"; {unknown} capacity results (sets not comparable)" if unknown else ""} '
                f'({time.time() - t1:.0f}s){"" if ok else "  FAIL"}{" (listing: see above)" if problem else ""}')
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
    ap.add_argument('--jobs', type=int, default=1, help='root jobs run in parallel (run under cpuslot -n N)')
    argv = sys.argv[1:]
    cfg = []
    if '--' in argv:
        i = argv.index('--')
        cfg, argv = argv[i + 1:], argv[:i]
    a = ap.parse_args(argv)
    if a.quick:
        a.suite = 'quick'
    TS.install_signal_handlers()
    t0 = time.time()
    try:
        worker = TS.Worker(a.worker)
        print(f'worker {worker.path}: src {worker.version.get("SRC_HASH", "?")[:16]} protocol '
              f'{worker.version.get("PROTOCOL", "<3")}', flush=True)
        if a.suite:
            todo = TS.select_only(SUITES[a.suite], a.only)
        elif cfg:
            todo = [('', cfg, a.layers)]
        else:
            ap.error('give a worker config after --, or --suite')
        fails = []
        for name, c, layers in todo:
            try:
                fails += run_config(worker, c, layers, a.timeout, name=name,
                                    out=lambda s: print(s, flush=True), jobs=max(1, a.jobs))
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
