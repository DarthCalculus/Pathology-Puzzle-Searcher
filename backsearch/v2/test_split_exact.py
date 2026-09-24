#!/usr/bin/env python3
"""Split exactness test for the v2 worker protocol (v2/DESIGN.md §5.1, v2/PROTOCOL3.md §2).

For one configuration (or every configuration of a --suite) it runs the REFERENCE search once,
monolithically, over the whole tree or over the subtree under --root SEED, and then re-runs the same
subtree as a tree of jobs: every job is split (--split-after S, --split-after-nodes N, or SIGINT after
S seconds), its REMAINING seeds become new jobs, until every job has exhausted.  It asserts:

  * levels: the union of the jobs' valid levels (BS_TRACE_VALID, canonical under the 8 symmetries)
    EQUALS the reference run's set.  A LOST level is a hole in the hand-back; an EXTRA level (found by
    a fresh-table job, missed by the reference run) means the reference run pruned a real level, i.e.
    a dedup / replay / prune unsoundness (review M87).  Both fail.  The best depth must agree.
  * traces: every run's trace has exactly SUMMARY.valid well-formed lines, each with depth == its path's
    token count (a torn trace is a failure).
  * protocol (PROTOCOL3 §2.2-2.3): status is exhausted or split and the exit code 0; an exhausted run
    prints no REMAINING, a split run at least one; every REMAINING path is well formed, at most
    path_tok_max tokens, and strictly extends its job's seed by whole tokens (never equal to it, never
    empty); SUMMARY.seed echoes the seed; one src_hash across all runs; SUMMARY.flags describe the
    config; at most one LEVEL line, at the run's best depth, and one of its traced levels; every
    UNRESOLVED line is deeper than the best, a node of the subtree, with cause pq|probe|big, and their
    number is SUMMARY.unresolved.
  * the split mode really split (at least one split run), and with --expect-evictions the reference run
    really evicted (for the small-table knob build).
Visit coverage (BS_TRACE_VISITS) is informational and off by default: --visits reports it, --strict
fails on reference nodes the tree never expanded (normally dedup twins, so --strict is for debugging).

The worker argv is the client's (volunteer.py worker_argv: --grid G --two-tables --exit E --seed-path S
--time 0 --split-after X --status-every 250 + the campaign's extra flags); the harness checks that it
still matches the client's when volunteer.py is importable.  The config after `--` must therefore give
--grid and --exit, plus the campaign flags (always --allow-exit-transit).

Resource bounds: one worker process at a time; every run is killed (process group) after --timeout
seconds and the test fails; trace files are deleted as soon as they are read; a run that traces more
than --max-levels levels aborts the test (memory bound).

  python3 test_split_exact.py WORKER [--split S | --split-nodes N | --sigint --split S] [--root SEED] \\
          [--timeout 120] [--expect-evictions] -- --grid 4x4 --exit 0 --allow-exit-transit --num-holes 3
  python3 test_split_exact.py WORKER --suite quick|campaign|all [--only NAME] [--src DIR]
  python3 test_split_exact.py WORKER --quick          (= --suite quick)

Suites (each config about 2 minutes or less on one CPU; see build_suites): 'quick' is a smoke test;
'campaign' is the review-M88 matrix: 0 holes with many blocks (4x4 and 4x5 whole exits, 5x5 and 6x6
subtrees), <=3 holes with unbounded blocks on 4x4, real campaign-#1 subtrees (deep paths, 8-11 blocks),
SIGINT splits, and eviction runs on a test-only build with tiny dedup tables (-DSHALLOW_LG2=12
-DRECENT_LG2=12, built from --src into a temp dir, its SRC_HASH marked TEST-BUILD and checked to differ
from the tested worker's, its KNOBS checked to be smaller; never a volunteer binary).  When the worker
has --split-after-nodes, 'campaign' also runs deterministic node-count splits (4x4, 5x5, 6x6).
The whole 'campaign' suite is about 10-12 CPU-minutes; run it config by config with --only if needed.
Exit status 0 = all passed, 1 = a failure (the message says which).
"""
import argparse, glob, json, os, re, shutil, signal, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_RE = re.compile(r"^(?:[URDL][123])(?:,[URDL][123])*$")
DEFAULT_PATH_TOK_MAX = 250      # workers before protocol 3 (no LIMITS line in --version)
STATUS_EVERY_MS = 250           # volunteer.py's value; refreshed from volunteer.py when importable


class Fail(Exception):
    pass


# ------------------------------------------------------------------------------------ canonical codes
M2C = {1: '7', 2: '8', 3: 'B', 4: '9', 5: 'J', 6: 'C', 7: 'E', 8: '6', 9: 'A', 10: 'I', 11: 'H', 12: 'D',
       13: 'G', 14: 'F', 15: '2'}
C2M = {v: k for k, v in M2C.items()}


def _tmask(m, swap, fr, fc):
    b = [m & 1, m & 2, m & 4, m & 8]            # U R D L
    if swap: b = [b[3], b[2], b[1], b[0]]       # transpose: U<->L, R<->D
    if fr: b = [b[2], b[1], b[0], b[3]]         # flip rows: U<->D
    if fc: b = [b[0], b[3], b[2], b[1]]         # flip columns: R<->L
    return (1 if b[0] else 0) | (2 if b[1] else 0) | (4 if b[2] else 0) | (8 if b[3] else 0)


_TRANSFORMS = {}


def _transforms(R, C):
    """The 8 dihedral images of an RxC code as (index list into the flat code + '/', char table)."""
    t = _TRANSFORMS.get((R, C))
    if t is None:
        t = []
        for swap in (0, 1):
            for fr in (0, 1):
                for fc in (0, 1):
                    g = [[r * C + c for c in range(C)] for r in range(R)]
                    if swap: g = [[g[r][c] for r in range(R)] for c in range(C)]
                    if fr: g = g[::-1]
                    if fc: g = [row[::-1] for row in g]
                    idx = []
                    for i, row in enumerate(g):
                        if i: idx.append(R * C)             # the '/' appended to the flat code
                        idx.extend(row)
                    tbl = str.maketrans({ch: M2C[_tmask(m, swap, fr, fc)] for ch, m in C2M.items()})
                    t.append((idx, tbl))
        _TRANSFORMS[(R, C)] = t
    return t


def canon(code):
    """Min over the 8 dihedral images of a level code (rows joined by '/'), block masks remapped
    (U=1 R=2 D=4 L=8), so mirrored twins compare equal.  Same result as the historical canon()."""
    rows = code.split('/')
    R, C = len(rows), len(rows[0])
    get = (''.join(rows) + '/').__getitem__
    return min(''.join(map(get, idx)).translate(tbl) for idx, tbl in _transforms(R, C))


def code_ok(code, R, C):
    rows = code.split('/')
    return len(rows) == R and all(len(r) == C for r in rows) and code.count('3') <= 1 and code.count('4') == 1


# ------------------------------------------------------------------------------------ the worker
def clean_env(extra=None):
    """The worker environment: no inherited BS_* debug switches, BS_ALLOW_DEBUG=1 so traces work in
    protocol mode (PROTOCOL3 §2.4), plus the given variables."""
    env = {k: v for k, v in os.environ.items() if not k.startswith('BS_')}
    env['BS_ALLOW_DEBUG'] = '1'
    env.update(extra or {})
    return env


class Worker:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise Fail(f'worker binary not found: {path}')
        self.version = {}
        self.path_tok_max = DEFAULT_PATH_TOK_MAX
        self._flags = {}
        try:
            out = subprocess.run([self.path, '--version'], capture_output=True, text=True, timeout=30,
                                 env=clean_env()).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Fail(f'cannot run {path} --version: {e}')
        for line in out.splitlines():
            f = line.split('\t', 1)
            if len(f) == 2:
                self.version[f[0]] = f[1]
        if 'LIMITS' in self.version:
            try:
                self.path_tok_max = int(json.loads(self.version['LIMITS'])['path_tok_max'])
            except (ValueError, KeyError, TypeError):
                raise Fail(f'unparsable LIMITS line: {self.version["LIMITS"]}')

    def supports(self, *flag_and_args):
        """True if the worker accepts these arguments (a trivial 2x3 run with them exits 0)."""
        if flag_and_args not in self._flags:
            cmd = [self.path, '--grid', '2x3', '--exit', '0', '--allow-exit-transit', '--num-holes', '0',
                   '--time', '0'] + list(flag_and_args)
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=clean_env())
                self._flags[flag_and_args] = p.returncode == 0 and '\nSUMMARY\t' in '\n' + p.stdout
            except subprocess.TimeoutExpired:
                self._flags[flag_and_args] = False
        return self._flags[flag_and_args]


def client_argv(worker, grid, exit_, seed, split_after, extra):
    """The client's worker argv (volunteer.py Volunteer.worker_argv), rebuilt here and cross-checked
    against the real function by check_client_argv().  The one deliberate difference: the split time is
    printed with %g, not %.1f, so tests can split after 0.02 s (the client never splits below 1 s)."""
    argv = [worker, '--grid', grid, '--two-tables', '--exit', str(exit_)]
    if seed != '':
        argv += ['--seed-path', seed]            # the client runs an exit root "" without --seed-path
    return argv + ['--time', '0', '--split-after', '%g' % split_after, '--status-every', str(STATUS_EVERY_MS)] + \
        list(extra)


def check_client_argv():
    """Compare client_argv with volunteer.py's own worker_argv (review M88 (a)).  Returns a note; raises
    Fail if they differ."""
    global STATUS_EVERY_MS
    sys.path.insert(0, HERE)
    try:
        import volunteer
    except Exception as e:                      # the client is owned elsewhere; do not fail on import
        return f'volunteer.py not importable ({type(e).__name__}: {e}); argv not cross-checked'
    finally:
        sys.path.pop(0)
    STATUS_EVERY_MS = getattr(volunteer, 'STATUS_EVERY_MS', STATUS_EVERY_MS)

    import threading
    flags = ['--allow-exit-transit', '--num-holes', '3']

    class Fake:                                  # the attributes worker_argv may read, in any client version
        campaign = {'grid': '4x4', 'extra': list(flags)}
        extra = list(flags)
        worker_base = ['W']
        lock = threading.RLock()
        args = None

        def param(self, k, d):
            v = self.campaign.get(k)
            return d if v is None else v

        def worker_argv_base(self):
            return ['W']
    fn = getattr(getattr(volunteer, 'Volunteer', None), 'worker_argv', None)
    if fn is None:
        return 'volunteer.Volunteer.worker_argv not found; argv not cross-checked'
    for seed in ('U1,L2', ''):
        try:
            theirs = fn(Fake(), {'exit': 0, 'id': 1, 'seed': seed}, seed, 1.5)
        except Exception as e:
            return f'volunteer.worker_argv raised {type(e).__name__}: {e}; argv not cross-checked'
        mine = client_argv('W', '4x4', 0, seed, 1.5, flags)
        if theirs != mine:
            raise Fail(f'the harness argv no longer matches the client:\n  client  {theirs}\n  harness {mine}\n'
                       'update client_argv() in test_split_exact.py')
    return 'argv matches volunteer.worker_argv'


def split_cfg(cfg):
    """Pull --grid and --exit out of a worker config; the rest are campaign flags."""
    grid = exit_ = None
    rest = []
    i = 0
    while i < len(cfg):
        if cfg[i] == '--grid' and i + 1 < len(cfg):
            grid = cfg[i + 1]; i += 2
        elif cfg[i] == '--exit' and i + 1 < len(cfg):
            exit_ = int(cfg[i + 1]); i += 2
        elif cfg[i] in ('--seed-path', '--split-after', '--split-after-nodes', '--time', '--status-every',
                        '--two-tables'):
            raise Fail(f'{cfg[i]} is set by the harness (use --root for a subtree)')
        else:
            rest.append(cfg[i]); i += 1
    if grid is None or exit_ is None:
        raise Fail('the config must give --grid RxC and --exit E')
    if '--allow-exit-transit' not in rest:
        raise Fail('campaign configs always pass --allow-exit-transit (owner rule)')
    if '--allow-block-on-exit' in rest:
        raise Fail('--allow-block-on-exit is never used (owner rule)')
    return grid, exit_, rest


class Run:
    """The parsed result of one worker run."""
    __slots__ = ('seed', 'rc', 'summary', 'remaining', 'levels', 'nlines', 'bad', 'visits', 'unresolved', 'tail',
                 'wall', 'level_lines')


def toks(p):
    return p.split(',') if p else []


_RUN_N = [0]


def run_job(worker, grid, exit_, extra, seed, tmp, timeout, split_s=0.0, split_nodes=0, sigint=False,
            visits=False, max_levels=3_000_000, keep_paths=False, argv_extra=()):
    """One worker run in the client's argv.  split_s > 0: --split-after split_s (or SIGINT after split_s
    with sigint); split_nodes > 0: --split-after-nodes N.  Returns a Run; raises Fail on a timeout."""
    R, C = (int(x) for x in grid.split('x'))
    _RUN_N[0] += 1
    tag = os.path.join(tmp, 'r%d' % _RUN_N[0])
    tv, tr = tag + '.tv', tag + '.tr'
    env = clean_env({'BS_TRACE_VALID': tv})
    if visits:
        env['BS_TRACE_VISITS'] = tr
    cmd = client_argv(worker.path, grid, exit_, seed, 0.0 if (sigint or split_nodes) else split_s, extra)
    cmd += list(argv_extra)
    if split_nodes:
        cmd += ['--split-after-nodes', str(split_nodes)]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         start_new_session=True)

    def kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        for f in glob.glob(tv + '.*') + glob.glob(tr + '.*'):
            os.remove(f)
    try:
        if sigint and split_s > 0:
            try:
                p.communicate(timeout=split_s)
            except subprocess.TimeoutExpired:
                p.send_signal(signal.SIGINT)
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill()
        raise Fail(f'worker run exceeded --timeout {timeout:.0f}s and was killed: seed {seed!r}')
    except BaseException:
        kill()
        raise
    r = Run()
    r.seed, r.rc, r.wall = seed, p.returncode, time.time() - t0
    r.summary, r.remaining, r.unresolved, r.level_lines = None, [], [], []
    r.tail = out[-600:] + '\n[stderr] ' + err[-600:]
    for line in out.split('\n'):
        if line.startswith('SUMMARY\t'):
            try:
                r.summary = json.loads(line[8:])
            except ValueError:
                raise Fail(f'unparsable SUMMARY (seed {seed!r}): {line[:300]}')
        elif line.startswith('REMAINING\t'):
            r.remaining.append(line[10:])
        elif line.startswith('UNRESOLVED\t'):
            r.unresolved.append(line[11:])
        elif line.startswith('LEVEL\t'):
            r.level_lines.append(line[6:])
    r.levels, r.nlines, r.bad = [], 0, 0
    files = glob.glob(tv + '.*')
    try:
        if len(files) > 1:
            raise Fail(f'{len(files)} trace files for one run (seed {seed!r})')
        for f in files:
            with open(f) as fh:
                for line in fh:
                    r.nlines += 1
                    if r.nlines > max_levels:
                        raise Fail(f'a run traced more than --max-levels {max_levels} levels (seed {seed!r}); '
                                   'pick a smaller subtree')
                    parts = line[:-1].split('\t') if line.endswith('\n') else None
                    if not parts or len(parts) != 3 or not parts[1].isdigit() or not code_ok(parts[2], R, C) or \
                            len(toks(parts[0])) != int(parts[1]):      # a node's depth is its path's token count
                        r.bad += 1
                        continue
                    # depth 0 is the root (player on the exit), not a level; the worker traces it only
                    # when given --seed-path '' (the client runs a root without --seed-path)
                    if parts[1] != '0':
                        r.levels.append((int(parts[1]), parts[2], parts[0] if keep_paths else None))
    finally:
        for f in files:
            os.remove(f)
    r.visits = None
    if visits:
        r.visits = set()
        for f in glob.glob(tr + '.*'):
            with open(f) as fh:
                r.visits.update(line.rstrip('\n') for line in fh)
            os.remove(f)
    return r


def check_run(worker, r, split_expected, grid=None, exit_=None, extra=None):
    """Protocol assertions on one run (see the module docstring).  With grid/exit_/extra, a protocol-3
    SUMMARY.flags must describe exactly that search (grid, exit, transit on, no block on the exit, the
    hole and block caps)."""
    s = r.summary
    where = f'seed {r.seed!r}'
    if s is None:
        raise Fail(f'no SUMMARY (exit code {r.rc}) for {where}\n{r.tail}')
    fl = s.get('flags')
    if fl is not None and grid is not None:
        want = {'grid': grid, 'exit': exit_, 'transit': 1, 'block_on_exit': 0}
        if extra and '--num-holes' in extra:
            want['max_holes'] = int(extra[extra.index('--num-holes') + 1])
        if extra and '--num-blocks' in extra:
            want['max_blocks'] = int(extra[extra.index('--num-blocks') + 1])
        bad = {k: (fl.get(k), v) for k, v in want.items() if fl.get(k) != v}
        if bad:
            raise Fail(f'SUMMARY.flags disagree with the config ({where}): ' +
                       ', '.join(f'{k} {a!r} != {b!r}' for k, (a, b) in bad.items()))
    st = s.get('status')
    if st not in ('exhausted', 'split'):
        raise Fail(f'status {st!r} (exit code {r.rc}) for {where}\n{r.tail}')
    if r.rc != 0:
        raise Fail(f'status {st} but exit code {r.rc} for {where}')
    if s.get('seed', r.seed) != r.seed:
        raise Fail(f'SUMMARY.seed {s.get("seed")!r} does not echo {where}')
    if st == 'exhausted' and r.remaining:
        raise Fail(f'exhausted run printed {len(r.remaining)} REMAINING lines ({where})')
    if st == 'split':
        if not split_expected:
            raise Fail(f'unsplit reference run ended with status split ({where})')
        if not r.remaining:
            raise Fail(f'split without REMAINING ({where})')
    st_t = toks(r.seed)
    for rem in r.remaining:
        if not SEED_RE.match(rem):
            raise Fail(f'malformed REMAINING {rem!r} ({where})')
        rt = rem.split(',')
        if len(rt) <= len(st_t) or rt[:len(st_t)] != st_t:
            raise Fail(f'REMAINING {rem!r} does not strictly extend the seed ({where}); the client voids such runs')
        if len(rt) > worker.path_tok_max:
            raise Fail(f'REMAINING of {len(rt)} tokens > path_tok_max {worker.path_tok_max} ({where})')
    if r.bad:
        raise Fail(f'{r.bad} malformed trace lines (torn, bad code, or depth != path tokens) ({where})')
    if 'valid' in s and r.nlines != s['valid']:
        raise Fail(f'trace has {r.nlines} lines but SUMMARY.valid is {s["valid"]} ({where})')
    # protocol-3 result lines (PROTOCOL3 §2.2): LEVEL = the run's verified best, one of its traced levels;
    # UNRESOLVED = candidates deeper than that best, each a node of this subtree, counted in SUMMARY
    if worker.version.get('PROTOCOL') == '3' and s.get('protocol') != 3:
        raise Fail(f'--version says PROTOCOL 3 but SUMMARY.protocol is {s.get("protocol")!r} ({where})')
    g = grid or (fl or {}).get('grid')
    R, C = (int(x) for x in g.split('x')) if g else (0, 0)
    best = s.get('best', 0)
    if len(r.level_lines) > 1:
        raise Fail(f'{len(r.level_lines)} LEVEL lines ({where})')
    if best > 0 and s.get('protocol') == 3 and not r.level_lines:
        raise Fail(f'best {best} but no LEVEL line ({where})')
    for ll in r.level_lines:
        f = ll.split('\t')
        if len(f) != 2 or not f[0].isdigit() or int(f[0]) != best or (g and not code_ok(f[1], R, C)):
            raise Fail(f'bad LEVEL line {ll!r} for best {best} ({where})')
        if g and canon(f[1]) not in set(canon(c) for d, c, _ in r.levels if d == best):
            raise Fail(f'LEVEL {ll!r} is not among the run\'s traced valid levels at depth {best} ({where})')
    for u in r.unresolved:
        f = u.split('\t')
        ok = len(f) == 4 and f[0].isdigit() and int(f[0]) > best and (not g or code_ok(f[1], R, C)) and \
            f[3] in ('pq', 'probe', 'big') and SEED_RE.match(f[2]) is not None
        if ok:
            ut = f[2].split(',')
            ok = len(ut) >= len(st_t) and ut[:len(st_t)] == st_t and len(ut) <= worker.path_tok_max and \
                len(ut) == int(f[0])                                   # the server requires depth == token count
        if not ok:
            raise Fail(f'bad UNRESOLVED line {u!r} (best {best}; want depth > best, a code, a path in the '
                       f'subtree of that many tokens, cause pq|probe|big) ({where})')
    if 'unresolved' in s and s['unresolved'] != len(r.unresolved):
        raise Fail(f'{len(r.unresolved)} UNRESOLVED lines but SUMMARY.unresolved is {s["unresolved"]} ({where})')


# ------------------------------------------------------------------------------------ one config
def run_config(worker, cfg, root='', split_s=0.02, split_nodes=0, sigint=False, timeout=120.0, max_runs=3000,
               visits=False, strict=False, expect_evictions=False, max_levels=3_000_000, name='', out=print,
               allow_unknown=False):
    """The whole test for one config.  Returns a result dict; raises Fail."""
    grid, exit_, extra = split_cfg(cfg)
    if root and not SEED_RE.match(root):
        raise Fail(f'--root {root!r} is not a seed path')
    if split_nodes and not worker.supports('--split-after-nodes', '5'):
        raise Fail('this worker has no --split-after-nodes')
    tmp = tempfile.mkdtemp(prefix='splitx_')
    t0 = time.time()
    try:
        ref = run_job(worker, grid, exit_, extra, root, tmp, timeout, visits=visits, max_levels=max_levels)
        check_run(worker, ref, split_expected=False, grid=grid, exit_=exit_, extra=extra)
        ref_flags = ref.summary.get('flags')
        unknown = ref.summary.get('unknown', 0)
        pops = ref.summary.get('max_pops')
        pend = ref.summary.get('max_pending')
        if ref.summary['status'] != 'exhausted':
            raise Fail(f'reference run did not exhaust: {ref.summary}')
        src_hash = ref.summary.get('src_hash')
        full = set((d, canon(code)) for d, code, _ in ref.levels)
        full_best = ref.summary['best']
        deepest = max((d for d, _ in full), default=0)
        if deepest != full_best:
            raise Fail(f'reference best {full_best} but deepest traced level {deepest}')
        ev = (ref.summary.get('evict_shallow', 0), ref.summary.get('evict_recent', 0))
        if expect_evictions and not (ev[0] or ev[1]):
            raise Fail(f'--expect-evictions: the reference run evicted nothing (evict_shallow/recent {ev})')
        full_vis, ref_t, ref_states = ref.visits, ref.wall, ref.summary.get('states')
        del ref
        jobs, union, extras = [root], set(), []
        runs = splits = best = maxdepth_seed = 0
        union_vis = set() if visits else None
        states = 0
        while jobs:
            seed = jobs.pop()
            r = run_job(worker, grid, exit_, extra, seed, tmp, timeout, split_s=split_s, split_nodes=split_nodes,
                        sigint=sigint, visits=visits, max_levels=max_levels, keep_paths=True)
            runs += 1
            check_run(worker, r, split_expected=True, grid=grid, exit_=exit_, extra=extra)
            if r.summary.get('flags') != ref_flags:
                raise Fail(f'SUMMARY.flags changed between runs: {ref_flags} vs {r.summary.get("flags")}')
            unknown += r.summary.get('unknown', 0)
            if pops is not None and r.summary.get('max_pops') is not None:
                pops, pend = max(pops, r.summary['max_pops']), max(pend, r.summary.get('max_pending', 0))
            if r.summary.get('src_hash') != src_hash:
                raise Fail(f'src_hash changed between runs: {src_hash} vs {r.summary.get("src_hash")}')
            if runs > max_runs:
                raise Fail(f'more than --max-runs {max_runs} runs')
            best = max(best, r.summary['best'])
            states += r.summary.get('states', 0)
            for d, code, path in r.levels:
                key = (d, canon(code))
                if key not in full and key not in union and len(extras) < 5:
                    extras.append((key, seed, path))
                union.add(key)
            if r.summary['status'] == 'split':
                splits += 1
                jobs.extend(r.remaining)
                maxdepth_seed = max([maxdepth_seed] + [len(toks(x)) for x in r.remaining])
            if visits:
                union_vis |= r.visits
        lost = sorted(k for k in full if k not in union)
        extra_n = sum(1 for k in union if k not in full)
        el = time.time() - t0
        res = dict(name=name, grid=grid, exit=exit_, flags=extra, root_tokens=len(toks(root)), ref_levels=len(full),
                   ref_best=full_best, ref_states=ref_states, ref_s=round(ref_t, 1), evict=list(ev), runs=runs,
                   splits=splits, tree_states=states, union=len(union), best=best, lost=len(lost), extra=extra_n,
                   deepest_remaining=maxdepth_seed, unknown=unknown, max_pops=pops, max_pending=pend,
                   elapsed=round(el, 1))
        mode = f'nodes {split_nodes}' if split_nodes else f'{"SIGINT" if sigint else "split"} {split_s}s'
        msg = (f"{name + ': ' if name else ''}{grid} exit {exit_} {' '.join(extra)}, root {len(toks(root))} tokens, "
               f"{mode}: reference {len(full)} canonical valid levels, best {full_best}, {ref_states} states, "
               f"evictions {ev[0]}/{ev[1]} ({ref_t:.1f}s); tree {runs} runs, {splits} splits, {states} states, "
               f"union {len(union)}, best {best}; lost {len(lost)} extra {extra_n}; deepest REMAINING "
               f"{maxdepth_seed} tokens" + (f"; unknown {unknown}, solver headroom max_pops {pops} max_pending {pend}"
                                            if pops is not None else '') + f" ({el:.0f}s)")
        if visits:
            missing = full_vis - union_vis
            res['missing_visits'] = len(missing)
            msg += f'; reference nodes not re-expanded {len(missing)} (dedup twins)'
        out(msg)
        if extras:
            out('  EXTRA levels (found by a job, not by the reference run):')
            for (d, c), sd, path in extras:
                out(f'    depth {d} {c}  job seed {sd!r} trace path {path}')
        if lost:
            out('  LOST levels (in the reference run, in no job):')
            want = set(lost[:5])
            again = run_job(worker, grid, exit_, extra, root, tmp, timeout, max_levels=max_levels, keep_paths=True)
            for d, code, path in again.levels:
                if (d, canon(code)) in want:
                    out(f'    depth {d} {code}: reference path {path}')
                    want.discard((d, canon(code)))
            for d, c in sorted(want):
                out(f'    depth {d} {c}')
        if lost or extra_n or best != full_best:
            raise Fail(f'level sets differ: lost {len(lost)} extra {extra_n}, best {best} vs {full_best}' +
                       (f' (with {unknown} capacity results: see PROTOCOL3 §1)' if unknown else ''))
        if unknown and not allow_unknown:
            raise Fail(f'{unknown} capacity results (SUMMARY.unknown): the sets are only bounded, V <= T <= V + U '
                       '(PROTOCOL3 §1); pick a config within capacity, or pass --allow-unknown')
        if splits == 0:
            raise Fail('no job was split: the split mode proved nothing (use a smaller --split / --split-nodes)')
        if visits and strict and res['missing_visits']:
            raise Fail(f'--strict: {res["missing_visits"]} reference nodes never expanded by the tree')
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------ suites
C1 = ['--allow-exit-transit', '--num-holes', '3']       # campaign #1 (5x5 at most 3 holes, any blocks)
Z = ['--allow-exit-transit', '--num-holes', '0']        # campaign #2's flags (6x6 0 holes)
# Seeds of real campaign-#1 jobs (jobs.db ids): deep seeds (15-54 tokens) whose best levels have 8-11
# blocks.  If one stops replaying (bad_seed), the tree semantics changed: that is a failure worth
# knowing about; re-pick seeds only deliberately.
SEEDS_C1 = {
    'job109788': (1, 'R2,U1,U3,L1,D1,L2,L2,U1,R1,U2,U2,L1,D1,L2,U1,R1,R3,D1,D1,L1,D2,D2,R2,U1,U1,L1,U1,D2,R1,U1,'
                     'R2,R2,D1,L1,L1,L1'),
    'job163789': (1, 'R2,U1,U3,L1,D1,L2,U1,R1,U2,U2,L1,D1,L2,D1,R2,R2,R2,D1,L1,L1,U1,D2,L1,U2,U1,R2,R2,U1,L1,L1,'
                     'D1,D1,L2,U1,R1,U2,L1,D1,D1,R1,R1,L2,L1,U1,U1,R1,R1,R1,D1,D1,D1,L1,D1,L1'),
    'job39336': (2, 'R2,U1,U3,U2,L1,D1,L2,L2,U2,U2,R1,D1,R1,D2,D1'),
    'job81420': (0, 'U1,U1,U3,L1,D1,L2,L2,U1,R1,U2,L1,D1,D1,R2,R2,R2,U1,L1,U1,L1,L1,D1,D1,R1,D2,L1,U2,U1,R1,R1'),
}
# 6x6 0-hole subtrees (campaign #2's flags), found by descending from the exit roots into pending
# entries of split runs (worker 6c531ee6, 2026-09-24): (exit, seed, split seconds); the timings are the
# monolithic subtree on the owner's laptop.  e14-t23 is the deep one (best 120, 11 blocks, ~40 s).
SEEDS_66 = {
    'e14-t10': (14, 'U2,U1,U1,R1,R1,D1,D1,D1,D1,D1', 0.5),                   # 339k states, best 86, 8 blocks, ~9 s
    'e14-t23': (14, 'U2,U1,U1,R1,R1,D1,D1,D1,D1,L1,L1,R2,R1,U1,U1,U1,U1,L1,L1,D1,D1,L1,D1', 3.0),  # 1.5M states, best 120
    'e0-t11': (0, 'U1,U1,U1,U1,L1,L1,U2,L2,L2,D1,D1', 0.5),                   # 294k states, best 50, 8 blocks, ~7 s
    'e7-t5': (7, 'U1,R2,U1,U1,U1', 0.5),                                      # 411k states, best 56, 9 blocks, ~4 s
}
# 5x5 0-hole subtrees (campaign #2's flags on campaign #1's grid), found the same way (worker 7f19cda6):
# (exit, seed, split seconds)
SEEDS_55H0 = {
    'e7-R2R1': (7, 'R2,R1', 0.3),      # 882k states, best 69, 6 blocks, ~3 s
    'e7-U2': (7, 'U2', 0.5),           # 2.2M states, best 57, 6 blocks, ~7 s
}
SUITES = {}


def build_suites(worker=None):
    SUITES['quick'] = [
        ('q4x4h3e0', ['--grid', '4x4', '--exit', '0'] + C1, '', dict(split_s=0.02), 'main'),
        ('q4x4h0e5', ['--grid', '4x4', '--exit', '5'] + Z, '', dict(split_s=0.002), 'main'),
        ('q-c1-job163789', ['--grid', '5x5', '--exit', '1'] + C1, SEEDS_C1['job163789'][1], dict(split_s=0.3), 'main'),
    ]
    camp = []
    # (1) <=3 holes with unbounded blocks on 4x4, every canonical exit; one SIGINT variant
    for e in (0, 1, 5):
        camp.append((f'4x4h3e{e}', ['--grid', '4x4', '--exit', str(e)] + C1, '', dict(split_s=0.02), 'main'))
    camp.append(('4x4h3e0-sigint', ['--grid', '4x4', '--exit', '0'] + C1, '', dict(split_s=0.05, sigint=True), 'main'))
    # (2) 0 holes, many blocks: 4x4 (exit 0 has under 1024 expansions, too few to split on time), 4x5
    for e in (1, 5):
        camp.append((f'4x4h0e{e}', ['--grid', '4x4', '--exit', str(e)] + Z, '', dict(split_s=0.002), 'main'))
    for e in (6, 7):
        camp.append((f'4x5h0e{e}', ['--grid', '4x5', '--exit', str(e)] + Z, '', dict(split_s=0.05), 'main'))
    # (3) real campaign-#1 subtrees: deep paths, 8-11 blocks; one SIGINT variant
    for nm, (e, seed) in SEEDS_C1.items():
        camp.append((f'c1-{nm}', ['--grid', '5x5', '--exit', str(e)] + C1, seed, dict(split_s=0.3), 'main'))
    camp.append(('c1-job81420-sigint', ['--grid', '5x5', '--exit', '0'] + C1, SEEDS_C1['job81420'][1],
                 dict(split_s=0.5, sigint=True), 'main'))
    # (4) 5x5 and 6x6 0-hole subtrees (campaign #2's flags)
    for nm, (e, seed, sp) in SEEDS_55H0.items():
        camp.append((f'55h0-{nm}', ['--grid', '5x5', '--exit', str(e)] + Z, seed, dict(split_s=sp), 'main'))
    for nm, (e, seed, sp) in SEEDS_66.items():
        camp.append((f'66-{nm}', ['--grid', '6x6', '--exit', str(e)] + Z, seed, dict(split_s=sp), 'main'))
    # (5) deterministic node-count splits, when the worker has --split-after-nodes (review N18)
    if worker is not None and worker.supports('--split-after-nodes', '5'):
        camp.append(('4x4h3e1-nodes', ['--grid', '4x4', '--exit', '1'] + C1, '', dict(split_nodes=2000), 'main'))
        camp.append(('c1-job39336-nodes', ['--grid', '5x5', '--exit', '2'] + C1, SEEDS_C1['job39336'][1],
                     dict(split_nodes=3000), 'main'))
        camp.append(('66-e0-t11-nodes', ['--grid', '6x6', '--exit', '0'] + Z, SEEDS_66['e0-t11'][1],
                     dict(split_nodes=10000), 'main'))
    # (6) evictions: tiny dedup tables (test-only build); the reference run must evict
    camp.append(('evict-4x4h3e0', ['--grid', '4x4', '--exit', '0'] + C1, '',
                 dict(split_s=0.02, expect_evictions=True), 'evict'))
    camp.append(('evict-4x5h0e7', ['--grid', '4x5', '--exit', '7'] + Z, '',
                 dict(split_s=0.05, expect_evictions=True), 'evict'))
    camp.append(('evict-66-e14-t10', ['--grid', '6x6', '--exit', '14'] + Z, SEEDS_66['e14-t10'][1],
                 dict(split_s=0.5, expect_evictions=True), 'evict'))
    camp.append(('evict-c1-job109788', ['--grid', '5x5', '--exit', '1'] + C1, SEEDS_C1['job109788'][1],
                 dict(split_s=0.3, expect_evictions=True), 'evict'))
    SUITES['campaign'] = camp
    SUITES['all'] = SUITES['quick'] + camp


EVICT_KNOBS = ['-DSHALLOW_LG2=12', '-DRECENT_LG2=12']
NN_STUB = r'''
int   nn_load(const char *p, float s, int r, int c, int ch) { (void)p;(void)s;(void)r;(void)c;(void)ch; return 0; }
float nn_score(const float *f) { (void)f; return 0.f; }
void  nn_score_batch(const float *f, int n, float *o) { (void)f; for (int i=0;i<n;i++) o[i]=0.f; }
void  nn_close(void) {}
int   nn_surrogate_load(const char *p, float s, int r, int c, int ch) { (void)p;(void)s;(void)r;(void)c;(void)ch; return 0; }
float nn_surrogate_score(const float *f) { (void)f; return 0.f; }
void  nn_surrogate_score_batch(const float *f, int n, float *o) { (void)f; for (int i=0;i<n;i++) o[i]=0.f; }
void  nn_surrogate_close(void) {}
'''


def build_worker(src, out_dir, knobs, name):
    """Build a TEST-ONLY worker from src with -D knobs (plain -O2, no PGO) into out_dir.  Its SRC_HASH_STR
    is 'TEST-BUILD-<knobs>' so it can never be mistaken for (or whitelisted as) a volunteer binary."""
    out = os.path.join(out_dir, name)
    stub = os.path.join(out_dir, 'nn_stub.c')
    with open(stub, 'w') as f:
        f.write(NN_STUB)
    srcs = [os.path.join(src, x) for x in ('backsearch.c', 'sokoban_bfs.c')]
    for s in srcs:
        if not os.path.isfile(s):
            raise Fail(f'worker source not found: {s} (pass --src)')
    tag = 'TEST-BUILD-' + '-'.join(k[2:] for k in knobs)
    cmd = [os.environ.get('CC', 'cc'), '-O2', '-w'] + knobs + [f'-DSRC_HASH_STR="{tag}"', '-I', src, '-o', out] + \
        srcs + [stub, '-lz', '-lm']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise Fail(f'build failed: {" ".join(cmd)}\n{p.stderr[-2000:]}')
    return out


def check_small_tables(ev, main_worker):
    """The eviction variant must really have smaller dedup tables than the worker under test (its
    --version KNOBS line, protocol 3; older workers have none and rely on --expect-evictions alone)."""
    def knobs(w):
        try:
            return json.loads(w.version.get('KNOBS', '{}'))
        except ValueError:
            raise Fail(f'unparsable KNOBS line from {w.path}')
    ke, km = knobs(ev), knobs(main_worker)
    for k in ('SHALLOW_LG2', 'RECENT_LG2'):
        if k in ke and k in km and not ke[k] < km[k]:
            raise Fail(f'eviction worker {ev.path} has {k} {ke[k]}, not below the tested worker\'s {km[k]}')
    if ev.version.get('SRC_HASH') and ev.version.get('SRC_HASH') == main_worker.version.get('SRC_HASH'):
        raise Fail('the eviction worker carries the tested worker\'s SRC_HASH: a knob build must never look like a '
                   'release build')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('worker')
    ap.add_argument('--split', type=float, default=0.02, help='split after S seconds (default 0.02)')
    ap.add_argument('--split-nodes', type=int, default=0, help='split after N expansions (--split-after-nodes)')
    ap.add_argument('--sigint', action='store_true', help='SIGINT after --split seconds instead of --split-after')
    ap.add_argument('--root', default='', help='test the subtree under this seed path')
    ap.add_argument('--max-runs', type=int, default=3000)
    ap.add_argument('--timeout', type=float, default=120.0, help='per worker run; the run is killed and the test fails')
    ap.add_argument('--visits', action='store_true', help='also trace visits (informational)')
    ap.add_argument('--strict', action='store_true', help='fail on reference nodes never re-expanded (implies --visits)')
    ap.add_argument('--expect-evictions', action='store_true')
    ap.add_argument('--max-levels', type=int, default=3_000_000)
    ap.add_argument('--allow-unknown', action='store_true', help='do not fail on capacity results (SUMMARY.unknown)')
    ap.add_argument('--suite', choices=['quick', 'campaign', 'all'])
    ap.add_argument('--quick', action='store_true', help='= --suite quick')
    ap.add_argument('--only', action='append', default=[], help='with --suite: only configs whose name contains this')
    ap.add_argument('--list', action='store_true', help='with --suite: list the configs and exit')
    ap.add_argument('--src', default=os.path.dirname(HERE), help='worker sources (for the eviction build)')
    ap.add_argument('--evict-worker', help='a prebuilt small-table worker (else built from --src)')
    ap.add_argument('--json', action='store_true', help='print one RESULT json line per config')
    argv = sys.argv[1:]
    cfg = []
    if '--' in argv:
        i = argv.index('--')
        cfg, argv = argv[i + 1:], argv[:i]
    a = ap.parse_args(argv)
    if a.quick:
        a.suite = 'quick'
    try:
        worker = Worker(a.worker)
        print(f'worker {worker.path}: src {worker.version.get("SRC_HASH", "?")[:16]} protocol '
              f'{worker.version.get("PROTOCOL", "<3")} path_tok_max {worker.path_tok_max}; {check_client_argv()}',
              flush=True)
        if not a.suite:
            if not cfg:
                ap.error('give a worker config after --, or --suite')
            res = run_config(worker, cfg, root=a.root, split_s=a.split, split_nodes=a.split_nodes, sigint=a.sigint,
                             timeout=a.timeout, max_runs=a.max_runs, visits=a.visits or a.strict, strict=a.strict,
                             expect_evictions=a.expect_evictions, max_levels=a.max_levels,
                             allow_unknown=a.allow_unknown)
            if a.json:
                print('RESULT\t' + json.dumps(res))
            print('OK')
            return 0
        build_suites(worker)
        todo = [c for c in SUITES[a.suite] if not a.only or any(o in c[0] for o in a.only)]
        if a.list:
            for c in todo:
                print(c[0], ' '.join(c[1]), f'root {len(toks(c[2]))} tokens', c[3], c[4])
            return 0
        variants = {'main': worker}
        build_dir = None
        failed = []
        try:
            for name, ccfg, root, mode, variant in todo:
                if variant not in variants:
                    if a.evict_worker:
                        variants['evict'] = Worker(a.evict_worker)
                    else:
                        build_dir = build_dir or tempfile.mkdtemp(prefix='splitx_build_')
                        t0 = time.time()
                        variants['evict'] = Worker(build_worker(a.src, build_dir, EVICT_KNOBS, 'bs_test_evict'))
                        print(f'built the test-only eviction worker ({" ".join(EVICT_KNOBS)}) in {time.time() - t0:.0f}s',
                              flush=True)
                    check_small_tables(variants['evict'], worker)
                m = dict(split_s=0.02, split_nodes=0, sigint=False, expect_evictions=False)
                m.update(mode)
                try:
                    res = run_config(variants[variant], ccfg, root=root, timeout=a.timeout, max_runs=a.max_runs,
                                     visits=a.visits or a.strict, strict=a.strict, max_levels=a.max_levels,
                                     name=name, allow_unknown=a.allow_unknown, **m)
                    if a.json:
                        print('RESULT\t' + json.dumps(res))
                except Fail as e:
                    print(f'FAIL {name}: {e}')
                    failed.append(name)
                sys.stdout.flush()
        finally:
            if build_dir:
                shutil.rmtree(build_dir, ignore_errors=True)
        if failed:
            print(f'FAILED {len(failed)}/{len(todo)}: {" ".join(failed)}')
            return 1
        print(f'OK ({len(todo)} configs)')
        return 0
    except Fail as e:
        print(f'FAIL: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
