#!/usr/bin/env python3
"""Dedup-key sensitivity test (review N13, test T6).

The generator's dedup tables tell states apart by a 96-bit key: a 64-bit hash k plus an independent
32-bit check value c (backsearch.c dedup_key).  64-bit collisions never happen in a test, so nothing
else notices when a lookup stops comparing c.  This test makes them happen: it builds two TEST-ONLY
workers from --src (test_split_exact.build_worker; SRC_HASH "TEST-BUILD-...", never a volunteer
binary) that keep only --kbits bits of k (-DDKEY_TEST_KBITS=N):

  * CHK build (c compared as usual): distinct states share k thousands of times per run, and the check
    value must tell every one apart.  Its level set (BS_TRACE_VALID, canonical under the 8 symmetries)
    and SUMMARY states / accepted / valid / best must EQUAL the tested worker's, and its stderr line
    "DKEY TEST: N lookups matched k but not the check value" must show N > 0 (the case was exercised).
  * NOCHK build (-DDKEY_TEST_NOCHK: c is always 0, states are told apart by the short k alone): the
    same runs must LOSE states and levels (false dedup hits prune real subtrees).  If they do not, the
    configuration cannot detect a missing compare and the test fails as insensitive.

Configurations: two protocol-mode subtrees in the client's argv (two-table dedup: the shallow and
recent tables) and one plain run with the single visited table (dedup_check_and_insert).  All run to
exhaustion, one worker process at a time, each well under a minute on one CPU; every run is killed
after --timeout seconds (the test then fails).

  python3 test_dkey.py WORKER [--src DIR] [--kbits 20] [--only NAME] [--timeout 120]

--src is the tested worker's source directory (default: this file's parent directory); its sha256
must equal the worker's SRC_HASH (--allow-src-mismatch downgrades that to a warning).
Exit status 0 = passed, 1 = a failure (the message says which).
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_split_exact as tse      # noqa: E402  (Worker, build_worker, run_job, canon, src checks)

Fail = tse.Fail
KMATCH_RE = re.compile(r'DKEY TEST: (\d+) lookups matched k but not the check value')
FIELDS = ('status', 'states', 'accepted', 'valid', 'best')
# (name, grid, exit, extra flags, seed): protocol-mode runs in the client's argv (--two-tables)
CONFIGS = [
    ('66h0-e7-t5', '6x6', 7, tse.Z, 'U1,R2,U1,U1,U1'),                       # campaign #2's flags, ~411k states
    ('c1-job39336', '5x5', 2, tse.C1, 'R2,U1,U3,U2,L1,D1,L2,L2,U2,U2,R1,D1,R1,D2,D1'),   # campaign #1
]
# (name, argv): plain runs, single visited table; compared by SUMMARY counts
PLAIN = [
    ('4x5-e0-d14-onetable', ['--grid', '4x5', '--exit', '0', '--max-depth', '14', '--time', '0']),
]


def levels_of(r):
    return {(d, tse.canon(code)) for d, code, _ in r.levels}


def kmatch(err_text, who):
    m = KMATCH_RE.search(err_text)
    if not m:
        raise Fail(f'{who}: no "DKEY TEST" line on stderr (is it a -DDKEY_TEST_KBITS build?)')
    return int(m.group(1))


def run_plain(worker, argv, timeout):
    """One plain (non-protocol) run: (SUMMARY dict, stderr)."""
    try:
        p = subprocess.run([worker.path] + argv, capture_output=True, text=True, timeout=timeout,
                           env=tse.clean_env())
    except subprocess.TimeoutExpired:
        raise Fail(f'{worker.path} {" ".join(argv)}: exceeded --timeout {timeout:.0f}s and was killed')
    s = None
    for line in p.stdout.split('\n'):
        if line.startswith('SUMMARY\t'):
            s = json.loads(line[8:])
    if p.returncode != 0 or s is None:
        raise Fail(f'{worker.path} {" ".join(argv)}: rc {p.returncode}, SUMMARY {"missing" if s is None else "present"}')
    return s, p.stderr


def compare(name, ref_s, chk_s, nochk_s, ref_lv=None, chk_lv=None, nochk_lv=None, n_kmatch=0, out=print):
    for f in FIELDS:
        if chk_s.get(f) != ref_s.get(f):
            raise Fail(f'{name}: the CHK build gives {f} {chk_s.get(f)}, the worker {ref_s.get(f)}: the check value '
                       'did not tell colliding states apart (a lookup ignores it?)')
    if ref_lv is not None and chk_lv != ref_lv:
        raise Fail(f'{name}: the CHK build\'s level set differs from the worker\'s ({len(ref_lv - chk_lv)} lost, '
                   f'{len(chk_lv - ref_lv)} extra)')
    if n_kmatch <= 0:
        raise Fail(f'{name}: no lookup matched k with a different check value: raise the collision rate (--kbits)')
    lost_states = ref_s['states'] - nochk_s['states']
    lost_levels = (len(ref_lv - nochk_lv) if ref_lv is not None else ref_s['valid'] - nochk_s['valid'])
    if lost_states <= 0 or lost_levels <= 0:
        raise Fail(f'{name}: the NOCHK build lost {lost_states} states and {lost_levels} levels: this configuration '
                   'cannot detect a lookup that ignores the check value')
    out(f'{name}: OK  worker/CHK states {ref_s["states"]} valid {ref_s["valid"]} best {ref_s["best"]}; '
        f'{n_kmatch} k-only matches resolved by the check value; NOCHK lost {lost_states} states, '
        f'{lost_levels} {"levels" if ref_lv is not None else "valid"} (best {nochk_s.get("best")})')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('worker')
    ap.add_argument('--src', default=os.path.dirname(HERE), help='the worker\'s source directory')
    ap.add_argument('--allow-src-mismatch', action='store_true')
    ap.add_argument('--kbits', type=int, default=20, help='bits of k the test builds keep (8..63, default 20)')
    ap.add_argument('--only', action='append', default=[], help='run only this configuration (repeatable)')
    ap.add_argument('--timeout', type=float, default=120.0, help='per worker run; the run is killed and the test fails')
    args = ap.parse_args()
    tse.install_signal_handlers()
    tmp = None
    try:
        if not 8 <= args.kbits <= 63:
            raise Fail('--kbits must be 8..63')
        worker = tse.Worker(args.worker)
        print(tse.check_src_matches(worker, args.src, args.allow_src_mismatch))
        names = [c[0] for c in CONFIGS] + [c[0] for c in PLAIN]
        for o in args.only:
            if o not in names:
                raise Fail(f'--only {o}: no such configuration ({", ".join(names)})')
        tmp = tempfile.mkdtemp(prefix='test_dkey_')
        kb = f'-DDKEY_TEST_KBITS={args.kbits}'
        chk = tse.Worker(tse.build_worker(args.src, tmp, [kb], 'bs_dkey_chk'))
        nochk = tse.Worker(tse.build_worker(args.src, tmp, [kb, '-DDKEY_TEST_NOCHK'], 'bs_dkey_nochk'))
        for w, want in ((chk, f'DKEY_TEST_KBITS={args.kbits} '), (nochk, 'DKEY_TEST_NOCHK ')):
            if want not in w.knobs().get('defs', '') + ' ':
                raise Fail(f'{w.path}: --version KNOBS defs lack {want.strip()!r}: the knob did not reach the build')
        print(f'test builds: k truncated to {args.kbits} bits, with and without the check value ({tmp})')
        for name, grid, exit_, extra, seed in CONFIGS:
            if args.only and name not in args.only:
                continue
            runs = [tse.run_job(w, grid, exit_, extra, seed, tmp, args.timeout) for w in (worker, chk, nochk)]
            for w, r in zip((worker, chk, nochk), runs):
                if r.rc != 0 or r.summary is None or r.summary.get('status') != 'exhausted':
                    raise Fail(f'{name}: {w.path} ended rc {r.rc}, status '
                               f'{(r.summary or {}).get("status")}, not exhausted:\n{r.tail}')
                if r.bad:
                    raise Fail(f'{name}: {w.path} wrote {r.bad} malformed trace lines')
            ref, a, b = runs
            compare(name, ref.summary, a.summary, b.summary, levels_of(ref), levels_of(a), levels_of(b),
                    kmatch(a.tail, name + ' CHK'))
        for name, argv in PLAIN:
            if args.only and name not in args.only:
                continue
            (rs, _), (as_, aerr), (bs, _) = (run_plain(w, argv, args.timeout) for w in (worker, chk, nochk))
            compare(name, rs, as_, bs, n_kmatch=kmatch(aerr, name + ' CHK'))
        print('test_dkey: all passed')
        return 0
    except Fail as e:
        print(f'FAIL: {e}')
        return 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
