#!/usr/bin/env python3
"""Big jobs end to end (client 3.2.0, server big jobs): the real server (node), the real client and real
workers on a small campaign (4x4, at most 3 holes, layer 2).

  python3 test_big.py ORDINARY_WORKER BIG_WORKER [--server-dir DIR] [--port P] [--keep]

ORDINARY_WORKER is a knob build whose solver caps are tiny (e.g. -DHP64_SIZE='(1<<2)'), so that runs end
as unknown_chain; BIG_WORKER is a big build (build_pgo.sh --big). The campaign whitelists both and marks
the big one as a big build. Client A runs the ordinary worker only; client B runs with --prefer-big.

Checks: the campaign completes and both clients exit by themselves; jobs became big at their first
unknown_chain (none quarantined, no failure other than unknown_chain); only client B ran big jobs, each
under the big build's hash, and never beside an ordinary job; the fresh audit is clean and exact and every
exit's best equals the big worker's monolithic run; no client logged '!!!'.
Run it under the CPU guard with two slots (a server, two clients, one worker each).
"""
import argparse, json, os, re, shutil, signal, socket, sqlite3, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GRID, EXTRA, LAYER = '4x4', ['--allow-exit-transit', '--num-holes', '3'], 2


def say(m):
    print('[test_big %s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def version(path):
    out = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=30).stdout
    return dict(l.split('\t', 1) for l in out.splitlines() if '\t' in l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ordinary'); ap.add_argument('big')
    ap.add_argument('--server-dir', default='/Users/george/PathologyRecords/server')
    ap.add_argument('--port', type=int, default=19470)
    ap.add_argument('--timeout', type=float, default=170)
    ap.add_argument('--keep', action='store_true')
    a = ap.parse_args()
    T = tempfile.mkdtemp(prefix='v2big_')
    procs, problems = [], []
    url = 'http://127.0.0.1:%d' % a.port
    s = socket.socket(); s.settimeout(0.5)
    if s.connect_ex(('127.0.0.1', a.port)) == 0:
        sys.exit('port %d is in use' % a.port)
    s.close()
    for d in ('data', 'records', 'bin', 'solverbin'):
        os.makedirs(os.path.join(T, d))
    ordw, bigw = os.path.join(T, 'bin', 'worker_ord'), os.path.join(T, 'bin', 'worker_big')
    shutil.copyfile(a.ordinary, ordw); shutil.copyfile(a.big, bigw)
    os.chmod(ordw, 0o755); os.chmod(bigw, 0o755)
    h_ord, h_big = version(ordw)['SRC_HASH'].strip(), version(bigw)['SRC_HASH'].strip()
    senv = dict(os.environ, PORT=str(a.port), JOBS_DATA_DIR=os.path.join(T, 'data'), RECORDS_DATA_DIR=os.path.join(T, 'records'),
                JOBS_BACKUP='0', V2_TOKEN_PER_MIN='100000', V2_IP_PER_MIN='100000', V2_IP_PER_HOUR='10000000',
                SOLVER_BIN_DIR=os.path.join(T, 'solverbin'), SOLVER_MAX_CONCURRENT='1')
    wenv = {k: v for k, v in os.environ.items() if not k.startswith('BS_')}
    try:
        base = [bigw, '--grid', GRID, '--two-tables', '--time', '0'] + EXTRA
        r = subprocess.run(base + ['--list-layer', str(LAYER)], capture_output=True, text=True, timeout=120, env=wenv)
        layer = os.path.join(T, 'layer.tsv'); open(layer, 'w').write(r.stdout)
        heads = [json.loads(l.split('\t', 1)[1]) for l in r.stdout.splitlines() if l.startswith('LAYERINFO\t')]
        exits = sorted(int(e) for h in heads for e in h['roots'])
        mono = {}
        for e in exits:
            r = subprocess.run(base + ['--exit', str(e)], capture_output=True, text=True, timeout=120, env=wenv)
            sm = [json.loads(l.split('\t', 1)[1]) for l in r.stdout.splitlines() if l.startswith('SUMMARY\t')]
            mono[e] = sm[-1]['best']
        say('exits %s, monolithic best %s; ordinary %s..., big %s...' % (exits, mono, h_ord[:8], h_big[:8]))
        plan = {'title': 'big e2e', 'grid': GRID, 'extra': EXTRA, 'layer': LAYER, 'hashes': [h_ord, h_big],
                'max_clients': 10, 'workers_max': 4, 'job_target_s': 1, 'split_after_s': 6, 'ramp_split_after_s': 6,
                'lease_s': 20, 'paused_max_s': 20, 'dup_fraction': 0, 'fail_regrant_s': 5, 'absorb_probe_s': 1,
                'absorb_total_s': 1, 'batch_interval_s': 1, 'lease_ahead_s': 2, 'heartbeat_s': 5, 'release': 'v-big'}
        pf = os.path.join(T, 'plan.json'); json.dump(plan, open(pf, 'w'))
        tool = lambda *args: subprocess.run(['node', os.path.join(a.server_dir, 'tools', args[0])] + list(args[1:]),
                                            capture_output=True, text=True, env=senv, timeout=60)
        r = tool('v2_seed.js', '--campaign-file', pf, '--layer-file', layer)
        if r.returncode:
            raise SystemExit('v2_seed: %s %s' % (r.stdout[-400:], r.stderr[-400:]))
        r = tool('v2_hashes.js', 'add', h_big, '--big')
        if r.returncode or 'big build' not in r.stdout:
            raise SystemExit('v2_hashes add --big: %s %s' % (r.stdout[-400:], r.stderr[-400:]))
        log = open(os.path.join(T, 'server.log'), 'w')
        srv = subprocess.Popen(['node', 'server.js'], cwd=a.server_dir, env=senv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        procs.append(srv)
        end = time.time() + 30
        while True:
            try:
                urllib.request.urlopen(url + '/api/v2/status', timeout=2).read(); break
            except OSError:
                if time.time() > end or srv.poll() is not None:
                    raise SystemExit('the server did not start; see %s/server.log' % T)
                time.sleep(0.2)
        clients = {}
        for name, workers, extra in (('A', 1, []), ('B', 2, ['--prefer-big', '--big-worker', bigw])):
            home = os.path.join(T, 'home_' + name); os.makedirs(home)
            env = dict(wenv, HOME=home, VOLUNTEER_ALLOW_DIRTY='1')
            argv = [sys.executable, os.path.join(HERE, 'volunteer.py'), '--name', 'big' + name, '--workers', str(workers), '--server', url,
                    '--worker', ordw, '--no-ui', '--outbox', os.path.join(T, 'outbox_' + name), '--complete-flush-s', '20'] + extra
            clog = open(os.path.join(T, 'client_%s.log' % name), 'w')
            clients[name] = subprocess.Popen(argv, env=env, stdout=clog, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            procs.append(clients[name])
            if name == 'A':
                time.sleep(4)        # A starts first: its runs turn jobs into big jobs for B
        end = time.time() + a.timeout
        while time.time() < end and any(p.poll() is None for p in clients.values()):
            time.sleep(0.5)
        for n, p in clients.items():
            if p.poll() is None:
                problems.append('client %s did not exit by itself within %.0f s' % (n, a.timeout))
            elif p.returncode != 0:
                problems.append('client %s exited with code %s' % (n, p.returncode))
        db = sqlite3.connect(os.path.join(T, 'data', 'jobs.db'))
        q = lambda sql, *args: db.execute(sql, args).fetchall()
        cid = q("SELECT max(id) FROM campaigns")[0][0]
        st = q("SELECT status FROM campaigns WHERE id = ?", cid)[0][0]
        if st != 'complete':
            problems.append('campaign status %s, not complete' % st)
        nbig = q("SELECT COUNT(*) FROM jobs WHERE campaign_id = ? AND big = 1", cid)[0][0]
        if nbig == 0:
            problems.append('no job became big (the ordinary worker never hit unknown_chain?)')
        quar = q("SELECT COUNT(*) FROM jobs WHERE campaign_id = ? AND status = 'quarantined'", cid)[0][0]
        if quar:
            problems.append('%d job(s) quarantined' % quar)
        reasons = q("SELECT reason, COUNT(*) FROM failures WHERE campaign_id = ? GROUP BY reason", cid)
        if any(r0 != 'unknown_chain' for r0, _ in reasons):
            problems.append('failures other than unknown_chain: %s' % reasons)
        tok = {n: q("SELECT token FROM clients WHERE name = ?", 'big' + n)[0][0] for n in 'AB'}
        bad_big = q("""SELECT j.id, r.client_token, r.src_hash FROM jobs j JOIN reports r ON r.job_id = j.id
                       WHERE j.campaign_id = ? AND j.big = 1 AND j.status IN ('done', 'split') AND (r.outcome LIKE 'done%' OR r.outcome LIKE 'split%')
                       AND (r.client_token != ? OR r.src_hash != ?)""", cid, tok['B'], h_big)
        if bad_big:
            problems.append('big job(s) finished by someone else or under another hash: %s' % bad_big[:5])
        nbig_done = q("SELECT COUNT(*) FROM jobs WHERE campaign_id = ? AND big = 1 AND status IN ('done', 'split')", cid)[0][0]
        checks = q("SELECT status, COUNT(*) FROM jobs WHERE campaign_id = ? AND check_of IS NOT NULL GROUP BY status", cid)
        verdicts = q("SELECT outcome, COUNT(*) FROM reports r JOIN jobs j ON j.id = r.job_id WHERE j.campaign_id = ? AND j.check_of IS NOT NULL GROUP BY outcome", cid)
        if not checks:
            problems.append('no check job was issued (the tiny-cap worker should leave candidates deeper than the best)')
        r = subprocess.run(['node', '-e', """
const { openJobsDB } = require(process.argv[1] + '/lib/jobs.js');
const J = openJobsDB(process.argv[2]);
const out = {};
for (const e of JSON.parse(process.argv[4])) { const x = J.audit(e, Number(process.argv[3]), { fresh: true });
  out[e] = { clean: x.clean, exact: x.exact, best: x.best, open_deeper: x.candidates.open_deeper, contradicted: x.candidates.contradicted, levelless: x.levelless.length }; }
console.log(JSON.stringify(out));""", a.server_dir, os.path.join(T, 'data'), str(cid), json.dumps(exits)], capture_output=True, text=True, timeout=60)
        aud = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        for e in exits:
            x = aud.get(str(e), {})
            # exact: the server turns every candidate deeper than an exit's best into a check job, which the big client settles
            if not (x.get('clean') and x.get('exact') and x.get('best') == mono[e]):
                problems.append('exit %d audit %s, monolithic best %s' % (e, x, mono[e]))
        la, lb = (open(os.path.join(T, 'client_%s.log' % n), errors='replace').read() for n in 'AB')
        if 'BIG job' in la:
            problems.append('client A ran a big job')
        if 'BIG job' not in lb:
            problems.append('client B never ran a big job')
        # (the hand-back of running ordinary windows is test_client.sh bg: here the 4x4 jobs take under a second)
        if 'backing off' in la + lb:
            problems.append('a client backed off (unknown_chain must not count as a failure streak)')
        for n, text in (('A', la), ('B', lb)):
            bang = [l for l in text.splitlines() if '!!!' in l and 'unknown_chain' not in l]
            if bang:
                problems.append('client %s logged: %s' % (n, bang[:3]))
        say('%d big job(s), %d finished by B under %s...; check jobs %s, verdicts %s; failures %s; audit %s' % (nbig, nbig_done, h_big[:8], checks, verdicts, reasons, aud))
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(20)
            except subprocess.TimeoutExpired:
                p.kill()
    if problems:
        say('FAIL (logs in %s):\n  - %s' % (T, '\n  - '.join(problems)))
        sys.exit(1)
    say('PASS')
    if not a.keep:
        shutil.rmtree(T, ignore_errors=True)


if __name__ == '__main__':
    main()
