#!/usr/bin/env python3
"""Integration chaos test (v2/DESIGN.md §5.2; review M90, M91, M96, T10): the real server (node), the
real client (volunteer.py) and the real worker on a small campaign, with a client SIGKILLed while it
holds leases, a client stopped with SIGINT in the middle of a window, and a server restart, each one
restarted, until the clients exit by themselves because the campaign is complete.

  python3 test_chaos.py WORKER [--config 4x4h3|6x6h0b1] [--server-dir DIR] [--port P] [--tmp DIR] [--keep]
  (bash test_chaos.sh WORKER [SERVER_DIR] runs the default config)

Run it under the CPU guard with two slots (a server, two clients and their workers run at once).

Setup
  - the campaign definition is an allowlisted protocol-3 plan with every canonical exit of the grid;
    its roots are the worker's own --list-layer K listing (whole stdout, LAYERINFO headers included),
    seeded with the server's tools/v2_seed.js into a temp jobs.db;
  - splitting is deterministic: the campaign sets split_after_nodes (the client passes
    --split-after-nodes to every run), so every root larger than N expansions splits;
  - lease 10 s, heartbeat 5 s, 1-s windows, a 1-s absorption budget, dup_fraction 0.2 (only a leased job
    that ends 'done' as a whole can be re-issued, about 40-70 per run: 0.2 makes a run with no compared
    duplicate a 1e-5 event; at 0.1 runs had 2 to 7);
  - two clients (1 worker each, their own HOME, outbox and name); the worker's BS_TRACE_VALID
    reaches the runs (VOLUNTEER_ALLOW_DIRTY=1 BS_ALLOW_DEBUG=1) and the client's development run log
    (VOLUNTEER_RUN_LOG) records pid, job, seed and outcome of every run;
  - the server runs on a temp data dir for BOTH databases (JOBS_DATA_DIR, RECORDS_DATA_DIR: the
    owner's records.db is never opened), without backups, with a site solver6 built into the temp dir
    for the champion checks (--solver6 none skips it).

Scenario (every step waits for its condition; the test fails if a condition never comes)
  1. client A is SIGKILLed while its worker runs and it holds at least one lease; restarted 1 s later
     (same outbox and token: its lapsed leases come back). Before the restart the test plants a stopped
     orphan (A's own pinned worker binary on a real search, SIGSTOPped, added to the pidfile A left):
     a real orphan dies by itself (broken pipe, or SIGHUP of its orphaned process group when paused), so
     this is the survivor the startup cleanup (M14) exists for; the restarted A must SIGKILL it (and any
     real orphan still alive) and log it;
  2. client B gets SIGINT while its worker runs a leased job: it must hand back its window (a report
     after 'stopping:'), exit 0 and hold no lease afterwards; restarted;
  3. the server is stopped with SIGTERM (it must exit 0) while work remains, left down 3 s (the
     clients must notice: a failed heartbeat or a batch kept in the outbox) and started again;
  4. both clients must exit by themselves with code 0 and the closing summary ('is complete.').

Assertions afterwards
  - the campaign is 'complete'; a FRESH audit of every exit is clean and exact (M90), with no
    mismatch, no quarantined job, no pending duplicate; at least one duplicate's fingerprint was
    really compared (a 'done:match' report of a dup job; a split or incomparable dup does not count)
    and no report came back 'mismatch'; no failure report was filed; at least one job split;
  - every exit's best equals the monolithic run's best (shallow best included), and champion levels
    the site solver checked agree with their depth;
  - M91: the union of the valid levels traced by ACCEPTED runs only (every done or split node of the
    coverage tree, tied to its run by exit, seed, states and valid through the run log) equals the
    monolithic run's canonical level set at depth >= K, per exit: nothing lost and nothing extra.
    Runs whose work was never reported (SIGKILLed client, void or demoted probes, re-checks) are
    listed and left out;
  - every client report was accepted (no REJECTED, nothing left in an outbox);
  - M96: no process of this test survives it (anything whose command line holds the temp dir), no
    process outside the test is ever signalled, a busy port is refused rather than freed.
"""
import argparse, collections, glob, json, os, re, shutil, signal, socket, sqlite3, subprocess, sys, tempfile, time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_split_exact import canon   # noqa: E402  (min over the dihedral images of a level code)

CONFIGS = {
    # 4x4, at most 3 holes: exits 0, 1, 5; about 700k states monolithic
    '4x4h3': dict(grid='4x4', extra=['--allow-exit-transit', '--num-holes', '3'], layer=2, nodes=500),
    # 6x6, no holes, at most one block: the campaign #2 grid and exits 0, 1, 2, 7, 8, 14; about 1.9M states
    '6x6h0b1': dict(grid='6x6', extra=['--allow-exit-transit', '--num-holes', '0', '--num-blocks', '1'], layer=3, nodes=600),
}
SOLVER6_DEFAULT = '/Users/george/PathologyRecords/solver/solver6.c'
SERVER_DIR_DEFAULT = '/Users/george/PathologyRecords/server'


class Fail(Exception):
    pass


def say(msg):
    print('[chaos %s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


class Chaos:
    def __init__(self, a):
        self.a = a
        self.cfg = dict(CONFIGS[a.config])
        if a.nodes:
            self.cfg['nodes'] = a.nodes
        self.T = tempfile.mkdtemp(prefix='v2chaos_%s_' % a.config, dir=a.tmp or None)
        self.procs = []          # every Popen this test started (clients, servers)
        self.server = None
        self.clients = {}        # name -> Popen
        self.url = 'http://127.0.0.1:%d' % a.port
        self.db_path = os.path.join(self.T, 'data', 'jobs.db')
        self.problems = []       # assertion failures collected after the run
        self.info = collections.OrderedDict()

    # ------------------------------------------------------------------ processes
    def worker_env(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('BS_')}
        env['BS_ALLOW_DEBUG'] = '1'
        return env

    def server_env(self):
        env = dict(os.environ)
        env.update(PORT=str(self.a.port), JOBS_DATA_DIR=os.path.join(self.T, 'data'),
                   RECORDS_DATA_DIR=os.path.join(self.T, 'records'), JOBS_BACKUP='0',
                   V2_TOKEN_PER_MIN='100000', V2_IP_PER_MIN='100000', V2_IP_PER_HOUR='10000000',
                   SOLVER_BIN_DIR=os.path.join(self.T, 'solverbin'), SOLVER_MAX_CONCURRENT='1')
        return env

    def port_busy(self):
        s = socket.socket()
        s.settimeout(0.5)
        try:
            s.connect(('127.0.0.1', self.a.port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def start_server(self):
        log = open(os.path.join(self.T, 'server.log'), 'a')
        log.write('----- server start %s\n' % time.strftime('%H:%M:%S')); log.flush()
        p = subprocess.Popen(['node', 'server.js'], cwd=self.a.server_dir, env=self.server_env(),
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.close()
        self.procs.append(p)
        self.server = p
        end = time.time() + 30
        while time.time() < end:
            if p.poll() is not None:
                raise Fail('the server exited at startup (code %s); see %s/server.log' % (p.returncode, self.T))
            try:
                with urllib.request.urlopen(self.url + '/api/v2/status', timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.2)
        raise Fail('the server did not answer within 30 s')

    def stop_server(self, expect_zero=True):
        p = self.server
        if p is None or p.poll() is not None:
            return p.returncode if p else None
        p.send_signal(signal.SIGTERM)
        try:
            rc = p.wait(20)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
            raise Fail('the server did not exit within 20 s of SIGTERM')
        if expect_zero and rc != 0:
            raise Fail('the server exited with code %s on SIGTERM (expected 0)' % rc)
        return rc

    def start_client(self, name):
        home = os.path.join(self.T, 'home_' + name)
        os.makedirs(home, exist_ok=True)
        os.makedirs(os.path.join(self.T, 'tv'), exist_ok=True)
        env = self.worker_env()
        env.update(HOME=home, VOLUNTEER_ALLOW_DIRTY='1', BS_ALLOW_DEBUG='1',
                   BS_TRACE_VALID=os.path.join(self.T, 'tv', name),
                   VOLUNTEER_RUN_LOG=os.path.join(self.T, 'runs_%s.jsonl' % name))
        log = open(os.path.join(self.T, 'client_%s.log' % name), 'a')
        log.write('----- client %s start %s\n' % (name, time.strftime('%H:%M:%S'))); log.flush()
        port = self.a.port + 10 + (ord(name[0]) - ord('A'))
        argv = [sys.executable, os.path.join(HERE, 'volunteer.py'), '--name', 'chaos' + name, '--workers', '1',
                '--server', self.url, '--worker', self.worker, '--no-ui', '--outbox', os.path.join(self.T, 'outbox_' + name),
                '--port', str(port), '--complete-flush-s', '30', '--stop-grace-s', '30']
        p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.close()
        self.procs.append(p)
        self.clients[name] = p
        return p

    def client_log(self, name):
        try:
            return open(os.path.join(self.T, 'client_%s.log' % name), errors='replace').read()
        except OSError:
            return ''

    def worker_pids(self, name):
        """Live worker processes of client `name`: they run the pinned copy in its outbox."""
        pat = os.path.join(self.T, 'outbox_' + name, '.bin') + '/'
        r = subprocess.run(['pgrep', '-f', pat], capture_output=True, text=True)
        return [int(x) for x in r.stdout.split() if x.strip().isdigit()]

    def test_pids(self):
        """Every live process whose command line holds this test's temp dir (never anything else)."""
        r = subprocess.run(['pgrep', '-f', self.T], capture_output=True, text=True)
        me = os.getpid()
        return [int(x) for x in r.stdout.split() if x.strip().isdigit() and int(x) != me]

    def cleanup(self):
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                    p.send_signal(signal.SIGCONT)     # a stopped process (the planted orphan) must see it
                except OSError:
                    pass
        end = time.time() + 10
        for p in self.procs:
            try:
                p.wait(max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    pass
        time.sleep(0.5)
        left = self.test_pids()
        for pid in left:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        return left

    # ------------------------------------------------------------------ DB (read-only, while the server runs)
    def q(self, sql, args=()):
        for _ in range(50):
            try:
                con = sqlite3.connect('file:%s?mode=ro' % self.db_path, uri=True, timeout=10)
                try:
                    return con.execute(sql, args).fetchall()
                finally:
                    con.close()
            except sqlite3.OperationalError as e:
                if 'locked' not in str(e) and 'busy' not in str(e):
                    raise
                time.sleep(0.1)
        raise Fail('jobs.db stayed locked')

    def token(self, name):
        r = self.q('SELECT token FROM clients WHERE name = ? ORDER BY created_at DESC LIMIT 1', ('chaos' + name,))
        return r[0][0] if r else None

    def leased(self, tok):
        return [x[0] for x in self.q("SELECT id FROM jobs WHERE status = 'leased' AND client_token = ?", (tok,))] if tok else []

    def finished(self):
        return self.q("SELECT COUNT(*) FROM jobs WHERE status IN ('done', 'split')")[0][0]

    def campaign_status(self):
        r = self.q('SELECT status FROM campaigns ORDER BY id DESC LIMIT 1')
        return r[0][0] if r else None

    def wait_until(self, what, pred, timeout, step=0.05):
        end = time.time() + timeout
        while time.time() < end:
            v = pred()
            if v:
                return v
            for n, p in self.clients.items():
                if p.poll() is not None:
                    raise Fail('client %s exited (code %s) while waiting for: %s; its log ends:\n%s'
                               % (n, p.returncode, what, self.client_log(n)[-1500:]))
            if self.server is not None and self.server.poll() is not None:
                raise Fail('the server exited (code %s) while waiting for: %s' % (self.server.returncode, what))
            time.sleep(step)
        raise Fail('timed out after %d s waiting for: %s' % (timeout, what))

    # ------------------------------------------------------------------ setup
    def setup(self):
        a, cfg = self.a, self.cfg
        if self.port_busy():
            raise Fail('port %d is in use: pass --port (this test never stops a process it did not start)' % a.port)
        for d in ('data', 'records', 'mono', 'tv', 'solverbin', 'bin'):
            os.makedirs(os.path.join(self.T, d), exist_ok=True)
        # the worker under test, copied into the temp dir so every process of this test is recognisable
        self.worker = os.path.join(self.T, 'bin', 'worker')
        shutil.copyfile(a.worker, self.worker); os.chmod(self.worker, 0o755)
        v = subprocess.run([self.worker, '--version'], capture_output=True, text=True, timeout=30).stdout
        info = dict(l.split('\t', 1) for l in v.splitlines() if '\t' in l)
        self.hash = info.get('SRC_HASH', '').strip()
        if info.get('PROTOCOL', '').strip() != '3' or not re.fullmatch(r'[0-9a-f]{64}', self.hash):
            raise Fail('the worker is not a protocol-3 build with a 64-hex SRC_HASH: %r' % v[:300])
        if a.solver6 != 'none':
            out = os.path.join(self.T, 'solverbin', 'solver6.exe')
            r = subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-w', '-o', out, a.solver6], capture_output=True, text=True)
            if r.returncode != 0:
                raise Fail('cannot build solver6: %s' % r.stderr[:300])
        grid, extra, K = cfg['grid'], cfg['extra'], cfg['layer']
        base = [self.worker, '--grid', grid, '--two-tables', '--time', '0'] + extra
        # roots: the worker's own listing, whole stdout
        r = subprocess.run(base + ['--list-layer', str(K)], capture_output=True, text=True, timeout=120, env=self.worker_env())
        if r.returncode != 0:
            raise Fail('--list-layer %d failed (code %s): %s' % (K, r.returncode, r.stderr[-300:]))
        layer = os.path.join(self.T, 'layer.tsv')
        open(layer, 'w').write(r.stdout)
        self.headers = [json.loads(l.split('\t', 1)[1]) for l in r.stdout.splitlines() if l.startswith('LAYERINFO\t')]
        self.exits = sorted(int(e) for h in self.headers for e in h['roots'])
        nroots = sum(1 for l in r.stdout.splitlines() if l.startswith('LAYER\t'))
        say('%s: %d roots at layer %d over exits %s; split after %d expansions' % (a.config, nroots, K, self.exits, cfg['nodes']))
        # the monolithic reference, per exit
        self.mono, self.mono_best, self.mono_n = {}, {}, {}
        for e in self.exits:
            env = self.worker_env(); env['BS_TRACE_VALID'] = os.path.join(self.T, 'mono', 'e%d' % e)
            r = subprocess.run(base + ['--exit', str(e)], capture_output=True, text=True, timeout=200, env=env)
            sm = [json.loads(l.split('\t', 1)[1]) for l in r.stdout.splitlines() if l.startswith('SUMMARY\t')]
            if r.returncode != 0 or not sm or sm[-1].get('status') != 'exhausted':
                raise Fail('the monolithic run of exit %d failed (code %s)' % (e, r.returncode))
            self.mono_best[e] = sm[-1]['best']
            files = glob.glob(os.path.join(self.T, 'mono', 'e%d.*' % e))
            keys, n = set(), 0
            for f in files:
                for line in open(f):
                    _, d, code = line.rstrip('\n').split('\t')
                    n += 1
                    if int(d) >= K:
                        keys.add(hash((int(d), canon(code))))
                os.unlink(f)            # the raw trace is large on 6x6: keep only the hashed set
            if n != sm[-1]['valid']:
                raise Fail('exit %d: the monolithic trace has %d lines but SUMMARY valid is %d' % (e, n, sm[-1]['valid']))
            self.mono[e] = keys
            self.mono_n[e] = n
        say('monolithic: ' + ', '.join('exit %d best %d (%d levels, %d at depth >= %d)'
                                       % (e, self.mono_best[e], self.mono_n[e], len(self.mono[e]), K) for e in self.exits))
        # the campaign
        plan = {'title': 'chaos %s' % a.config, 'grid': grid, 'extra': extra, 'layer': K, 'hashes': [self.hash],
                'max_clients': 10, 'workers_max': 4, 'job_target_s': 1, 'split_after_s': 1, 'ramp_split_after_s': 1,
                'lease_s': 10, 'paused_max_s': 20, 'dup_fraction': 0 if a.e2e else 0.2, 'fail_regrant_s': 5,
                'absorb_probe_s': 1, 'absorb_total_s': 1, 'batch_interval_s': 1, 'lease_ahead_s': 2, 'heartbeat_s': 5,
                'split_after_nodes': cfg['nodes'], 'release': 'v-chaos'}
        pf = os.path.join(self.T, 'plan.json')
        json.dump(plan, open(pf, 'w'))
        r = subprocess.run(['node', os.path.join(a.server_dir, 'tools', 'v2_seed.js'), '--campaign-file', pf, '--layer-file', layer],
                           capture_output=True, text=True, env=self.server_env(), timeout=60)
        if r.returncode != 0:
            raise Fail('v2_seed refused the campaign: %s %s' % (r.stdout[-500:], r.stderr[-500:]))
        say('seeded: ' + r.stdout.strip().splitlines()[-1][:200])

    # ------------------------------------------------------------------ the scenario
    def scenario(self):
        t0 = time.time()
        self.start_server()
        if self.a.e2e:
            return self.scenario_e2e(t0)
        self.start_client('A'); self.start_client('B')
        # 1. SIGKILL A while its worker runs and it holds leases
        def a_busy():
            pids = self.worker_pids('A')
            if not pids:
                return None
            held = self.leased(self.token('A'))
            if not held:
                return None
            self.clients['A'].send_signal(signal.SIGKILL)      # at once: its worker is running now
            return held, pids
        held, wpids = self.wait_until('client A running a leased job', a_busy, 90, step=0.01)
        self.clients['A'].wait(10)
        self.killA = {'leases': held, 'workers': wpids, 'at': round(time.time() - t0, 1)}
        say('1. SIGKILLed client A at %.1f s holding leases %s, worker pid(s) %s' % (time.time() - t0, held, wpids))
        time.sleep(1.0)
        self.orphans_left = self.worker_pids('A')
        planted = self.plant_orphan('A')
        mark_a = len(self.client_log('A'))
        self.start_client('A')
        self.check_orphans('A', planted, mark_a)
        # 2. SIGINT B while its worker runs a leased job
        def b_busy():
            tok = self.token('B')
            held = self.leased(tok)
            if not held or not self.worker_pids('B'):
                return None
            wins = re.findall(r'worker \d+: job (\d+) exit \d+ window', self.client_log('B'))
            return (held, int(wins[-1])) if wins and int(wins[-1]) in held else None
        held_b, job_b = self.wait_until('client B running a window of a leased job', b_busy, 60)
        mark = len(self.client_log('B'))
        self.clients['B'].send_signal(signal.SIGINT)
        try:
            rc = self.clients['B'].wait(60)
        except subprocess.TimeoutExpired:
            raise Fail('client B did not exit within 60 s of SIGINT')
        after = self.client_log('B')[mark:]
        tok_b = self.token('B')
        left_b = self.leased(tok_b)
        row = self.q('SELECT status, client_token FROM jobs WHERE id = ?', (job_b,))
        self.intB = {'rc': rc, 'job': job_b, 'job_after': row[0][0] if row else None, 'leases_after': left_b,
                     'report_after_stop': bool(re.search(r'stopping:.*?\n(?:.*\n)*?.*job %d (split|done) in' % job_b, after)),
                     'at': round(time.time() - t0, 1)}
        say('2. SIGINT to client B during job %d: exit code %s, job %d is now %s, B holds %s'
            % (job_b, rc, job_b, self.intB['job_after'], left_b or 'no lease'))
        if rc != 0:
            raise Fail('client B exited with code %s after SIGINT (expected 0); log:\n%s' % (rc, after[-1500:]))
        if 'stopping:' not in after:
            raise Fail("client B's log shows no 'stopping:' after SIGINT")
        if left_b:
            raise Fail('client B still holds leases %s after its Stop (the stop heartbeat must release them)' % left_b)
        if not self.intB['report_after_stop'] and self.intB['job_after'] != 'open':
            raise Fail('job %d that B was running is %s after B stopped: neither reported nor handed back'
                       % (job_b, self.intB['job_after']))
        self.start_client('B')
        # 3. server restart while work remains
        f0 = self.finished()
        self.wait_until('3 more finished jobs after the restarts', lambda: self.finished() >= f0 + 3, 90)
        if self.campaign_status() != 'open':
            raise Fail('the campaign completed before the server restart: make the campaign larger (--nodes)')
        remain = self.q("SELECT COUNT(*) FROM jobs WHERE status IN ('open', 'leased')")[0][0]
        marks = {n: len(self.client_log(n)) for n in self.clients}
        rc = self.stop_server()
        down_at = time.time()
        time.sleep(3.0)
        self.start_server()
        noticed = []
        for n in self.clients:
            if re.search(r'heartbeat failed|kept in the outbox|cannot reach|lease failed|unreachable|Connection refused',
                         self.client_log(n)[marks[n]:]):
                noticed.append(n)
        self.restart = {'rc': rc, 'down_s': round(time.time() - down_at, 1), 'open_or_leased': remain, 'noticed_by': noticed,
                        'at': round(down_at - t0, 1)}
        say('3. server stopped (code %s) with %d open or leased job(s), up again after %.1f s'
            % (rc, remain, time.time() - down_at))
        # 4. both clients exit by themselves once the campaign is complete
        end = time.time() + self.a.timeout
        while time.time() < end and any(p.poll() is None for p in self.clients.values()):
            if self.server.poll() is not None:
                raise Fail('the server exited (code %s) during the run' % self.server.returncode)
            time.sleep(0.25)
        for n, p in self.clients.items():
            if p.poll() is None:
                raise Fail('client %s did not exit within %d s: the campaign never completed (status %s)'
                           % (n, self.a.timeout, self.campaign_status()))
        for n in self.clients:
            if not self.restart['noticed_by'] and re.search(r'heartbeat failed|kept in the outbox|cannot reach',
                                                            self.client_log(n)[marks[n]:]):
                self.restart['noticed_by'].append(n)
        self.exit_codes = {n: p.returncode for n, p in self.clients.items()}
        self.wall = round(time.time() - t0, 1)
        say('4. clients exited with %s after %.1f s' % (self.exit_codes, self.wall))
        try:
            with urllib.request.urlopen(self.url + '/api/v2/status', timeout=5) as r:
                self.status = json.loads(r.read().decode())
        except (OSError, ValueError):
            self.status = None
        self.stop_server()

    def plant_orphan(self, name):
        """A real worker from a dead client run that is still alive when the client comes back (M14): a
        running orphan dies by itself of the broken pipe and a paused one of the orphaned process group's
        SIGHUP, so the survivor the startup cleanup exists for is made here: the client's own pinned binary,
        started on a real search, SIGSTOPped at once and added to the pidfile the killed client left behind.
        The restarted client must kill it before it pins its binary again (check_orphans)."""
        pf = os.path.join(self.T, 'outbox_' + name, '.workers.pid')
        try:
            d = json.load(open(pf))
        except (OSError, ValueError) as e:
            raise Fail('client %s left no readable pidfile %s after SIGKILL: %s' % (name, pf, e))
        binp = d.get('bin') if isinstance(d, dict) else None
        if not binp or not os.path.isfile(binp) or not binp.startswith(os.path.join(self.T, 'outbox_' + name) + '/'):
            raise Fail("client %s's pidfile names no pinned worker in its outbox: %r" % (name, d))
        p = subprocess.Popen([binp, '--grid', self.cfg['grid'], '--two-tables', '--time', '0'] + self.cfg['extra']
                             + ['--exit', str(self.exits[0])], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, env=self.worker_env(), start_new_session=True)
        os.kill(p.pid, signal.SIGSTOP)
        self.procs.append(p)
        d['workers'] = list(d.get('workers') or []) + [p.pid]
        tmp = pf + '.chaos'
        json.dump(d, open(tmp, 'w'))
        os.replace(tmp, pf)
        say('   planted a stopped orphan worker pid %d (%s) in the pidfile of %s' % (p.pid, os.path.basename(binp), name))
        return p

    def check_orphans(self, name, planted, mark):
        """The restarted client killed every worker of its dead predecessor still alive: the planted one and
        any real orphan (review T2: the chaos run itself exercises the startup cleanup)."""
        def gone():
            if planted.poll() is None:
                return None
            live = set(self.orphans_left) & set(self.worker_pids(name))   # still running A's pinned binary
            return None if live else True
        self.wait_until('restarted client %s killing the orphaned workers of its dead run' % name, gone, 30)
        text = self.client_log(name)[mark:]
        m = re.search(r'killed (\d+) worker\(s\) left behind by an earlier client run: \[([0-9, ]*)\]', text)
        killed = [int(x) for x in m.group(2).split(',') if x.strip()] if m else []
        self.orphan_cleanup = {'planted': planted.pid, 'rc': planted.returncode, 'killed_logged': killed,
                               'real_orphans': self.orphans_left}
        say('   restarted %s killed the orphans %s (planted pid %d ended with %s)'
            % (name, killed or 'none logged', planted.pid, planted.returncode))
        if planted.pid not in killed:
            raise Fail('restarted client %s did not log killing the planted orphan %d; its log after restart:\n%s'
                       % (name, planted.pid, text[-1500:]))
        if planted.returncode != -signal.SIGKILL:
            raise Fail('the planted orphan ended with %s, not SIGKILL from the client' % planted.returncode)

    def scenario_e2e(self, t0):
        """--e2e: one client, no disruption; it must work the whole campaign and exit by itself (code 0)
        once the server marks the campaign complete (the owner's 'return control when done' feature)."""
        self.start_client('A')
        p = self.clients['A']
        try:
            p.wait(self.a.timeout)
        except subprocess.TimeoutExpired:
            raise Fail('the client did not exit within %d s: the campaign never completed (status %s)'
                       % (self.a.timeout, self.campaign_status()))
        self.exit_codes = {'A': p.returncode}
        self.wall = round(time.time() - t0, 1)
        text = self.client_log('A')
        m = re.search(r'campaign complete\. jobs done (\d+)', text)
        say('the client exited by itself with code %s after %.1f s%s' % (p.returncode, self.wall,
                                                                       ' (%s windows)' % m.group(1) if m else ''))
        if not m:
            raise Fail("the client's log does not say it stopped because the campaign is complete")
        self.stop_server()

    # ------------------------------------------------------------------ checks
    def check(self):
        P = self.problems
        K = self.cfg['layer']
        for n, rc in self.exit_codes.items():
            if rc != 0:
                P.append('client %s exited with code %s (expected 0 on completion)' % (n, rc))
            text = self.client_log(n)
            if 'is complete.' not in text.split('----- client %s start' % n)[-1]:
                P.append("client %s's last run shows no closing summary ('is complete.')" % n)
            if 'REJECTED' in text:
                P.append('client %s: the server REJECTED a report: %s' % (n, re.findall(r'.*REJECTED.*', text)[:2]))
            ob = os.path.join(self.T, 'outbox_' + n)
            left = [x for x in os.listdir(ob) if x.endswith('.json')] if os.path.isdir(ob) else []
            rej = os.listdir(os.path.join(ob, 'rejected')) if os.path.isdir(os.path.join(ob, 'rejected')) else []
            if left or rej:
                P.append('client %s: outbox not empty at exit: %s, rejected %s' % (n, left[:3], rej[:3]))
        if not self.a.e2e and not self.restart['noticed_by']:
            P.append('no client noticed the server outage (no failed heartbeat or kept batch): the restart exercised nothing')
        # the fresh audit (M90): computed from the DB, not the server's cache
        js = os.path.join(self.T, 'audit.js')
        open(js, 'w').write(
            "const {openJobsDB} = require(process.argv[2] + '/lib/jobs.js');\n"
            "const J = openJobsDB(process.argv[3]);\n"
            "const a = J.audit(null, Number(process.argv[4]), {fresh: true});\n"
            "const per = {}; for (const e of (a.exits || [])) per[e.exit] = J.audit(e.exit, Number(process.argv[4]), {fresh: true});\n"
            "console.log(JSON.stringify({all: a, per}));\n")
        cid = self.q('SELECT id FROM campaigns ORDER BY id DESC LIMIT 1')[0][0]
        r = subprocess.run(['node', js, self.a.server_dir, os.path.join(self.T, 'data'), str(cid)], capture_output=True, text=True,
                           env=self.server_env(), timeout=120)
        if r.returncode != 0:
            raise Fail('the audit script failed: %s' % r.stderr[-800:])
        au = json.loads(r.stdout.strip().splitlines()[-1])
        self.audit = au
        A = au['all']
        if A.get('campaign_state') != 'complete':
            P.append('campaign_state is %s, not complete' % A.get('campaign_state'))
        for x in A.get('exits', []):
            e = x['exit']
            bad = {k: x[k] for k in ('clean', 'exact', 'mismatches', 'quarantined', 'pending_dups', 'uncovered_roots',
                                    'unknown_hash_jobs', 'verify_mismatches', 'levelless', 'candidates_open_deeper')
                   if x.get(k) not in (True, 0, None) or (k in ('clean', 'exact') and x.get(k) is not True)}
            if bad:
                pe = au['per'].get(str(e), {})
                P.append('exit %d audit: %s; details: uncovered %s mismatches %s' % (e, bad, str(pe.get('uncovered_roots'))[:300],
                                                                                   str(pe.get('mismatches'))[:300]))
            if x.get('best') != self.mono_best.get(e):
                P.append('exit %d: audit best %s != monolithic best %s' % (e, x.get('best'), self.mono_best.get(e)))
            if x.get('roots') != x.get('expected_roots'):
                P.append('exit %d: %s roots, LAYERINFO says %s' % (e, x.get('roots'), x.get('expected_roots')))
        if sorted(x['exit'] for x in A.get('exits', [])) != self.exits:
            P.append('the audit covers exits %s, the listing %s' % ([x['exit'] for x in A.get('exits', [])], self.exits))
        # the DB after the run
        rows = self.q('SELECT id, exit, seed, status, states, valid, best, dup_of, mismatch, level_code, verified_moves, client_token '
                      'FROM jobs')
        by_status = collections.Counter((r[3], r[7] is not None) for r in rows)
        self.info['jobs'] = {('dup ' if d else '') + s: n for (s, d), n in sorted(by_status.items())}
        splits = sum(n for (s, d), n in by_status.items() if s == 'split' and not d)
        if splits < 1:
            P.append('no job split: the split path was not exercised')
        # duplicates: the server compares fingerprints only when the dup and its original both ended 'done'
        # with the same src_hash (report outcome 'done:match' / 'done:mismatch'); a split or otherwise
        # incomparable dup ('...:incomparable') compared nothing, so only real comparisons count (T2 review)
        dup_out = dict(self.q('SELECT r.outcome, COUNT(*) FROM reports r JOIN jobs j ON j.id = r.job_id '
                              'WHERE j.dup_of IS NOT NULL GROUP BY r.outcome'))
        self.info['dup_reports'] = dup_out or 'none'
        compared = dup_out.get('done:match', 0)
        if compared < 1 and not self.a.e2e:       # one client: a dup never goes back to its original's client (M32)
            P.append('no duplicate was re-run and compared (no done:match report; dup reports %s)' % (dup_out or 'none'))
        bad_out = self.q("SELECT job_id, outcome FROM reports WHERE outcome LIKE '%mismatch%'")
        if bad_out:
            P.append('fingerprint mismatch reports: %s' % bad_out[:10])
        mism = [r[0] for r in rows if r[8]]
        if mism:
            P.append('fingerprint mismatches on jobs %s' % mism[:10])
        fails = self.q('SELECT job_id, reason, detail FROM failures')
        self.info['failures'] = len(fails)
        if fails:
            P.append('%d failure report(s) were filed: %s' % (len(fails), fails[:5]))
        ev = self.q('SELECT kind, COUNT(*) FROM events GROUP BY kind')
        self.info['events'] = dict(ev)
        # the jobs A held when it was killed were finished by someone
        st = dict((r[0], r[3]) for r in rows)
        notdone = [j for j in getattr(self, 'killA', {}).get('leases', []) if st.get(j) not in ('done', 'split')]
        if notdone:
            P.append('jobs %s that A held when it was killed never finished' % notdone)
        # champions the site solver checked
        ver = [(r[1], r[6], r[10]) for r in rows if r[9] and r[10] is not None]
        wrong = [v for v in ver if v[1] != v[2]]
        self.info['champion_checks'] = '%d level(s) re-solved by the site solver, %d disagree' % (len(ver), len(wrong))
        if wrong:
            P.append('the site solver disagrees with reported levels (exit, depth, solver): %s' % wrong[:5])
        # M91: the level union over accepted runs only
        self.union_check(rows, K)

    def union_check(self, rows, K):
        P = self.problems
        runs = collections.defaultdict(list)     # (exit, seed, states, valid, status) -> [pid...]
        nruns, kinds = 0, collections.Counter()
        for n in self.clients:
            f = os.path.join(self.T, 'runs_%s.jsonl' % n)
            if not os.path.exists(f):
                P.append('client %s wrote no run log (VOLUNTEER_RUN_LOG)' % n)
                continue
            for line in open(f):
                r = json.loads(line)
                nruns += 1
                kinds[(r['kind'], r.get('reason') or r.get('status'))] += 1
                if r['kind'] == 'ok':
                    runs[(r['exit'], r['seed'], r['states'], r['valid'], r['status'])].append((n, r['pid']))
        want_status = {'done': 'exhausted', 'split': 'split'}
        union = collections.defaultdict(set)
        used, ambiguous, missing = set(), 0, []
        tree = [r for r in rows if r[7] is None and r[3] in ('done', 'split')]
        for (jid, e, seed, status, states, valid, best, dup_of, mm, code, vm, tok) in tree:
            cands = runs.get((e, seed, states, valid, want_status[status]), [])
            sets = []
            for (n, pid) in cands:
                f = os.path.join(self.T, 'tv', '%s.%d' % (n, pid))
                if not os.path.exists(f):
                    continue
                keys, cnt = set(), 0
                for line in open(f):
                    _, d, c = line.rstrip('\n').split('\t')
                    cnt += 1
                    keys.add(hash((int(d), canon(c))))
                if cnt != valid:
                    continue
                sets.append(keys)
                used.add((n, pid))
            if valid and not sets:
                missing.append((jid, e, seed, status, valid))
                continue
            if len(sets) > 1 and any(s != sets[0] for s in sets[1:]):
                ambiguous += 1
            if sets:
                union[e] |= set.intersection(*sets)
        all_traces = glob.glob(os.path.join(self.T, 'tv', '*.*'))
        self.info['runs'] = '%d worker runs logged by the clients, outcomes %s' % (nruns, dict(kinds.most_common(8)))
        self.info['traces'] = ('%d trace files, %d tied to the %d accepted done/split nodes (a run is tied when its exit, seed, '
                               'status, states and valid equal the node\'s; an unreported run with the same fingerprint is a '
                               'deterministic re-run with the same trace), %d left out (never reported: killed client, void or '
                               'demoted probes, re-checks)' % (len(all_traces), len(used), len(tree), len(all_traces) - len(used)))
        if missing:
            P.append('%d accepted node(s) have no matching run trace (exit, seed, status, valid): %s' % (len(missing), missing[:5]))
        if ambiguous:
            self.info['ambiguous'] = '%d node(s) matched runs with different traces (intersection used)' % ambiguous
        for e in self.exits:
            lost = self.mono[e] - union[e]
            extra = union[e] - self.mono[e]
            say('exit %d: monolithic %d levels at depth >= %d, accepted-run union %d, lost %d, extra %d'
                % (e, len(self.mono[e]), K, len(union[e]), len(lost), len(extra)))
            if lost or extra:
                P.append('exit %d: the accepted runs lost %d and added %d canonical levels' % (e, len(lost), len(extra)))

    def owner_db_state(self):
        """M96: size and mtime of the owner's records.db files next to the server checkout (never opened here)."""
        d = os.path.join(os.path.dirname(os.path.abspath(self.a.server_dir)), 'data')
        out = {}
        for n in ('records.db', 'records.db-wal', 'records.db-shm', 'jobs.db', 'jobs.db-wal', 'jobs.db-shm'):
            try:
                st = os.stat(os.path.join(d, n))
                out[n] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        return out

    def run(self):
        ok = False
        leaks = []
        owner0 = self.owner_db_state()
        try:
            self.setup()
            self.scenario()
            self.check()
            ok = not self.problems
        except Fail as e:
            self.problems.insert(0, str(e))
        finally:
            leaks = self.cleanup()
        if leaks:
            self.problems.append('M96: %d process(es) of this test were still alive at the end and were killed: %s'
                                 % (len(leaks), leaks[:10]))
            ok = False
        if self.owner_db_state() != owner0:
            self.problems.append("M96: the owner's databases next to the server checkout changed during the test")
            ok = False
        elif not os.path.exists(os.path.join(self.T, 'records', 'records.db')) and hasattr(self, 'wall'):
            self.problems.append('M96: the test server did not create its own records.db (RECORDS_DATA_DIR ignored?)')
            ok = False
        for k in ('killA', 'orphan_cleanup', 'intB', 'restart'):
            if hasattr(self, k):
                say('%s: %s' % (k, getattr(self, k)))
        if getattr(self, 'orphans_left', None) is not None:
            say('orphaned workers of A still running 1 s after the kill: %s' % (self.orphans_left or 'none'))
        for k, v in self.info.items():
            say('%s: %s' % (k, v))
        if self.problems:
            for p in self.problems:
                say('FAIL: ' + p)
        say('%s %s (logs in %s)' % ('PASS' if ok else 'FAIL', self.a.config, self.T))
        if ok and not self.a.keep:
            for d in ('tv', 'mono'):
                shutil.rmtree(os.path.join(self.T, d), ignore_errors=True)
        return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('worker')
    ap.add_argument('--config', default='4x4h3', choices=sorted(CONFIGS))
    ap.add_argument('--server-dir', default=SERVER_DIR_DEFAULT)
    ap.add_argument('--port', type=int, default=int(os.environ.get('V2_CHAOS_PORT', '19410')))
    ap.add_argument('--tmp', default=os.environ.get('TMPDIR'), help='where the temp dir goes')
    ap.add_argument('--nodes', type=int, default=0, help='override the config split_after_nodes')
    ap.add_argument('--timeout', type=float, default=150, help='seconds the clients get to finish after the restart')
    ap.add_argument('--solver6', default=SOLVER6_DEFAULT, help="site solver6 source for the champion checks, or 'none'")
    ap.add_argument('--keep', action='store_true', help='keep the traces of a passing run')
    ap.add_argument('--e2e', action='store_true', help='end-to-end completion run: one client, no disruption, it must exit '
                    'by itself with code 0 when the campaign is complete (same audit and level checks)')
    a = ap.parse_args()
    a.worker = os.path.abspath(a.worker)

    def on_term(sig, frm):
        raise SystemExit(128 + sig)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    sys.exit(Chaos(a).run())


if __name__ == '__main__':
    main()
