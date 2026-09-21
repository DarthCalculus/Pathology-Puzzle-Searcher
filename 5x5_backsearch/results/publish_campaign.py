#!/usr/bin/env python3
"""
publish_campaign.py DIR [--max-blocks B] [--max-holes H] [--dry]

Turn a finished campaign.py directory into site proofs and champion levels:
  * refuses unless every job in jobs.tsv is in done.tsv with status exhausted;
  * one proof per exit (5x5, the given caps, max depth over the exit's jobs);
  * re-runs the job that found each exit's maximum to print the level, converts
    it and submits it as an ordinary level attributed to Panacea (the site
    refuses ties/duplicates itself), paced to the 60/hour limit.
Logs to DIR/publish.log.
"""
import csv, json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
os.chdir(ROOT)
args = sys.argv[1:]
D = args[0]; dry = '--dry' in args
B = int(args[args.index('--max-blocks') + 1]) if '--max-blocks' in args else None
H = int(args[args.index('--max-holes') + 1]) if '--max-holes' in args else None
log = open(os.path.join(D, 'publish.log'), 'a')
def L(m): log.write(time.strftime('[%F %T] ') + m + '\n'); log.flush(); print(m)

cfg = json.load(open(os.path.join(D, 'config.json')))
extra = cfg['extra']
jobs = [l.split('\t') for l in open(os.path.join(D, 'jobs.tsv')).read().split('\n') if l]
rows = list(csv.DictReader(open(os.path.join(D, 'done.tsv')), delimiter='\t'))
done = {(r['exit'], r['path']): r for r in rows}
missing = [j for j in jobs if (j[0], j[1]) not in done]
bad = [r for r in rows if r['status'] not in ('exhausted', 'split')]
if missing or bad:
    L(f"NOT complete: {len(missing)} missing, {len(bad)} non-exhausted -> nothing published"); sys.exit(1)
best, bestjob, states, cpu = {}, {}, {}, {}
for r in rows:
    e = int(r['exit']); b = int(r['best'])
    states[e] = states.get(e, 0) + int(r['states']); cpu[e] = cpu.get(e, 0.0) + float(r['elapsed'])
    if b > best.get(e, -1): best[e] = b; bestjob[e] = r['path']
desc = ' and '.join(x for x in [f"at most {B} movable block{'s' if B != 1 else ''}" if B is not None else None,
                               f"at most {H} hole{'s' if H != 1 else ''}" if H is not None else None] if x)
method = "exhaustive backward search (backsearch_worker + campaign.py, real rules: blocks may transit the exit, no block starts on the exit)"
proofs = [dict(rows=5, cols=5, exitCell=e, maxBlocks=B, maxHoles=H, maxMoves=best[e], method=method, attribution="Panacea",
               states=states[e], cpuSeconds=round(cpu[e]),
               note=f"Every 5x5 level with the exit at cell {e} using {desc} (any walls) was enumerated; {best[e]} moves is the longest optimal solution. Campaign {D}, finished {time.strftime('%Y-%m-%d')}.")
          for e in sorted(best)]
pf = os.path.join(D, 'proofs.json'); json.dump(proofs, open(pf, 'w'), indent=1)
L("maxima: " + ', '.join(f"e{e}:{best[e]}" for e in sorted(best)))
if dry: sys.exit(0)
key = os.path.expanduser('~/.ssh/id_ed25519')
subprocess.run(['scp', '-q', '-i', key, '-o', 'BatchMode=yes', pf, f'root@2.29.8.112:/opt/pathology/server/tools/proofs_{os.path.basename(D)}.json'], check=True)
r = subprocess.run(['ssh', '-i', key, '-o', 'BatchMode=yes', 'root@2.29.8.112',
                    f'cd /opt/pathology && chown pathology:pathology server/tools/proofs_{os.path.basename(D)}.json && '
                    f'runuser -u pathology -- node server/tools/add_proof.js --file server/tools/proofs_{os.path.basename(D)}.json'],
                   capture_output=True, text=True)
L("add_proof: " + r.stdout.strip().replace('\n', ' | ')[:1500] + (' ERR ' + r.stderr.strip()[-300:] if 'REFUSED' in r.stdout or r.returncode else ''))
for e in sorted(best):
    m = best[e]
    out = subprocess.run(['./backsearch_worker_nt', '--grid', '5x5', '--two-tables'] + extra + ['--exit', str(e), '--seed-path', bestjob[e], '--time', '7200'],
                         capture_output=True, text=True).stdout
    mm = re.search(r'^%d \([^)]*\)\n((?:  .*\n)+)' % m, out, re.M)
    if not mm: L(f"exit {e}: could not extract the {m}-move level from {bestjob[e]}"); continue
    open(os.path.join(D, f'champion_e{e}.txt'), 'w').write(mm.group(1))
    code = subprocess.run(['./level_to_pathology'], input=mm.group(1), capture_output=True, text=True).stdout.strip()
    while True:
        res = subprocess.run(['curl', '-s', '-m', '300', '-D', '-', '-H', 'Content-Type: application/json',
                              '--data', json.dumps({"code": code, "attribution": "Panacea"}), 'https://pathology.georgespahn.com/api/submit'],
                             capture_output=True, text=True).stdout
        hdr, body = re.split(r'\r?\n\r?\n', res, 1) if re.search(r'\r?\n\r?\n', res) else (res, '')
        if ' 429 ' in hdr.split('\n')[0]: time.sleep(130); continue
        try: j = json.loads(body); msg = f"accepted id {j['level']['id']}" if j.get('accepted') else f"refused: {j.get('reason')}"
        except Exception: msg = body[:120]
        L(f"champion exit {e} ({m}) {code.replace(chr(10), '/')}: {msg}"); break
    time.sleep(61)
L("done")
