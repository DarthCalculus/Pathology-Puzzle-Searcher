#!/usr/bin/env python3
"""
compile_classes.py — turn finished results/classes/b{B}_h{H}_e{E}.log runs into
site proofs and champion submissions.

  python3 results/classes/compile_classes.py            # print table only
  python3 results/classes/compile_classes.py --publish  # also add proofs + submit champions

A class is published only when all six exits are exhausted.  Since every hole
needs a block (un-consume creates them in pairs), a hole cap >= the block cap is
recorded as "any holes".  Already-published classes (results/classes/published.txt)
are skipped.  Champions are ordinary submissions attributed to Panacea; ties and
dominated levels are refused by the site, which is fine.
"""
import glob, json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
os.chdir(ROOT)
EXITS = [0, 1, 2, 6, 7, 12]
METHOD = "exhaustive backward search (backsearch_worker, real rules: blocks may transit the exit, no block starts on the exit)"
SSH = ['ssh', '-i', os.path.expanduser('~/.ssh/id_ed25519'), '-o', 'BatchMode=yes', 'root@2.29.8.112']

def parse(f):
    t = open(f).read()
    if '--- Exit' not in t: return None
    g = lambda rx: re.search(rx, t)
    st = g(r'--- Exit \d+ \((.*?)\)').group(1)
    return dict(status=st, elapsed=float(g(r'elapsed:\s+([\d.]+)').group(1)),
                states=int(g(r'states checked: (\d+)').group(1)), best=int(g(r'best depth:\s+(\d+)').group(1)),
                text=t)

def level_block(text, depth):
    m = re.search(r'^%d \([^)]*\)\n((?:  .*\n)+)' % depth, text, re.M)
    return m.group(1) if m else None

classes = {}
for f in glob.glob('results/classes/b*_h*_e*.log'):
    m = re.match(r'.*/b(\w+)_h(\w+)_e(\d+)\.log', f)
    classes.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = parse(f)

published = set(open('results/classes/published.txt').read().split()) if os.path.exists('results/classes/published.txt') else set()
publish = '--publish' in sys.argv
proofs, champs, ready = [], [], []
for (B, H), ex in sorted(classes.items(), key=lambda kv: (kv[0][0] != 'x', kv[0][0], kv[0][1])):
    done = all(e in ex and ex[e] and ex[e]['status'] == 'exhausted' for e in EXITS)
    label = f"b{B}_h{H}"
    row = ' '.join(f"e{e}:{ex[e]['best'] if e in ex and ex[e] else '..'}" for e in EXITS)
    cpu = sum(ex[e]['elapsed'] for e in EXITS if e in ex and ex[e])
    print(f"{label:8s} {'DONE ' if done else 'part '} {row}  cpu {cpu/3600:.2f} h" + ("  (published)" if label in published else ''))
    if not done or label in published: continue
    ready.append(label)
    hcap = None if (H == 'x' or (B != 'x' and int(H) >= int(B))) else int(H)
    bcap = None if B == 'x' else int(B)
    for e in EXITS:
        r = ex[e]
        desc = ' and '.join(x for x in [f"at most {bcap} movable block{'s' if bcap != 1 else ''}" if bcap is not None else None,
                                        f"at most {hcap} hole{'s' if hcap != 1 else ''}" if hcap is not None else None] if x)
        proofs.append(dict(rows=5, cols=5, exitCell=e, maxBlocks=bcap, maxHoles=hcap, maxMoves=r['best'], method=METHOD,
                           attribution="Panacea", states=r['states'], cpuSeconds=round(r['elapsed']),
                           note=f"Every 5x5 level with the exit at cell {e} using {desc} (any walls) was enumerated; {r['best']} moves is the longest optimal solution. {time.strftime('%Y-%m-%d')}."))
        if r['best'] > 0:
            lb = level_block(r['text'], r['best'])
            if lb: champs.append((label, e, r['best'], lb))
if not publish or not proofs:
    print(f"\n{len(ready)} class(es) ready to publish: {ready}" if ready else "\nnothing new to publish"); sys.exit(0)

json.dump(proofs, open('results/classes/proofs_pending.json', 'w'), indent=1)
subprocess.run(['scp', '-q', '-i', os.path.expanduser('~/.ssh/id_ed25519'), '-o', 'BatchMode=yes',
                'results/classes/proofs_pending.json', 'root@2.29.8.112:/opt/pathology/server/tools/proofs_pending.json'], check=True)
r = subprocess.run(SSH + ['cd /opt/pathology && chown pathology:pathology server/tools/proofs_pending.json && '
                          'runuser -u pathology -- node server/tools/add_proof.js --file server/tools/proofs_pending.json'],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-1000:].replace('X11 forwarding request failed on channel 0\n', ''))
for label, e, m, lb in champs:
    code = subprocess.run(['./level_to_pathology'], input=lb, capture_output=True, text=True).stdout.strip()
    payload = json.dumps({"code": code, "attribution": "Panacea"})
    res = subprocess.run(['curl', '-s', '-m', '300', '-H', 'Content-Type: application/json', '--data', payload,
                          'https://pathology.georgespahn.com/api/submit'], capture_output=True, text=True).stdout
    try: j = json.loads(res); msg = f"accepted id {j['level']['id']}" if j.get('accepted') else f"refused: {j.get('reason')}"
    except Exception: msg = res[:120]
    print(f"champion {label} exit {e} ({m}): {code.replace(chr(10), '/')} -> {msg}")
    time.sleep(61)   # the site allows 60 submissions per hour per IP
with open('results/classes/published.txt', 'a') as f: f.write(' '.join(ready) + '\n')
print("published:", ready)
