#!/bin/bash
# Auto-finish for the <=2-hole campaign (results/camp_h2).
# Waits for the campaign driver to exit, then — only if every job is exhausted —
#   1. records one proof per exit on the live site (5x5, exit e, <=2 holes, max M),
#   2. submits each exit's champion level as an ORDINARY submission (never playground),
#   3. writes results/camp_h2_FINAL.txt.
# Everything is logged to results/autofinish_h2.log. Kill with: pkill -f autofinish_h2.sh
cd "$(dirname "$0")/.."
LOG=results/autofinish_h2.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
SSH="ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes root@2.29.8.112"
log "watching for the camp_h2 driver to exit"
while pgrep -f "campaign.py --out results/camp_h2" >/dev/null; do sleep 120; done
log "driver exited"
python3 campaign.py --out results/camp_h2 --status >> "$LOG" 2>&1

python3 - <<'EOF' > results/camp_h2_summary.json
import csv, json, collections
jobs=[l.split('\t') for l in open('results/camp_h2/jobs.tsv').read().split('\n') if l]
rows=list(csv.DictReader(open('results/camp_h2/done.tsv'), delimiter='\t'))
done={(r['exit'],r['path']):r for r in rows}
missing=[j for j in jobs if (j[0],j[1]) not in done]
bad=[r for r in rows if r['status'] not in ('exhausted','split')]
best=collections.defaultdict(int); bestjob={}; states=collections.Counter(); cpu=collections.Counter()
for r in rows:
    e=int(r['exit']); b=int(r['best']); states[e]+=int(r['states']); cpu[e]+=float(r['elapsed'])
    if b>best[e]: best[e]=b; bestjob[e]=r['path']
print(json.dumps({"complete": not missing and not bad, "missing": len(missing), "bad": len(bad),
                  "best": best, "bestjob": bestjob, "states": states, "cpu": {e: round(c) for e,c in cpu.items()}}))
EOF
cat results/camp_h2_summary.json >> "$LOG"
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('results/camp_h2_summary.json'))['complete'] else 1)"; then
  log "campaign NOT complete (missing or non-exhausted jobs) — nothing submitted"; exit 1
fi

# --- 1. proofs ---------------------------------------------------------------
python3 - <<'EOF' > results/proofs_5x5_h2.json
import json
s=json.load(open('results/camp_h2_summary.json'))
method="exhaustive backward search (backsearch_worker + campaign.py, real rules: blocks may transit the exit, no block starts on the exit)"
out=[]
for e,m in sorted(s['best'].items(), key=lambda kv:int(kv[0])):
    out.append({"rows":5,"cols":5,"exitCell":int(e),"maxHoles":2,"maxMoves":int(m),"method":method,"attribution":"Panacea",
                "states":int(s['states'][e]),"cpuSeconds":int(s['cpu'][e]),
                "note":f"Every 5x5 level with the exit at cell {e} and at most 2 holes (any blocks, any walls) was enumerated; {m} moves is the longest optimal solution. Campaign results/camp_h2, finished {__import__('datetime').date.today()}."})
json.dump(out, open('results/proofs_5x5_h2.json','w'), indent=1); print(len(out), "proofs")
EOF
scp -q -i "$HOME/.ssh/id_ed25519" -o BatchMode=yes results/proofs_5x5_h2.json root@2.29.8.112:/opt/pathology/server/tools/proofs_5x5_h2.json 2>>"$LOG"
$SSH 'cd /opt/pathology && chown pathology:pathology server/tools/proofs_5x5_h2.json && runuser -u pathology -- node server/tools/add_proof.js --file server/tools/proofs_5x5_h2.json' >> "$LOG" 2>&1
log "proofs step done (see above; a REFUSED line means a stored level contradicts the search)"

# --- 2. champions ------------------------------------------------------------
python3 - <<'EOF'
import json, subprocess, os, re, time
s=json.load(open('results/camp_h2_summary.json'))
def log(m): open('results/autofinish_h2.log','a').write(time.strftime('[%F %T] ')+m+'\n')
for e,m in sorted(s['best'].items(), key=lambda kv:int(kv[0])):
    path=s['bestjob'][e]
    # re-run the job that found the maximum to print the level (cap 3 h; new-best lines stream early)
    cmd=['./backsearch_worker_nt','--grid','5x5','--two-tables','--allow-exit-transit','--num-holes','2','--exit',e,'--seed-path',path,'--time','10800']
    out=subprocess.run(cmd,capture_output=True,text=True).stdout
    mm=re.search(r'^%s \([^)]*\)\n((?:  .*\n)+)'%m, out, re.M)
    if not mm: log(f"exit {e}: could not extract the {m}-move level from job {path}"); continue
    lvl=mm.group(1); open(f'results/camp_h2_champion_e{e}.txt','w').write(lvl)
    code=subprocess.run(['./level_to_pathology'],input=lvl,capture_output=True,text=True).stdout.strip()
    payload=json.dumps({"code":code,"attribution":"Panacea"})
    r=subprocess.run(['curl','-s','-m','300','-H','Content-Type: application/json','--data',payload,'https://pathology.georgespahn.com/api/submit'],capture_output=True,text=True).stdout
    log(f"exit {e} max {m}: level {code.replace(chr(10),'/')} -> submit: {r[:300]}")
    time.sleep(8)
EOF

# --- 3. summary -------------------------------------------------------------
{
  echo "<=2-hole campaign (5x5, real rules) finished $(date '+%F %T')"
  python3 campaign.py --out results/camp_h2 --status
  echo; echo "per-exit maxima / champion jobs:"; cat results/camp_h2_summary.json; echo
  echo "proofs + submissions: see results/autofinish_h2.log"
} > results/camp_h2_FINAL.txt
echo "- $(date '+%F'): camp_h2 finished; autofinish_h2.sh recorded proofs + submitted champions (see results/autofinish_h2.log, results/camp_h2_FINAL.txt)" >> "$HOME/.claude/projects/-Users-george-PathologyRecords/memory/project_backsearch.md"
log "all done"
