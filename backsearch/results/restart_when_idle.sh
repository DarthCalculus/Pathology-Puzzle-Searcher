#!/bin/bash
# Cap is 0, so the driver starts nothing new. When its last worker finishes,
# restart it under the fixed campaign.py with 3 workers and re-arm autofinish.
cd "$(dirname "$0")/.."
LOG=results/autofinish_h2.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
D=$(pgrep -f "campaign.py --out results/camp_h2" | head -1)
log "idle watcher: waiting for driver $D to have no running workers"
while [ -n "$(pgrep -P "$D" 2>/dev/null)" ]; do sleep 30; done
log "driver idle -> restarting under fixed campaign.py"
pkill -f autofinish_h2.sh; kill "$D"; sleep 2
echo 3 > results/camp_h2/workers
nohup python3 campaign.py --out results/camp_h2 --workers 3 --worker ./backsearch_worker_nt --shuffle >> results/camp_h2.out 2>&1 &
sleep 3
nohup bash results/autofinish_h2.sh >/dev/null 2>&1 &
log "driver restarted (pid $(pgrep -f 'campaign.py --out results/camp_h2' | head -1)); autofinish re-armed"
