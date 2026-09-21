#!/bin/bash
# Restart the camp_h2 driver under the fixed (non-starving) campaign.py as soon as
# no depth-11 child is running, so the long child jobs already in flight are not lost.
cd "$(dirname "$0")/.."
LOG=results/autofinish_h2.log
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "restart watcher: waiting for all running depth-11 children to finish"
while ps -eo args= | grep "backsearch_worker_nt --grid" | grep -v grep | grep -oE -- "--seed-path [^ ]+" | awk '{print $2}' | grep -q -E "^([^,]+,){10,}"; do sleep 60; done
log "no child in flight -> restarting driver under fixed campaign.py"
pkill -f autofinish_h2.sh
pkill -f "campaign.py --out results/camp_h2"; sleep 1
pkill -f "backsearch_worker_nt --grid 5x5 --two-tables --exit"; sleep 2
echo 3 > results/camp_h2/workers
nohup python3 campaign.py --out results/camp_h2 --workers 3 --worker ./backsearch_worker_nt --shuffle >> results/camp_h2.out 2>&1 &
sleep 3
nohup bash results/autofinish_h2.sh >/dev/null 2>&1 &
log "driver restarted (pid $(pgrep -f 'campaign.py --out results/camp_h2' | head -1)); autofinish re-armed"
