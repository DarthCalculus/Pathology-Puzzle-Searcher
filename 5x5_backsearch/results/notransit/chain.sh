#!/bin/bash
# Auto-pilot: when the exit-2 campaign finishes, give exit 1 three workers;
# when exit 1 finishes, start the exit-0 campaign.  Pause anything with
# `echo 0 > results/camp_eN/workers`; kill this chain with: pkill -f results/chain.sh
cd "$(dirname "$0")/.."
until grep -q "jobs done 2738/2738" results/camp_e2.out 2>/dev/null; do sleep 30; done
echo 3 > results/camp_e1/workers
echo "[chain $(date)] exit 2 campaign done; exit 1 -> 3 workers" >> results/chain.log
while pgrep -f "campaign.py --out results/camp_e1" >/dev/null; do sleep 60; done
echo "[chain $(date)] exit 1 campaign driver finished; starting exit 0 campaign" >> results/chain.log
nohup python3 campaign.py --out results/camp_e0 --exits 0 --layer 8 --workers 3 \
    --worker ./backsearch_worker_nt --shuffle --split-after 900 >> results/camp_e0.out 2>&1
echo "[chain $(date)] exit 0 campaign driver finished" >> results/chain.log
