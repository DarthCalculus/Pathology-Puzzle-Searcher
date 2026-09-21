#!/bin/bash
# Auto-pilot (real rules: transit on): after the <=4-block campaign finishes,
# start the <=2-hole campaign.  Pause: echo 0 > results/camp_*/workers.  Kill: pkill -f results/chain.sh
cd "$(dirname "$0")/.."
while pgrep -f "campaign.py --out results/camp_b4" >/dev/null; do sleep 60; done
echo "[chain $(date)] camp_b4 driver finished; starting camp_h2" >> results/chain.log
nohup python3 campaign.py --out results/camp_h2 --exits 0,1,2,6,7,12 --extra "--num-holes 2" --layer 8 --workers 3 \
    --worker ./backsearch_worker_nt --shuffle --split-after 900 >> results/camp_h2.out 2>&1
echo "[chain $(date)] camp_h2 driver finished" >> results/chain.log
