#!/bin/bash
# Stage B: <=5 blocks with hole caps 0/1/2 (monolithic per exit, 3 workers), then
# <=1 hole with any blocks as a resumable campaign (results/camp_h1), then publish.
cd "$(dirname "$0")/../.."
CLASSES="5_0 5_1 5_2" P=3 bash results/classes/run_classes.sh
python3 results/classes/compile_classes.py --publish >> results/classes/publish_B.log 2>&1
python3 campaign.py --out results/camp_h1 --exits 0,1,2,6,7,12 --extra "--num-holes 1" --layer 8 --workers 3 \
    --worker ./backsearch_worker_nt --shuffle >> results/camp_h1.out 2>&1
echo "$(date '+%F %T') stage B finished (camp_h1 driver exited)" >> results/classes/run.log
