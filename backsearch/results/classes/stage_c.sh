#!/bin/bash
# Stage C (real rules): <=6 blocks with 0/1 holes monolithic, publish; then
# <=6 blocks <=2 holes as a campaign, publish.  Waits for the final worker build.
cd "$(dirname "$0")/../.."
S=/private/tmp/claude-502/-Users-george-PathologyRecords/92d6b5c4-ca51-4888-97d7-50af2caf7f30/scratchpad
until [ -f "$S/worker_final" ]; do sleep 10; done
echo "$(date '+%F %T') stage C start" >> results/classes/run.log
CLASSES="6_0 6_1" P=2 bash results/classes/run_classes.sh
python3 results/classes/compile_classes.py --publish >> results/classes/publish_C.log 2>&1
python3 campaign.py --out results/camp_b6h2 --exits 0,1,2,6,7,12 --extra "--num-blocks 6 --num-holes 2" --layer 8 --workers 2 \
    --worker ./backsearch_worker_nt --shuffle >> results/camp_b6h2.out 2>&1
python3 results/publish_campaign.py results/camp_b6h2 --max-blocks 6 --max-holes 2 >> results/camp_b6h2_publish.out 2>&1
echo "$(date '+%F %T') stage C finished" >> results/classes/run.log
