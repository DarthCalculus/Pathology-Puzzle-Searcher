#!/bin/bash
# Server load test (DESIGN.md §7): a local server on a temp jobs.db seeded with
# synthetic roots, hammered by simulated clients (test_load.py).  No workers.
#
#   bash test_load.sh [CLIENTS=40] [WORKERS=32] [SECONDS=60] [JOB_S=2] [SERVER_DIR]
set -u
CLIENTS=${1:-40}; WORKERS=${2:-32}; SECONDS_=${3:-60}; JOB_S=${4:-2}; SERVER_DIR=${5:-/Users/george/PathologyRecords/server}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/v2load.XXXX); export JOBS_DATA_DIR=$T/data; mkdir -p "$T/data"
PORT=3198; URL=http://127.0.0.1:$PORT
HASH=$(printf 'ab%.0s' $(seq 1 32))
cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; }
trap cleanup EXIT

python3 - "$T" "$HASH" <<'EOF'
import itertools, json, random, sys
T, H = sys.argv[1], sys.argv[2]
random.seed(1)
toks = [d + v for d in 'URDL' for v in '123']
allp = [','.join(p) for p in itertools.product(toks, repeat=5)]
random.shuffle(allp)
open(T + '/layer.tsv', 'w').write('\n'.join(f"LAYER\t12\t{s}\t5\t2\t1" for s in allp[:50000]) + '\n')
json.dump({"title": "load test", "grid": "5x5", "extra": ["--allow-exit-transit", "--num-holes", "3"], "exits": [12], "layer": 5,
           "hashes": [H], "max_clients": 45, "workers_max": 32, "job_target_s": 1200, "split_after_s": 1800, "lease_s": 3600,
           "paused_max_s": 43200, "dup_fraction": 0.02}, open(T + '/plan.json', 'w'))
EOF
node "$SERVER_DIR/tools/v2_seed.js" --campaign-file "$T/plan.json" --layer-file "$T/layer.tsv" | tail -1
lsof -ti :$PORT | xargs -r kill 2>/dev/null; sleep 0.3
(cd "$SERVER_DIR" && exec env PORT=$PORT node server.js > "$T/server.log" 2>&1) & SRV=$!
for i in $(seq 1 50); do curl -s "$URL/api/v2/status" >/dev/null 2>&1 && break; sleep 0.2; done

# server CPU sampling in the background
( while kill -0 "$SRV" 2>/dev/null; do ps -o %cpu= -p "$SRV"; sleep 5; done ) > "$T/cpu.txt" &
python3 "$HERE/test_load.py" --server "$URL" --clients "$CLIENTS" --workers "$WORKERS" --seconds "$SECONDS_" --job-s "$JOB_S" --hash "$HASH"
RC=$?
echo "server CPU% samples: $(tr '\n' ' ' < "$T/cpu.txt")"
echo "jobs.db size: $(du -h "$T/data/jobs.db" | cut -f1); rows: $(sqlite3 "$T/data/jobs.db" 'select count(*) from jobs')"
echo "status (cached): $(curl -s "$URL/api/v2/status" | head -c 300)"
echo "logs in $T"; exit $RC
