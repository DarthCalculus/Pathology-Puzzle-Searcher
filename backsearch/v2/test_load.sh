#!/bin/bash
# Server load test (DESIGN.md §7, review H8/M30): a local server on a temp jobs.db seeded with
# synthetic roots (with est_s, so the est_s lease order and the pool estimate are exercised),
# hammered by simulated protocol-3 clients (test_load.py). No workers.
#
#   bash test_load.sh [CLIENTS=40] [WORKERS=32] [SECONDS=60] [JOB_S=2] [SERVER_DIR]
#
# Environment: V2_LOAD_PORT (default 19430; a busy port is refused, never freed), TMPDIR (where the temp dir goes), STATUS_POLL_S
# (default 3: one hunt.html-style poller of /api/v2/status; 0 = none).
# Prints per-route latency (median / p95 / max), the server's CPU% samples (ps, every 5 s),
# its cumulative CPU time, and the rows written.
set -u
CLIENTS=${1:-40}; WORKERS=${2:-32}; SECONDS_=${3:-60}; JOB_S=${4:-2}; SERVER_DIR=${5:-/Users/george/PathologyRecords/server}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/v2load.XXXX"); export JOBS_DATA_DIR=$T/data RECORDS_DATA_DIR=$T/records; mkdir -p "$T/data" "$T/records"
PORT=${V2_LOAD_PORT:-19430}; URL=http://127.0.0.1:$PORT
HASH=$(printf 'ab%.0s' $(seq 1 32))
cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; [ -n "${CPU:-}" ] && kill "$CPU" 2>/dev/null; }
trap cleanup EXIT

# a protocol-3 campaign (allowlisted definition, canonical exits) with 50,000 depth-5 roots at the
# centre exit and one at each other exit, est_s spread over 4 orders of magnitude
node - "$SERVER_DIR" "$HASH" <<'EOF' || exit 1
const [dir, hash] = process.argv.slice(2);
const { openJobsDB } = require(dir + '/lib/jobs.js');
const J = openJobsDB(process.env.JOBS_DATA_DIR);
const c = J.createCampaign({ title: 'load test', grid: '5x5', extra: ['--allow-exit-transit', '--num-holes', '3'], layer: 5, hashes: [hash],
  max_clients: 45, workers_max: 32, job_target_s: 1200, split_after_s: 1800, lease_s: 3600, paused_max_s: 43200, dup_fraction: 0.02 });
let x = 0x2545F491;                     // xorshift32 (exact in 32-bit integer arithmetic)
const rnd = () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return (x >>> 0) / 4294967296; };
const toks = []; for (const d of 'URDL') for (const v of '123') toks.push(d + v);
const seeds = new Set();
while (seeds.size < 50000) seeds.add(Array.from({ length: 5 }, () => toks[Math.floor(rnd() * 12)]).join(','));
const roots = [...seeds].map(seed => ({ exit: 12, seed, est_s: Math.round(Math.pow(10, 4 * rnd())) }));
for (const e of [0, 1, 2, 6, 7]) roots.push({ exit: e, seed: 'U1,U1,U1,U1,U1', est_s: 10 });
const r = J.addRootJobs(c.id, roots);
console.log(`campaign #${c.id}: ${r.inserted} roots`);
J.close();
EOF
# M96: never stop a process this script did not start; a busy port is an error
if curl -s -o /dev/null --max-time 1 "$URL/" 2>/dev/null || lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT is in use: set V2_LOAD_PORT" >&2; exit 2
fi
(cd "$SERVER_DIR" && exec env PORT=$PORT JOBS_BACKUP=0 SOLVER_BIN_DIR="$T/no_solver" node server.js > "$T/server.log" 2>&1) & SRV=$!
for i in $(seq 1 50); do curl -s "$URL/api/v2/status" >/dev/null 2>&1 && break; sleep 0.2; done

# server CPU sampling in the background
( while kill -0 "$SRV" 2>/dev/null; do ps -o %cpu= -p "$SRV"; sleep 5; done ) > "$T/cpu.txt" & CPU=$!
python3 "$HERE/test_load.py" --server "$URL" --clients "$CLIENTS" --workers "$WORKERS" --seconds "$SECONDS_" --job-s "$JOB_S" --hash "$HASH" \
  --status-poll "${STATUS_POLL_S:-3}"
RC=$?
echo "server CPU% samples: $(tr '\n' ' ' < "$T/cpu.txt")"
echo "server CPU time: $(ps -o time= -p "$SRV" 2>/dev/null | tr -d ' ') (cumulative, whole run)"
echo "jobs.db size: $(du -h "$T/data/jobs.db" | cut -f1); rows: $(sqlite3 "$T/data/jobs.db" 'select count(*) from jobs'); events: $(sqlite3 "$T/data/jobs.db" 'select count(*) from events' 2>/dev/null)"
echo "status (cached): $(curl -s "$URL/api/v2/status" | head -c 300)"
grep -E "stats worker|failed|error" "$T/server.log" | head -5
echo "logs in $T"; exit $RC
