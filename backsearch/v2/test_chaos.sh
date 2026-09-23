#!/bin/bash
# Integration chaos test (v2/DESIGN.md §5.2): real server (node), real worker,
# real client.  Small campaign (5x5 exit 7, <=3 blocks, ~4 s monolithic, layer-2
# roots so several jobs exceed split_after = 1 s and must split), lease 5 s.  Two clients; one is SIGKILLed and restarted,
# one is stopped with SIGINT and restarted, until the audit is clean.  Then
# the union of the workers' valid-level traces must equal the monolithic
# run's canonical level set and the best depth must agree.
#
#   bash test_chaos.sh WORKER_BINARY [SERVER_DIR]
set -u
WORKER=${1:?worker binary}; SERVER_DIR=${2:-/Users/george/PathologyRecords/server}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d /tmp/v2chaos.XXXX); export JOBS_DATA_DIR=$T/data; mkdir -p "$T/data" "$T/tv"
PORT=3199; URL=http://127.0.0.1:$PORT
# second-long test jobs would trip the production rate limits (30/min per token, 240/min per IP)
export V2_TOKEN_PER_MIN=100000 V2_IP_PER_MIN=100000 V2_IP_PER_HOUR=10000000
CFG=(--grid 5x5 --exit 7 --num-blocks 3 --allow-exit-transit)
HASH=$("$WORKER" --version | awk -F'\t' '/^SRC_HASH/{print $2}')
cleanup() { pkill -P $$ 2>/dev/null; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; }
trap cleanup EXIT

# 1. campaign + roots
cat > "$T/plan.json" <<EOF
{"title":"chaos","grid":"5x5","extra":["--allow-exit-transit","--num-blocks","3"],"exits":[7],"layer":2,
 "hashes":["$HASH"],"max_clients":5,"workers_max":4,"job_target_s":1,"split_after_s":1,"ramp_split_after_s":1,"lease_s":30,"paused_max_s":20,"dup_fraction":0.05,
 "absorb_probe_s":1,"absorb_total_s":5,"batch_interval_s":2,"lease_ahead_s":10}
EOF
"$WORKER" "${CFG[@]}" --two-tables --time 0 --list-layer 2 2>/dev/null | grep '^LAYER' > "$T/layer.tsv"
node "$SERVER_DIR/tools/v2_seed.js" --campaign-file "$T/plan.json" --layer-file "$T/layer.tsv" | tail -2

# 2. server
lsof -ti :$PORT | xargs -r kill 2>/dev/null; sleep 0.3   # a stale server from an aborted run would silently take the clients
(cd "$SERVER_DIR" && exec env PORT=$PORT node server.js > "$T/server.log" 2>&1) & SRV=$!
for i in $(seq 1 50); do curl -s "$URL/api/v2/status" >/dev/null 2>&1 && break; sleep 0.2; done

# 3. monolithic reference
BS_TRACE_VALID=$T/full "$WORKER" "${CFG[@]}" --two-tables --time 0 > "$T/full.out" 2>/dev/null
FULL_BEST=$(awk -F'\t' '/^LEVEL/{print $2}' "$T/full.out")

client() {  # name workers
  BS_TRACE_VALID=$T/tv/$1 VOLUNTEER_ALLOW_DIRTY=1 HOME=$T/home_$1 python3 "$HERE/volunteer.py" --name "$1" --workers "$2" --server "$URL" \
     --worker "$WORKER" --no-ui --outbox "$T/outbox_$1" --port $((8800 + RANDOM % 100)) >> "$T/client_$1.log" 2>&1 &
  echo $!
}
mkdir -p "$T/home_A" "$T/home_B"
audit_clean() { curl -s "$URL/api/v2/audit?exit=7" | python3 -c 'import json,sys; a=json.load(sys.stdin); print("1" if a.get("clean") else "0")'; }

A=$(client A 2); B=$(client B 2)
sleep 4;  kill -9 "$A" 2>/dev/null; pkill -9 -f "volunteer.py --name A" 2>/dev/null   # crash A (workers orphaned -> leases expire)
sleep 3;  A=$(client A 2)
sleep 4;  kill -INT "$B" 2>/dev/null                                                  # graceful stop B (hands back splits)
sleep 3;  B=$(client B 1)
for i in $(seq 1 120); do
  [ "$(audit_clean)" = 1 ] && break
  sleep 2
done
kill -INT "$A" "$B" 2>/dev/null; sleep 3
curl -s "$URL/api/v2/status" | python3 -c 'import json,sys; s=json.load(sys.stdin); t=s["totals"]; print("status:", {k:t[k] for k in ("jobs","open","leased","done","split","dups","best")}); sys.exit(0 if t["split"] > 0 else 3)' || { echo "FAIL: no job was split -- the test did not exercise the split path"; exit 3; }
SB=$(curl -s "$URL/api/v2/status" | python3 -c 'import json,sys; print(json.load(sys.stdin)["totals"]["best"])'); [ "$SB" = "$FULL_BEST" ] || { echo "FAIL: server best $SB != monolithic best $FULL_BEST (a champion level was lost on the way to the server)"; exit 4; }
echo "audit clean: $(audit_clean)"

# 4. invariant: canonical valid-level sets
python3 - "$T" "$FULL_BEST" "$HERE" 2 <<'EOF'
import glob, sys, os
T, full_best, here, LAYER = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.argv = ['x', 'dummy']  # test_split_exact parses argv at import; give it something harmless
src = open(os.path.join(here, 'test_split_exact.py')).read().split('def run(seed, split):')[0]
ns = {}; exec(src.replace("a = ap.parse_args(argv)", "a = None").replace("tmp = tempfile.mkdtemp(prefix='splitx_')", ""), ns)
canon = ns['canon']
def load(pattern):
    out = set(); best = 0
    for f in glob.glob(pattern):
        for l in open(f):
            _, d, code = l.rstrip('\n').split('\t')
            if int(d) < LAYER: continue   # levels above the root layer belong to no subtree job
            out.add((int(d), canon(code))); best = max(best, int(d))
    return out, best
full, fb = load(os.path.join(T, 'full.[0-9]*'))
jobs, jb = load(os.path.join(T, 'tv', '*'))
lost = full - jobs
print(f"full: {len(full)} levels best {fb}; jobs union: {len(jobs)} levels best {jb}; lost {len(lost)} extra {len(jobs - full)}")
sys.exit(0 if not lost and jb == fb else 'FAIL')
EOF
RC=$?
echo "logs in $T"; exit $RC
