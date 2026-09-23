#!/usr/bin/env bash
# Client tests for volunteer.py against fake_server.py + fake_worker.py.
#
#   ./test_client.sh            run all tests, one after the other
#   ./test_client.sh a|b|c|d    run one test
#
# a  stop_clean    2 workers for 15 s, SIGINT: clean exit; a second run covers every root.
# b  kill9         kill -9 the client mid-run, restart: no job lost (audit clean).
# c  pause_resume  /pause and /resume flip the workers' process state (ps STAT 'T');
#                  /workers changes the count; /stop exits cleanly; POST without the
#                  same-origin header is refused.
# d  no_summary    a worker that dies without SUMMARY is logged and does not spin.
#
# Each test finishes in under a minute, uses at most 2 fake workers, and never
# touches ~/.pathology_volunteer.json (HOME is redirected to a temp dir).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
SPORT="${TEST_SERVER_PORT:-18931}"
UPORT="${TEST_UI_PORT:-18766}"
SERVER="http://127.0.0.1:$SPORT"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/volunteer_test.XXXXXX")"
export HOME="$TMP/home"; mkdir -p "$HOME"
export VOLUNTEER_ALLOW_DIRTY=1
export FAKE_MIN_S=1 FAKE_MAX_S=4
SERVER_PID=""; CLIENT_PID=""; FAIL=0

say() { printf '\n== %s\n' "$*"; }
ok() { printf '   ok   %s\n' "$*"; }
bad() { printf '   FAIL %s\n' "$*"; FAIL=1; }

cleanup() {
  [ -n "$CLIENT_PID" ] && kill -9 "$CLIENT_PID" 2>/dev/null
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  pkill -f "$HERE/fake_worker.py" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

http_get() { "$PY" -c 'import sys,urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())' "$1" 2>/dev/null; }
# http_post URL [json] [header-value]; prints "<status> <body>"
http_post() {
  "$PY" - "$1" "${2:-{\}}" "${3:-}" <<'EOF'
import sys, urllib.request, urllib.error
url, body, hdr = sys.argv[1], sys.argv[2], sys.argv[3]
h = {"Content-Type": "application/json"}
if hdr: h["X-Volunteer"] = hdr
req = urllib.request.Request(url, data=body.encode(), headers=h, method="POST")
try:
    with urllib.request.urlopen(req, timeout=5) as r: print(r.status, r.read().decode())
except urllib.error.HTTPError as e: print(e.code, e.read().decode())
EOF
}
audit() { http_get "$SERVER/api/v2/audit"; }
audit_clean() {
  audit | "$PY" -c 'import sys,json; a=json.load(sys.stdin); c=a["counts"]; sys.exit(0 if not a["uncovered_roots"] and c["open"]==0 and c["leased"]==0 else 1)'
}
state_field() { http_get "http://127.0.0.1:$UPORT/state" | "$PY" -c "import sys,json; s=json.load(sys.stdin); print($1)"; }

start_server() {  # start_server ROOTS LEASE_S SPLIT_AFTER_S
  "$PY" "$HERE/fake_server.py" --port "$SPORT" --roots "$1" --lease-s "$2" --split-after-s "$3" 2>"$TMP/server.log" &
  SERVER_PID=$!
  for _ in $(seq 1 60); do http_get "$SERVER/api/v2/status" >/dev/null 2>&1 && return 0; sleep 0.5; done
  bad "fake server did not come up"; cat "$TMP/server.log"; return 1
}
stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }

start_client() {  # start_client LOGNAME extra-args...
  local logname="$1"; shift
  "$PY" "$HERE/volunteer.py" --name "Tester" --server "$SERVER" --worker "$HERE/fake_worker.py" \
      --outbox "$TMP/outbox" --port "$UPORT" "$@" 2>"$TMP/$logname" &
  CLIENT_PID=$!
  for _ in $(seq 1 40); do grep -q -e "registered as" -e "reusing registration" "$TMP/$logname" 2>/dev/null && return 0; sleep 0.5; done
  bad "client did not register within 20 s"; tail -5 "$TMP/$logname"; return 1
}
wait_exit() {  # wait_exit PID SECONDS -> 0 if exited
  local i; for i in $(seq 1 $(( $2 * 2 ))); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.5; done; return 1
}
wait_audit_clean() {  # wait_audit_clean SECONDS
  local i; for i in $(seq 1 "$1"); do audit_clean && return 0; sleep 1; done; return 1
}
fake_workers_alive() { pgrep -f "$HERE/fake_worker.py" 2>/dev/null | wc -l | tr -d ' '; }

# ------------------------------------------------------------------------------
test_a() {
  say "test a: stop_clean"
  start_server 10 5 2.5 || return
  FAKE_MAX_S=5 start_client a1.log --workers 2 --no-ui || return
  sleep 12   # workers are still busy: SIGINT forces at least one split
  kill -INT "$CLIENT_PID"
  if wait_exit "$CLIENT_PID" 25; then wait "$CLIENT_PID"; rc=$?; else bad "client did not exit within 25 s of SIGINT"; return; fi
  [ "$rc" = 0 ] && ok "client exited 0 after SIGINT" || bad "client exit code $rc"
  grep -q "stopped\. jobs done" "$TMP/a1.log" && ok "final summary logged: $(grep -o 'jobs done.*' "$TMP/a1.log")" || bad "no final summary in log"
  grep -q '!!!' "$TMP/a1.log" && bad "client logged an error: $(grep '!!!' "$TMP/a1.log" | head -2)" || ok "no protocol errors logged"
  [ "$(fake_workers_alive)" = 0 ] && ok "no fake workers left behind" || bad "fake workers still running"
  a=$(audit); echo "   audit after run 1: $a"
  echo "$a" | grep -q '"split": [1-9]' && ok "SIGINT produced split reports" || bad "no split after SIGINT"
  sleep 6   # leases of never-started jobs expire (lease_s = 5)
  start_client a2.log --workers 2 --no-ui || return
  if wait_audit_clean 25; then ok "audit clean after run 2: $(audit)"; else bad "audit not clean after run 2: $(audit)"; fi
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20 || bad "second client did not exit"
  CLIENT_PID=""; stop_server
}

test_b() {
  say "test b: kill9"
  start_server 6 5 2.5 || return
  start_client b1.log --workers 2 --no-ui || return
  sleep 6
  kill -9 "$CLIENT_PID"; wait "$CLIENT_PID" 2>/dev/null; CLIENT_PID=""
  ok "client killed with SIGKILL mid-run; audit: $(audit)"
  sleep 3
  n=$(fake_workers_alive)
  [ "$n" = 0 ] && ok "orphaned fake workers died on their own (broken pipe)" || bad "$n orphaned fake workers still alive"
  start_client b2.log --workers 2 --no-ui || return
  if wait_audit_clean 35; then ok "audit clean after restart (no job lost): $(audit)"; else bad "audit not clean: $(audit)"; fi
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20 || bad "client did not exit"
  CLIENT_PID=""; stop_server
}

test_c() {
  say "test c: pause_resume"
  start_server 6 60 60 || return
  FAKE_MIN_S=25 FAKE_MAX_S=30 start_client c.log --workers 2 --no-browser || return
  pids=""
  for _ in $(seq 1 30); do pids=$(state_field "' '.join(str(w['pid']) for w in s['slots'] if w['pid'])" 2>/dev/null); [ "$(echo $pids | wc -w | tr -d ' ')" = 2 ] && break; sleep 0.5; done
  [ "$(echo $pids | wc -w | tr -d ' ')" = 2 ] && ok "two workers running (pids $pids)" || { bad "expected 2 running workers, state: $(http_get http://127.0.0.1:$UPORT/state | head -c 300)"; return; }
  r=$(http_post "http://127.0.0.1:$UPORT/pause" '{}' '')
  case "$r" in 403*) ok "POST without the same-origin header is refused (403)";; *) bad "POST without header answered: $r";; esac
  http_post "http://127.0.0.1:$UPORT/pause" '{}' 1 >/dev/null; sleep 1
  st=$(for p in $pids; do ps -o stat= -p "$p"; done | tr -d ' \n')
  case "$st" in *T*T*) ok "after /pause both workers are stopped (STAT $st)";; *) bad "after /pause STAT is '$st'";; esac
  [ "$(state_field "s['paused']")" = True ] && ok "/state reports paused" || bad "/state does not report paused"
  http_post "http://127.0.0.1:$UPORT/resume" '{}' 1 >/dev/null; sleep 1
  st=$(for p in $pids; do ps -o stat= -p "$p"; done | tr -d ' \n')
  case "$st" in *T*) bad "after /resume STAT still '$st'";; *) ok "after /resume workers run again (STAT $st)";; esac
  http_post "http://127.0.0.1:$UPORT/workers" '{"workers": 1}' 1 >/dev/null; sleep 0.5
  [ "$(state_field "s['workers']")" = 1 ] && ok "/workers changed the count to 1" || bad "/workers did not change the count"
  http_post "http://127.0.0.1:$UPORT/stop" '{}' 1 >/dev/null
  if wait_exit "$CLIENT_PID" 20; then wait "$CLIENT_PID"; rc=$?; [ "$rc" = 0 ] && ok "/stop: client exited 0" || bad "/stop: exit code $rc"; else bad "client did not exit after /stop"; fi
  CLIENT_PID=""
  a=$(audit); echo "   audit: $a"
  echo "$a" | grep -q '"split": [1-9]' && ok "interrupted jobs were reported as splits" || bad "no split reported after /stop"
  stop_server
}

test_d() {
  say "test d: no_summary (worker dies)"
  start_server 4 60 60 || return
  FAKE_DIE=1 start_client d.log --workers 1 --no-ui || return
  sleep 12
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 10 || bad "client did not exit"; CLIENT_PID=""
  n=$(grep -c "WITHOUT a SUMMARY" "$TMP/d.log")
  [ "$n" -ge 3 ] && ok "$n runs without SUMMARY were logged loudly" || bad "expected >= 3 loud logs, got $n"
  [ "$n" -le 5 ] && ok "no spinning: only $n launches in 12 s (backoff after 3 failures)" || bad "client spun: $n launches"
  grep -q "backing off" "$TMP/d.log" && ok "backoff engaged" || bad "no backoff message"
  a=$(audit); echo "$a" | grep -q '"done": 0' && ok "no job was reported (all left to lease expiry)" || bad "unexpected report: $a"
  stop_server
}

which="${1:-all}"
start=$(date +%s)
case "$which" in
  a) test_a;; b) test_b;; c) test_c;; d) test_d;;
  all) test_a; test_b; test_c; test_d;;
  *) echo "usage: $0 [a|b|c|d|all]"; exit 2;;
esac
echo
echo "logs in $TMP  (elapsed $(( $(date +%s) - start )) s)"
if [ "$FAIL" = 0 ]; then echo "ALL PASSED"; else echo "SOME TESTS FAILED"; exit 1; fi
