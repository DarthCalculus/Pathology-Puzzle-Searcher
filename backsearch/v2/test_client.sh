#!/usr/bin/env bash
# Client tests for volunteer.py (protocol 3) against fake_server.py + fake_worker.py.
#
#   ./test_client.sh               run all tests, one after the other
#   ./test_client.sh a c h ...     run some tests
#
# a  stop_clean    2 workers, 4 s windows, SIGINT mid-window: clean exit, tree reports with
#                  absorbed children, the final heartbeat releases everything (stopping, holding []);
#                  a second run covers every root and exits 0 on its own (campaign complete).
# b  kill9         kill -9 mid-window, restart: no job lost; kill -9 while PAUSED leaves no
#                  stopped worker behind; a worker orphaned by an earlier run is killed at startup.
# c  pause_resume  /pause and /resume flip the workers' process state; /workers; /exit; /stop;
#                  panel security: foreign Host / Origin and a missing session secret get 403.
# d  no_summary    a worker that dies without SUMMARY: loud log, backoff, 'crash' failure reports.
# e  coalesce      60 trivial roots: one-node reports, batched into few POST /reports, few leases.
# f  closed_410    the campaign is closed and a new one opened: the running client re-registers
#                  into it; a restart with the stored (closed) token does the same.
# g  too_old_426   the server wants a newer client: the server's message is printed, exit 2.
# h  complete      the campaign completes: closing summary, final heartbeat, exit 0; a restart with
#                  no campaign running exits 0 with a message.
# i  keep_going    --keep-going: waits after completion and joins the next campaign.
# j  instances     a second client on the same outbox refuses to start; a client with its own
#                  outbox gets its own token and the two never release each other's jobs.
# k  extra_refuse  campaign flags outside the allowlist (--harvest PATH, --allow-block-on-exit,
#                  no --allow-exit-transit) make the client refuse to run; no file is created.
# l  foldback      a split with 6000 REMAINING lines is never truncated (job handed back
#                  untouched, failure 'too_big'); fold-back of nested split trees (unit test of fit_tree).
# m  failures      bad_seed and a malformed REMAINING line become failure reports, never splits;
#                  REMAINING == seed: no failure on Stop, 'no_progress' on the worker's own timer.
# n  watchdog      a silent worker that ignores SIGINT is killed and reported as 'hang'; a slow
#                  SIGINT answer (3 s) is still a normal split.
# o  batch413      a server body cap below the batch size: 413, the client halves, all delivered.
# p  unresolved    UNRESOLVED lines are forwarded per node as `unresolved` candidates.
# q  dedupe        a lease grant of a job the client already holds is skipped (M19).
# r  empty_root    a root with the empty seed "" runs without --seed-path.
# s  units         pure functions: extra allowlist, BS_* stripping, HTTP status classification.
# t  exit_check    --exit outside the campaign falls back to any exit.
# u  outbox        reports written while the server is down are delivered after a restart.
# v  complete_410  a complete campaign's token gets 410 once a newer campaign is open: at startup
#                  the client joins the new campaign; while running it ends with the summary, exit 0.
# w  sleep         client and worker frozen across the split time (a closed lid): a normal split, no
#                  'hang'; frozen after Stop: the stop grace does not count the sleep.
# x  grace_kill    an absorption probe killed after Stop's grace leaves its node open and the window
#                  is still reported; a lease with absorb_total_s 0 (endgame) runs no probes.
# y  absorb_order  M18: probes go deepest first; the first probe that splits is kept as a split node,
#                  the pass goes on into its children only; shallower siblings are never probed.
#                  A budget-cut probe that splits with its children never probed goes back open.
# z  lease         M108: at most 2 jobs per worker held; a queued job starts with the server's CURRENT
#                  window (not the one of its lease); N18: sub-second windows (min_window_s) and
#                  --split-after-nodes reach the worker; workers 3 -> 1 while the retiring ones are
#                  still busy: the kept worker goes on leasing (the hold cap counts only kept workers).
# pw pause_workers M56: Pause, fewer workers, Resume leaves no worker stopped; fewer workers, then Pause
#                  freezes the retiring ones too; M61: Ctrl-Z (SIGTSTP) freezes the workers, fg thaws them.
# ui panel         the panel's script in node with a small DOM stand-in: 6x6 boards, the global board with
#                  the server's newline codes (M55), 'Any exit' = the longest best (M59), no redraw of an
#                  unchanged board (M63), the completion banner, paused_max_s (M67). Skipped without node.
#
# Ports: TEST_SERVER_PORT (default 19131) and TEST_UI_PORT (default 19166, +1 for a second
# client); the script refuses to run when one is already taken. FAKE_SEED seeds the fake
# workers (reproducible runs). HOME is redirected to a temp dir. Every test ends its processes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
SPORT="${TEST_SERVER_PORT:-19131}"
UPORT="${TEST_UI_PORT:-19166}"
UPORT2=$((UPORT + 1))
SERVER="http://127.0.0.1:$SPORT"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/volunteer_test.XXXXXX")"
export HOME="$TMP/home"; mkdir -p "$HOME"
export VOLUNTEER_ALLOW_DIRTY=1
export FAKE_MIN_S=1 FAKE_MAX_S=4 FAKE_SEED="${FAKE_SEED:-1}"
SERVER_PID=""; CLIENT_PID=""; CLIENT2_PID=""; EXTRA_PIDS=""; FAIL=0; NSERV=0; SLOG="$TMP/server-0.log"
FAST=(--campaign-poll-s 1 --complete-flush-s 5)

T0=0
say() { T0=$(date +%s); printf '\n== %s\n' "$*"; }
done_test() { local e=$(( $(date +%s) - T0 )); [ "$e" -lt "${1:-60}" ] && ok "finished in $e s" || bad "took $e s (limit ${1:-60})"; }
ok() { printf '   ok   %s\n' "$*"; }
bad() { printf '   FAIL %s\n' "$*"; FAIL=1; }

# the client runs a pinned copy of the worker from <outbox>/.bin/, so this pattern matches only
# workers of this test run (never another checkout's or another agent's processes)
WPAT="$TMP/[^ ]*\.bin/fake_worker\.py"
fake_workers_alive() { pgrep -f "$WPAT" 2>/dev/null | wc -l | tr -d ' '; }
cleanup() {
  for p in $CLIENT_PID $CLIENT2_PID $EXTRA_PIDS; do kill -9 "$p" 2>/dev/null; done
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  for p in $(pgrep -f "$WPAT" 2>/dev/null); do kill -CONT "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done
  wait 2>/dev/null
}
# after every test: a test that gave up early (return) must not leave its server, clients or
# workers behind for the next one (its port would be taken)
end_test_procs() { cleanup; SERVER_PID=""; CLIENT_PID=""; CLIENT2_PID=""; EXTRA_PIDS=""; }
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

port_free() { "$PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) else 1)' "$1"; }
for p in "$SPORT" "$UPORT" "$UPORT2"; do
  port_free "$p" || { echo "port $p is already in use (a stale test server?): set TEST_SERVER_PORT / TEST_UI_PORT"; exit 2; }
done

http_get() {  # http_get URL [HOST]
  "$PY" - "$1" "${2:-}" <<'EOF' 2>/dev/null
import sys, urllib.request, urllib.error
url, host = sys.argv[1], sys.argv[2]
h = {"Host": host} if host else {}
try:
    print(urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=5).read().decode())
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()); sys.exit(1)
EOF
}
# http_post URL [json] [X-Volunteer value] [Host] [Origin]; prints "<status> <body>"
http_post() {
  "$PY" - "$1" "${2:-{\}}" "${3:-}" "${4:-}" "${5:-}" <<'EOF'
import sys, urllib.request, urllib.error
url, body, hdr, host, origin = sys.argv[1:6]
h = {"Content-Type": "application/json"}
if hdr: h["X-Volunteer"] = hdr
if host: h["Host"] = host
if origin: h["Origin"] = origin
req = urllib.request.Request(url, data=body.encode(), headers=h, method="POST")
try:
    with urllib.request.urlopen(req, timeout=5) as r: print(r.status, r.read().decode())
except urllib.error.HTTPError as e: print(e.code, e.read().decode())
except Exception as e: print(0, e)
EOF
}
header_of() {  # header_of URL NAME
  "$PY" -c 'import sys,urllib.request; r=urllib.request.urlopen(sys.argv[1], timeout=5); print(r.headers.get(sys.argv[2]) or "")' "$1" "$2" 2>/dev/null
}
ui_secret() { http_get "http://127.0.0.1:$1/" | sed -n 's/.*name="volunteer-secret" content="\([0-9a-f]*\)".*/\1/p' | head -1; }
admin() { http_post "$SERVER/api/v2/_admin" "$1" >/dev/null; }
audit() { http_get "$SERVER/api/v2/audit${1:+?campaign=$1}"; }
aq() {  # aq PYTHON-EXPR [campaign]: evaluate EXPR with a = the audit JSON
  audit "${2:-}" | "$PY" -c "import sys,json; a=json.load(sys.stdin); print($1)"
}
stat() { aq "a['stats']['$1']"; }
audit_clean() { [ "$(aq "a['clean'] and a['counts']['open']==0 and a['counts']['leased']==0" "${1:-}")" = True ]; }
state_field() { http_get "http://127.0.0.1:${2:-$UPORT}/state" | "$PY" -c "import sys,json; s=json.load(sys.stdin); print($1)"; }

start_server() {  # start_server ROOTS LEASE_S SPLIT_AFTER_S [more fake_server flags]
  local roots="$1" lease="$2" split="$3"; shift 3
  port_free "$SPORT" || { bad "port $SPORT taken before the fake server started"; return 1; }
  NSERV=$((NSERV + 1)); SLOG="$TMP/server-$NSERV.log"
  "$PY" "$HERE/fake_server.py" --port "$SPORT" --roots "$roots" --lease-s "$lease" --split-after-s "$split" "$@" 2>"$SLOG" &
  SERVER_PID=$!
  local i
  for i in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || { bad "fake server died: $(tail -3 "$SLOG")"; SERVER_PID=""; return 1; }
    grep -q "listening on" "$SLOG" 2>/dev/null && http_get "$SERVER/api/v2/status" >/dev/null && return 0
    sleep 0.25
  done
  bad "fake server did not come up"; cat "$SLOG"; return 1
}
stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }

start_client() {  # start_client LOGNAME extra-args...   (outbox $TMP/outbox, port UPORT)
  local logname="$1"; shift
  "$PY" "$HERE/volunteer.py" --name "Tester" --server "$SERVER" --worker "$HERE/fake_worker.py" \
      --outbox "${OUTBOX:-$TMP/outbox}" --port "$UPORT" "${FAST[@]}" "$@" 2>"$TMP/$logname" &
  CLIENT_PID=$!
  wait_log "$logname" "registered as|reusing registration" 20 || { bad "client did not register within 20 s"; tail -5 "$TMP/$logname"; return 1; }
}
wait_log() {  # wait_log LOGNAME ERE SECONDS
  local i; for i in $(seq 1 $(( $3 * 4 ))); do grep -Eq -e "$2" "$TMP/$1" 2>/dev/null && return 0; sleep 0.25; done; return 1
}
wait_exit() {  # wait_exit PID SECONDS -> 0 if exited
  local i; for i in $(seq 1 $(( $2 * 4 ))); do kill -0 "$1" 2>/dev/null || return 0; sleep 0.25; done; return 1
}
wait_rc() {  # wait_rc PID SECONDS: sets RC to the exit code (or "timeout"); never in a subshell
  if wait_exit "$1" "$2"; then wait "$1"; RC=$?; else RC=timeout; fi
}
wait_true() {  # wait_true SECONDS CMD...: poll until CMD succeeds
  local s="$1" i; shift; for i in $(seq 1 $(( s * 4 ))); do "$@" && return 0; sleep 0.25; done; return 1
}
wait_audit_clean() { wait_true "$1" audit_clean "${2:-}"; }
wait_leases_expired() { wait_true "$1" eval '[ "$(aq "a[\"counts\"][\"leased\"]")" = 0 ]'; }
no_bang() {  # no_bang LOGNAME: the client logged no '!!!' line
  grep -q '!!!' "$TMP/$1" && bad "client logged an error: $(grep '!!!' "$TMP/$1" | head -2)" || ok "no protocol errors logged"
}
final_heartbeat_ok() {
  local fh; fh=$(aq "a['last_heartbeat'].get('Tester', {})")
  case "$fh" in *"'holding': []"*"'running': []"*"'stopping': True"*) ok "final heartbeat: stopping, holding [], running []";;
    *) bad "final heartbeat was $fh";; esac
}

# ------------------------------------------------------------------------------
tree_ok() {  # assert absorbed children exist and were never leased as separate jobs
  local ad ga; ad=$(stat absorbed_done); ga=$(stat grants_of_absorbed)
  [ "$ad" -ge 1 ] && ok "$ad trivial children were absorbed locally (arrived done inside tree reports)" || bad "no child was absorbed locally"
  [ "$ga" = 0 ] && ok "none of them was ever leased as a separate server job" || bad "$ga absorbed children were leased"
}

test_a() {
  say "test a: stop_clean (windows, tree reports, final heartbeat, completion)"
  # window 4 s, probes 2 s (<= 6 s total), leases 14 s > window + absorption. Roots take 4.5-6 s,
  # so every root splits at the 4 s window; the last REMAINING child of each split is 3 tokens
  # deeper (0.6^3 as long) and always absorbs in a 2 s probe.
  start_server 4 14 4 --absorb-total-s 6 --absorb-probe-s 2 --batch-interval-s 3 --heartbeat-s 5 || return
  FAKE_MIN_S=4.5 FAKE_MAX_S=6 start_client a1.log --workers 2 --no-ui || return
  wait_log a1.log "job [0-9]+ split in" 25 || bad "no window finished within 25 s"
  sleep 2    # the next windows are running: SIGINT forces splits into the tree report
  kill -INT "$CLIENT_PID"
  wait_rc "$CLIENT_PID" 25; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "client exited 0 after SIGINT" || bad "client exit code $rc"
  grep -q "stopped\. jobs done" "$TMP/a1.log" && ok "final summary logged: $(grep -o 'jobs done.*' "$TMP/a1.log" | head -1)" || bad "no final summary in log"
  no_bang a1.log
  [ "$(fake_workers_alive)" = 0 ] && ok "no fake workers left behind" || bad "fake workers still running"
  final_heartbeat_ok
  tr=$(stat tree_reports); rb=$(stat report_bodies); rp=$(stat report_posts)
  [ "$tr" -ge 1 ] && ok "$tr of $rb reports carried a tree (max $(stat max_nodes) nodes; one report per window)" || bad "no tree report (bodies $rb)"
  [ "$rp" = 0 ] && ok "no single /report POST: everything went through /reports batches" || bad "$rp single /report POSTs"
  tree_ok
  sn=$(stat split_nodes); sl=$(stat split_nodes_with_level); ol=$(stat open_nodes_with_level)
  [ "$sl" -ge 1 ] && [ "$sl" = "$sn" ] && ok "every split node ($sn) carried its level; $ol open probe node(s) carried one too" || bad "split nodes $sn, with level $sl"
  cv=$(aq "a['client_versions']"); want_cv=$(sed -n 's/^CLIENT_VERSION = "\(.*\)"/\1/p' "$HERE/volunteer.py")
  case "$cv" in *"'$want_cv'"*) ok "requests carried client_version $want_cv ($cv)";; *) bad "client versions seen: $cv (want $want_cv)";; esac
  wait_leases_expired 20 && ok "no lease left on the server after Stop" || bad "leases still held after 20 s"
  FAKE_MIN_S=4.5 FAKE_MAX_S=6 start_client a2.log --workers 2 --no-ui || return
  wait_rc "$CLIENT_PID" 75; rc=$RC; CLIENT_PID=""
  if audit_clean; then ok "audit clean after run 2: $(audit | cut -c1-100)"; else bad "audit not clean after run 2: $(audit | cut -c1-300)"; fi
  [ "$rc" = 0 ] && grep -q "is complete" "$TMP/a2.log" && ok "run 2 finished the campaign and exited 0 by itself" || bad "run 2: exit $rc, $(tail -2 "$TMP/a2.log")"
  stop_server; done_test 100
}

test_b() {
  say "test b: kill9 (running and paused), orphan cleanup at startup"
  # lease 10 s > window 4 s + absorption 4 s; the restart waits for the killed client's leases
  # release grace 3 s: the restarts reuse the token, and the server releases what they do not hold
  start_server 4 10 4 --absorb-total-s 4 --absorb-probe-s 2 --batch-interval-s 3 --release-grace-s 3 --paused-max-s 5 || return
  FAKE_MIN_S=4.5 FAKE_MAX_S=6 start_client b1.log --workers 2 --no-ui || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 2 ]' || bad "workers did not start"
  kill -9 "$CLIENT_PID"; wait "$CLIENT_PID" 2>/dev/null; CLIENT_PID=""
  ok "client killed with SIGKILL mid-window"
  wait_true 5 eval '[ "$(fake_workers_alive)" = 0 ]' && ok "orphaned running workers died on their own (broken pipe)" || bad "$(fake_workers_alive) orphaned fake workers still alive"
  wait_leases_expired 20 && ok "leases of the killed client expired" || bad "leases still held after 20 s"
  # kill -9 while paused: the workers are SIGSTOPped; their orphaned process group gets SIGHUP+SIGCONT
  FAKE_MIN_S=30 FAKE_MAX_S=40 start_client b2.log --workers 2 --no-browser || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 2 ]' || bad "workers did not start (paused test)"
  sec=$(ui_secret "$UPORT")
  http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" >/dev/null
  wait_true 5 eval 'ps -o stat= -p "$(pgrep -f "$WPAT" | head -1)" | grep -q T' && ok "workers paused (SIGSTOP)" || bad "workers not paused"
  kill -9 "$CLIENT_PID"; wait "$CLIENT_PID" 2>/dev/null; CLIENT_PID=""
  wait_true 5 eval '[ "$(fake_workers_alive)" = 0 ]' && ok "paused workers did not outlive the killed client" || bad "$(fake_workers_alive) stopped workers left behind"
  # an orphan from an earlier run (recorded in the pidfile) is killed when the client starts
  bin="$TMP/outbox/.bin/fake_worker.py"
  FAKE_MIN_S=300 FAKE_MAX_S=300 "$PY" "$bin" --seed-path U1 --split-after 0 >/dev/null 2>&1 &
  orphan=$!; EXTRA_PIDS="$EXTRA_PIDS $orphan"; sleep 0.5; kill -STOP "$orphan"
  printf '{"client_pid": 1, "bin": "%s", "workers": [%d]}' "$bin" "$orphan" > "$TMP/outbox/.workers.pid"
  FAKE_MIN_S=4.5 FAKE_MAX_S=6 start_client b3.log --workers 2 --no-ui || return
  gone() { case "$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')" in ""|Z*) return 0;; *) return 1;; esac; }
  wait_true 5 gone "$orphan" && grep -q "left behind by an earlier client run" "$TMP/b3.log" \
    && ok "a stopped worker from an earlier run was killed at startup" || bad "orphan $orphan still alive"
  if wait_audit_clean 60; then ok "audit clean after restart (no job lost): $(audit | cut -c1-100)"; else bad "audit not clean: $(audit | cut -c1-300)"; fi
  kill -INT "$CLIENT_PID" 2>/dev/null; wait_exit "$CLIENT_PID" 20 || bad "client did not exit"
  CLIENT_PID=""; stop_server; done_test 100
}

test_c() {
  say "test c: pause_resume and panel security"
  start_server 6 60 60 --heartbeat-s 5 || return
  FAKE_MIN_S=25 FAKE_MAX_S=30 start_client c.log --workers 2 --no-browser || return
  pids=""
  for _ in $(seq 1 60); do pids=$(state_field "' '.join(str(w['pid']) for w in s['slots'] if w['pid'])" 2>/dev/null); [ "$(echo $pids | wc -w | tr -d ' ')" = 2 ] && break; sleep 0.25; done
  [ "$(echo $pids | wc -w | tr -d ' ')" = 2 ] && ok "two workers running (pids $pids)" || { bad "expected 2 running workers, state: $(http_get http://127.0.0.1:$UPORT/state | head -c 300)"; return; }
  f=$(state_field "','.join(k for k in ('cur_seed','window_left','stack_size','nodes_done','phase') if k in s['slots'][0])")
  [ "$f" = "cur_seed,window_left,stack_size,nodes_done,phase" ] && ok "/state carries the window fields per worker (window_left $(state_field "s['slots'][0]['window_left']") s)" || bad "window fields missing: $f"
  sec=$(ui_secret "$UPORT")
  [ ${#sec} = 32 ] && ok "the served page carries a per-session secret" || bad "no secret in the page: '$sec'"
  [ "$(header_of "http://127.0.0.1:$UPORT/" X-Frame-Options)" = DENY ] && ok "X-Frame-Options: DENY" || bad "no X-Frame-Options header"
  r=$(http_get "http://127.0.0.1:$UPORT/state" "evil.example:$UPORT"); case "$r" in *403*) ok "GET /state with a foreign Host is refused (403)";; *) bad "foreign Host answered: $(echo "$r" | head -c 80)";; esac
  r=$(http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" "evil.example:$UPORT"); case "$r" in 403*) ok "POST with a foreign Host is refused (DNS rebinding)";; *) bad "foreign Host POST: $r";; esac
  r=$(http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" "" "http://evil.example"); case "$r" in 403*) ok "POST with a foreign Origin is refused";; *) bad "foreign Origin POST: $r";; esac
  r=$(http_post "http://127.0.0.1:$UPORT/pause" '{}' 1); case "$r" in 403*) ok "POST without the session secret is refused (403)";; *) bad "POST without secret answered: $r";; esac
  r=$(http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" "localhost:$UPORT"); case "$r" in 200*) ok "POST /pause with the secret and Host localhost:PORT accepted";; *) bad "/pause answered: $r";; esac
  wait_true 5 eval 'st=$(for p in $pids; do ps -o stat= -p "$p"; done | tr -d " \n"); case "$st" in *T*T*) true;; *) false;; esac' \
    && ok "after /pause both workers are stopped" || bad "after /pause STAT is '$(for p in $pids; do ps -o stat= -p "$p"; done | tr -d ' \n')'"
  [ "$(state_field "s['paused']")" = True ] && ok "/state reports paused" || bad "/state does not report paused"
  http_post "http://127.0.0.1:$UPORT/resume" '{}' "$sec" >/dev/null
  wait_true 5 eval 'st=$(for p in $pids; do ps -o stat= -p "$p"; done | tr -d " \n"); case "$st" in *T*) false;; *) true;; esac' \
    && ok "after /resume workers run again" || bad "after /resume STAT still stopped"
  http_post "http://127.0.0.1:$UPORT/workers" '{"workers": 1}' "$sec" >/dev/null; sleep 0.5
  [ "$(state_field "s['workers']")" = 1 ] && ok "/workers changed the count to 1" || bad "/workers did not change the count"
  pw=$(state_field "s['pending_flags']['workers']")
  case "$pw" in *finishing*) ok "pending flag: \"$pw\"";; *) bad "no pending workers phase: '$pw'";; esac
  r=$(http_post "http://127.0.0.1:$UPORT/exit" '{"exit": 2}' "$sec")
  case "$r" in 200*) ok "/exit 2 accepted immediately";; *) bad "/exit answered: $r";; esac
  [ "$(state_field "s['exit_pref']")" = 2 ] && ok "/state shows exit preference 2" || bad "exit_pref not 2"
  pe=$(state_field "s['pending_flags']['exit']")
  case "$pe" in *"new leases use exit 2"*) ok "pending flag: \"$pe\"";; *) bad "no pending exit phase: '$pe'";; esac
  r=$(http_post "http://127.0.0.1:$UPORT/exit" '{"exit": 99}' "$sec")
  case "$r" in 400*) ok "/exit 99 (not in the campaign) refused with 400";; *) bad "/exit 99 answered: $r";; esac
  http_post "http://127.0.0.1:$UPORT/exit" '{"exit": null}' "$sec" >/dev/null
  [ "$(state_field "s['exit_pref'] is None and s['pending_flags']['exit'] is None")" = True ] && ok "/exit null clears the preference and its pending flag" || bad "exit not cleared"
  wait_true 10 eval '[ "$(state_field "s[\"me\"] is not None")" = True ]'
  me=$(state_field "sorted(s['me'].keys())[:4]")
  case "$me" in *best*contributors*cpu_s*) ok "/state carries me from the heartbeat (${me} ...)";; *) bad "me missing: $me";; esac
  ex=$(state_field "s['exits']['0']['roots'], s['exits']['0']['roots_covered'], 'best' in s['exits']['0']")
  case "$ex" in "2 0 True") ok "/state carries exits from the heartbeat (exit 0: roots 2, covered 0)";; *) bad "exits missing or wrong: $ex";; esac
  http_post "http://127.0.0.1:$UPORT/stop" '{}' "$sec" >/dev/null
  ps_=$(state_field "s['pending_flags']['stop']" 2>/dev/null)
  case "$ps_" in *waiting*|*sending*|*exiting*) ok "pending flag: \"$ps_\"";; *) bad "no pending stop phase: '$ps_'";; esac
  wait_rc "$CLIENT_PID" 20; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "/stop: client exited 0" || bad "/stop: exit code $rc"
  echo "   audit: $(audit | cut -c1-150)"
  [ "$(stat split_nodes)" -ge 1 ] && ok "interrupted jobs were reported as split trees ($(stat tree_reports) tree reports, batch sent at Stop)" || bad "no split reported after /stop"
  final_heartbeat_ok
  stop_server; done_test
}

test_d() {
  say "test d: no_summary (worker dies): failure reports, backoff"
  start_server 4 60 60 || return
  FAKE_DIE=1 start_client d.log --workers 1 --no-ui || return
  wait_log d.log "backing off" 20 || bad "no backoff within 20 s"
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 15; rc=$RC; CLIENT_PID=""
  n=$(grep -c "run void (crash): exited with code 1 WITHOUT a SUMMARY" "$TMP/d.log")
  [ "$n" -ge 3 ] && ok "$n runs without SUMMARY were logged loudly" || bad "expected >= 3 loud logs, got $n"
  [ "$n" -le 5 ] && ok "no spinning: only $n launches (backoff after 3 failures)" || bad "client spun: $n launches"
  fr=$(aq "a['failure_reasons'].get('crash', 0)")
  [ "$fr" -ge 3 ] && ok "$fr 'crash' failure reports reached the server" || bad "crash failure reports: $fr ($(aq "a['failure_reasons']"))"
  [ "$(aq "a['counts']['done'] + a['counts']['split']")" = 0 ] && ok "no job was reported done or split" || bad "unexpected report: $(audit | cut -c1-200)"
  [ "$(aq "max(a['fail_counts'].values())")" -ge 1 ] && ok "the server counts failures per job ($(aq "a['fail_counts']"))" || bad "no fail counts"
  stop_server; done_test
}

test_e() {
  say "test e: coalesce (trivial jobs, one-node reports, batched)"
  start_server 60 30 4 || return
  FAKE_MIN_S=0.2 FAKE_MAX_S=0.5 start_client e.log --workers 2 --no-ui --exit 2 || return
  wait_true 20 eval '[ "$(stat lease_requests)" -ge 1 ]'
  sleep 11
  kill -INT "$CLIENT_PID"
  wait_rc "$CLIENT_PID" 20; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "client exited 0" || { bad "exit code $rc"; return; }
  rb=$(stat report_bodies); bp=$(stat batch_posts); one=$(stat one_node_reports); lr=$(stat lease_requests); mb=$(stat max_batch)
  [ "$rb" -ge 10 ] && ok "$rb reports in ~11 s with 2 workers" || bad "only $rb reports"
  [ "$one" -ge 10 ] && ok "$one of them were one-node reports (trivial jobs)" || bad "one-node reports: $one"
  [ "$bp" -ge 1 ] && [ $(( bp * 3 )) -le "$rb" ] && ok "sent as $bp batch(es) (largest $mb): at most one POST /reports per 10 s" || bad "$bp batches for $rb reports"
  # M108: at most 2 jobs per worker held, so trivial jobs are leased in small batches: one request per
  # batch interval, plus one sooner (>= 2 s apart) when a worker is idle with nothing queued
  [ "$lr" -le 8 ] && ok "$lr lease request(s) in ~11 s (one per 10 s, plus early ones >= 2 s apart for idle workers)" || bad "$lr lease requests"
  # (the server's own held count also includes finished jobs whose report waits for the next batch)
  [ "$(stat max_lease_n)" -le 4 ] && ok "requests of at most $(stat max_lease_n) jobs (2 workers: at most 2 per worker running + queued)" \
    || bad "lease sizes: max n $(stat max_lease_n)"
  grep -q "exit=2" "$SLOG" && ok "lease requests carried the exit preference (--exit 2)" || bad "no exit preference in lease requests"
  no_bang e.log
  stop_server; done_test
}

test_f() {
  say "test f: closed_410 (re-register into the next campaign, running and at restart)"
  start_server 6 60 60 --heartbeat-s 2 --batch-interval-s 2 || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client f1.log --workers 1 --no-ui || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]' || bad "worker did not start"
  admin '{"op": "close"}'
  admin '{"op": "open", "roots": 4, "split_after_s": 4}'
  wait_log f1.log "for campaign 2" 20 && ok "the running client re-registered into campaign 2 after the 410" || bad "no re-registration: $(tail -3 "$TMP/f1.log")"
  grep -q "closed by its owner" "$TMP/f1.log" && ok "the client said why (campaign closed)" || bad "no 410 message"
  wait_true 20 eval '[ "$(aq "a[\"counts\"][\"leased\"] + a[\"counts\"][\"done\"] + a[\"counts\"][\"split\"]" 2)" -ge 1 ]' \
    && ok "it leases and works on campaign 2's jobs" || bad "no campaign-2 activity: $(audit 2 | cut -c1-200)"
  [ "$(ls "$TMP/outbox/closed" 2>/dev/null | wc -l | tr -d ' ')" -ge 1 ] && ok "the undeliverable campaign-1 split was kept in outbox/closed/, never sent to campaign 2" \
    || bad "nothing in outbox/closed/"
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 20; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "client exited 0 after SIGINT" || bad "exit code $rc"
  # restart with the stored campaign-2 token after campaign 2 closed and 3 opened
  admin '{"op": "close"}'; admin '{"op": "open", "roots": 3, "split_after_s": 4}'
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client f2.log --workers 1 --no-ui || return
  grep -q "no longer valid" "$TMP/f2.log" && wait_log f2.log "for campaign 3" 10 \
    && ok "a restart with the closed campaign's token registered into campaign 3 without --reregister" || bad "restart: $(tail -4 "$TMP/f2.log")"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  stop_server; done_test
}

test_g() {
  say "test g: too_old_426"
  start_server 4 60 60 --min-client-version 9.0.0 || return
  "$PY" "$HERE/volunteer.py" --name Tester --server "$SERVER" --worker "$HERE/fake_worker.py" --outbox "$TMP/outbox_g" \
      --no-ui "${FAST[@]}" 2>"$TMP/g.log" &
  CLIENT_PID=$!
  wait_rc "$CLIENT_PID" 15; rc=$RC; CLIENT_PID=""
  [ "$rc" = 2 ] && ok "client exited 2" || bad "exit code $rc"
  grep -q "too old" "$TMP/g.log" && grep -q "git checkout v9.0.0" "$TMP/g.log" && ok "the server's message (naming the release) was printed" || bad "no 426 message: $(tail -3 "$TMP/g.log")"
  stop_server; done_test
}

test_h() {
  say "test h: complete (closing summary, exit 0), then no campaign"
  start_server 4 30 4 --heartbeat-s 2 --batch-interval-s 1 --absorb-total-s 2 --absorb-probe-s 1 || return
  FAKE_MIN_S=0.2 FAKE_MAX_S=0.6 start_client h1.log --workers 2 --no-browser || return
  wait_rc "$CLIENT_PID" 45; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "the client exited 0 on its own when the campaign completed" || bad "exit code $rc: $(tail -3 "$TMP/h1.log")"
  grep -q "is complete" "$TMP/h1.log" && grep -q "Your contribution as Tester" "$TMP/h1.log" && grep -q "Result per exit" "$TMP/h1.log" \
    && ok "closing summary printed: $(grep -A1 'Result per exit' "$TMP/h1.log" | tail -1 | sed 's/^ *//')" || bad "no closing summary: $(tail -8 "$TMP/h1.log")"
  [ "$(grep -A3 'Result per exit' "$TMP/h1.log" | grep -c 'moves by Tester (exact; ')" = 3 ] && ok "one row per exit, each marked exact (from the heartbeat's exits)" || bad "result rows: $(grep -A3 'Result per exit' "$TMP/h1.log")"
  [ "$(aq "a['campaign_state']")" = complete ] && ok "server campaign state: complete" || bad "server state $(aq "a['campaign_state']")"
  final_heartbeat_ok
  [ -z "$(ls "$TMP/outbox"/*.json 2>/dev/null)" ] && ok "every report was delivered (outbox empty)" || bad "outbox not empty"
  no_bang h1.log
  start_client_noreg() { "$PY" "$HERE/volunteer.py" --name Tester --server "$SERVER" --worker "$HERE/fake_worker.py" --outbox "$TMP/outbox" --no-ui "${FAST[@]}" 2>"$TMP/$1" & CLIENT_PID=$!; }
  start_client_noreg h2.log
  wait_rc "$CLIENT_PID" 15; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && grep -q "No campaign is running" "$TMP/h2.log" && ok "a restart with no campaign running exits 0 with a message" || bad "restart: exit $rc, $(tail -2 "$TMP/h2.log")"
  stop_server; done_test
}

test_i() {
  say "test i: keep_going (join the next campaign)"
  start_server 3 30 4 --heartbeat-s 2 --batch-interval-s 1 --absorb-total-s 2 --absorb-probe-s 1 || return
  FAKE_MIN_S=0.2 FAKE_MAX_S=0.6 start_client i.log --workers 2 --no-browser --keep-going || return
  wait_log i.log "waiting for the next campaign" 45 && ok "campaign 1 complete: the client waits for the next one" || bad "no wait: $(tail -3 "$TMP/i.log")"
  kill -0 "$CLIENT_PID" 2>/dev/null && ok "still running (--keep-going)" || bad "client exited"
  [ "$(state_field "(s.get('completion') or {}).get('campaign_id')")" = 1 ] && ok "the panel shows campaign 1's closing summary while waiting" \
    || bad "panel completion while waiting: $(state_field "s.get('completion')")"
  admin '{"op": "open", "roots": 40, "split_after_s": 4}'
  wait_log i.log "for campaign 2" 15 && ok "it registered into campaign 2" || bad "did not join campaign 2"
  wait_log i.log "campaign 2: grid" 5
  wait_true 1 eval '[ "$(state_field "(s.get(\"completion\") or {}).get(\"campaign_id\", 0) != 1")" = True ]' \
    && ok "campaign 1's summary is gone from the panel once campaign 2 runs" \
    || bad "the panel still shows campaign 1's summary during campaign 2: $(state_field "s.get('completion')")"
  wait_true 45 eval '[ "$(grep -c "is complete" "$TMP/i.log")" -ge 2 ]' && ok "and completed it too" || bad "campaign 2 not completed: $(audit 2 | cut -c1-200)"
  wait_log i.log "waiting for the next campaign.*" 10
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 10; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "Ctrl-C while waiting: exit 0" || bad "exit code $rc"
  stop_server; done_test 100
}

test_j() {
  say "test j: instances (outbox lock, one token per outbox)"
  start_server 12 60 60 --heartbeat-s 2 --release-grace-s 3 --batch-interval-s 2 --no-steal || return
  FAKE_MIN_S=25 FAKE_MAX_S=30 start_client j1.log --workers 2 --no-ui || return
  "$PY" "$HERE/volunteer.py" --name Tester --server "$SERVER" --worker "$HERE/fake_worker.py" --outbox "$TMP/outbox" \
      --no-ui "${FAST[@]}" 2>"$TMP/j2.log" &
  p2=$!
  wait_rc "$p2" 10; rc=$RC
  [ "$rc" = 2 ] && grep -q -- "--outbox" "$TMP/j2.log" && ok "a second client on the same outbox refused to start (exit 2, says --outbox)" || bad "second instance: exit $rc $(tail -2 "$TMP/j2.log")"
  kill -0 "$CLIENT_PID" 2>/dev/null && ok "the first client is unaffected" || bad "first client died"
  FAKE_MIN_S=25 FAKE_MAX_S=30 "$PY" "$HERE/volunteer.py" --name Tester --server "$SERVER" --worker "$HERE/fake_worker.py" \
      --outbox "$TMP/outbox_j" --port "$UPORT2" --no-ui --workers 2 "${FAST[@]}" 2>"$TMP/j3.log" &
  CLIENT2_PID=$!
  wait_log j3.log "registered as" 15 && ok "a client with its own outbox registered its own token" || bad "second outbox client did not register"
  sleep 9   # several heartbeats of both clients, release grace 3 s
  [ "$(stat released)" = 0 ] && ok "no job was released by the other instance's heartbeats" || bad "$(stat released) job(s) released"
  [ "$(stat drops)" = 0 ] && ok "no drop" || bad "$(stat drops) drop(s)"
  kill -INT "$CLIENT_PID" "$CLIENT2_PID"; wait_exit "$CLIENT_PID" 20; wait_exit "$CLIENT2_PID" 20; CLIENT_PID=""; CLIENT2_PID=""
  stop_server; done_test
}

test_k() {
  say "test k: extra_refuse (worker flags allowlist)"
  for extra in "--allow-exit-transit --harvest $TMP/pwned.bin" "--allow-exit-transit --allow-block-on-exit" "--num-holes 3"; do
    start_server 2 60 60 --extra "$extra" || return
    "$PY" "$HERE/volunteer.py" --name Tester --server "$SERVER" --worker "$HERE/fake_worker.py" --outbox "$TMP/outbox_k" \
        --no-ui "${FAST[@]}" 2>"$TMP/k.log" &
    CLIENT_PID=$!
    wait_rc "$CLIENT_PID" 15; rc=$RC; CLIENT_PID=""
    [ "$rc" = 2 ] && grep -q "Refusing to run" "$TMP/k.log" && ok "extra '$extra': refused (exit 2): $(grep -o 'Refusing to run: [^.]*' "$TMP/k.log" | head -1)" || bad "extra '$extra': exit $rc $(tail -2 "$TMP/k.log")"
    [ "$(stat lease_requests)" = 0 ] && ok "  no job was leased" || bad "  jobs were leased"
    stop_server
  done
  [ ! -e "$TMP/pwned.bin" ] && ok "no file was created by a server-supplied path flag" || bad "pwned.bin exists"
  done_test
}

test_l() {
  say "test l: foldback (never truncate; 6000 REMAINING; nested trees)"
  start_server 1 30 2 --absorb-total-s 2 --absorb-probe-s 1 --batch-interval-s 1 --exits 0 || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 FAKE_REMAINING_N=6000 start_client l.log --workers 1 --no-ui || return
  wait_log l.log "handed back untouched" 20 && ok "a 6000-child split is handed back untouched: $(grep -o 'its own REMAINING[^;]*' "$TMP/l.log" | head -1)" || bad "no hand-back: $(tail -3 "$TMP/l.log")"
  wait_true 10 eval '[ "$(aq "a[\"failure_reasons\"].get(\"too_big\", 0)")" -ge 1 ]' && ok "and reported to the server as failure 'too_big' (counted, quarantined if it repeats)" || bad "no too_big failure: $(aq "a['failure_reasons']")"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  [ "$(stat max_nodes)" -lt 5000 ] && [ "$(stat split_nodes)" = 0 ] && ok "the server never saw a (truncated) split of it" || bad "server saw max_nodes $(stat max_nodes), splits $(stat split_nodes)"
  stop_server
  "$PY" - "$HERE" <<'EOF' && ok "fit_tree: nested splits fold back deepest-first, every split keeps all its children" || bad "fit_tree unit test failed"
import sys; sys.path.insert(0, sys.argv[1]); import volunteer as v
J = "U1,U1"
tree = [{"seed": J, "parent": None, "status": "split", "summary": {"status": "split"}, "level": {"depth": 9, "code": "3"}}]
for c in ("R1", "R2", "R3"):
    tree.append({"seed": J + "," + c, "parent": J, "status": "split" if c != "R3" else "open"})
A, B = J + ",R1", J + ",R2"
tree[1].update(summary={"status": "split"}); tree[2].update(summary={"status": "split"}, level={"depth": 12, "code": "30"},
                                                            unresolved=[{"depth": 20, "code": "3", "path": B, "cause": "pq"}])
T12 = ["U1", "U2", "U3", "R1", "R2", "R3", "D1", "D2", "D3", "L1", "L2", "L3"]
def ext(k): return [T12[(k // 12 ** i) % 12] for i in range(4)]
def kids(p, n): return [{"seed": p + "," + ",".join(ext(k)), "parent": p, "status": "open"} for k in range(n)]
tree += kids(A, 2600) + kids(B, 2600)
assert len(tree) == 5204
out = v.fit_tree(tree, 5000, 10 ** 9)
by = {n["seed"]: n for n in out}
assert len(out) == 2604, len(out)
assert by[B]["status"] == "open" and "summary" not in by[B] and "unresolved" not in by[B] and by[B]["level"]["depth"] == 12
assert by[A]["status"] == "split" and sum(1 for n in out if n["parent"] == A) == 2600
assert not any(n["parent"] == B for n in out) and sum(1 for n in out if n["parent"] == J) == 3
assert tree[2]["status"] == "split"                       # the input is not modified
out2 = v.fit_tree(tree, 10 ** 6, 5000)                   # byte cap: both A and B demoted
assert [n["status"] for n in out2] == ["split", "open", "open", "open"], [n["status"] for n in out2]
assert v.fit_tree([tree[0]] + kids(J, 6000), 5000, 10 ** 9) is None
wire = v.clean_node(by[A]); assert set(wire) == {"seed", "parent", "status", "summary"}
# candidate cap (server CAND_MAX_PER_REPORT): no split left, so the done node with most candidates is demoted
c = lambda p, n: [{"depth": 30, "code": "3", "path": p, "cause": "big"}] * n
t3 = [{"seed": J, "parent": None, "status": "split", "summary": {"status": "split", "unresolved": 0}},
      {"seed": A, "parent": J, "status": "done", "summary": {"status": "exhausted", "unresolved": 700}, "unresolved": c(A, 700)},
      {"seed": B, "parent": J, "status": "done", "summary": {"status": "exhausted", "unresolved": 400}, "unresolved": c(B, 400)}]
o3 = {n["seed"]: n for n in v.fit_tree(t3, 5000, 10 ** 9, max_cands=1000)}
assert o3[A]["status"] == "open" and "unresolved" not in o3[A] and o3[B]["status"] == "done" and len(o3[B]["unresolved"]) == 400
assert v.fit_tree(t3, 5000, 10 ** 9) == t3 or len(v.fit_tree(t3, 5000, 10 ** 9)) == 3
EOF
  done_test
}

test_m() {
  say "test m: failures (bad_seed, bad REMAINING, REMAINING == seed)"
  start_server 3 60 3 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_STATUS=bad_seed start_client m1.log --workers 1 --no-ui || return
  wait_true 20 eval '[ "$(aq "a[\"failure_reasons\"].get(\"bad_seed\", 0)")" -ge 2 ]' && ok "bad_seed runs became failure reports: $(aq "a['failure_reasons']")" || bad "no bad_seed failures: $(aq "a['failure_reasons']")"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  [ "$(stat split_nodes)" = 0 ] && [ "$(aq "a['counts']['done']")" = 0 ] && ok "nothing from those runs was used" || bad "a failed run was reported as coverage"
  stop_server
  start_server 3 60 3 --batch-interval-s 1 --exits 0 --heartbeat-s 2 --absorb-total-s 1 --absorb-probe-s 1 || return
  FAKE_BAD_REMAINING=1 FAKE_MIN_S=20 FAKE_MAX_S=30 start_client m2.log --workers 1 --no-ui || return
  wait_true 20 eval '[ "$(aq "a[\"failure_reasons\"].get(\"bad_remaining\", 0)")" -ge 1 ]' && ok "a malformed REMAINING line voided the run: failure 'bad_remaining'" || bad "no bad_remaining failure: $(aq "a['failure_reasons']")"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  [ "$(stat split_nodes)" = 0 ] && ok "no partial split was ever reported" || bad "$(stat split_nodes) split(s) reported"
  stop_server
  start_server 3 60 30 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_REMAINING_SELF=1 FAKE_MIN_S=40 FAKE_MAX_S=50 start_client m3.log --workers 1 --no-ui || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]'; sleep 1
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  grep -q "interrupted before expanding its root" "$TMP/m3.log" && [ "$(stat failures)" = 0 ] && ok "REMAINING == seed after Stop: job untouched, no failure" || bad "stop case: failures $(stat failures)"
  stop_server
  start_server 2 60 2 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_REMAINING_SELF=1 FAKE_MIN_S=40 FAKE_MAX_S=50 start_client m4.log --workers 1 --no-ui || return
  wait_true 20 eval '[ "$(aq "a[\"failure_reasons\"].get(\"no_progress\", 0)")" -ge 1 ]' && ok "REMAINING == seed on the worker's own timer: failure 'no_progress'" || bad "no no_progress failure: $(aq "a['failure_reasons']")"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  stop_server; done_test 100
}

test_n() {
  say "test n: watchdog (hang) and slow SIGINT"
  start_server 1 60 2 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_HANG_AFTER_S=1 FAKE_SIGINT_IGNORE=1 FAKE_MIN_S=40 FAKE_MAX_S=50 start_client n1.log --workers 1 --no-ui \
      --watchdog-slack-s 2 --stop-grace-s 2 || return
  wait_log n1.log "sending SIGINT" 20 && ok "watchdog fired: $(grep -o 'no output for [0-9]* s after its split time' "$TMP/n1.log" | head -1)" || bad "no watchdog"
  wait_true 15 eval '[ "$(aq "a[\"failure_reasons\"].get(\"hang\", 0)")" -ge 1 ]' && ok "the hung run was killed and reported as failure 'hang'" || bad "no hang failure: $(aq "a['failure_reasons']")"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  [ "$(fake_workers_alive)" = 0 ] && ok "no hung worker left behind" || bad "hung worker still alive"
  stop_server
  start_server 2 60 60 --batch-interval-s 1 --exits 0 || return
  FAKE_SIGINT_DELAY_S=3 FAKE_MIN_S=40 FAKE_MAX_S=50 start_client n2.log --workers 1 --no-ui || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]'; sleep 1
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 25; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && [ "$(stat split_nodes)" -ge 1 ] && ok "a worker that answers SIGINT after 3 s is reported as a normal split" || bad "slow SIGINT: exit $rc, splits $(stat split_nodes)"
  stop_server; done_test
}

test_o() {
  say "test o: batch413 (halving)"
  start_server 16 30 4 --batch-interval-s 3 --body-max 2500 --exits 0 --heartbeat-s 2 || return
  FAKE_MIN_S=0.2 FAKE_MAX_S=0.4 start_client o.log --workers 2 --no-ui || return
  wait_rc "$CLIENT_PID" 60; rc=$RC; CLIENT_PID=""
  [ "$(stat http_413)" -ge 1 ] && ok "$(stat http_413) batch(es) refused with 413; the client halved them" || bad "no 413 happened (max batch $(stat max_batch_bytes) bytes)"
  [ "$rc" = 0 ] && audit_clean && ok "every report was delivered in smaller batches (campaign complete, exit 0)" || bad "exit $rc; audit $(audit | cut -c1-200)"
  [ -z "$(ls "$TMP/outbox/rejected" 2>/dev/null)" ] && ok "nothing went to rejected/" || bad "rejected/: $(ls "$TMP/outbox/rejected")"
  stop_server; done_test
}

test_p() {
  say "test p: unresolved candidates forwarded per node"
  start_server 2 30 4 --batch-interval-s 1 --exits 0 --heartbeat-s 2 --absorb-total-s 1 --absorb-probe-s 1 || return
  FAKE_UNRESOLVED=3 FAKE_MIN_S=0.2 FAKE_MAX_S=0.4 start_client p.log --workers 1 --no-ui || return
  wait_rc "$CLIENT_PID" 40; rc=$RC; CLIENT_PID=""
  nc=$(stat candidates)
  [ "$nc" -ge 6 ] && ok "$nc candidates stored by the server: $(aq "a['candidates'][0]")" || bad "candidates: $nc"
  [ "$rc" = 0 ] && ok "campaign complete, exit 0" || bad "exit $rc"
  stop_server; done_test
}

test_q() {
  say "test q: dedupe of lease grants"
  start_server 6 60 60 --dup-grants --batch-interval-s 1 --heartbeat-s 2 || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client q.log --workers 2 --no-ui --exit 0 || return
  wait_true 20 eval '[ "$(stat dup_grants_sent)" -ge 1 ]'
  wait_log q.log "skipping duplicate grant" 20 && ok "the client skipped the re-granted job: $(grep -o 'skipping duplicate grant.*' "$TMP/q.log" | head -1)" || bad "no dedupe log"
  jobs_run=$(grep -Eo 'worker [0-9]+: job [0-9]+ exit' "$TMP/q.log" | awk '{print $4}' | sort | uniq -d)
  [ -z "$jobs_run" ] && ok "no job ran twice" || bad "jobs started twice: $jobs_run"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  stop_server; done_test
}

test_r() {
  say "test r: empty root seed"
  start_server 0 30 4 --empty-root --exits 0 --batch-interval-s 1 --heartbeat-s 2 || return
  FAKE_MIN_S=0.2 FAKE_MAX_S=0.4 start_client r.log --workers 1 --no-ui || return
  wait_rc "$CLIENT_PID" 30; rc=$RC; CLIENT_PID=""
  grep -q "seed (exit root)" "$TMP/r.log" && ok "the exit-root job ran (no --seed-path)" || bad "no exit-root run: $(tail -3 "$TMP/r.log")"
  [ "$rc" = 0 ] && audit_clean && ok "its report was accepted and the campaign completed" || bad "exit $rc, audit $(audit | cut -c1-200)"
  stop_server; done_test
}

test_s() {
  say "test s: units (allowlist, env, classification)"
  "$PY" - "$HERE" "$TMP" <<'EOF' 2>"$TMP/s.err" && ok "parse_extra / worker_env / classify / valid_seed / parse_unresolved / run judging / wake baselines / 413 requeue / absorb order / queue order / window at start / lease sizing / panel flags behave as specified" \
    || { bad "unit checks failed"; tail -5 "$TMP/s.err"; }
import sys; sys.path.insert(0, sys.argv[1]); import volunteer as v
ok = lambda x: v.parse_extra(x)[1] is None
assert v.parse_extra(["--allow-exit-transit", "--num-holes", "0"]) == (["--allow-exit-transit", "--num-holes", "0"], None)
assert ok("--allow-exit-transit --num-blocks 9 --min-walls 2 --num-holes 3")
for badx in (["--num-holes", "3"], ["--allow-exit-transit", "--harvest", "x"], ["--allow-exit-transit", "--allow-block-on-exit"],
             ["--allow-exit-transit", "--num-holes", "-1"], ["--allow-exit-transit", "--num-holes"], ["--allow-exit-transit", "--num-holes", "3; rm"],
             ["--allow-exit-transit", "--allow-exit-transit"], ["--allow-exit-transit", "--grid", "9x9"], ["--allow-exit-transit", "--seed-path", "U1"]):
    assert not ok(badx), badx
env = v.worker_env({"PATH": "/bin", "BS_DUMP_SEED": "1", "BS_TRACE_VISITS": "x"}, False)
assert env == {"PATH": "/bin"}, env
assert "BS_DUMP_SEED" in v.worker_env({"BS_DUMP_SEED": "1"}, True)
E = v.ApiError
assert v.classify(E(410, {"error": "campaign_closed"}, False)) == "closed"
assert v.classify(E(426, {"error": "client_too_old"}, False)) == "too_old"
assert v.classify(E(403, {"error": "unknown_token"}, False)) == "reregister"
assert v.classify(E(403, {"error": "revoked", "revoked": True}, False)) == "revoked"
assert v.classify(E(404, {"error": "no_campaign"}, False)) == "no_campaign"
assert v.classify(E(413, {}, False)) == "too_large"
assert v.classify(E(503, {}, True)) == "transient"
assert v.valid_seed("", 1024) and v.valid_seed("U1,R3", 1024) and not v.valid_seed("U1,X3", 1024) and not v.valid_seed("U1,U1", 1)
assert v.extends("U1,R2,D3", ["U1"], 1024) == ["U1", "R2", "D3"] and v.extends("U2,R2", ["U1"], 1024) is None and v.extends("U1,R2", ["U1"], 1) is None
assert v.campaign_state_of({"campaign_state": "complete"}) == "complete" and v.campaign_state_of({"campaign": {"status": "open"}}) == "running"
# UNRESOLVED lines: depth == path tokens, the path extends the seed, cause pq|probe|big (else the run is void)
U = lambda d, p, c="pq": ["UNRESOLVED", str(d), "3000/0000", p, c]
assert v.parse_unresolved(U(3, "U1,R1,R1"), "U1", 1024) == {"depth": 3, "code": "3000/0000", "path": "U1,R1,R1", "cause": "pq"}
assert v.parse_unresolved(U(1, "U1"), "U1", 1024) is not None                      # the seed itself
for bad in (U(4, "U1,R1,R1"), U(3, "U2,R1,R1"), U(3, "U1,R1,R1", "heap"), U(0, ""), ["UNRESOLVED", "x", "3", "U1", "pq"], U(3, "U1,R1")):
    assert v.parse_unresolved(bad, "U1", 1024) is None, bad
# judging a finished run: a protocol-3 SUMMARY needs flags and unresolved, and unresolved == the lines received
import argparse
vol = v.Volunteer(argparse.Namespace(server="http://127.0.0.1:9", workers=1, outbox=sys.argv[2] + "/s_outbox", exit=None,
                                     watchdog_slack_s=60.0))
vol.worker_info = {"PROTOCOL": 3}; vol.src_hash = "fa4e" + "0" * 60
vol.campaign = {"grid": "5x5"}; vol.extra = ["--allow-exit-transit"]
slot, job = v.Slot(0), {"id": 1, "exit": 0, "seed": "U1"}
flags = {"grid": "5x5", "exit": 0, "transit": 1, "block_on_exit": 0, "max_holes": 64, "max_blocks": 32, "min_walls": 0, "bulk_walk": 1}
cand = v.parse_unresolved(U(3, "U1,R1,R1"), "U1", 1024)
def judge(summary, unres, remaining=(), rc=0):
    return vol._judge(slot, job, "U1", False, rc, False, False, False, "", vol.src_hash, False, summary, False, [], False, 0,
                      list(remaining), unres, None)
base = {"status": "exhausted", "states": 5, "flags": flags, "unresolved": 1, "protocol": 3}
assert judge(base, [cand]).kind == "ok"
r = judge(dict(base, unresolved=2), [cand]); assert (r.kind, r.reason) == ("void", "bad_unresolved"), r.reason
r = judge({k: x for k, x in base.items() if k != "flags"}, [cand]); assert (r.kind, r.reason) == ("void", "bad_summary"), r.reason
r = judge(dict(base, states=0), [cand]); assert (r.kind, r.reason) == ("void", "bad_summary"), r.reason
r = judge(dict(base, status="split"), [cand], ["U1,R1"]); assert r.kind == "ok" and r.remaining == ["U1,R1"]
r = judge(dict(base, status="split"), [cand]); assert (r.kind, r.reason) == ("void", "bad_remaining"), r.reason
r = judge(dict(base, status="path_overflow"), [], rc=4); assert (r.kind, r.reason, r.failure) == ("void", "path_overflow", True)
r = judge(None, [], rc=-11); assert (r.kind, r.reason) == ("void", "crash")
# Stop's grace ran out (SIGKILL): only this run is lost, no failure; a drop/lease-loss kill is 'killed'
r = vol._judge(slot, job, "U1", False, -9, "grace", False, True, "", None, False, None, False, [], False, 0, [], [], None)
assert (r.kind, r.reason) == ("none", "stop_grace"), (r.kind, r.reason)
r = vol._judge(slot, job, "U1", False, -9, "client", False, False, "", None, False, None, False, [], False, 0, [], [], None)
assert (r.kind, r.reason) == ("none", "killed")
slot.watchdog_why = "watchdog: no output for 75 s after its split time"
r = vol._judge(slot, job, "U1", False, 0, None, True, False, "", vol.src_hash, False, dict(base, status="split"), False, [], False, 0, ["U1,R1"], [cand], None)
assert (r.kind, r.reason) == ("void", "hang") and "no output for 75 s" in r.detail, r.detail
# a sleep across the split time (the main loop did not run): the watchdog and the stop grace count from the wake
import time as _t
class P:
    pid = None
    def poll(self): return 0
def fresh(now):
    s = v.Slot(0); s.proc = P(); s.state = "running"; s.job = job
    s.split_after_s = 6.0; s.split_at = now - 20.0; s.last_line_at = now - 25.0; s.run_started_at = now - 26.0
    return s
now = _t.time()
vol.slots = [fresh(now)]; vol._supervise(now, 2.75, 90.0)
assert vol.slots[0].watchdog_fired and "no output for 20 s" in vol.slots[0].watchdog_why, vol.slots[0].watchdog_why
vol.slots = [fresh(now)]; vol.shift_baselines(29.5, now); vol._supervise(now + 1.0, 2.75, 90.0)
s0 = vol.slots[0]
assert not s0.watchdog_fired and s0.last_line_at == now and abs(s0.split_at - (now + 9.5)) < 1e-6, (s0.watchdog_fired, s0.split_at - now)
s0.state = "finishing"; s0.stop_sent_at = now - 95.0; vol.shift_baselines(60.0, now); vol._supervise(now, 2.75, 90.0)
assert not s0.grace_killed and s0.stop_sent_at == now - 35.0
vol._supervise(now + 56.0, 2.75, 90.0); assert s0.grace_killed and not s0.killed_by_client
vol.slots = []
# 413 halving with no outbox file (the disk refused it): the delivered half is not queued again
vol.token = "t"; vol.campaign = {"id": 1}
vol.outbox_write = lambda *a, **k: None
sent = []
def post(reports):
    if len(reports) > 2: return "too_large", E(413, {}, False)
    sent.append([r["job_id"] for r in reports])
    return ("ok", None) if len(sent) == 1 else ("retry", E(503, {}, True))
vol.post_batch = post
vol.pending = [{"job_id": i, "nodes": []} for i in range(4)]
assert vol.send_pending() is False
assert sent == [[0, 1], [2, 3]] and [r["job_id"] for r in vol.pending] == [2, 3], (sent, vol.pending)
assert v.nonneg_float(0) == 0.0 and v.nonneg_float("12.5") == 12.5 and v.nonneg_float(-1) is None and v.nonneg_float(True) is None
# M18: absorption probes deepest first, stable (the cursor stays first among equals)
assert v.absorb_order(["U1,R1", "U1,R1,R1,R1", "U1,L2", "U1,R2,R2"]) == ["U1,R1,R1,R1", "U1,R2,R2", "U1,R1", "U1,L2"]
# M108: the local queue runs largest est_s first (then shallowest, then grant order); grants without
# est_s run in grant order (the server's own lease order), never re-sorted by depth
q = [{"depth": 9, "seq": 1}, {"depth": 5, "seq": 2}, {"depth": 5, "seq": 3}, {"depth": 12, "seq": 4, "est_s": 50.0},
     {"depth": 3, "seq": 5, "est_s": 50.0}, {"depth": 20, "seq": 6, "est_s": 80.0}]
assert [x["seq"] for x in sorted(q, key=v.queue_key)] == [6, 5, 4, 1, 2, 3], [x["seq"] for x in sorted(q, key=v.queue_key)]
# M18: the pass's last kept probe split goes back open when it kept less than 1 s of work per extra
# open child (a fold-back: children dropped, the deepest level kept); else it stays a split
def tree():
    ns = [{"seed": "U1", "parent": None, "status": "split"},
          {"seed": "U1,R1", "parent": "U1", "status": "split", "summary": {"status": "split"}, "level": {"depth": 5, "code": "a"},
           "unresolved": [{"depth": 9}]}]
    ks = [{"seed": "U1,R1," + t, "parent": "U1,R1", "status": "open"} for t in ("R1", "R2", "R3", "L1", "L2")]
    ks[0].update(status="done", summary={"status": "exhausted"}, level={"depth": 9, "code": "b"})
    ns += ks + [{"seed": "U1,L1", "parent": "U1", "status": "open"}]
    return ns, {n["seed"]: n for n in ns}
sl = v.Slot(0); sl.nodes_done = 3
ns, ix = tree()
r = vol._settle_last_split(sl, ns, ix, {"seed": "U1,R1", "saved": 2.9}, 4)      # 2.9 s < 1 s x (4 - 1)
assert r == {"done": 1, "unprobed": 4, "children": 5, "saved": 2.9}, r
assert [n["seed"] for n in ns] == ["U1", "U1,R1", "U1,L1"] and set(ix) == {"U1", "U1,R1", "U1,L1"}, ns
assert ns[1]["status"] == "open" and "summary" not in ns[1] and "unresolved" not in ns[1] and ns[1]["level"]["depth"] == 9, ns[1]
assert sl.nodes_done == 2 and vol.stats["probe_splits_demoted"] == 1
ns, ix = tree()
assert vol._settle_last_split(sl, ns, ix, {"seed": "U1,R1", "saved": 3.0}, 4) is None and len(ns) == 8 and ns[1]["status"] == "split"
# M108: the window is chosen when it starts (the latest lease/heartbeat window), a re-run keeps its own
vol.slots = []; vol.campaign = {"grid": "5x5", "split_after_s": 1800, "absorb_total_s": 120}
job2 = {"id": 2, "exit": 0, "seed": "U1", "depth": 1, "split_after_s": 1800.0, "absorb_total_s": 120.0, "window_mode": "full"}
assert vol.window_for(job2) == (1800.0, 120.0, "full")
vol.note_window(120, 60, "ramp", "lease")
assert vol.window_for(job2) == (120.0, 60.0, "ramp"), vol.window_for(job2)
assert vol.window_for(dict(job2, split_after_fixed=3600.0))[0] == 3600.0
assert vol.window_for(dict(job2, absorb_fixed=0.0))[1] == 0.0
# N18: windows are clamped to min_window_s (1 s unless the campaign lowers it); --split-after keeps 3 decimals
vol.note_window(0.3, 0, "endgame", "heartbeat")
assert vol.window_for(job2)[0] == 1.0
vol.campaign["min_window_s"] = 0.05
assert vol.window_for(job2)[0] == 0.3
vol.worker_base = ["w"]; vol.campaign["split_after_nodes"] = 7
a = vol.worker_argv(job2, "U1", 0.3)
assert a[a.index("--split-after") + 1] == "0.300" and a[a.index("--split-after-nodes") + 1] == "7", a
del vol.campaign["split_after_nodes"]; assert "--split-after-nodes" not in vol.worker_argv(job2, "U1", 0.3)
a = vol.worker_argv(job2, "", 0.0004); assert a[a.index("--split-after") + 1] == "0.001"
# M108: lease sizing holds at most 2 jobs per worker (running + queued), an idle worker always gets one
class Th:
    def is_alive(self): return True
def slots(n, busy):
    out = []
    for i in range(n):
        s = v.Slot(i); s.thread = Th()
        if i < busy:
            s.job = {"id": 100 + i, "exit": 0, "seed": "U1", "depth": 1}; s.started_at = _t.time() - 1; s.window_end = _t.time() + 1799
        out.append(s)
    return out
vol.campaign = {"grid": "5x5", "split_after_s": 1800, "absorb_total_s": 120, "lease_ahead_s": 300}; vol.window_now = None
vol.depth_hist = {}; vol.grant_depths.clear(); vol.queue.clear()
vol.slots = slots(4, 0); assert vol.lease_want() == (4, True), vol.lease_want()
vol.slots = slots(4, 4); assert vol.lease_want()[0] == 0          # long windows far from their end: nothing ahead
vol.depth_hist = {1: v.collections.deque([0.2, 0.3, 0.25])}; vol.grant_depths.extend([1] * 5)
vol.slots = slots(4, 4); assert vol.lease_want()[0] == 4          # trivial jobs: capped at 2 x 4 - 4 running
vol.queue.extend({"id": 200 + i, "depth": 1, "seed": "U1"} for i in range(3))
assert vol.lease_want()[0] == 1 and vol.lease_want()[1] is False
vol.queue.clear(); vol.depth_hist = {}; vol.grant_depths.clear()
# the hold cap counts only the workers that stay: fewer workers while the retiring ones still run their
# jobs must not leave the kept ones idle (3 -> 1, 8 -> 2, 8 -> 3 with the kept workers idle)
for n, keep in ((3, 1), (8, 2), (8, 3)):
    vol.slots = slots(n, n)
    for s_ in vol.slots[:keep]:
        s_.job = None
    for s_ in vol.slots[keep:]:
        s_.retire = True
    assert vol.lease_want() == (keep, True), (n, keep, vol.lease_want())
vol.slots = slots(3, 3); vol.slots[1].retire = vol.slots[2].retire = True
assert vol.lease_want()[0] == 0                                    # the kept worker is busy, far from its end
# M57: redundant clicks leave no stale request timestamp
vol.slots = []; vol.paused = False; vol.stopping = False
vol.pause_requested_at = 5.0; vol.pause(); vol.pause_requested_at = 6.0; vol.pause()
assert vol.pause_requested_at == 0.0 and vol.paused
vol.resume_requested_at = 7.0; vol.resume(); vol.resume_requested_at = 8.0; vol.resume()
assert vol.resume_requested_at == 0.0 and not vol.paused and vol.pending_flags(_t.time())["pause"] is None
# M58: the first lease response after an exit change settles the pending flag
vol.campaign = {"grid": "5x5", "exits": [0, 1], "split_after_s": 1800, "batch_interval_s": 0}
vol.set_exit(1); assert vol.pending_flags(_t.time())["exit"]
vol.lease_want = lambda: (1, True); vol.last_lease_at = 0.0; vol.no_jobs_until = 0.0
vol.api.post = lambda path, body, timeout=20: {"jobs": [{"id": 9, "exit": 0, "seed": "U1"}], "split_after_s": 60, "exit_fallback": True}
vol._lease_once()
assert vol.pending_flags(_t.time())["exit"] is None and vol.exit_fallback and not vol.exit_applied
vol.set_exit(1); vol.last_lease_at = 0.0
vol.api.post = lambda path, body, timeout=20: {"jobs": [{"id": 10, "exit": 1, "seed": "U2"}], "split_after_s": 60}
vol._lease_once(); assert vol.exit_applied and not vol.exit_fallback and not vol.exit_pending
assert vol.window_now["split_after_s"] == 60.0 and [j["id"] for j in vol.queue] == [9, 10]
vol.queue.clear()
# a failed lease request waits a whole batch interval, also for an idle worker (no retry every 2 s)
vol.campaign["batch_interval_s"] = 10
def down(path, body, timeout=20):
    raise E(503, {"error": "busy"}, True)
vol.api.post = down; vol.last_lease_at = 0.0; vol.no_jobs_until = 0.0
vol._lease_once()
assert vol.no_jobs_until >= _t.time() + 9.5, vol.no_jobs_until - _t.time()
vol.last_lease_at = 0.0; calls = []
vol.api.post = lambda path, body, timeout=20: calls.append(1) or {"jobs": []}
vol._lease_once(); assert calls == [], "leased again before the batch interval"
vol.no_jobs_until = 0.0
# M66: a LEVEL line feeds the last-hour and last-second boards (short runs print no STATUS)
vol._note_level({"id": 3, "exit": 1}, {"depth": 40, "code": "3000/0000"})
st = vol.state()
assert st["best_hour"]["depth"] == 40 and st["best_second"]["depth"] == 40 and st["best_session"]["depth"] == 40, st["best_hour"]
# M60: the status age counts from the last successful fetch
vol.campaign_status = {"totals": {}}; vol.campaign_status_ok_at = _t.time() - 120
st = vol.state(); assert st["campaign_status_age"] >= 119 and st["campaign_status_stale"]
# closing summary: unresolved candidates per exit when the server sends a count
assert v.open_candidates({"candidates_open_deeper": 2}) == 2 and v.open_candidates({"candidates": {"open_deeper": 1}}) == 1
vol.campaign_result = None; vol.exits = {"0": {"best": {"moves": 50, "by": "X"}, "exact": False, "clean": True, "candidates_open_deeper": 3}}
assert "3 unresolved candidate(s)" in vol.result_rows()[0], vol.result_rows()
EOF
  done_test
}

test_t() {
  say "test t: exit_check (--exit outside the campaign)"
  start_server 4 60 60 --exits 0,1 --batch-interval-s 2 || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client t.log --workers 1 --no-ui --exit 7 || return
  wait_log t.log "exit 7 is not part of this campaign" 5 && ok "warned: $(grep -o 'exit 7 is not part[^;]*' "$TMP/t.log" | head -1)" || bad "no warning"
  wait_true 15 eval '[ "$(stat lease_grants)" -ge 1 ]' && ok "and leased jobs of any exit" || bad "no lease: $(grep lease "$SLOG" | tail -2)"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  stop_server; done_test
}

test_u() {
  say "test u: outbox (reports written while the server is down are delivered later)"
  start_server 2 60 3 --exits 0 --batch-interval-s 1 --heartbeat-s 2 --absorb-total-s 1 --absorb-probe-s 1 --no-steal || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client u1.log --workers 1 --no-ui || return
  tok=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print([v["token"] for k,v in d.items() if isinstance(v,dict) and v.get("token")][0])' "$HOME/.pathology_volunteer.json")
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]'
  stop_server                                    # the window ends while the server is down
  wait_true 20 eval 'ls "$TMP/outbox"/*.json >/dev/null 2>&1' && ok "the finished window's report is in the outbox" || bad "no outbox file"
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 30; rc=$RC; CLIENT_PID=""
  n=$(ls "$TMP/outbox"/*.json 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 1 ] && ok "after Stop with the server down, $n report file(s) stay on disk" || bad "outbox empty after stop"
  start_server 2 60 3 --exits 0 --batch-interval-s 1 --heartbeat-s 2 --absorb-total-s 1 --absorb-probe-s 1 --no-steal --preregister "Tester:$tok" || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client u2.log --workers 1 --no-ui || return
  wait_true 20 eval '[ -z "$(ls "$TMP/outbox"/*.json 2>/dev/null)" ]' && ok "the outbox was delivered after the restart" || bad "outbox still holds $(ls "$TMP/outbox"/*.json 2>/dev/null | wc -l) file(s)"
  [ "$(stat split_nodes)" -ge 1 ] && ok "the server has the split from the outbox" || bad "no split at the server"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  stop_server; done_test
}

test_v() {
  say "test v: complete_410 (a complete campaign's token after a newer campaign opened)"
  start_server 3 30 4 --heartbeat-s 2 --batch-interval-s 1 --absorb-total-s 2 --absorb-probe-s 1 || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client v1.log --workers 1 --no-ui || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]'
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 20; CLIENT_PID=""        # client A stops and keeps its token
  OUTBOX="$TMP/outbox_v" FAKE_MIN_S=0.2 FAKE_MAX_S=0.4 start_client v2.log --workers 2 --no-ui || return
  wait_rc "$CLIENT_PID" 45; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && grep -q "is complete" "$TMP/v2.log" && ok "a second client finished campaign 1 and exited 0" || bad "campaign 1 not completed: exit $rc"
  admin '{"op": "open", "roots": 2, "split_after_s": 4}'
  FAKE_MIN_S=0.2 FAKE_MAX_S=0.4 start_client v3.log --workers 1 --no-ui || return
  grep -q "no longer valid.*410" "$TMP/v3.log" && grep -q "for campaign 2" "$TMP/v3.log" \
    && ok "client A restarted with campaign 1's token: 410, then it joined campaign 2" || bad "restart: $(sed -n 3,5p "$TMP/v3.log")"
  wait_rc "$CLIENT_PID" 45; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "and exited 0 when campaign 2 completed" || bad "exit $rc"
  admin '{"op": "open", "roots": 4, "split_after_s": 30}'
  FAKE_MIN_S=40 FAKE_MAX_S=50 start_client v4.log --workers 1 --no-ui || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]'
  admin '{"op": "complete"}'; admin '{"op": "open", "roots": 2, "split_after_s": 4}'
  wait_rc "$CLIENT_PID" 30; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && grep -q "a newer campaign is open (HTTP 410)" "$TMP/v4.log" && grep -q "is complete" "$TMP/v4.log" \
    && ok "a running client whose campaign completed while a newer one opened: summary, exit 0" || bad "running 410-complete: exit $rc, $(tail -3 "$TMP/v4.log")"
  [ "$(fake_workers_alive)" = 0 ] && ok "no worker left behind" || bad "workers left"
  stop_server; done_test 100
}

test_w() {
  say "test w: sleep (a system sleep across the split time is not a hang; neither is one across the stop grace)"
  # 6 s windows. Part 1: the client and its worker are frozen with SIGSTOP (like a closed lid) for 9 s
  # across the split time; the client wakes first and the worker 1.5 s later. With a watchdog slack
  # of 2 s the watchdog fired at once on wake before the fix (a void run and a 'hang' failure).
  # Part 2: Stop, then a 7 s freeze while the worker takes 3 s to answer SIGINT (grace 4 s).
  start_server 1 60 6 --batch-interval-s 1 --exits 0 --heartbeat-s 2 --absorb-total-s 0 || return
  FAKE_SIGINT_DELAY_S=3 FAKE_MIN_S=40 FAKE_MAX_S=50 start_client w1.log --workers 1 --no-ui \
      --watchdog-slack-s 2 --stop-grace-s 4 || return
  wait_true 15 eval '[ "$(fake_workers_alive)" = 1 ]' || { bad "no worker started"; return; }
  local wp wp2; wp=$(pgrep -f "$WPAT" | head -1)
  sleep 2
  kill -STOP "$CLIENT_PID" "$wp"; sleep 9
  kill -CONT "$CLIENT_PID"; sleep 1.5; kill -CONT "$wp"
  wait_log w1.log "job [0-9]+ split in" 15 && ok "the window whose split time passed during the sleep was reported as a split" \
    || bad "no split after the wake: $(grep -E 'void|SIGINT' "$TMP/w1.log" | head -2)"
  grep -Eq "sending SIGINT|run void" "$TMP/w1.log" && bad "the watchdog fired after the wake: $(grep -E 'sending SIGINT|run void' "$TMP/w1.log" | head -2)" \
    || ok "no watchdog and no void run after the wake"
  wait_true 10 eval '[ "$(stat split_nodes)" -ge 1 ]' && [ "$(aq "a['failure_reasons'].get('hang', 0)")" = 0 ] \
    && ok "the server got the split and no 'hang' failure" || bad "splits $(stat split_nodes), failures $(aq "a['failure_reasons']")"
  wait_true 15 eval 'wp2=$(pgrep -f "$WPAT" | head -1); [ -n "$wp2" ] && [ "$wp2" != "$wp" ]' || { bad "no second window"; return; }
  sleep 1
  kill -INT "$CLIENT_PID"
  wait_log w1.log "stopping:" 5 || bad "no stop logged"
  kill -STOP "$CLIENT_PID" "$wp2"; sleep 7
  kill -CONT "$CLIENT_PID"; sleep 1; kill -CONT "$wp2"
  wait_rc "$CLIENT_PID" 20; rc=$RC; CLIENT_PID=""
  grep -q "did not print SUMMARY" "$TMP/w1.log" && bad "the stop grace killed the worker after the sleep" \
    || ok "the stop grace did not count the sleep"
  [ "$rc" = 0 ] && [ "$(stat split_nodes)" -ge 2 ] && ok "the stopped window was reported as a split; exit 0" \
    || bad "exit $rc, splits $(stat split_nodes)"
  [ "$(fake_workers_alive)" = 0 ] && ok "no worker left behind" || bad "workers left"
  stop_server; done_test
}

test_x() {
  say "test x: grace_kill and endgame (a probe killed after Stop's grace keeps the window; absorb_total_s 0 per lease)"
  # J (8 tokens, 40-50 s) splits at its 3 s window into 2 children of 24-30 s; the first 5 s absorption
  # probe ignores SIGINT and is killed after the 2 s grace: its node stays open, J's split is reported.
  start_server 1 60 3 --absorb-total-s 30 --absorb-probe-s 5 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_SIGINT_IGNORE=1 FAKE_REMAINING_N=2 FAKE_MIN_S=40 FAKE_MAX_S=50 start_client x1.log --workers 1 --no-ui \
      --stop-grace-s 2 || return
  wait_true 20 pgrep -f "$WPAT.*--split-after 5\.0" >/dev/null || { bad "no absorption probe started"; return; }
  sleep 0.5
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 20; rc=$RC; CLIENT_PID=""
  grep -q "stays open; the rest of the window is reported" "$TMP/x1.log" && ok "the probe was killed after the grace: $(grep -o 'Node [^;]*stays open' "$TMP/x1.log" | head -1)" \
    || bad "no grace kill of the probe: $(grep -E 'SUMMARY|abandoned' "$TMP/x1.log" | head -2)"
  ! grep -q "abandoned" "$TMP/x1.log" && [ "$rc" = 0 ] && [ "$(stat split_nodes)" = 1 ] && [ "$(aq "a['counts']['open']")" = 2 ] \
    && ok "the window was reported (J split, both children open), not abandoned; exit 0" \
    || bad "exit $rc, splits $(stat split_nodes), open $(aq "a['counts']['open']"), $(grep abandoned "$TMP/x1.log" | head -1)"
  [ "$(aq "len(a['failure_reasons'])")" = 0 ] && ok "no failure reported for the killed probe" || bad "failures $(aq "a['failure_reasons']")"
  [ "$(fake_workers_alive)" = 0 ] && ok "no worker left behind" || bad "workers left"
  stop_server
  # the lease says absorb_total_s 0 (jobs.js endgame): no probe runs although the campaign's budget is 30 s
  start_server 2 60 2 --lease-absorb-total-s 0 --absorb-total-s 30 --absorb-probe-s 5 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_MIN_S=2.5 FAKE_MAX_S=3 start_client x2.log --workers 1 --no-ui || return
  wait_rc "$CLIENT_PID" 45; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && audit_clean && ok "the campaign completed (exit 0)" || bad "exit $rc; audit $(audit | cut -c1-200)"
  [ "$(stat split_nodes)" -ge 1 ] && [ "$(stat absorbed_done)" = 0 ] && grep -q "(endgame)" "$TMP/x2.log" \
    && ok "$(stat split_nodes) split(s), no child absorbed: the per-lease budget 0 skipped the absorption pass" \
    || bad "splits $(stat split_nodes), absorbed $(stat absorbed_done), endgame logged: $(grep -c '(endgame)' "$TMP/x2.log")"
  stop_server; done_test
}

test_y() {
  say "test y: absorb_order (deepest first; a split probe is kept and only its children follow)"
  # J (depth 8, 60-80 s) splits at its 3 s window into children +5,+4,+3 (cursor and stack top: trivial)
  # and three +1 (stack bottom: 3-4 s each, a 2 s probe splits). Budget 6 s (the new pass needs about 4.5 s).
  # The old shallow-first order spent it all on the three +1 children (every probe split and was dropped)
  # and never probed the deep ones.
  start_server 1 60 3 --absorb-total-s 6 --absorb-probe-s 2 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_REMAINING_DEPTHS=5,4,3,1,1,1 FAKE_DEPTH_DECAY=0.05 FAKE_MIN_S=60 FAKE_MAX_S=80 start_client y.log --workers 1 --no-ui || return
  wait_true 30 eval '[ "$(aq "len(a[\"report_log\"])")" -ge 1 ]' || { bad "no report within 30 s: $(tail -3 "$TMP/y.log")"; return; }
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  r=$(aq "a['report_log'][0]['nodes']")
  "$PY" - "$r" <<'EOF' && ok "J's report: its deep children done, one +1 probe kept as a split with all its children done, the other two +1 never probed: $r" || bad "report nodes: $r"
import sys, ast
nodes = ast.literal_eval(sys.argv[1])
by = {}
for d, st, el in nodes:
    by.setdefault(d, []).append(st)
assert nodes[0][:2] == [8, "split"], nodes[0]
assert sorted(by[9]) == ["open", "open", "split"], by[9]          # the probe that split is kept, siblings untouched
assert all(st == "done" for d, sts in by.items() if d >= 10 for st in sts), by   # every deeper node absorbed
assert sum(len(v) for d, v in by.items() if d >= 10) == 9, by     # 3 deep children of J + 6 children of the probe
EOF
  grep -q "absorption: 9 node(s) finished, 1 probe(s) split and kept" "$TMP/y.log" && ok "logged: $(grep -o 'absorption: .*' "$TMP/y.log" | head -1)" || bad "no absorption log: $(grep absorption "$TMP/y.log" | head -2)"
  [ "$(stat grants_of_absorbed)" = 0 ] && ok "no absorbed node was ever leased" || bad "$(stat grants_of_absorbed) absorbed nodes leased"
  no_bang y.log
  stop_server
  # The same tree with a budget of 2.6 s: the three deep children finish (about 0.6 s), the first +1
  # probe gets its 2 s (or what is left), splits into six children and the budget is gone (at most one
  # of them probed). Kept, it would hand 5-6 never-probed children to the pool for about 2 s of kept
  # work (under 1 s per extra job): it goes back open.
  rm -rf "$TMP/outbox" "$HOME/.pathology_volunteer.json"
  start_server 1 60 3 --absorb-total-s 2.6 --absorb-probe-s 2 --batch-interval-s 1 --exits 0 --heartbeat-s 2 || return
  FAKE_REMAINING_DEPTHS=5,4,3,1,1,1 FAKE_DEPTH_DECAY=0.05 FAKE_MIN_S=60 FAKE_MAX_S=80 start_client y2.log --workers 1 --no-ui || return
  wait_true 30 eval '[ "$(aq "len(a[\"report_log\"])")" -ge 1 ]' || { bad "no report within 30 s: $(tail -3 "$TMP/y2.log")"; return; }
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  r=$(aq "a['report_log'][0]['nodes']")
  "$PY" - "$r" <<'EOF2' && ok "J's report: its deep children done, the +1 probe that split as the budget ran out is back to open with no children: $r" || bad "report nodes: $r"
import sys, ast
nodes = ast.literal_eval(sys.argv[1])
assert nodes[0][:2] == [8, "split"], nodes[0]
assert sorted(st for d, st, el in nodes if d == 9) == ["open", "open", "open"], nodes    # the split probe went back open
assert sorted(d for d, st, el in nodes if d >= 10) == [11, 12, 13] and all(st == "done" for d, st, el in nodes if d >= 10), nodes
EOF2
  grep -Eq "the last split probe goes back open \([56] of its 6 children never probed" "$TMP/y2.log" \
    && ok "logged: $(grep -o 'absorption: .*' "$TMP/y2.log" | head -1)" || bad "no demotion logged: $(grep absorption "$TMP/y2.log" | head -2)"
  no_bang y2.log
  stop_server; done_test
}

test_z() {
  say "test z: lease (2 jobs per worker; the window is chosen when it starts; N18 sub-second and node splits)"
  # 1 worker, 5-6 s jobs, 30 s windows: one job runs, one waits (hold 2). The owner then shortens the
  # window to 2 s; the waiting job, leased with 30 s, must start with 2 s.
  start_server 6 60 30 --batch-interval-s 1 --exits 0 --heartbeat-s 1 --heartbeat-window --absorb-total-s 0 || return
  FAKE_MIN_S=5 FAKE_MAX_S=6 start_client z1.log --workers 1 --no-ui || return
  wait_true 15 eval '[ "$(stat lease_grants)" -ge 2 ]' || bad "no second job leased ahead: $(grep 'lease ' "$SLOG" | tail -2)"
  admin '{"op": "set", "params": {"split_after_s": 2, "ramp_split_after_s": 2}}'
  wait_log z1.log "window 2 s" 20 || bad "no window of 2 s: $(grep -o 'window [0-9.]* s' "$TMP/z1.log" | sort | uniq -c | tr '\n' ' ')"
  jid=$(grep -o 'job [0-9]* exit [0-9]* window 2 s' "$TMP/z1.log" | head -1 | awk '{print $2}')
  [ -n "$jid" ] && grep -Eq "lease Tester .*-> \[([0-9]+, )*$jid(, [0-9]+)*\] \(split_after_s 30" "$SLOG" \
    && ok "job $jid, leased with a 30 s window, started with the server's current 2 s window" \
    || bad "job '$jid' was not leased with 30 s: $(grep 'lease Tester' "$SLOG" | head -3)"
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  [ "$(stat max_lease_n)" -le 2 ] && [ "$(stat max_held)" -le 3 ] \
    && ok "1 worker: requests of at most $(stat max_lease_n) job(s), at most $(stat max_held) held (2 + a report in flight)" \
    || bad "max n $(stat max_lease_n), max held $(stat max_held)"
  no_bang z1.log
  stop_server
  # N18: min_window_s 0.1 lets a 0.6 s window through (no clamp to 1 s), and split_after_nodes 1 reaches
  # the worker: J splits after one step (about 0.1 s), long before its 0.6 s split time
  start_server 1 60 0.6 --min-window-s 0.1 --split-after-nodes 1 --batch-interval-s 1 --exits 0 --heartbeat-s 2 --absorb-total-s 0 || return
  FAKE_MIN_S=60 FAKE_MAX_S=80 start_client z2.log --workers 1 --no-ui || return
  wait_log z2.log "window 0.6 s" 10 && ok "a 0.6 s window ran as such (campaign min_window_s 0.1)" || bad "window: $(grep -o 'window [0-9.]* s' "$TMP/z2.log" | head -2)"
  wait_true 15 eval '[ "$(aq "len(a[\"report_log\"])")" -ge 1 ]'
  kill -INT "$CLIENT_PID"; wait_exit "$CLIENT_PID" 20; CLIENT_PID=""
  j=$(aq "a['report_log'][0]['nodes'][0]")
  "$PY" -c 'import sys,ast; d,st,el=ast.literal_eval(sys.argv[1]); sys.exit(0 if st=="split" and el is not None and el < 0.45 else 1)' "$j" \
    && ok "J split after one step (--split-after-nodes 1): $j" || bad "J: $j"
  stop_server
  # The hold cap counts only the workers that stay. 3 workers on 2 s windows; the retiring workers 2 and 3
  # are frozen here (so their windows cannot end) and the count goes 3 -> 1: worker 1 must go on leasing.
  # With their jobs counted (2 x 1 - 2 running) it ran out of queued jobs and sat idle.
  rm -rf "$TMP/outbox" "$HOME/.pathology_volunteer.json"
  start_server 60 120 2 --batch-interval-s 1 --exits 0 --heartbeat-s 1 --absorb-total-s 0 || return
  FAKE_MIN_S=20 FAKE_MAX_S=30 start_client z3.log --workers 3 --no-browser || return
  sec=$(ui_secret "$UPORT")
  frozen=""
  for _ in 1 2 3 4 5 6; do
    p12=$(state_field "' '.join(str(w['pid']) for w in s['slots'] if w['idx'] in (1, 2) and w['pid'] and w['job_id'] is not None)" 2>/dev/null)
    if [ "$(echo $p12 | wc -w | tr -d ' ')" = 2 ]; then
      kill -STOP $p12 2>/dev/null
      again=$(state_field "' '.join(str(w['pid']) for w in s['slots'] if w['idx'] in (1, 2) and w['pid'])" 2>/dev/null)
      st12=$(for p in $p12; do ps -o stat= -p "$p" 2>/dev/null | cut -c1; done | tr -d ' \n')
      [ "$again" = "$p12" ] && [ "$st12" = TT ] && { frozen="$p12"; break; }
      kill -CONT $p12 2>/dev/null
    fi
    sleep 0.3
  done
  if [ -z "$frozen" ]; then bad "could not freeze workers 2 and 3 mid-window"; else
    EXTRA_PIDS="$EXTRA_PIDS $frozen"
    http_post "http://127.0.0.1:$UPORT/workers" '{"workers": 1}' "$sec" >/dev/null
    n0=$(grep -c "worker 0: job .* window" "$TMP/z3.log")
    g0=$(stat lease_grants)
    wait_true 25 eval '[ $(( $(grep -c "worker 0: job .* window" "$TMP/z3.log") - n0 )) -ge 6 ]' \
      && ok "workers 3 -> 1 with the 2 retiring ones busy: worker 1 ran $(( $(grep -c "worker 0: job .* window" "$TMP/z3.log") - n0 )) more windows, $(( $(stat lease_grants) - g0 )) job(s) leased meanwhile" \
      || bad "worker 1 ran only $(( $(grep -c "worker 0: job .* window" "$TMP/z3.log") - n0 )) window(s) after 3 -> 1; lease grants $g0 -> $(stat lease_grants)"
    [ "$(state_field "sum(1 for w in s['slots'] if w['retiring'] and w['job_id'] is not None)")" = 2 ] \
      && ok "the 2 retiring workers still held their jobs all along" || bad "retiring: $(state_field "[(w['retiring'], w['job_id']) for w in s['slots']]")"
    kill -CONT $frozen 2>/dev/null
  fi
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 30; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "Stop: exit 0" || bad "exit $rc"
  no_bang z3.log
  stop_server; done_test 130
}

test_pw() {
  say "test pw: pause_workers (M56) and Ctrl-Z (M61)"
  start_server 8 60 60 --heartbeat-s 5 || return
  FAKE_MIN_S=40 FAKE_MAX_S=50 start_client pw.log --workers 3 --no-browser || return
  pids=""
  for _ in $(seq 1 60); do pids=$(state_field "' '.join(str(w['pid']) for w in s['slots'] if w['pid'])" 2>/dev/null); [ "$(echo $pids | wc -w | tr -d ' ')" = 3 ] && break; sleep 0.25; done
  [ "$(echo $pids | wc -w | tr -d ' ')" = 3 ] || { bad "expected 3 running workers: $(http_get http://127.0.0.1:$UPORT/state | head -c 200)"; return; }
  stats_of() { for p in $pids; do ps -o stat= -p "$p" 2>/dev/null | cut -c1; done | tr -d ' \n'; }
  all_T() { [ "$(stats_of)" = TTT ]; }
  none_T() { case "$(stats_of)" in *T*) return 1;; *) return 0;; esac; }
  sec=$(ui_secret "$UPORT")
  http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" >/dev/null
  wait_true 5 all_T && ok "Pause: all 3 workers stopped" || bad "after Pause: '$(stats_of)'"
  http_post "http://127.0.0.1:$UPORT/workers" '{"workers": 1}' "$sec" >/dev/null
  http_post "http://127.0.0.1:$UPORT/resume" '{}' "$sec" >/dev/null
  wait_true 5 none_T && ok "Pause, workers 3 -> 1, Resume: no worker left stopped (the 2 retiring ones run again)" || bad "after Resume: '$(stats_of)'"
  [ "$(state_field "sum(1 for w in s['slots'] if w['retiring'])")" = 2 ] && ok "the panel shows 2 workers retiring" || bad "retiring: $(state_field "[w['retiring'] for w in s['slots']]")"
  http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" >/dev/null
  wait_true 5 all_T && ok "workers 3 -> 1, then Pause: the retiring workers are frozen too" || bad "after the second Pause: '$(stats_of)'"
  pf=$(state_field "s['pending_flags']['pause']")
  [ "$pf" = None ] && ok "no stale 'freezing' flag once all are frozen" || bad "pause flag: $pf"
  http_post "http://127.0.0.1:$UPORT/pause" '{}' "$sec" >/dev/null      # a redundant click (M57)
  http_post "http://127.0.0.1:$UPORT/resume" '{}' "$sec" >/dev/null
  wait_true 5 none_T || bad "after the second Resume: '$(stats_of)'"
  sleep 0.5
  pf=$(state_field "s['pending_flags']['pause']")
  [ "$pf" = None ] && ok "a redundant Pause click leaves no 'freezing…' flag after Resume (M57)" || bad "pause flag after Resume: $pf"
  kill -TSTP "$CLIENT_PID"
  wait_true 5 all_T && ok "Ctrl-Z (SIGTSTP): the workers stopped with the client" || bad "after SIGTSTP: '$(stats_of)'"
  case "$(ps -o stat= -p "$CLIENT_PID" | cut -c1)" in T) ok "the client itself is stopped";; *) bad "client state $(ps -o stat= -p "$CLIENT_PID")";; esac
  kill -CONT "$CLIENT_PID"
  wait_true 5 none_T && ok "SIGCONT (fg): the workers run again" || bad "after SIGCONT: '$(stats_of)'"
  kill -INT "$CLIENT_PID"; wait_rc "$CLIENT_PID" 30; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "Stop: exit 0" || bad "exit $rc"
  [ "$(fake_workers_alive)" = 0 ] && ok "no worker left behind" || bad "workers left"
  stop_server; done_test
}

test_ui() {
  say "test ui: the panel script in node (6x6 boards, newline codes, completion)"
  command -v node >/dev/null 2>&1 || { ok "skipped: node is not installed"; return; }
  start_server 4 60 60 --grid 6x6 --exits 0,1,2,7,8,14 --extra "--allow-exit-transit --num-holes 0" --paused-max-s 7200 \
      --heartbeat-s 1 --batch-interval-s 1 --absorb-total-s 0 || return
  FAKE_MIN_S=1.5 FAKE_MAX_S=2.5 start_client ui.log --workers 1 --no-browser || return
  wait_true 20 eval '[ "$(state_field "any(e.get(\"best\") for e in (s[\"exits\"] or {}).values())")" = True ]' || { bad "no best level in /state"; return; }
  http_get "http://127.0.0.1:$UPORT/state" > "$TMP/ui_state1.json"
  "$PY" - "$TMP/ui_state1.json" <<'EOF' && ok "/state: campaign paused_max_s 7200 (M67); the server's best codes use newlines (as jobs.js)" || bad "state checks failed: $(head -c 300 "$TMP/ui_state1.json")"
import sys, json
s = json.load(open(sys.argv[1]))
assert s["campaign"]["paused_max_s"] == 7200, s["campaign"]
codes = [e["best"]["code"] for e in s["exits"].values() if e.get("best")]
assert codes and all("\n" in c and "/" not in c for c in codes), codes
assert all(len(r) == 6 for c in codes for r in c.split("\n")) and all(len(c.split("\n")) == 6 for c in codes), codes
EOF
  for _ in $(seq 1 120); do                      # follow the client to its completion (the panel's view of it)
    http_get "http://127.0.0.1:$UPORT/state" > "$TMP/ui_state2.json.tmp" 2>/dev/null || break
    [ -s "$TMP/ui_state2.json.tmp" ] && mv "$TMP/ui_state2.json.tmp" "$TMP/ui_state2.json"
    grep -q '"completion": {' "$TMP/ui_state2.json" 2>/dev/null && break
    sleep 0.25
  done
  wait_rc "$CLIENT_PID" 30; rc=$RC; CLIENT_PID=""
  [ "$rc" = 0 ] && ok "the campaign completed, exit 0" || bad "exit $rc"
  grep -q '"completion": {' "$TMP/ui_state2.json" && ok "the panel saw the completion state" || bad "no completion in the last /state"
  cat > "$TMP/ui_test.js" <<'EOF'
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[2], 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const st1 = JSON.parse(fs.readFileSync(process.argv[3], 'utf8')), st2 = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
class CL { constructor(e) { this.e = e; this.s = new Set(); } add(c) { this.s.add(c); } remove(c) { this.s.delete(c); } toggle(c, on) { if (on === undefined ? !this.s.has(c) : on) this.s.add(c); else this.s.delete(c); } contains(c) { return this.s.has(c); } }
class El {
  constructor(tag) { this.tagName = tag; this.children = []; this.style = {}; this.attrs = {}; this._inner = ''; this._text = ''; this.classList = new CL(this); this.hidden = false; this.className = ''; this.value = ''; this.disabled = false; this.sets = 0; this.max = ''; }
  set innerHTML(v) { this._inner = String(v); this.children = []; this._text = ''; this.sets++; }
  get innerHTML() { return this._inner; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); c.parentNode = this; if (c.id) byId[c.id] = c; return c; }
  removeChild(c) { this.children = this.children.filter(x => x !== c); if (c.id && byId[c.id] === c) delete byId[c.id]; }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener() {} matches() { return false; } contains() { return false; } closest() { return null; }
}
const byId = {};           // the page's own ids exist from the start; others only once appended (as in a browser)
for (const m of html.split('<script>')[0].matchAll(/ id="([^"]+)"/g)) byId[m[1]] = Object.assign(new El('div'), { id: m[1] });
const document = { getElementById: id => byId[id] || null, createElement: t => new El(t),
  querySelector: () => null, addEventListener() {}, body: new El('body'), activeElement: null, execCommand() {} };
const states = [st1];
const ctx = { document, window: { innerWidth: 1200, getSelection: () => ({ rangeCount: 0, isCollapsed: true }), addEventListener() {} },
  navigator: {}, console, JSON, Math, Date, Number, String, Array, Object, Set, Map, WeakMap, Promise, Error,
  setInterval() {}, setTimeout: (f, ms) => 0, clearTimeout() {}, AbortController, alert() {}, confirm: () => false,
  fetch: async () => ({ json: async () => states[0] }) };
vm.createContext(ctx);
vm.runInContext(script, ctx);
(async () => {
  await new Promise(r => setImmediate(r)); await new Promise(r => setImmediate(r));
  const run = code => vm.runInContext(code, ctx);
  // pure helpers
  assert.strictEqual(JSON.stringify(run(`levelRows('030000\\n000400\\n0?0000')`)), '["030000","000400","0?0000"]');
  assert.strictEqual(run(`codeText('0300\\n0040')`), '0300/0040');
  assert.strictEqual(run(`codeText('0300/0040')`), '0300/0040');
  assert.strictEqual(run(`levelRows('03<b>')`), null);
  assert.strictEqual(run(`globalExit({'0': {best: {moves: 127}}, '1': {best: {moves: 149}}, '2': {best: null}}, null)`), '1');
  assert.strictEqual(run(`globalExit({'0': {best: {moves: 127}}, '1': {best: {moves: 149}}}, 0)`), '0');
  assert.strictEqual(run(`openCands({candidates_open_deeper: 3})`), 3);
  assert.strictEqual(run(`openCands({candidates: {open_deeper: 2}})`), 2);
  assert.strictEqual(run(`openCands({})`), null);
  // the page rendered st1 at load (poll); render again: an unchanged board is not redrawn (M63)
  const g = byId.bGlobal;
  assert.ok(g && g._key, 'the global board was drawn');
  const tiles = el => (el.className === 'tile' ? 1 : 0) + el.children.reduce((n, c) => n + tiles(c), 0);
  assert.strictEqual(tiles(g), 36, 'a 6x6 global board has 36 tiles, got ' + tiles(g));
  const codeEl = (function find(el) { if (el.tagName === 'code') return el; for (const c of el.children) { const x = find(c); if (x) return x; } return null; })(g);
  assert.ok(codeEl && /^[0-9A-Z?]{6}(\/[0-9A-Z?]{6}){5}$/.test(codeEl.textContent), 'the global code is shown with / rows: ' + (codeEl && codeEl.textContent));
  assert.ok(/best known overall \(exit \d+\)/.test(g.children[0].innerHTML), 'Any exit: caption ' + g.children[0].innerHTML);
  const before = g.children[0];
  run('render(' + JSON.stringify(st1) + ')');
  assert.strictEqual(g.children[0], before, 'an unchanged board was redrawn');
  assert.ok(/7200|2\.\d h|2 h/.test(byId.pausedMax.textContent) || /h after each lease/.test(byId.pausedMax.textContent), 'paused max: ' + byId.pausedMax.textContent);
  for (const s of st1.slots) { const w = byId['w' + s.idx]; assert.ok(w, 'worker card ' + s.idx); }
  // worker cards: the numbers that tick every poll live in their own spans; the seed block is not rewritten
  const busy = JSON.parse(JSON.stringify(st1));
  busy.slots = [Object.assign({}, st1.slots[0] || {}, { idx: 0, job_id: 77, exit: 0, seed: 'U1,R2', cur_seed: 'U1,R2', phase: 'window',
    state: 'running', frozen: false, retiring: false, pid: 4242, window_left: 12, stack_size: 1, nodes_done: 0, elapsed: 3, cur: null })];
  run('render(' + JSON.stringify(busy) + ')');
  const facts = byId.w0.children[0], top = facts.children[0], sets = top.sets;
  assert.ok(/U1,R2/.test(top.innerHTML) && /window, 12 s left/.test(facts.children[1].textContent), 'card: ' + top.innerHTML + ' | ' + facts.children[1].textContent);
  busy.slots[0].window_left = 11; busy.slots[0].elapsed = 4; busy.slots[0].nodes_done = 2;
  run('render(' + JSON.stringify(busy) + ')');
  assert.strictEqual(top.sets, sets, 'the seed block was rewritten for a tick');
  assert.ok(/window, 11 s left/.test(facts.children[1].textContent) && /nodes done 2 · running 4 s/.test(facts.children[1].textContent),
            'ticking numbers: ' + facts.children[1].textContent);
  busy.slots[0].phase = 'absorb'; busy.slots[0].stack_size = 5;
  run('render(' + JSON.stringify(busy) + ')');
  assert.ok(/absorbing, 11 s left/.test(facts.children[1].textContent) && /still to probe 5/.test(facts.children[1].textContent), 'absorb: ' + facts.children[1].textContent);
  // completion: the banner says so and says the window can be closed
  run('render(' + JSON.stringify(st2) + ')');
  const b = byId.banner.innerHTML;
  assert.ok(!byId.banner.hidden && /Campaign complete/.test(b) && /You can close this window/.test(b), 'completion banner: ' + b.slice(0, 300));
  assert.ok(/Result per exit/.test(b), 'the result is in the banner');
  // the client exits: after 8 failed polls the banner keeps the result
  run('lastState = ' + JSON.stringify(st2) + '; failures = 8; lastOkAt = Date.now() - 5000; gone = false; freshness();');
  assert.ok(/the client has exited/.test(byId.banner.innerHTML) && /You can close this window/.test(byId.banner.innerHTML), 'gone banner: ' + byId.banner.innerHTML.slice(0, 200));
  assert.ok(document.body.classList.contains('stale'), 'the page is marked stale');
  console.log('ui ok: 36 tiles, code ' + codeEl.textContent);
})().catch(e => { console.error(e && e.stack || e); process.exit(1); });
EOF
  out=$(node "$TMP/ui_test.js" "$HERE/volunteer_ui.html" "$TMP/ui_state1.json" "$TMP/ui_state2.json" 2>&1) \
    && ok "panel script: $out" || bad "panel script failed: $(echo "$out" | head -8)"
  stop_server; done_test
}

ALL="a b c d e f g h i j k l m n o p q r s t u v w x y z pw ui"
which="${*:-all}"
[ "$which" = all ] && which="$ALL"
start=$(date +%s)
for t in $which; do
  case " $ALL " in *" $t "*) rm -rf "$TMP/outbox" "$TMP/outbox_"* "$HOME/.pathology_volunteer.json"; "test_$t"; end_test_procs;;
    *) echo "usage: $0 [$ALL|all]"; exit 2;; esac
done
echo
echo "logs in $TMP  (elapsed $(( $(date +%s) - start )) s)"
if [ "$FAIL" = 0 ]; then echo "ALL PASSED"; else echo "SOME TESTS FAILED"; exit 1; fi
