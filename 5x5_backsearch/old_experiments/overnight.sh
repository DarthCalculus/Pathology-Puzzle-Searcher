#!/bin/bash
# Overnight backsearch experiment (exit-0 / seed-path geometry).
#
#   Phase 1: per-restart P scan {0,10,25,50}, 50 episodes each -> settle P.
#   Phase 2: depth chase with the two best P values, ~800 episodes each,
#            to see how deep this geometry reaches unguided.
#
# Sequential (one worker at a time: no CPU contention).  Self-terminating
# via --rollout-restarts caps; --time backstops only stop a runaway episode.
# bash 3.2 compatible (macOS).  Worker args go through an array + "${...[@]}"
# so multi-token args can never be mis-split.
set -u
cd "$(dirname "$0")"

W=./backsearch_worker2
BASE=(--grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3)

OUT=/tmp/overnight
mkdir -p "$OUT"
LOG="$OUT/summary.log"
: > "$LOG"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] $*" | tee -a "$LOG"; }
med()   { printf '%s\n' $1 | sort -n | awk 'NF{a[NR]=$0} END{print (NR? a[int((NR+1)/2)] : "-")}'; }
mx()    { printf '%s\n' $1 | sort -n | awk 'NF{v=$0} END{print (NR? v : "-")}'; }

# run <label> <P> <restarts> <timecap_s>
run() {
  label=$1; P=$2; R=$3; T=$4
  o="$OUT/${label}.out"; e="$OUT/${label}.err"
  log "START $label  P=$P restarts=$R time=${T}s"
  "$W" "${BASE[@]}" --rollout "8,400,40,$P" --rollout-restarts "$R" --time "$T" \
       --rollout-trace >"$o" 2>"$e"
  el=$(grep -aoE 'elapsed: *[0-9.]+' "$o" | grep -aoE '[0-9.]+' | tail -1)
  rs=$(grep -aoE 'restarts=[0-9]+' "$e" | grep -aoE '[0-9]+' | tail -1)
  best=$(grep -aiE 'Best depth' "$o" | tail -1 | grep -aoE '[0-9]+' | head -1)
  des=$(grep -aoE 'DEAD-END at depth [0-9]+' "$e" | grep -aoE '[0-9]+$')
  spr=$(awk -v a="${el:-0}" -v b="${rs:-0}" 'BEGIN{printf "%.2f",(b>0?a/b:0)}')
  log "DONE  $label  best=${best:-?}  restarts=${rs:-0}  avg/restart=${spr}s  elapsed=${el:-0}s  deadend med=$(med "$des") max=$(mx "$des")"
}

log "==== PHASE 1: per-restart P scan (50 episodes each) ===="
for p in 0 10 25 50; do
  run "scan_p$p" "$p" 50 3600
done

log "==== PHASE 2: depth chase (P=0 ceiling, P=25 consistency) ===="
run "chase_p0"  0  800 21600
run "chase_p25" 25 800 21600

log "==== AGGREGATE: deepest puzzle found across all runs ===="
bestall=0; bestfile=""
for f in "$OUT"/*.out; do
  b=$(grep -aiE 'Best depth' "$f" | tail -1 | grep -aoE '[0-9]+' | head -1)
  [ -z "$b" ] && b=0
  if [ "$b" -gt "$bestall" ]; then bestall=$b; bestfile=$f; fi
done
log "GLOBAL BEST depth=$bestall  in $bestfile"
log "ALL DONE.  Per-run logs in $OUT/*.out (streamed new-best puzzles are inside)."
