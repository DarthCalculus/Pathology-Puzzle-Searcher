#!/bin/bash
# Overnight depth chase on the fw67 geometry.
# Phase 1: A/B three arms, equal wall-clock each, time-based.
# Phase 2: commit the rest of the night to whichever arm reached the deepest.
# Sequential (one worker at a time) per the no-parallel-sweeps rule.
# No --seed (long run aggregates many restarts; depth lives in the tail).
set -u

PROJ="/Users/george/Documents/Claude Projects/project45/5x5_backsearch"
cd "$PROJ" || { echo "cannot cd to PROJ"; exit 1; }
W=./backsearch_worker                       # relative => no space-in-path word-split grief
OUT=/tmp/overnight
mkdir -p "$OUT"
LOG="$OUT/summary.log"
RES="$OUT/results.tsv"

GEOM="--grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3 --fixedwalls 6,7"

T1=3600        # phase-1 seconds per arm (3 arms => 3h)
HALF1=1800     # phase-1 halfway mark (secondary metric)
T2=21600       # phase-2 seconds for the winner (6h)  => ~9h total

# arm = "tag|extra rollout flags".  Model path is RELATIVE (no space) on purpose.
ARMS=(
"lean|--rollout 4,200,10,10 --rollout-rand-gens 5"
"tight|--rollout 4,400,40,10 --rollout-rand-gens 4"
"nn|--rollout 4,200,10,10 --rollout-rand-gens 5 --nn-value-model checkpoints/value_v4.ts --nn-blend 0.5"
)

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(stamp)] $*" | tee -a "$LOG"; }

# parse one arm's stdout -> "best besthalf": deepest depth, deepest by HALF1 secs.
parse(){
  awk -v half="$1" '
    /^[0-9]+ \([0-9.]+[sm]\)$/ {
      d=$1+0; t=$2; gsub(/[()]/,"",t);
      u=substr(t,length(t),1); v=substr(t,1,length(t)-1)+0;
      sec=(u=="m")?v*60:v;
      if(d>best)best=d;
      if(sec<=half && d>bh)bh=d;
    } END{ printf "%d %d\n", best+0, bh+0 }' "$2"
}

# extract the deepest puzzle block (the last/highest new-best block) from stdout.
deepest_puzzle(){
  awk '
    /^[0-9]+ \([0-9.]+[sm]\)$/ { d=$1+0; if(d>=best){best=d; cap=1; buf=$0"\n"; next} else {cap=0} }
    cap && /^[[:space:]]*$/ { cap=0; next }
    cap { buf=buf $0"\n" }
    END{ printf "%s", buf }' "$1"
}

CUR_PID=""
cleanup(){ [ -n "$CUR_PID" ] && kill "$CUR_PID" 2>/dev/null; log "INTERRUPTED, killed $CUR_PID"; exit 130; }
trap cleanup INT TERM

: > "$LOG"
printf 'phase\ttag\tbest\thalf\tsecs\n' > "$RES"
log "START overnight A/B.  geom: $GEOM"
log "Phase 1: ${T1}s/arm x ${#ARMS[@]} arms; Phase 2: ${T2}s for winner."

# ---------- Phase 1 ----------
win_tag=""; win_extra=""; win_best=-1
for a in "${ARMS[@]}"; do
  tag="${a%%|*}"; extra="${a#*|}"
  o="$OUT/p1_$tag.out"; e="$OUT/p1_$tag.err"
  log "PHASE1 START $tag  ($extra)"
  t0=$(date +%s)
  "$W" $GEOM $extra --time "$T1" >"$o" 2>"$e" &
  CUR_PID=$!; wait "$CUR_PID"; CUR_PID=""
  dt=$(( $(date +%s) - t0 ))
  set -- $(parse "$HALF1" "$o"); best=$1; half=$2
  printf 'p1\t%s\t%s\t%s\t%s\n' "$tag" "$best" "$half" "$dt" >> "$RES"
  log "PHASE1 DONE  $tag  best=$best  half=$half  (${dt}s)"
  if [ "$best" -gt "$win_best" ]; then win_best=$best; win_tag=$tag; win_extra=$extra; fi
done
log "PHASE1 WINNER: $win_tag  (best=$win_best)  -> $win_extra"

# ---------- Phase 2 ----------
o="$OUT/p2_$win_tag.out"; e="$OUT/p2_$win_tag.err"
log "PHASE2 START $win_tag for ${T2}s"
t0=$(date +%s)
"$W" $GEOM $win_extra --time "$T2" >"$o" 2>"$e" &
CUR_PID=$!; wait "$CUR_PID"; CUR_PID=""
dt=$(( $(date +%s) - t0 ))
set -- $(parse 0 "$o"); best=$1
printf 'p2\t%s\t%s\t%s\t%s\n' "$win_tag" "$best" "0" "$dt" >> "$RES"
log "PHASE2 DONE  $win_tag  best=$best  (${dt}s)"

# ---------- Report ----------
log "===== DEEPEST PUZZLE (phase 2, depth $best) ====="
deepest_puzzle "$o" | tee -a "$LOG"
log "===== RESULTS ====="
column -t -s$'\t' "$RES" | tee -a "$LOG"
log "ALL DONE."
