#!/bin/bash
# Phase 3: confirm top configs at a longer budget; push M; recheck tailwidth
# and a longer K at the winning population size.
set -u
cd "$(dirname "$0")"

BASE="--grid 6x6 --exit 8 --num-holes 0 --two-tables --allow-exit-transit"
TIME=20
REPS=5

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$0} END{print (NR%2)?a[(NR+1)/2]:int((a[NR/2]+a[NR/2+1])/2)}'; }
mx() { printf '%s\n' "$@" | sort -n | tail -1; }

run_cfg() {  # $1 = K,M,C,P  $2 = metric flag ("" or --beam-score-tailwidth)
  local spec="$1" metric="$2" depths d
  depths=""
  for r in $(seq 1 $REPS); do
    d=$(./backsearch_worker $BASE --rollout "$spec" $metric --time $TIME 2>&1 \
        | grep -aiE "^Best depth" | grep -aoE "[0-9]+" | head -1)
    [ -z "$d" ] && d=0
    depths="$depths $d"
  done
  printf '%-18s %-22s | runs:%-26s max:%-4s med:%-4s\n' \
    "$spec" "${metric:-states_popped}" "$depths" "$(mx $depths)" "$(median $depths)"
}

echo "=== Phase 3: top configs @ ${TIME}s, ${REPS} reps ==="
run_cfg "8,400,40,10" ""
run_cfg "8,200,20,10" ""
run_cfg "8,800,40,10" ""
run_cfg "8,400,40,10" "--beam-score-tailwidth"
run_cfg "12,400,40,10" ""
run_cfg "8,400,40,25" ""
echo "=== sweep done ==="
