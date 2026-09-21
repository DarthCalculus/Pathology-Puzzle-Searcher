#!/bin/bash
# Sweep --rollout params on the 6x6 test case. Stochastic search => multiple
# reps per config; report the depth list, max, and median.
set -u
cd "$(dirname "$0")"

BASE="--grid 6x6 --exit 8 --num-holes 0 --two-tables --allow-exit-transit"
TIME=10
REPS=3

median() {  # args: numbers
  printf '%s\n' "$@" | sort -n | awk '{a[NR]=$0} END{print (NR%2)?a[(NR+1)/2]:int((a[NR/2]+a[NR/2+1])/2)}'
}
mx() { printf '%s\n' "$@" | sort -n | tail -1; }

run_cfg() {  # $1 = rollout spec, $2 = metric flag ("" or --beam-score-tailwidth)
  local spec="$1" metric="$2" depths d
  depths=""
  for r in $(seq 1 $REPS); do
    d=$(./backsearch_worker $BASE --rollout "$spec" $metric --time $TIME 2>&1 \
        | grep -aiE "^Best depth" | grep -aoE "[0-9]+" | head -1)
    [ -z "$d" ] && d=0
    depths="$depths $d"
  done
  printf '%-22s %-22s | runs:%-20s max:%-4s med:%-4s\n' \
    "$spec" "${metric:-states_popped}" "$depths" "$(mx $depths)" "$(median $depths)"
}

echo "=== Phase 1: K x P x metric  (M=100 C=20, ${REPS} reps, ${TIME}s each) ==="
for K in 3 5 8; do
  for P in 0 10 25; do
    run_cfg "$K,100,20,$P" ""
    run_cfg "$K,100,20,$P" "--beam-score-tailwidth"
  done
done
echo "=== sweep done ==="
