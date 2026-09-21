#!/bin/bash
# Phase 2: sweep M (population) x C (children) at the phase-1 winner
# K=8, P=10, states_popped. 5 reps, 10s each.
set -u
cd "$(dirname "$0")"

BASE="--grid 6x6 --exit 8 --num-holes 0 --two-tables --allow-exit-transit"
TIME=10
REPS=5

median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$0} END{print (NR%2)?a[(NR+1)/2]:int((a[NR/2]+a[NR/2+1])/2)}'; }
mx() { printf '%s\n' "$@" | sort -n | tail -1; }

run_cfg() {  # $1 = K,M,C,P
  local spec="$1" depths d
  depths=""
  for r in $(seq 1 $REPS); do
    d=$(./backsearch_worker $BASE --rollout "$spec" --time $TIME 2>&1 \
        | grep -aiE "^Best depth" | grep -aoE "[0-9]+" | head -1)
    [ -z "$d" ] && d=0
    depths="$depths $d"
  done
  printf '%-18s | runs:%-26s max:%-4s med:%-4s\n' "$spec" "$depths" "$(mx $depths)" "$(median $depths)"
}

echo "=== Phase 2: M x C  (K=8 P=10 states_popped, ${REPS} reps, ${TIME}s) ==="
for M in 50 100 200 400; do
  for C in 10 20 40; do
    run_cfg "8,$M,$C,10"
  done
done
echo "=== sweep done ==="
