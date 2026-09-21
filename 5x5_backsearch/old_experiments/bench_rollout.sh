#!/bin/bash
# Benchmark a backsearch_worker invocation over N reps and report best-depth
# and dead-end-depth statistics.
#
# Usage:
#   ./bench_rollout.sh <reps> <label> -- <backsearch_worker args...>
#
# Example:
#   ./bench_rollout.sh 5 both -- --grid 6x6 --exit 8 --num-holes 0 --two-tables \
#       --allow-exit-transit --rollout 8,400,40,10 --rollout-pushoff \
#       --rollout-dfs-fill 800 --time 60
#
# WHY THIS SCRIPT EXISTS: everything after `--` is forwarded to the worker
# verbatim via "$@", so multi-token arguments can NEVER be mis-split.  Do not
# pack worker args into a shell string/variable and expand it unquoted — under
# zsh (Claude Code's Bash-tool shell) unquoted variables are NOT word-split, so
# `ARGS="--a --b"; worker $ARGS` passes the single bogus arg "--a --b".  Always
# either type args literally as separate tokens or route them through this
# script's "$@".  bash 3.2 compatible (macOS /bin/bash): no mapfile/declare -A.
set -u
cd "$(dirname "$0")"

if [ $# -lt 3 ]; then
  echo "usage: $0 <reps> <label> -- <worker args...>" >&2
  exit 2
fi
reps=$1; label=$2; shift 2
if [ "${1:-}" = "--" ]; then shift; fi

med() { printf '%s\n' $1 | sort -n | awk 'NF{a[NR]=$0} END{print (NR? a[int((NR+1)/2)] : "-")}'; }
mx()  { printf '%s\n' $1 | sort -n | awk 'NF{v=$0} END{print (NR? v : "-")}'; }

depths=""
deadends=""
tot_el=0      # summed worker-reported elapsed seconds across reps
tot_rs=0      # summed restart (episode) count across reps
for r in $(seq 1 "$reps"); do
  ./backsearch_worker "$@" --rollout-trace \
      >"/tmp/bench_${label}_$r.out" 2>"/tmp/bench_${label}_$r.err"
  d=$(grep -aiE 'best depth' "/tmp/bench_${label}_$r.out" | tail -1 | grep -aoE '[0-9]+' | head -1)
  [ -z "$d" ] && d=0
  depths="$depths $d"
  de=$(grep -aoE 'DEAD-END at depth [0-9]+' "/tmp/bench_${label}_$r.err" | grep -aoE '[0-9]+$')
  deadends="$deadends $de"
  # avg time/restart inputs: worker prints "elapsed: X.XXX s" and "restarts=N"
  el=$(grep -aoE 'elapsed: *[0-9.]+' "/tmp/bench_${label}_$r.out" | grep -aoE '[0-9.]+' | tail -1)
  rs=$(grep -aoE 'restarts=[0-9]+' "/tmp/bench_${label}_$r.err" | grep -aoE '[0-9]+' | tail -1)
  tot_el=$(awk -v a="$tot_el" -v b="${el:-0}" 'BEGIN{printf "%.3f", a+b}')
  tot_rs=$((tot_rs + ${rs:-0}))
done

spr=$(awk -v e="$tot_el" -v r="$tot_rs" 'BEGIN{printf "%.2f", (r>0 ? e/r : 0)}')
printf '%-10s reps=%s  best: [%s ] med=%s max=%s\n' \
  "$label" "$reps" "$depths" "$(med "$depths")" "$(mx "$depths")"
printf '%-10s deadend-depths: med=%s max=%s\n' "" "$(med "$deadends")" "$(mx "$deadends")"
printf '%-10s restarts=%s  elapsed=%ss  avg/restart=%ss\n' "" "$tot_rs" "$tot_el" "$spr"
