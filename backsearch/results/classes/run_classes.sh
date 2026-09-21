#!/bin/bash
# Exhaust small 5x5 classes (block cap B, hole cap H; "x" = unconstrained), one
# monolithic worker per (class, exit), 3 at a time.  Real rules (transit on).
# Logs: results/classes/b{B}_h{H}_e{E}.log ; finished logs are skipped, so rerun to resume.
cd "$(dirname "$0")/../.."
CLASSES="${CLASSES:-0_x 1_0 1_x 2_0 2_1 2_x 3_0 3_1 3_2 3_x 4_0 4_1 4_2 x_0}"
P="${P:-3}"
jobs=""
for c in $CLASSES; do for e in 0 1 2 6 7 12; do
  f=results/classes/b${c%_*}_h${c#*_}_e$e.log
  grep -q "^--- Exit" "$f" 2>/dev/null || jobs="$jobs $c:$e"
done; done
echo "$(date '+%F %T') running $(echo $jobs | wc -w) (class,exit) jobs on $P workers" >> results/classes/run.log
printf '%s\n' $jobs | xargs -P "$P" -I{} bash -c '
  c="${1%:*}"; e="${1#*:}"; B="${c%_*}"; H="${c#*_}"
  args="--grid 5x5 --two-tables --allow-exit-transit --exit $e --time 0"
  [ "$B" != x ] && args="$args --num-blocks $B"
  [ "$H" != x ] && args="$args --num-holes $H"
  f=results/classes/b${B}_h${H}_e$e.log
  ./backsearch_worker_nt $args > "$f.part" 2>&1 && mv "$f.part" "$f"
  echo "$(date "+%F %T") done b$B h$H e$e: $(grep -A7 "^--- Exit" "$f" | grep "elapsed\|best depth" | tr -s " " | tr "\n" " ")" >> results/classes/run.log
' _ {}
echo "$(date '+%F %T') all done" >> results/classes/run.log
