#!/bin/bash
# Automated proving loop (real rules), one exit at a time, easiest first.
#   1. For every class in results/prove_queue.txt ("B H", x = unconstrained) and
#      every canonical exit, size the (class, exit) tree with the estimator
#      (dedup-free lower bound; ~7x low on past campaigns) unless it is already
#      marked DONE in results/prove_loop.log.  Estimates go to
#      results/prove_estimates.tsv (class exit est_h) -- edit to reorder.
#   2. Repeatedly pick the cheapest undone (class, exit), run it as its own
#      campaign (results/camp_b{B}h{H}_e{E}, resumable, WORKERS workers), publish
#      its proof + champion immediately, mark it DONE, repeat.
# Stop with: touch results/prove_loop.stop   (takes effect between jobs).
# Re-estimate everything: rm results/prove_estimates.tsv before restarting.
cd "$(dirname "$0")/.."
LOG=results/prove_loop.log; EST=results/prove_estimates.tsv; WORKERS=${WORKERS:-2}
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
# done, or covered by a finished (class, exit) with the same exit and caps at
# least as loose (x = unconstrained): its proof already bounds this class.
is_done() {
  local tag=$1 e=$2 B H B2 H2 t2
  grep -q " DONE $tag e$e:" "$LOG" 2>/dev/null && return 0
  B=${tag#b}; B=${B%%_*}; H=${tag##*_h}
  for t2 in $(grep -oE " DONE b[0-9x]+_h[0-9x]+ e$e:" "$LOG" 2>/dev/null | awk '{print $2}' | sort -u); do
    B2=${t2#b}; B2=${B2%%_*}; H2=${t2##*_h}
    { [ "$B2" = x ] || { [ "$B" != x ] && [ "$B2" -ge "$B" ]; }; } && { [ "$H2" = x ] || { [ "$H" != x ] && [ "$H2" -ge "$H" ]; }; } || continue
    # the superclass proof bounds this class; it settles it only if one of its
    # champions lies inside this class (same maximum, witnessed)
    w=$(python3 results/covered_by_witness.py "$B" "$H" "$e" "$B2" "$H2") && { echo "$(date '+%F %T') COVERED $tag e$e by $t2: $w" >> "$LOG"; return 0; }
    echo "$(date '+%F %T') not covered: $tag e$e vs $t2: $w -- running it" >> "$LOG"
  done
  return 1
}
extra_for() { local B=$1 H=$2 x=""; [ "$B" != x ] && x="$x --num-blocks $B"; [ "$H" != x ] && x="$x --num-holes $H"; echo "$x"; }
log "loop start (workers $WORKERS, one exit at a time, easiest first)"
while pgrep -f 'campaign.py --out' >/dev/null; do sleep 30; done   # an earlier loop's campaign may still be finishing
if [ ! -f "$EST" ]; then
  while read -r B H; do
    [ -z "$B" ] && continue; tag="b${B}_h${H}"; extra=$(extra_for $B $H)
    for e in 0 1 2 6 7 12; do
      is_done $tag $e && continue
      h=$(./backsearch_worker_nt --grid 5x5 --two-tables --allow-exit-transit --exit $e $extra --estimate 1500000 --estimate-depth 10 2>&1 | grep 'est. DFS time' | sed -E 's/.*= ([0-9.]+) h .*/\1/')
      echo -e "$tag\t$e\t${h:-0}\t1" >> "$EST"   # 4th column: priority (0 runs before 1), then by estimate
    done
    log "$tag estimates (h, dedup-free): $(grep "^$tag" "$EST" | awk '{printf "e%s:%s ", $2, $3}')"
  done < results/prove_queue.txt
fi
while true; do
  [ -f results/prove_loop.stop ] && { log "stop file seen"; exit 0; }
  pick=""
  while IFS=$'\t' read -r tag e h pr; do
    is_done $tag $e && continue
    pick="$tag $e $h"; break
  done < <(sort -t$'\t' -k4,4n -k3,3g "$EST")
  [ -z "$pick" ] && { log "nothing left to run"; exit 0; }
  set -- $pick; tag=$1; e=$2; h=$3; B=${tag#b}; B=${B%%_*}; H=${tag##*_h}; extra=$(extra_for $B $H)
  dir=results/camp_${tag}_e$e
  log "START $tag e$e (est ${h} h dedup-free, ~$(python3 -c "print(round($h*7,1))") CPU-h)"
  python3 campaign.py --out $dir --exits $e --extra "$extra" --layer 8 --workers $WORKERS --worker ./backsearch_worker_nt --shuffle >> $dir.out 2>&1
  pub=""; [ "$B" != x ] && pub="$pub --max-blocks $B"; [ "$H" != x ] && pub="$pub --max-holes $H"
  if python3 results/publish_campaign.py $dir $pub >> ${dir}_publish.out 2>&1; then
    log "DONE $tag e$e: $(grep maxima $dir/publish.log | tail -1 | sed -E 's/.*maxima: //')  cpu $(python3 campaign.py --out $dir --status 2>/dev/null | head -1 | sed -E 's/.*cpu ([^ ]+).*/\1/')"
  else
    log "PUBLISH FAILED $tag e$e (campaign not exhausted?) -- retry in 10 min"; sleep 600
  fi
  [ -f results/prove_loop.stop ] && { log "stop file seen after $tag e$e"; exit 0; }
done
