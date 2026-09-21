#!/bin/bash
# M-scaling A/B: isolate population size M at hour-scale on fw67.
# The depth-240 result confounded M(800 vs 400), C(20 vs 40), rg(5 vs 4).
# Here we hold K=4, C=20, P=10, rg5 FIXED and vary ONLY M ∈ {400,800,1600},
# equal long time per arm, recording depth-over-time so we compare CURVES.
# Tests the "large M sustains the deep tail; small M flattens" hypothesis.
set -u
PROJ="/Users/george/Documents/Claude Projects/project45/5x5_backsearch"
cd "$PROJ" || { echo "cannot cd"; exit 1; }
W=./backsearch_worker
OUT=/tmp/mscale
mkdir -p "$OUT"
LOG="$OUT/summary.log"; RES="$OUT/results.tsv"
GEOM="--grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3 --fixedwalls 6,7"
T=${T:-10800}     # seconds per arm (default 3h => 9h total)

ARMS=(
"M400|--rollout 4,400,20,10 --rollout-rand-gens 5"
"M800|--rollout 4,800,20,10 --rollout-rand-gens 5"
"M1600|--rollout 4,1600,20,10 --rollout-rand-gens 5"
)

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(stamp)] $*" | tee -a "$LOG"; }
# print best depth + crossing times (first time each milestone hit)
report(){
  awk '
    /^[0-9]+ \([0-9.]+[sm]\)$/ {
      d=$1+0; t=$2; gsub(/[()]/,"",t);
      u=substr(t,length(t),1); v=substr(t,1,length(t)-1)+0; sec=(u=="m")?v*60:v;
      if(d>best)best=d;
      for(m=180;m<=260;m+=20){ if(d>=m && cross[m]==0) cross[m]=sec }
    }
    END{ printf "best=%d", best+0;
         for(m=180;m<=260;m+=20) if(cross[m]>0) printf "  %d@%.0fs", m, cross[m];
         printf "\n" }' "$1"
}

CUR_PID=""
cleanup(){ [ -n "$CUR_PID" ] && kill "$CUR_PID" 2>/dev/null; log "INTERRUPTED, killed $CUR_PID"; exit 130; }
trap cleanup INT TERM

: > "$LOG"
printf 'tag\tbest\tsecs\textra\n' > "$RES"
log "START M-scaling A/B  (fixed K4 C20 P10 rg5; vary M)  ${T}s/arm"

for a in "${ARMS[@]}"; do
  tag="${a%%|*}"; extra="${a#*|}"
  o="$OUT/$tag.out"; e="$OUT/$tag.err"
  log "START $tag  (extra: ${extra})"
  t0=$(date +%s)
  "$W" $GEOM $extra --time "$T" >"$o" 2>"$e" &
  CUR_PID=$!; wait "$CUR_PID"; CUR_PID=""
  dt=$(( $(date +%s) - t0 ))
  set -- $(awk '/^[0-9]+ \([0-9.]+[sm]\)$/{d=$1+0; if(d>b)b=d} END{print b+0}' "$o"); best=$1
  printf '%s\t%s\t%s\t%s\n' "$tag" "$best" "$dt" "${extra}" >> "$RES"
  log "DONE  $tag  $(report "$o")  (${dt}s)"
done

log "===== RESULTS ====="
{ head -1 "$RES"; tail -n +2 "$RES" | sort -t$'\t' -k2,2nr; } | column -t -s$'\t' | tee -a "$LOG"
log "ALL DONE."
