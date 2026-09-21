#!/bin/bash
# P-schedule A/B: does a non-monotone percentile (keep branchy states in the
# early band gens, where record-depth levels sit high-percentile) beat flat P=10
# at equal M?  Base = the 240 regime (4,800,20,10 rg5).  Sequential, time-based.
# Records depth-over-time so we compare CURVES (crossing times), not just endpoint.
set -u
PROJ="/Users/george/Documents/Claude Projects/project45/5x5_backsearch"
cd "$PROJ" || { echo "cannot cd"; exit 1; }
W=./backsearch_worker
OUT=/tmp/pwin
mkdir -p "$OUT"
LOG="$OUT/summary.log"; RES="$OUT/results.tsv"
GEOM="--grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3 --fixedwalls 6,7"
BASE="--rollout 4,800,20,10 --rollout-rand-gens 5"
T=${T:-5400}     # seconds per arm (default 1.5h => 4.5h total)

ARMS=(
"control|"
"win40_50|--rollout-pct-window 40,50,90"
"win24_50|--rollout-pct-window 24,50,90"
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
      for(m=160;m<=240;m+=20){ if(d>=m && cross[m]==0) cross[m]=sec }
    }
    END{ printf "best=%d", best+0;
         for(m=160;m<=240;m+=20) if(cross[m]>0) printf "  %d@%.0fs", m, cross[m];
         printf "\n" }' "$1"
}

CUR_PID=""
cleanup(){ [ -n "$CUR_PID" ] && kill "$CUR_PID" 2>/dev/null; log "INTERRUPTED, killed $CUR_PID"; exit 130; }
trap cleanup INT TERM

: > "$LOG"
printf 'tag\tbest\tsecs\textra\n' > "$RES"
log "START P-schedule A/B  base: $BASE   ${T}s/arm"

for a in "${ARMS[@]}"; do
  tag="${a%%|*}"; extra="${a#*|}"
  o="$OUT/$tag.out"; e="$OUT/$tag.err"
  log "START $tag  (extra: ${extra:-none})"
  t0=$(date +%s)
  "$W" $GEOM $BASE $extra --time "$T" >"$o" 2>"$e" &
  CUR_PID=$!; wait "$CUR_PID"; CUR_PID=""
  dt=$(( $(date +%s) - t0 ))
  set -- $(awk '/^[0-9]+ \([0-9.]+[sm]\)$/{d=$1+0; if(d>b)b=d} END{print b+0}' "$o"); best=$1
  printf '%s\t%s\t%s\t%s\n' "$tag" "$best" "$dt" "${extra:-none}" >> "$RES"
  log "DONE  $tag  $(report "$o")  (${dt}s)"
done

log "===== RESULTS ====="
{ head -1 "$RES"; tail -n +2 "$RES" | sort -t$'\t' -k2,2nr; } | column -t -s$'\t' | tee -a "$LOG"
log "ALL DONE."
