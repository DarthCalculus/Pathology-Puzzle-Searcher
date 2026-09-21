#!/bin/bash
# bf-target A/B: does selecting rollout candidates by MATCHING the 240 level's
# absolute branch_factor per depth beat flat P=10 at equal M?  This is the
# decisive test of the "force branchiness that matches the 240" hypothesis.
# Base = the 240 regime (4,800,20,10 rg5).  Sequential, time-based.
# Two bf-target arms: W=0 (raw depth-40 cliff) and W=10 (smoothed ramp).
# Records depth-over-time so we compare CURVES (crossing times), not endpoint.
set -u
cd "$(dirname "$0")" || { echo "cannot cd"; exit 1; }
W=./backsearch_worker
CSV=/tmp/p240_path.csv
OUT=/tmp/bftarget
mkdir -p "$OUT"
LOG="$OUT/summary.log"; RES="$OUT/results.tsv"
T=${T:-10800}     # seconds per arm (default 3h => 9h total)

# GEOM + BASE inlined per-arm below (no quoted-var word-split under zsh).
ARMS=(
"control|"
"bft_w0|--rollout-bf-target $CSV --rollout-bf-smooth 0"
"bft_w10|--rollout-bf-target $CSV --rollout-bf-smooth 10"
)

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(stamp)] $*" | tee -a "$LOG"; }
report(){
  awk '
    /^[0-9]+ \([0-9.]+[sm]\)$/ {
      d=$1+0; t=$2; gsub(/[()]/,"",t);
      u=substr(t,length(t),1); v=substr(t,1,length(t)-1)+0; sec=(u=="m")?v*60:v;
      if(d>best)best=d;
      for(m=160;m<=260;m+=20){ if(d>=m && cross[m]==0) cross[m]=sec }
    }
    END{ printf "best=%d", best+0;
         for(m=160;m<=260;m+=20) if(cross[m]>0) printf "  %d@%.0fs", m, cross[m];
         printf "\n" }' "$1"
}

CUR_PID=""
cleanup(){ [ -n "$CUR_PID" ] && kill "$CUR_PID" 2>/dev/null; log "INTERRUPTED, killed $CUR_PID"; exit 130; }
trap cleanup INT TERM

: > "$LOG"
printf 'tag\tbest\tsecs\textra\n' > "$RES"
log "START bf-target A/B  base: 4,800,20,10 rg5   ${T}s/arm"

for a in "${ARMS[@]}"; do
  tag="${a%%|*}"; extra="${a#*|}"
  o="$OUT/$tag.out"; e="$OUT/$tag.err"
  log "START $tag  (extra: ${extra:-none})"
  t0=$(date +%s)
  "$W" --grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3 \
       --fixedwalls 6,7 --rollout 4,800,20,10 --rollout-rand-gens 5 \
       $extra --time "$T" >"$o" 2>"$e" &
  CUR_PID=$!; wait "$CUR_PID"; CUR_PID=""
  dt=$(( $(date +%s) - t0 ))
  set -- $(awk '/^[0-9]+ \([0-9.]+[sm]\)$/{d=$1+0; if(d>b)b=d} END{print b+0}' "$o"); best=$1
  printf '%s\t%s\t%s\t%s\n' "$tag" "$best" "$dt" "${extra:-none}" >> "$RES"
  log "DONE  $tag  $(report "$o")  (${dt}s)"
done

log "===== RESULTS ====="
{ head -1 "$RES"; tail -n +2 "$RES" | sort -t$'\t' -k2,2nr; } | column -t -s$'\t' | tee -a "$LOG"
log "ALL DONE."
