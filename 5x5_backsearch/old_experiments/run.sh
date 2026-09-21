#!/bin/bash
# Single entry point for every backsearch_worker invocation.
#
# WHY THIS EXISTS: the Bash tool's shell is zsh, which does NOT word-split an
# unquoted variable expansion. Packing multi-token args into one $VAR and
# writing `./worker $VAR` passes the whole string as a SINGLE argv, which the C
# parser correctly rejects as `unknown arg`. This script removes the failure
# mode entirely: geometry lives in real bash arrays here, and every caller adds
# only literal, individually-quoted tokens after a preset name. Nothing is ever
# re-split.
#
# USAGE:
#   bash run.sh <preset> [extra worker args...]
#
#   <preset> selects a geometry array (see below). Everything after it is
#   forwarded verbatim to the worker via "$@" (array expansion, never re-split).
#
# EXAMPLES:
#   bash run.sh fw67 --rollout 8,400,40,10 --rollout-restarts 25 --time 3600
#   bash run.sh fw67 --rollout 8,400,40,10 --rollout-rand-gens 3 --rollout-trace
#   bash run.sh raw  --grid 6x6 --two-tables --exit 0        # no preset geometry
#
set -u
cd "$(dirname "$0")"

WORKER="${WORKER:-./backsearch_worker2}"

preset="${1:-}"
if [ -z "$preset" ]; then
  echo "usage: bash run.sh <preset> [worker args...]" >&2
  echo "presets: fw67, base, raw" >&2
  exit 2
fi
shift

# --- Geometry presets (each is a real array; tokens stay separate) -----------
case "$preset" in
  fw67)
    # 6x6, two-tables, holes, seeded path, fixed walls at 6,7 (the sweep geom).
    GEOM=(--grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3 --fixedwalls 6,7)
    ;;
  base)
    # 6x6, two-tables, holes, seeded path (no fixed walls).
    GEOM=(--grid 6x6 --two-tables --exit 0 --fixedholes 1 --seed-path L1,L1,L3)
    ;;
  raw)
    # No preset geometry; caller supplies everything.
    GEOM=()
    ;;
  *)
    echo "unknown preset: $preset (known: fw67, base, raw)" >&2
    exit 2
    ;;
esac

if [ ! -x "$WORKER" ]; then
  echo "worker not found/executable: $WORKER" >&2
  exit 3
fi

# Array expansions keep every token separate -- no word-splitting possible.
# Set RUN_TRACE=1 to echo the exact argv (useful when debugging invocations).
[ "${RUN_TRACE:-0}" = 1 ] && set -x
exec "$WORKER" "${GEOM[@]}" "$@"
