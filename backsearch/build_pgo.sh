#!/bin/bash
# Profile-guided build of backsearch_worker.
#
# The forward solver is most of the runtime and is branch-heavy; a PGO build
# is measurably faster than plain -O3.  This script:
#   1. builds an instrumented worker,
#   2. runs it on two campaign-shaped jobs (the campaign's own flags, protocol
#      mode, a fixed amount of work: a few seconds each) to collect a profile:
#      a 5x5 <=3-hole layer-4 root with exit transit (block-on-exit states, the
#      64-bit solver instance) and a 6x6 0-hole subtree whose checks are about
#      a third wider than 64 bits (10+ blocks: the 16-block-byte instance), so
#      no campaign hot path is compiled as cold code,
#   3. checks that both runs completed and that the profile is not empty, and
#      (clang) that both solver instances ran: code the training never runs is
#      optimised for size, which measured 27% SLOWER than plain -O3 on heavy
#      6x6 jobs when a 6x6 training run never reached the wide instance,
#   4. rebuilds with that profile.
# The binary is portable (no -march/-mcpu=native: a binary copied to another
# CPU must not die with SIGILL).  NATIVE=1 opts in to the host's ISA.
#
# Usage:
#   ./build_pgo.sh                 # -> ./backsearch_worker
#   ./build_pgo.sh -o mybinary     # custom output path
#   ./build_pgo.sh --no-torch      # force the no-NN stub even if libtorch is present
#   ./build_pgo.sh --train "--grid 6x6 --exit 0 --time 6"   # custom first training run
#   ./build_pgo.sh --train2 ""     # skip the second (6x6) training run
#   CC=gcc-13 ./build_pgo.sh   # pick the compiler (clang and GCC PGO are both handled)
#   NATIVE=1 ./build_pgo.sh    # also -mcpu=native (arm64) / -march=native (x86): this machine only
#   KNOBS="-DHP64_SIZE=(1<<9) -DPATH_TOK_MAX=16" ./build_pgo.sh -o /tmp/knob_worker
#                              # a "knob build": compile-time table-size overrides (tests)
#   ./build_pgo.sh --print-hash      # print the SRC_HASH this build would carry, then exit
#
# SRC_HASH (the v2 protocol's build identity, v2/PROTOCOL3.md §2.1) is the sha256
# of backsearch.c + sokoban_bfs.c + sokoban_bfs.h, PLUS the sorted list of KNOBS
# overrides when any are given:
#   no KNOBS : sha256(cat backsearch.c sokoban_bfs.c sokoban_bfs.h)
#   KNOBS    : sha256(the same bytes, then "\0KNOBS\0", then the -D tokens sorted
#              (LC_ALL=C) one per line, each followed by "\n")
# so a knob build can never carry (and be whitelisted under) the release hash.
# -D/-U/-O/-f flags smuggled in through CC are refused: they would change the
# search without changing the hash.  The binary's --version prints the effective
# knob values (KNOBS line).
#
# libtorch: if nn_inference.o exists and `python3 -c 'import torch'` works,
# the NN hooks are linked against the pip-installed libtorch exactly as the
# README describes.  Otherwise a stub that reports "no model loaded" is
# linked, and --nn-* flags are inert.
set -euo pipefail
cd "$(dirname "$0")"

OUT=./backsearch_worker
# Training runs: campaign flags (always --allow-exit-transit), protocol mode
# (--status-every), a fixed number of expansions (--split-after-nodes).
TRAIN_ARGS="--grid 5x5 --two-tables --allow-exit-transit --num-holes 3 --exit 7 --seed-path D2,R1,U1,U1 --time 0 --split-after-nodes 500000 --status-every 1000"
TRAIN2_ARGS="--grid 6x6 --two-tables --allow-exit-transit --num-holes 0 --exit 14 --seed-path R2,D2,L2,L2,L2,L2,U2,U2,R1,U2,R2,R1 --time 0 --status-every 1000"
USE_TORCH=auto
PRINT_HASH=0
while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    --no-torch) USE_TORCH=no; shift ;;
    --torch) USE_TORCH=yes; shift ;;
    --train) TRAIN_ARGS="$2"; shift 2 ;;
    --train2) TRAIN2_ARGS="$2"; shift 2 ;;
    --print-hash) PRINT_HASH=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- knob overrides (KNOBS="-DNAME=VALUE ...") ---------------------------------
KNOBS=${KNOBS:-}
for k in $KNOBS; do
  case "$k" in
    -D[A-Za-z_]*) ;;
    *) echo "KNOBS may only contain -DNAME or -DNAME=VALUE tokens (got '$k')" >&2; exit 2 ;;
  esac
done
case " ${CC:-cc} " in
  *" -D"*|*" -U"*|*" -O"*|*" -f"*|*" -W"*|*" -m"*|*" -include"*)
    echo "CC='${CC:-}' carries compiler flags: put compile-time overrides in KNOBS=\"-D...\" (they are folded into SRC_HASH)" >&2; exit 2 ;;
esac

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "")
# Source hash for the v2 job protocol (v2/PROTOCOL3.md §2.1): sha256 of the search
# sources (plus the sorted knob overrides, if any), independent of compiler/PGO so
# every honest build of the same code agrees.
sha256_stdin() { { shasum -a 256 2>/dev/null || sha256sum; } | cut -c1-64; }
if [ -z "$KNOBS" ]; then
  SRC_HASH=$(cat backsearch.c sokoban_bfs.c sokoban_bfs.h | sha256_stdin)
else
  # shellcheck disable=SC2086
  SORTED_KNOBS=$(printf '%s\n' $KNOBS | LC_ALL=C sort)
  SRC_HASH=$( { cat backsearch.c sokoban_bfs.c sokoban_bfs.h; printf '\0KNOBS\0'; printf '%s\n' "$SORTED_KNOBS"; } | sha256_stdin)
fi
if [ "$PRINT_HASH" = 1 ]; then echo "$SRC_HASH"; exit 0; fi
[ -n "$KNOBS" ] && echo "knob build: $KNOBS (SRC_HASH $SRC_HASH)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- NN hook source: real libtorch or stub -----------------------------------
CC=${CC:-cc}
NN_OBJ=""; NN_LINK=""; CXX_LINK=$CC   # link with the same compiler driver (its PGO runtime must match the objects)
if [ "$USE_TORCH" != no ]; then
  TORCH_DIR=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null || true)
  if [ -n "$TORCH_DIR" ] && [ -d "$TORCH_DIR/lib" ]; then
    echo "libtorch: $TORCH_DIR"
    c++ -O3 -std=c++17 -c nn_inference.cpp -o "$TMP/nn_inference.o" \
        -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include"
    NN_OBJ="$TMP/nn_inference.o"
    NN_LINK="-L$TORCH_DIR/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$TORCH_DIR/lib"
    CXX_LINK=c++
  elif [ "$USE_TORCH" = yes ]; then
    echo "libtorch requested but not found (pip install torch)" >&2; exit 1
  fi
fi
if [ -z "$NN_OBJ" ]; then
  echo "libtorch: not linked (NN flags inert)"
  cat > "$TMP/nn_stub.c" <<'EOF'
int   nn_load(const char *p, float s, int r, int c, int ch) { (void)p;(void)s;(void)r;(void)c;(void)ch; return 0; }
float nn_score(const float *f) { (void)f; return 0.f; }
void  nn_score_batch(const float *f, int n, float *o) { (void)f; for (int i=0;i<n;i++) o[i]=0.f; }
void  nn_close(void) {}
int   nn_surrogate_load(const char *p, float s, int r, int c, int ch) { (void)p;(void)s;(void)r;(void)c;(void)ch; return 0; }
float nn_surrogate_score(const float *f) { (void)f; return 0.f; }
void  nn_surrogate_score_batch(const float *f, int n, float *o) { (void)f; for (int i=0;i<n;i++) o[i]=0.f; }
void  nn_surrogate_close(void) {}
EOF
  ${CC:-cc} -O3 -c "$TMP/nn_stub.c" -o "$TMP/nn_stub.o"
  NN_OBJ="$TMP/nn_stub.o"
fi

ARCHFLAGS=""
if [ "${NATIVE:-0}" = 1 ]; then
  case "$(uname -m)" in arm64|aarch64) ARCHFLAGS="-mcpu=native" ;; *) ARCHFLAGS="-march=native" ;; esac
  if ! echo 'int main(void){return 0;}' | ${CC:-cc} -x c - $ARCHFLAGS -o "$TMP/archtest" 2>/dev/null; then
    echo "NATIVE=1: the compiler does not accept $ARCHFLAGS" >&2; exit 1
  fi
  echo "NATIVE=1: $ARCHFLAGS (this binary may not run on other CPUs)"
fi
CFLAGS="-O3 $ARCHFLAGS $KNOBS -DGIT_SHA_STR=\"$SHA\" -DSRC_HASH_STR=\"$SRC_HASH\""
LIBS="-lz -lm"

# --- compiler family: clang (macOS, or clang on Linux) vs GCC ------------------
# clang: -fprofile-instr-generate/-use with an llvm-profdata merge step.
# gcc:   -fprofile-generate/-use; the .gcda files are named after the OBJECT
#        path, so both stages must compile to the same object paths.
if $CC --version 2>/dev/null | grep -qi clang; then FAMILY=clang; else FAMILY=gcc; fi
echo "compiler: $($CC --version | head -1) [$FAMILY PGO]"

if [ "$FAMILY" = clang ]; then
  PROFDATA=$(command -v llvm-profdata || (command -v xcrun >/dev/null && echo "xcrun llvm-profdata") || ls /usr/bin/llvm-profdata-* /usr/lib/llvm-*/bin/llvm-profdata 2>/dev/null | sort -V | tail -1)
  if [ -z "$PROFDATA" ]; then
    echo "llvm-profdata not found: falling back to a plain -O3 build (install llvm tools for PGO)" >&2
    $CC $CFLAGS -c backsearch.c  -o "$TMP/bs.o"
    $CC $CFLAGS -c sokoban_bfs.c -o "$TMP/sb.o"
    $CXX_LINK "$TMP/bs.o" "$TMP/sb.o" $NN_OBJ -o "$OUT" $LIBS $NN_LINK
    echo "built $OUT (no PGO)"; exit 0
  fi
  GEN="-fprofile-instr-generate"; USE="-fprofile-instr-use=$TMP/w.profdata"
else
  GEN="-fprofile-generate"; USE="-fprofile-use -fprofile-correction -Wno-missing-profile"
fi

# --- 1. instrumented build ---------------------------------------------------
echo "[1/4] instrumented build"
$CC $CFLAGS $GEN -c backsearch.c  -o "$TMP/bs.o"
$CC $CFLAGS $GEN -c sokoban_bfs.c -o "$TMP/sb.o"
$CXX_LINK $GEN "$TMP/bs.o" "$TMP/sb.o" $NN_OBJ -o "$TMP/worker.gen" $LIBS $NN_LINK

# --- 2. training runs --------------------------------------------------------
# Each must exit 0 and end with a SUMMARY of a completed run (status split or
# exhausted); anything else fails the build instead of producing a build
# trained on nothing.
train() {   # train N ARGS
  local n=$1; shift
  echo "      training run $n: $*"
  local rc=0
  LLVM_PROFILE_FILE="$TMP/w$n.profraw" "$TMP/worker.gen" "$@" > "$TMP/train$n.out" 2> "$TMP/train$n.err" || rc=$?
  if [ $rc -ne 0 ] || ! grep -Eq '^SUMMARY.*"status":"(split|exhausted)"' "$TMP/train$n.out"; then
    echo "training run $n failed (exit $rc): $*" >&2
    tail -5 "$TMP/train$n.err" >&2; grep '^SUMMARY' "$TMP/train$n.out" >&2 || true
    exit 1
  fi
}
echo "[2/4] training runs"
# shellcheck disable=SC2086
train 1 $TRAIN_ARGS
# shellcheck disable=SC2086
if [ -n "$TRAIN2_ARGS" ]; then train 2 $TRAIN2_ARGS; fi

# --- 3. the profile must not be empty ------------------------------------------
echo "[3/4] checking the profile"
if [ "$FAMILY" = clang ]; then
  # clang merges the raw profiles
  $PROFDATA merge -output="$TMP/w.profdata" "$TMP"/w*.profraw
  MAXC=$( { $PROFDATA show "$TMP/w.profdata" 2>/dev/null || true; } | sed -n 's/^Maximum function count: *\([0-9]*\).*/\1/p' | head -1 || true)
  if [ -z "$MAXC" ] || [ "$MAXC" -lt 100000 ]; then
    echo "the training profile is empty or nearly so (maximum function count '${MAXC:-?}'): refusing to build" >&2; exit 1
  fi
  echo "      maximum function count $MAXC"
  # Code the training never ran is compiled for size: the solver instances the
  # training runs are meant to cover must have run (the second run is the one
  # that reaches the 16-block-byte instance; skip it with --train2 "").
  fcount() { { $PROFDATA show --function="$1" --counts "$TMP/w.profdata" 2>/dev/null || true; } | sed -n 's/^ *Function count: *\([0-9]*\).*/\1/p' | head -1 || true; }
  NEED="solve_push_cutoff_w1 solve_multi_w1"
  if [ -n "$TRAIN2_ARGS" ]; then NEED="$NEED solve_push_cutoff_w2"; fi
  for f in $NEED; do
    c=$(fcount "$f")
    if [ -z "$c" ]; then echo "      (profile check: no count found for $f; skipped)"; continue; fi
    if [ "$c" -eq 0 ]; then
      echo "the training runs never reached $f: its code would be compiled as cold (choose a training run that does)" >&2; exit 1
    fi
    echo "      $f: $c calls"
  done
else
  # gcc accumulates both runs in $TMP/*.gcda and reads them directly
  if ! ls "$TMP"/*.gcda >/dev/null 2>&1 || [ -z "$(find "$TMP" -name '*.gcda' -size +0c)" ]; then
    echo "the training runs left no profile data (.gcda): refusing to build" >&2; exit 1
  fi
fi

# --- 4. optimised build (same object paths as stage 1, see above) -------------
echo "[4/4] optimised build -> $OUT"
$CC $CFLAGS $USE -c backsearch.c  -o "$TMP/bs.o"
$CC $CFLAGS $USE -c sokoban_bfs.c -o "$TMP/sb.o"
$CXX_LINK "$TMP/bs.o" "$TMP/sb.o" $NN_OBJ -o "$OUT" $LIBS $NN_LINK
echo "built $OUT"
