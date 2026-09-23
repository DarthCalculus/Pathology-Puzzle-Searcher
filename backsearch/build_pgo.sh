#!/bin/bash
# Profile-guided build of backsearch_worker.
#
# The forward solver is ~90% of runtime and is branch-heavy; a PGO build is
# measurably faster (~10-20% on the solver replay benchmark) than plain -O3.
# This script:
#   1. builds an instrumented worker,
#   2. runs it briefly on a representative search to collect a profile,
#   3. rebuilds with that profile.
#
# Usage:
#   ./build_pgo.sh                 # -> ./backsearch_worker
#   ./build_pgo.sh -o mybinary     # custom output path
#   ./build_pgo.sh --no-torch      # force the no-NN stub even if libtorch is present
#   ./build_pgo.sh --train "--grid 6x6 --exit 0 --time 6"   # custom training run
#   CC=gcc-13 ./build_pgo.sh   # pick the compiler (clang and GCC PGO are both handled)
#
# libtorch: if nn_inference.o exists and `python3 -c 'import torch'` works,
# the NN hooks are linked against the pip-installed libtorch exactly as the
# README describes.  Otherwise a stub that reports "no model loaded" is
# linked, and --nn-* flags are inert.
set -euo pipefail
cd "$(dirname "$0")"

OUT=./backsearch_worker
TRAIN_ARGS="--grid 5x5 --exit 0 --time 4"
USE_TORCH=auto
while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT="$2"; shift 2 ;;
    --no-torch) USE_TORCH=no; shift ;;
    --torch) USE_TORCH=yes; shift ;;
    --train) TRAIN_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "")
# Source hash for the v2 job protocol (v2/DESIGN.md): sha256 of the search sources,
# independent of compiler/PGO so every honest build of the same code agrees.
SRC_HASH=$(cat backsearch.c sokoban_bfs.c sokoban_bfs.h | { shasum -a 256 2>/dev/null || sha256sum; } | cut -c1-64)
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

CFLAGS="-O3 -DGIT_SHA_STR=\"$SHA\" -DSRC_HASH_STR=\"$SRC_HASH\""
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
echo "[1/3] instrumented build"
$CC $CFLAGS $GEN -c backsearch.c  -o "$TMP/bs.o"
$CC $CFLAGS $GEN -c sokoban_bfs.c -o "$TMP/sb.o"
$CXX_LINK $GEN "$TMP/bs.o" "$TMP/sb.o" $NN_OBJ -o "$TMP/worker.gen" $LIBS $NN_LINK

# --- 2. training run ---------------------------------------------------------
echo "[2/3] training run: $TRAIN_ARGS"
# shellcheck disable=SC2086
LLVM_PROFILE_FILE="$TMP/w.profraw" "$TMP/worker.gen" $TRAIN_ARGS >/dev/null 2>&1 || true
if [ "$FAMILY" = clang ]; then $PROFDATA merge -output="$TMP/w.profdata" "$TMP/w.profraw"; fi   # gcc reads $TMP/*.gcda directly

# --- 3. optimised build (same object paths as stage 1, see above) -------------
echo "[3/3] optimised build -> $OUT"
$CC $CFLAGS $USE -c backsearch.c  -o "$TMP/bs.o"
$CC $CFLAGS $USE -c sokoban_bfs.c -o "$TMP/sb.o"
$CXX_LINK "$TMP/bs.o" "$TMP/sb.o" $NN_OBJ -o "$OUT" $LIBS $NN_LINK
echo "built $OUT"
