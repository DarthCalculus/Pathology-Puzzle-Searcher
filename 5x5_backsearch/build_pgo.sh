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
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- NN hook source: real libtorch or stub -----------------------------------
NN_OBJ=""; NN_LINK=""; CXX_LINK=cc
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
  cc -O3 -c "$TMP/nn_stub.c" -o "$TMP/nn_stub.o"
  NN_OBJ="$TMP/nn_stub.o"
fi

CFLAGS="-O3 -DGIT_SHA_STR=\"$SHA\""

# --- 1. instrumented build ---------------------------------------------------
echo "[1/3] instrumented build"
cc $CFLAGS -fprofile-instr-generate -c backsearch.c  -o "$TMP/bs.gen.o"
cc $CFLAGS -fprofile-instr-generate -c sokoban_bfs.c -o "$TMP/sb.gen.o"
$CXX_LINK -fprofile-instr-generate "$TMP/bs.gen.o" "$TMP/sb.gen.o" $NN_OBJ -o "$TMP/worker.gen" -lz $NN_LINK

# --- 2. training run ---------------------------------------------------------
echo "[2/3] training run: $TRAIN_ARGS"
# shellcheck disable=SC2086
LLVM_PROFILE_FILE="$TMP/w.profraw" "$TMP/worker.gen" $TRAIN_ARGS >/dev/null 2>&1 || true
PROFDATA=$(command -v llvm-profdata || echo "xcrun llvm-profdata")
$PROFDATA merge -output="$TMP/w.profdata" "$TMP/w.profraw"

# --- 3. optimised build ------------------------------------------------------
echo "[3/3] optimised build -> $OUT"
cc $CFLAGS -fprofile-instr-use="$TMP/w.profdata" -c backsearch.c  -o "$TMP/bs.o"
cc $CFLAGS -fprofile-instr-use="$TMP/w.profdata" -c sokoban_bfs.c -o "$TMP/sb.o"
$CXX_LINK "$TMP/bs.o" "$TMP/sb.o" $NN_OBJ -o "$OUT" -lz $NN_LINK
echo "built $OUT"
