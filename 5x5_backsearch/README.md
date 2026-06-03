# 5x5_backsearch

Backward-DFS Sokoban puzzle generator with task-partitioned parallelism.

## What's here

| File | Role |
|---|---|
| `backsearch.c` | Worker source: backward DFS, dedup, shortcut pruning, partition enumeration |
| `sokoban_bfs.c` / `sokoban_bfs.h` | Forward solver used by the worker for shortcut checks and verification |
| `backsearch` | Bash wrapper: auto-partitions the search and runs workers in parallel |
| `_backsearch_filter.py` | Stream filter the wrapper pipes worker output through |
| `harvest_format.h` | On-disk record layout for `--harvest` binary logs |
| `harvest_load.py` | Post-processor: reads a `--harvest` log, computes derived stats, writes SQLite |

## Building

The C code uses `clock_gettime(CLOCK_MONOTONIC, ...)` — a POSIX call that isn't available in plain MSVC. Builds need either a POSIX-compatible toolchain (macOS, Linux, MinGW, MSYS2, WSL) or a small port of the timing functions.

### macOS

The worker is the only piece that needs compilation. The wrapper script and Python filter run as-is.

```bash
cd 5x5_backsearch
SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
cc -O3 -DGIT_SHA_STR=\"$SHA\" -o backsearch_worker backsearch.c sokoban_bfs.c -lz
```

`cc` on macOS is Apple Clang. The only external dep is zlib, which ships with Xcode/macOS. `GIT_SHA_STR` is embedded into `--harvest` logs so we can trace data back to a build; the build still works if it's empty.

For a debug build with symbols:

```bash
cc -O0 -g -DGIT_SHA_STR=\"$SHA\" -o backsearch_dbg backsearch.c sokoban_bfs.c -lz
```

### Windows

Three options, easiest to hardest:

**Option A — WSL (recommended).** Install Windows Subsystem for Linux, then build exactly as on macOS:

```bash
sudo apt install build-essential python3       # one-time setup
cd 5x5_backsearch
cc -O3 -o backsearch_worker backsearch.c sokoban_bfs.c -lz
./backsearch --grid 5x5 --time 60
```

The wrapper, Python filter, and FIFO-based merging all work transparently inside WSL.

**Option B — MSYS2 / Git Bash with MinGW-w64.** Provides bash and a POSIX-ish gcc on native Windows.

```bash
# In an MSYS2 shell (after installing mingw-w64-x86_64-gcc):
cd 5x5_backsearch
gcc -O3 -o backsearch_worker.exe backsearch.c sokoban_bfs.c -lz
./backsearch --grid 5x5 --time 60
```

The wrapper script will resolve `backsearch_worker.exe` if you rename it (or just keep `backsearch_worker` without the extension on MSYS2 — that works too).

**Option C — Cross-compile from Mac/Linux with MinGW.** If you have `mingw-w64` installed:

```bash
# macOS: brew install mingw-w64
x86_64-w64-mingw32-gcc -O3 -o backsearch_worker.exe backsearch.c sokoban_bfs.c -lz   # add -static if linking zlib statically
```

You get a Windows `.exe`, but you'll still need MSYS2 or Git Bash on the target machine to run the wrapper script.

**Plain MSVC is not supported** — the worker's `clock_gettime` and `CLOCK_MONOTONIC` calls would need a Windows-native replacement (`QueryPerformanceCounter`).

## Runtime requirements

- The compiled `backsearch_worker` binary has no dependencies beyond libc.
- The `backsearch` wrapper requires bash 3.2 or newer (macOS `/bin/bash` is 3.2 — fine). It does *not* use any bash 4+ features.
- The wrapper invokes `python3` for output filtering, so Python 3 must be on `PATH`.
- Standard Unix tools: `awk`, `tee`, `grep`, `mkfifo`, `pkill`. All present by default on macOS, Linux, and MSYS2.

## Quick usage

```bash
# Default 5×5, all canonical exits, run until queue drains
./backsearch --grid 5x5 --time 0

# 60-second cap, allow blocks to transit the exit, dedup with two tables
./backsearch --grid 5x5 --time 60 --allow-exit-transit --two-tables

# Run with 4 worker processes instead of the default 6
./backsearch --grid 5x5 --time 30 --num-threads 4

# Tight-wall config (only depth-1 puzzles fit)
./backsearch --grid 5x5 --num-walls 23 --time 0
```

For the full flag list, `./backsearch --help` forwards to the worker's help.

## Harvest workflow

`--harvest FILE` logs every visited state — accepted *and* every prune outcome — to a binary log for downstream ML training. The log carries id, parent_id, depth, outcome (`A` accepted, `S` shortcut-pruned, `D` dedup-pruned, `E` solver-error, `W` walls-cap, `X` depth-cap), the cutoff forward-solve value (so pruned states contribute labeled `(state → forward_solve)` pairs), the canonical state key, and the full state blob.

If the filename ends in `.gz` the log is gzip-compressed on the fly (~6× smaller, ~3% extra CPU).

### Single-run usage

```bash
./backsearch_worker --grid 6x6 --exit 0 --allow-exit-transit --two-tables \
  --time 30 --harvest /tmp/harvest.bin.gz
python3 harvest_load.py /tmp/harvest.bin.gz --db /tmp/harvest.sqlite
```

`harvest_load.py` reads the binary, bubbles `max_descendant_depth` and `n_descendants` up through parent edges, and writes a per-run SQLite. Useful for one-off analysis. Record layout: see `harvest_format.h`.

### Central corpus

For long-term collection across many runs, use `harvest_ingest.py` to append into a central SQLite at `corpus/corpus.sqlite`:

```bash
python3 harvest_ingest.py /tmp/harvest.bin.gz
# → adds a row to `runs`, bulk-loads states (run_id-scoped), rebuilds canonical_states
```

`harvest_ingest.py` is idempotent on `(hostname, started_at)`. Re-ingesting the same harvest is a no-op unless `--force`.

### Schema

```
runs(run_id, started_at, ended_at, hostname, code_sha, argv,
     grid_rows, grid_cols, exit_pos, flags_text,
     states_visited, best_depth, exit_reason)

states(run_id FK, state_id, parent_id, canonical_key,
       depth, outcome, forward_solve,
       nblocks, nholes, player_pos, exit_pos,
       committed_empty, blocks, holes,
       max_descendant_depth, n_descendants)
       -- primary key: (run_id, state_id)

canonical_states(canonical_key PK,
                 best_max_descendant_depth, tightest_forward_solve,
                 n_observations, n_runs_seen,
                 exemplar_run_id, exemplar_state_id)
       -- rebuilt by harvest_index.py after every ingest
```

Indexes: `canonical_key`, `(run_id, depth)`, `(run_id, parent_id)`, `outcome`, `max_descendant_depth`.

### Common queries

```bash
# Top 20 deepest accepted states across the whole corpus
python3 harvest_sample.py --top-subtree 20

# Pruned-at-cutoff hard negatives for solver-surrogate training (6x6)
python3 harvest_sample.py --near-cutoff 0 --grid 6x6 --limit 5000 \
    --out jsonl --output hard_negatives.jsonl

# Cross-run dedup view: keys observed in more than one run
sqlite3 corpus/corpus.sqlite \
    "SELECT canonical_key, n_observations, n_runs_seen
     FROM canonical_states WHERE n_runs_seen > 1
     ORDER BY best_max_descendant_depth DESC LIMIT 10"
```

### Quick-look plots

```bash
python3 corpus_viz.py summary          # text overview
python3 corpus_viz.py depth-hist       # writes /tmp/corpus_depth-hist.png
python3 corpus_viz.py outcome-by-depth
python3 corpus_viz.py subtree-cdf
```

### ML training

`corpus_features.state_to_tensor(row, R, C)` returns a `(9, R, C)` float32 tensor:
`[player, exit, block-U, block-R, block-D, block-L, hole, committed_empty, unknown]`.

`corpus_features.CorpusDataset(db, where=...)` is a duck-typed PyTorch `Dataset`. Pass it to `torch.utils.data.DataLoader` (torch is imported lazily).

### Storage

~22.5 bytes per state gzip-compressed. A 12-hour 6x6 run is ~22 GB. Records are fixed 144 B uncompressed, addressable by `np.frombuffer(...).view(RECORD_DTYPE)` after the file header — see `harvest_load.py:RECORD_DTYPE`.

### Tests

```bash
python3 -m pytest tests/test_harvest.py -v
```

## Value-head NN (Phase 4)

`--nn-value-model PATH` replaces `beam_score()` with predictions from a libtorch TorchScript model (trained on the harvest corpus to predict `max_descendant_depth`). Use `--nn-blend α` (default 1.0) to weight: `α=0` is pure hand-tuned, `α=1` is pure NN, intermediate blends mix.

Pipeline:

```bash
# Train (200k subsample → ~26 min on CPU)
python3 train_value.py --grid 6x6 --epochs 8 --target-mode extra \
    --max-states 200000 --device cpu --out checkpoints/value.pt

# Export to TorchScript
python3 export_torchscript.py --checkpoint checkpoints/value.pt \
    --out checkpoints/value.ts

# Search with NN
./backsearch_worker --grid 6x6 --exit 0 --allow-exit-transit --two-tables \
    --beam 1000 --time 30 --nn-value-model checkpoints/value.ts --nn-blend 0.5
```

### Building with libtorch

The worker uses libtorch from the pip-installed PyTorch — no separate download:

```bash
TORCH_DIR=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
SHA=$(git rev-parse HEAD 2>/dev/null || echo "")

cc  -O3 -c sokoban_bfs.c -o sokoban_bfs.o
cc  -O3 -DGIT_SHA_STR=\"$SHA\" -c backsearch.c -o backsearch.o
c++ -O3 -std=c++17 -c nn_inference.cpp -o nn_inference.o \
    -I$TORCH_DIR/include -I$TORCH_DIR/include/torch/csrc/api/include
c++ -O3 backsearch.o sokoban_bfs.o nn_inference.o -o backsearch_worker \
    -L$TORCH_DIR/lib -ltorch -ltorch_cpu -lc10 -lz \
    -Wl,-rpath,$TORCH_DIR/lib
```

(`-Wl,-rpath` embeds the libtorch dir into the binary so DYLD_LIBRARY_PATH isn't needed at runtime.)

### Current state of the model

Inference is batched once per beam level — single libtorch call regardless of beam width, so the NN adds negligible per-state overhead (only the feature-extraction cost).

The first model trained on hand-tuned-harvested data **actively hurts beam depth** at any non-zero blend on 6x6 e=0 (depth 108 at α=0 vs 43 at α=1). The model learns the training distribution but that distribution was produced by hand-tuned search; predictions don't generalise to NN-guided exploration. The fix is iterative self-play: harvest with NN guidance, retrain, repeat. That loop is the natural next step.
