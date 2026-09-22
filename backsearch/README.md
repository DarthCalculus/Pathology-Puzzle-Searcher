# backsearch

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

### macOS and Linux

The worker is the only piece that needs compilation; the Python tooling runs as-is.
`build_pgo.sh` works with clang (macOS, or clang + llvm tools on Linux) and with
GCC (`-fprofile-generate/-use`); pick the compiler with `CC=gcc-13 ./build_pgo.sh`.
Linux needs zlib headers (`apt install build-essential zlib1g-dev`, plus `llvm`
if you use clang, otherwise the script falls back to a plain `-O3` build).

**Recommended — profile-guided build** (the forward solver is ~90% of runtime and PGO is worth 10–20% on top of `-O3`; links libtorch automatically if `import torch` works, otherwise a no-op NN stub):

```bash
cd backsearch
./build_pgo.sh                      # -> ./backsearch_worker
./build_pgo.sh -o worker_test --no-torch
```

Plain build, any C11 compiler (clang or gcc), no PGO:

```bash
cc -O3 -o backsearch_worker_nt backsearch.c sokoban_bfs.c nn_stub.c -lz -lm
```

**Solver regression / benchmark:** `solvebench.c` replays every forward-solver call recorded in a `--harvest` file and checks each answer against the recorded one (exit status 3 on any mismatch). Build and usage are in its header comment. Use it before and after any change to `sokoban_bfs.c`.

**End-to-end equivalence checks.** The DFS is deterministic, so an exact change must reproduce the number of *accepted* states and the best depth (the worker prints `accepted:`).  `states checked` / `solver calls` may drop when a new prune fires before the solver (the shortest-walk-segment prune in `expand()` does that); `elapsed` may change.

```bash
# shallow, ~3 s:   accepted 1289049, best depth 16
./backsearch_worker --grid 4x5 --exit 0 --max-depth 16 --time 20
# deep, ~15 s:     accepted 2362998, best depth 94   (1 eviction per table; was 2362700 before the 2026-09-18 solver changes)
./backsearch_worker --grid 5x5 --two-tables --exit 0 --num-holes 1 --num-blocks 3 --time 55
# no-transit exit 7, ~9 s:            accepted 2877948,  best depth 43
./backsearch_worker --grid 5x5 --two-tables --exit 7 --time 55
# REAL RULES (transit), ~40 s:         accepted 2286977 (2292601 with --no-exit-block-prune), valid levels 473037, best depth 58
./backsearch_worker --grid 5x5 --two-tables --allow-exit-transit --exit 7 --num-blocks 3 --time 55
# REAL RULES, ~1 min, needs -DSHALLOW_LG2=24 -DRECENT_LG2=24 for zero evictions: accepted 5279939, best depth 58   (was 5279938, same reason)
./backsearch_worker --grid 5x5 --two-tables --allow-exit-transit --exit 12 --num-blocks 4 --time 0
```

The accepted count is a strict invariant only when the dedup tables never evict
(`evicts 0 / 0` in the stats).  With evictions, two exact builds can differ by a
few hundredths of a percent because a prune that fires before the dedup insert
changes table contents and hence how much redundant re-exploration the two-table
scheme does; the best depth still has to match.  Check a transit config as well
as the no-transit ones — the walk-segment prune's first version passed all three
no-transit configs and failed under transit (a hand-built transit root left its
segment anchor unset).

**Bulk walk-back generation (default on; `--no-bulk-walk` to disable).** Every
walk-back descendant of a state that stays on already-committed cells shares the
parent's forward puzzle, so `sokoban_solve_multi()` decides the whole player
component in one solve: a lockstep Dijkstra carrying one label per start cell,
expanding each forward state once (the expansion does not depend on the start or
the cost) and replaying the cached successor list for every start that reaches
it, with shortcut decisions propagated to the cells behind a pruned cell, and
with ancestor dominance: a start only explores forward states it reaches at
least (chain distance - 1) cheaper than every ancestor on its chain, because
otherwise the ancestor would have the shortcut instead (see the comment above
`sokoban_solve_multi`).  The
survivors are pushed directly with depth = parent depth + walk distance; their own
walk-backs over committed cells are already covered, so they only generate pushes
and walks onto new cells.  `multitest.c` checks it against single solves (0
mismatches on 1.7M starts) and all equivalence configs match with it on or off.
Measured 2026-09-18 on the depth-21 precursor of the 149 with <=3 holes
(back-to-back pairs, same binary): 214 s -> 171 s before dominance and 212 s ->
177 s with it; the committed-walk solver bucket went 92 s -> 53 s -> 43 s (label
pops 636M -> 304M on 200M distinct expansions).  Pushing the nearest cells last
so the DFS explores them first also cut push-type checks by 1.5% through less
dedup re-exploration.  6% on exit 12 <=4 blocks; within noise on the small
configs; a depth threshold for enabling bulk (--bulk-min-depth) gives nothing.  A `-DWBPROF` build prints the per-child-kind
solver time split this was built on.

**Ancestor-table reuse (default on; `--no-parent-table` to disable).** An
accepted state's shortcut check ran to exhaustion, so its (state, cost) table
proves that no solution of length <= depth-2 passes through any of those states:
for every table state Y, rest(Y) >= depth - g(Y).  The table is exported as an
open-addressing hash (`SokRefTable`, built once, ~1.5K entries, peak ~30 live
tables / 11 MB on the deep benchmark), reference-counted in `g_ptab` while the
state waits in the DFS queue, and attached by pointer for the whole expansion of
that state.  A child at chain offset k (cutoff = owner's + k) whose forward
search reaches a table state Y at cost nc can skip it iff nc > g(Y) + k - 2; for
k = 1 that is "only states the child reaches strictly cheaper than the parent",
i.e. the states closer to the child's start than to the parent's -- about half.
Which children may use the parent's table:

* walk-backs over committed cells (the bulk multi-solve, one lookup per
  candidate shared by all starts; k = distance);
* push-backs of an existing block along a direction already in its mask;
* un-consumes: once the new block is consumed into the new hole the board is
  the parent's exactly, so the child's keys for that part of its search equal
  the parent's up to a constant XOR (the consumed block's Zobrist term; hole
  terms are keyed by cell so an extra filled hole contributes nothing) --
  `sokoban_ref_set_xor`;
* a child solved with a reference exports its own table too: its labels plus
  the reference's entries at +k are all upper bounds on its distances, so its
  children reference it at k = 1 instead of inheriting the owner's at k + 1.

Children whose puzzle gained floor cells use the table with a puzzle delta (every
new cell must be too far to be used within the budget); measured, that prunes
0.03% of their pops -- the new cell is always in reach -- so walk/new-cell and
push/new-cell children effectively get nothing.  New-block children and
push-backs that add a direction change the puzzle in ways the parent's proof
does not cover and never use a table.

Measured 2026-09-18 on the depth-36 seed of the 149 (deterministic: 185569
accepted), `-DWBPROF` counts: without tables 275M solver pops; first version
(nearest parent only, per-check table rebuilds) 207M pops + 444M hash ops; after
this review 179M pops + 367M hash ops.  Per referenced check, a k = 1 reference
removes 47-49% of the pops (the predicted half), k = 2 27-32%, k >= 3 16-37%;
un-consume checks lose 61% of their pops.  Why the gain is capped: two thirds of
solver time is spent on children whose forward puzzle differs from the parent's
(new block, new push direction, newly committed cell), where the parent's proof
says nothing; the eligible third loses about half.  Wall time on this laptop was
too noisy to quote on the day (load 3-5 from other work); the earlier quiet
depth-21 measurement was 163 s -> 141 s for the first version.  `-DREF_CHECK`
re-runs every check made with a reference without it and compares decisions
(0 mismatches on 2.2M checks across the benchmark configs); `-DBULK_CHECK`
does the same for the multi-solve.  Zobrist hole keys are per cell (not per
hole index) since this change.

Tried and rejected (2026-09-18): a hole-crossing lower bound (when the player
cannot reach the exit with the unfilled holes as walls, every solution must
first bring some block to some hole; the minimum over pairs of walk + pushes +
walk is admissible).  Exact, but it prunes only 1-2% of pops on 5x5 and costs
~25% per solve; kept behind `-DHOLEBOUND`.

Exact pruning/solver changes so far (2026-09-14): small L2 solver table with big-table fallback; A* walk-distance bound; slack-0 pop skip and depth-limited walk BFS; shortest-walk-segment prune before the solver call (generalises the anti-wiggle rule; ~3% of solver calls on deep configs); decision mode (`sokoban_set_decision_only`: return on the first solution within the cutoff, queue ordered by g + h; ~16% fewer solver states; off automatically when `--harvest`/`--trace-csv`/`--bf-dump` need exact lengths — `solvebench --decision` replays it).

**Permanently stuck exit block (2026-09-22; `--no-exit-block-prune` to disable).**
A state with a block on the exit is not a level; it leads to levels only through a
later backward push-back of that block off the exit, which needs the cell it moves
to and the cell the player lands on to be free.  Going backward, blocks and holes
are never removed and the grid edge / fixed walls never change, so a block whose
every pull needs an off-grid, fixed-wall, hole, or itself-stuck cell can never
move (mutual locks such as blocks on 6, 8, 16, 18 of a 5x5 are found by
un-sticking from "all stuck" to a fixpoint; uncommitted cells count as free).
Such states are pruned before the solver call.  The summary now prints
`valid levels` (accepted states with no block on the exit), which is the
invariant to compare across this prune: unchanged on every config, best depths
unchanged; accepted drops 0.25% (exit 7 <= 3 blocks) to 2.3% (the depth-36
seed of the 149).  Note the proportions: on that seed only 4,789 of 185,569
accepted states are levels; the rest carry a block on the exit.

## DFS-order ranges, checkpoints and chunks (2026-09-21)

`expand()` pushes a node's children in a fixed order (bulk walk-backs first,
then single steps by direction and variant) and the stack pops them in reverse,
so the DFS visits the tree in one fixed order and every node's path (its
`--seed-path` string; a bulk walk-back child carries its whole walk) is its
position in that order.  Hence:

* **Checkpoint.** Ctrl-C (SIGINT/SIGTERM) or the `--time` cap finish the current
  expansion and print `CURSOR <exit> <path> <depth>`: everything before that
  node is done.  `--from <path>` resumes there (the node's subtree included);
  only the ancestors along the path are re-expanded.
* **Ranges.** `--from P --until Q` searches exactly the nodes visited between P
  (inclusive) and Q (exclusive); Q and everything after it belong to another
  run.  Cut points can be at any depth, and `--list-layer` / `--estimate-dump`
  emit layer nodes in this visiting order, so consecutive layer nodes are
  consecutive ranges.  Verified by visit traces (`BS_TRACE_VISITS=file`): a
  full run's node sequence equals the concatenation of its ranges, and an
  interrupted run plus its resume, up to the re-expanded ancestors.
* **Driver.** `campaign.py` accepts range jobs (`FROM..UNTIL`, either side
  empty); Ctrl-C or `--split-after S` checkpoints the running workers, records
  the job as `continued` and appends the continuation `CURSOR..UNTIL` to
  jobs.tsv, so re-running the same command continues from the checkpoints.
* **Chunks (collective proofs).** `results/chunk_plan.py` estimates the layer
  (re-estimating heavy prefixes at a deeper layer, spliced in place) and cuts
  the DFS order into contiguous chunks of roughly equal estimated work; each
  chunk is a range `[range_from, range_until)` whose interior cut points are its
  parallel sub-ranges.  `run_chunks.py` runs chunks from the tracker at
  pathology.georgespahn.com and prints the report block it accepts.

The explicit push-off-exit seeds were removed the same day: the root's own
new-block push-back children are the same states, and a node must have one
path.  The transit equivalence counts rose by the number of those root children
(exit 7 <= 3 blocks: 2292598 -> 2292601; exit 12 <= 4 blocks: 5279938 -> 5279939);
best depths and everything else are unchanged.

## Exhaustive campaigns (proving a grid's record)

**Rule set.** Pathology lets blocks be pushed over the exit and the record levels
rely on it (the 5x5 149 and 146 both do), so every record-relevant run must pass
`--allow-exit-transit`.  A level may not start with a block on the exit, so
`--allow-block-on-exit` is never used.  `campaign.py` adds transit automatically
and refuses block-on-exit.  Results produced without transit are a restricted
class only and live in `results/notransit/`.

Three tools turn "run until the queue drains" into a plannable, resumable job:

```bash
# 1. Size the tree first.  Knuth random-probe estimator: exact depth-K layer
#    (with dedup), then ~N probes through expand()/try_successor() with dedup
#    off.  Prints estimated nodes, states checked, wall time, a depth profile
#    and the heaviest layer nodes.  Runs LOW (deep corridors are under-sampled):
#    actual/estimate was 1x-4x on the runs checked so far — a lower bound.
#    Use K >= 10 and >= 10 probes per layer node.
./backsearch_worker --grid 5x5 --two-tables --allow-exit-transit --exit 0 --estimate 1500000 --estimate-depth 10

# 2. Split the exit into independent sub-jobs: the dedup'd depth-K layer as
#    --seed-path strings.  Their subtrees plus the (depth < K) ancestors are
#    the whole tree, so the union of exhaustive per-job runs is an exhaustive
#    run (verified: 867 jobs reproduce the monolithic max depth and count).
./backsearch_worker --grid 5x5 --two-tables --allow-exit-transit --exit 0 --list-layer 8 | grep -c LAYER

# 3. Run them: resumable, parallel, records every job in DIR/done.tsv, keeps the
#    full worker output of deep finds, --status for progress.  Use the no-torch
#    build (./build_pgo.sh -o backsearch_worker_nt --no-torch) for fast startup.
python3 campaign.py --out results/camp_b4 --exits 0,1,2,6,7,12 --extra "--num-blocks 4" \
    --layer 7 --workers 3 --worker ./backsearch_worker_nt --shuffle --split-after 600
python3 campaign.py --out results/camp_b4 --status
echo 2 > results/camp_b4/workers        # change concurrency live (0 pauses)
bash status.sh                          # everything at a glance
```

The DFS tree is extremely unbalanced (a handful of depth-7 subtrees hold most of
an exit's work), so `--split-after SEC` kills any job still running after SEC
seconds and replaces it with its depth+3 children (`--seed-path P --list-layer 3`);
the parent is recorded as `split` in done.tsv and the children are appended to
jobs.tsv, so resuming still works.  Waste is at most SEC per split.

Cross-job dedup loss is small (dedup off entirely costs ~1.4x states), and dedup
table size barely matters (2x larger tables changed a 74M-state run by 1.5%).
Keep concurrent workers to what RAM allows (~250 MB each).

**5x5 under real rules (2026-09-14 estimates, dedup-free lower bounds):** exit 0
45 h, exit 1 124 h, exit 2 33 h, exit 6 ~8000 h, exit 7 216 h, exit 12 105 h of
single-core time — a full proof is out of reach on a laptop.  Tractable classes:
<=4 blocks (all exits ~0.5 h lower bound) and <=2 holes (~10 h lower bound),
which extend the already-exhausted <=3-block and <=1-hole classes.

**Done (real rules, exhaustive):**
- <=4 blocks, any holes, all six exits (2026-09-14): exit 0 **127**, exit 1 119,
  exit 2 92, exit 6 80, exit 7 76, exit 12 58 (`results/camp_b4`, 106k jobs,
  2h33m CPU, 1.04e9 states; exits 7 and 12 cross-checked against monolithic runs).
- <=2 holes, any blocks, all six exits (2026-09-17): exit 0 124, exit 1 **149**,
  exit 2 112, exit 6 124, exit 7 103, exit 12 80 (`results/camp_h2`, 225,878 jobs,
  134h45m CPU, 8.3e9 states; exit 12 cross-checked against a monolithic run).
  So the 149-move "Capital C" is the longest 5x5 level with at most 2 holes.

The estimator's lower bounds were 6x (<=4 blocks) and ~13x (<=2 holes) low.
Also exhausted (2026-09-17, `results/classes/`, 20 min total): every combination
of block cap 0-4 / unconstrained and hole cap 0-2 / unconstrained (hole cap below
the block cap), 84 per-exit proofs; e.g. <=3 blocks 110, <=2 blocks 75, no holes
69, <=4 blocks <=1 hole 106.  All proofs are on the live site with derived
whole-size entries.  `results/classes/run_classes.sh` runs a class list per exit
(resumable) and `compile_classes.py --publish` turns finished classes into proofs
and champion submissions.  Note the site's submit limit is 60/hour.

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
cd backsearch
cc -O3 -o backsearch_worker backsearch.c sokoban_bfs.c -lz
./backsearch --grid 5x5 --time 60
```

The wrapper, Python filter, and FIFO-based merging all work transparently inside WSL.

**Option B — MSYS2 / Git Bash with MinGW-w64.** Provides bash and a POSIX-ish gcc on native Windows.

```bash
# In an MSYS2 shell (after installing mingw-w64-x86_64-gcc):
cd backsearch
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

The worker (`backsearch_worker_nt`) is single-threaded and has no thread flag.
Parallelism comes from running several workers: `campaign.py --workers N` is the
recommended way (resumable, exact, one exit split into seed-path jobs); the
`backsearch` wrapper below is the older alternative and partitions one run into a
handful of tasks with `--num-threads N`.

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
