# 5x5_backsearch

Backward-DFS Sokoban puzzle generator with task-partitioned parallelism.

## What's here

| File | Role |
|---|---|
| `backsearch.c` | Worker source: backward DFS, dedup, shortcut pruning, partition enumeration |
| `sokoban_bfs.c` / `sokoban_bfs.h` | Forward solver used by the worker for shortcut checks and verification |
| `backsearch` | Bash wrapper: auto-partitions the search and runs workers in parallel |
| `_backsearch_filter.py` | Stream filter the wrapper pipes worker output through |

## Building

The C code uses `clock_gettime(CLOCK_MONOTONIC, ...)` — a POSIX call that isn't available in plain MSVC. Builds need either a POSIX-compatible toolchain (macOS, Linux, MinGW, MSYS2, WSL) or a small port of the timing functions.

### macOS

The worker is the only piece that needs compilation. The wrapper script and Python filter run as-is.

```bash
cd 5x5_backsearch
cc -O3 -o backsearch_worker backsearch.c sokoban_bfs.c
```

That's it. `cc` on macOS is Apple Clang. No external dependencies. The `backsearch` wrapper script and `_backsearch_filter.py` are invoked directly — Python 3 is already on macOS.

For a debug build with symbols:

```bash
cc -O0 -g -o backsearch_dbg backsearch.c sokoban_bfs.c
```

### Windows

Three options, easiest to hardest:

**Option A — WSL (recommended).** Install Windows Subsystem for Linux, then build exactly as on macOS:

```bash
sudo apt install build-essential python3       # one-time setup
cd 5x5_backsearch
cc -O3 -o backsearch_worker backsearch.c sokoban_bfs.c
./backsearch --grid 5x5 --time 60
```

The wrapper, Python filter, and FIFO-based merging all work transparently inside WSL.

**Option B — MSYS2 / Git Bash with MinGW-w64.** Provides bash and a POSIX-ish gcc on native Windows.

```bash
# In an MSYS2 shell (after installing mingw-w64-x86_64-gcc):
cd 5x5_backsearch
gcc -O3 -o backsearch_worker.exe backsearch.c sokoban_bfs.c
./backsearch --grid 5x5 --time 60
```

The wrapper script will resolve `backsearch_worker.exe` if you rename it (or just keep `backsearch_worker` without the extension on MSYS2 — that works too).

**Option C — Cross-compile from Mac/Linux with MinGW.** If you have `mingw-w64` installed:

```bash
# macOS: brew install mingw-w64
x86_64-w64-mingw32-gcc -O3 -o backsearch_worker.exe backsearch.c sokoban_bfs.c
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
