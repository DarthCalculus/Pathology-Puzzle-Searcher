# Sokoban Puzzle Search — Project Summary

## Problem Definition

Find a Sokoban-variant puzzle on a **4×5 grid** whose *shortest* solution (measured in player moves) is as long as possible.

### Rules
- The player wins by reaching an **exit tile** (not by pushing a block onto a goal)
- **Blocks** obstruct the player and can be pushed; a block cannot start on the exit tile
- Each block has a per-side **pushability bitmask** (`U=1 R=2 D=4 L=8`) — zero means immovable
- **Holes** consume any block pushed into them; the hole disappears and the cell becomes walkable
- **Walls** are a bitmask over cells (currently no walls in the search — immovable blocks serve the same purpose)
- Blocks on edge/corner cells have geometrically restricted push directions (pushing would require the player to stand off-grid)

### Grid Layout

```
Cell indices (row-major, COLS=5):
 0  1  2  3  4
 5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
```

Symmetry reduction: only the 6 exit cells in the top-left quadrant (`{0,1,2,5,6,7}`) are tried; the rest are equivalent by reflection.

---

## Current Best Result: **51 moves** (3 blocks, 4×5 grid)

```
  EO...   block A push=5 [UD]
  AO@O.   block B push=f [URDL]
  .BC..   block C push=7 [URD]
  .....
  exit=0  player_start=7  walls=00000
```

Found by exhaustive search over all 3-block configurations.

---

## Codebase

### Core Files

| File | Role |
|------|------|
| `sokoban_bfs.h` | Public header: `Puzzle` struct, constants, `sokoban_solve` declaration |
| `sokoban_bfs.c` | BFS solver implementation |
| `puzzle_search.c` | Exhaustive search: iterates all (exit, holes, block positions, player start, push masks) |
| `test_puzzle.c`  | Ad-hoc test harness for specific puzzle configurations |

### Legacy / Experimental Files

These predate the current architecture (4×4 grid, earlier approach):

`sokoban_fast.c`, `sokoban_fast2.c`, `sokoban_sa.c`, `sokoban_pair.c`, `sokoban_big.c`, `sokoban_ga.c`, `sokoban_opt.c`, `sokoban_opt2.c`, `sokoban_trace.c`, `sokoban_45.c`, `sokoban_45_table.c`, `sokoban_exitguard.c`, `sokoban_exitguard2.c`

---

## Architecture

### BFS Solver (`sokoban_bfs.c`)

```c
int sokoban_solve(const Puzzle *pz, uint8_t *used_dirs);
// Returns minimum moves to exit, or -1 (unsolvable), or -2 (queue overflow).
// If used_dirs != NULL, fills per-block bitmask of directions actually pushed
// on the optimal path.
```

**State representation** — packed into a `uint64_t`:
- Player position: 5 bits (0–20, 20 = CONSUMED sentinel)
- Each block position: 5 bits each
- Hole mask: 1 bit per hole (up to 5 bits)

**Per-thread storage** (`_Thread_local SolverState *`):
- `htk[16M]` + `ht_gen[16M]`: visited-set hash table with generation-counter O(1) clear (~192 MB)
- `qs[16M]`: BFS queue of packed states (~128 MB)
- `qused[16M]`: parallel array tracking cumulative push directions used on the path to each queued state (~64 MB); only populated when `used_dirs != NULL`

Total: **~384 MB per thread** (allocated once, reused across all BFS calls via generation counter).

**Hash table design:**
- Open addressing with linear probing, probe limit 128
- Two-round splitmix64 finalizer for good bit diffusion of the structured low-bit state
- Generation counter (`ht_seq`) makes logical clear O(1) — no memset between BFS calls

**BFS loop optimisations:**
- `block_occ` bitmask — O(1) "is there a block here?" check
- `active_holes` bitmask — O(1) hole check
- Free moves patch player bits in-place (`cur & ~0x1F | np`), no re-pack
- Level-boundary tracking with two ints (`ql`, `dist`) instead of a per-entry depth array

### Exhaustive Search (`puzzle_search.c`)

**Enumeration order:** for each `(exit_cell, hole_count)` work item, iterate over all hole placements, block placements, and player start positions. For each board geometry, run the **bitmask search**.

**Bitmask search** (`try_bitmasks`): enumerates all pushability mask vectors top-down (most-pushable first). Exploits two monotonicity facts:
- *Fact 1:* If mask A is solvable in d steps, any subset B ⊆ A is solvable in ≥ d steps
- *Fact 2:* If mask A is unsolvable, every subset B ⊆ A is also unsolvable

Pruning:
- **Local antichain** (per-level): masks found unsolvable at block i prune their subsets for the same block
- **Cross-call (XC) antichain**: unsolvable masks discovered for block i+1 under one value of block i are inherited when block i gets a harder mask
- **Global antichain** (per board): maximal unsolvable full mask vectors — skips BFS calls that are provably unsolvable
- **KnownSolvable (KS) table** (per board): stores `(packed_used_dirs, dist)` from previous BFS calls. Before each leaf BFS, checks if any stored entry's `used_dirs` is a componentwise subset of the current `block_pushable` masks — if so, the same solution path still works and BFS is skipped entirely (returns stored distance)

**Parallelism:**
- Default mode: `(exit_cell, hole_count)` pairs form a work queue; 8 worker threads consume it
- `--shard` mode: 6 threads, one per exit cell, each processes all hole counts independently; prints per-exit notable results (new records or > 60 steps)

**Precomputed tables:**
- `cell_masks[p]` / `cell_nmasks[p]`: valid push masks per cell in popcount-descending order
- `superset_list[m]` / `superset_cnt[m]`: 4-bit mask supersets, used to efficiently seed the XC antichain

---

## Performance (3-block exhaustive search, 8 threads, Apple Silicon)

| Metric | Value |
|--------|-------|
| Solver calls | 67,888,195 |
| Wall time | ~35s (clean machine) |
| User CPU time | ~78s (8 threads × ~10s each) |

The KnownSolvable optimisation was the largest single win: reduced calls from ~1.78B to ~67.9M (~26×) and wall time from ~350s to ~35s.

---

## Build

```sh
clang -O2 -o puzzle_search puzzle_search.c sokoban_bfs.c -lpthread

./puzzle_search 3           # 3-block exhaustive search
./puzzle_search 3 --shard   # shard mode (one thread per exit)
```

---

## Open Questions / Potential Next Steps

- Run exhaustive 4-block search (expected: much longer runtime)
- Investigate whether wall tiles add meaningfully longer solutions (currently excluded — immovable blocks cover the same space)
- Robin Hood hashing as alternative to linear probing (reduces probe chain variance)
- A* or IDA* with an admissible heuristic to focus on hard positions without exhaustive enumeration
