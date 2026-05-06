/*
 * backsearch.c — backward-search puzzle generator.
 *
 * Constructs hard puzzles by walking backward from the solved state
 * (player on exit cell 0).  Each backward step either undoes a forward
 * walk or undoes a forward push (introducing a new block constraint
 * if needed).  After every step we run sokoban_solve on the partial
 * puzzle as a shortcut check: if the optimal forward solve is shorter
 * than the backward depth, the branch is pruned.
 *
 * The shortcut-check puzzle uses (a) walls = current unconstrained
 * cells (the *most* walls we'll ever have) and (b) masks = current
 * min-mask (the *fewest* push directions we'll ever allow).  These
 * choices maximise forward solve length, so the result is an upper
 * bound on the final puzzle's forward solve.
 *
 * Grid size is configurable at runtime via --grid RxC (R,C in [1..5]).
 * The underlying sokoban_bfs.c stays a 5x5 solver; smaller grids are
 * emulated by treating cells outside the R*C window as permanent walls.
 *
 * Win condition (verified in sokoban_bfs.c:682) is "player ends turn
 * on exit_pos" — the existing header comment about blocks reaching
 * the exit is misleading.
 */

#include "sokoban_bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>

#define DEFAULT_CAP_S    60.0

/* The six canonical exit cells (D4 fundamental domain on a 5x5 grid):
 *   (0,0)=0   (0,1)=1   (0,2)=2
 *             (1,1)=6   (1,2)=7
 *                       (2,2)=12
 * Every other cell is equivalent to one of these by some D4 transform. */
#define NUM_EXIT_CELLS 6
static const int CANONICAL_EXITS[NUM_EXIT_CELLS] = { 0, 1, 2, 6, 7, 12 };

/* Adjacency table — duplicates the static one in sokoban_bfs.c.
 * Directions: 0=Up, 1=Right, 2=Down, 3=Left. */
static const int8_t adj[NCELLS][4] = {
    {-1,  1,  5, -1}, {-1,  2,  6,  0}, {-1,  3,  7,  1},
    {-1,  4,  8,  2}, {-1, -1,  9,  3},
    { 0,  6, 10, -1}, { 1,  7, 11,  5}, { 2,  8, 12,  6},
    { 3,  9, 13,  7}, { 4, -1, 14,  8},
    { 5, 11, 15, -1}, { 6, 12, 16, 10}, { 7, 13, 17, 11},
    { 8, 14, 18, 12}, { 9, -1, 19, 13},
    {10, 16, 20, -1}, {11, 17, 21, 15}, {12, 18, 22, 16},
    {13, 19, 23, 17}, {14, -1, 24, 18},
    {15, 21, -1, -1}, {16, 22, -1, 20}, {17, 23, -1, 21},
    {18, 24, -1, 22}, {19, -1, -1, 23},
};

/* -------------------------------------------------------------------------
 * Global config
 * ------------------------------------------------------------------------- */

static int      g_grid_rows   = 5;
static int      g_grid_cols   = 5;
static uint32_t g_active_mask = 0;          /* set in main() from --grid    */
static double   g_time_cap_s  = DEFAULT_CAP_S;
static int      g_exit_pos    = 0;          /* current exit cell being searched */
static int      g_only_exit   = -1;         /* -1 = iterate canonical exits */
/* --exitloc: explicit list of exit cells to search (overrides default
 * canonical set when non-empty).  --exit takes priority over --exitloc
 * if both are given. */
static int      g_n_only_exits = 0;
static int      g_only_exit_list[NCELLS];
static int      g_allow_exit_transit = 0;   /* 1 = block may transit through exit */
static int      g_holeless           = 0;   /* 1 = forbid all holes (no variant 4) */
static int      g_two_tables         = 0;   /* 1 = use shallow+recent two-table dedup */

/* Fixed holes: when non-empty, holes are *only* allowed at these cells.
 * Variant 4 (un-consume) skips placing a hole at any cell not in this
 * mask.  No pre-placement — the search may choose to put 0..N holes at
 * any subset of these allowed cells. */
static int      g_fixed_nholes = 0;
static int      g_fixed_hole_pos[MAX_HOLES];
static uint32_t g_fixed_holes_mask = 0;

/* Fixed walls: cells that must be walls in the reported puzzle setup.
 * The search prevents these cells from ever entering committed_empty —
 * the player never walks on them, no block or hole is introduced there. */
static uint32_t g_fixed_walls_mask = 0;

/* Effective walkable region = active region minus fixed walls.  This is
 * what expand() consults to decide if a cell can be entered or have a
 * block/hole placed on it. */
static uint32_t g_walkable_mask = 0;

/* Lower bound on wall count in the reported puzzle.  The search prunes
 * any successor whose committed_empty grows so large that fewer than
 * this many cells of the active region remain walls.  Default: 0 (no
 * constraint). */
static int      g_min_walls         = 0;
static int      g_max_committed_in_active = INT_MAX;  /* derived from g_min_walls */

/* Upper bounds on dynamically-introduced blocks/holes (--num-blocks,
 * --num-holes).  Variants 3 and 4 in expand() will not introduce new
 * blocks/holes once the bound is reached.  --num-holes counts the TOTAL
 * (including --fixedholes); --num-blocks counts the total nblocks. */
static int      g_max_blocks        = MAX_BLOCKS;
static int      g_max_holes         = MAX_HOLES;

/* Upper bound on a state's depth.  Successors with depth > this are
 * pruned in try_successor.  Used by the wrapper's "shallow scan" pass
 * (--max-depth 2) to cover depths the depth-3 task partition can't
 * reach when --num-walls is tight.  Default: INT_MAX (no cap). */
static int      g_max_depth         = INT_MAX;

/* Task partitioning.  When --task-id is set, the search starts not from
 * the standard depth-0 roots but from a specific subset of depth-2
 * states grouped by:
 *   (transit, hole_loc)
 * where transit ∈ {0, 1} indicates whether the 1st backstep involved an
 * exit-transit (block moving from adj[exit][D] to exit, only possible
 * when --allow-exit-transit is set), and hole_loc ∈ {-1, cell index}
 * indicates whether the 2nd backstep introduced a new hole (and at
 * which cell).
 *
 * Tasks are auto-enumerated per exit; --list-tasks prints how many
 * tasks each searched exit has, then exits.  --task-id N selects the
 * N-th task (global, across the iterated exits).
 *
 * Use case: shell-level parallelism.
 *     N=$(./backsearch ... --list-tasks)
 *     for i in $(seq 0 $((N-1))); do
 *         ./backsearch ... --task-id $i &
 *     done; wait
 */
static int g_only_task   = -1;   /* -1 = run all tasks merged (default) */
static int g_list_tasks  = 0;

/* Dedup horizon: states at depth <= this value are deduped against the
 * visited table; states deeper than this skip dedup entirely (free-fly).
 * Trade-off: free-flying deep states do redundant subtree exploration
 * (cheap because subtrees are small at high depth) but the dedup table
 * stays small and never overflows.  Default: unlimited (current behaviour). */
static int       g_dupe_threshold = INT_MAX;
static long long g_skipped_dedup  = 0;

/* -------------------------------------------------------------------------
 * Backward state
 * ------------------------------------------------------------------------- */

typedef struct {
    int8_t   player_pos;
    int8_t   nblocks;
    int8_t   nholes;                     /* count of currently-active holes  */
    int8_t   block_pos [MAX_BLOCKS];
    uint8_t  block_mask[MAX_BLOCKS];
    int8_t   hole_pos  [MAX_HOLES];      /* sorted ascending; all active     */
    uint32_t committed_empty;            /* bit i: cell i is known not-wall */
    int32_t  depth;
} BState;

/* Sort blocks ascending by (pos, mask).  Insertion sort — nblocks is small. */
static void sort_blocks(int8_t *bp, uint8_t *bm, int n) {
    for (int i = 1; i < n; i++) {
        int8_t  p = bp[i];
        uint8_t m = bm[i];
        int j = i - 1;
        while (j >= 0 && (bp[j] > p || (bp[j] == p && bm[j] > m))) {
            bp[j+1] = bp[j];
            bm[j+1] = bm[j];
            j--;
        }
        bp[j+1] = p;
        bm[j+1] = m;
    }
}

/* Sort holes ascending by position.  Holes have no per-hole flag — all
 * tracked holes are active. */
static void sort_holes(int8_t *hp, int n) {
    for (int i = 1; i < n; i++) {
        int8_t p = hp[i];
        int j = i - 1;
        while (j >= 0 && hp[j] > p) { hp[j+1] = hp[j]; j--; }
        hp[j+1] = p;
    }
}

/* -------------------------------------------------------------------------
 * Visited hash set
 *
 * Open addressing, linear probing.  Stores 64-bit hash of canonical state
 * representation plus the depth at which it was first reached.  Two states
 * with the same hash are treated as equal — collision probability ≈
 * states_visited / 2^64, negligible at the scales reachable in 60s.
 * ------------------------------------------------------------------------- */

/* Visited table size — compile-time constant for hot-path speed.
 * Default 24 (16M slots, 256 MB) balances cache locality and per-exit
 * memset cost.  Recompile with -DHASH_LG2=26 (1 GB) for long 5x5 runs
 * that would otherwise hit the graceful-overflow path. */
#ifndef HASH_LG2
#define HASH_LG2 24
#endif
#define HASH_CAP  ((size_t)1 << HASH_LG2)
#define HASH_MASK (HASH_CAP - 1)

typedef struct { uint64_t key; int32_t depth; } HashSlot;
static HashSlot *g_visited = NULL;
static long long g_visited_count = 0;
static int       g_dedup_full   = 0;    /* set when table can't accept more */

/* -------------------------------------------------------------------------
 * Two-table dedup mode (--two-tables).
 *
 * Idea: split the dedup budget between two tables with different eviction
 * policies, both consulted on every lookup, both inserted on every miss:
 *
 *  - SHALLOW table: keeps the lowest-depth states ever seen.  When ~80%
 *    full it evicts entries with depth above a histogram-derived
 *    threshold, halving the table.
 *  - RECENT table: LRU-style.  Each slot stores a 64-bit clock value;
 *    when ~80% full it evicts entries whose clock is below a histogram-
 *    derived threshold, halving the table.
 *
 * The shallow table catches structurally important duplicates (large
 * subtrees below them); the recent table catches the temporally hot
 * frontier of the DFS.  Together they keep dedup useful indefinitely
 * without ever overflowing.
 * ------------------------------------------------------------------------- */
#ifndef SHALLOW_LG2
#define SHALLOW_LG2 22
#endif
#ifndef RECENT_LG2
#define RECENT_LG2  22
#endif
#define SHALLOW_CAP  ((size_t)1 << SHALLOW_LG2)
#define SHALLOW_MASK (SHALLOW_CAP - 1)
#define RECENT_CAP   ((size_t)1 << RECENT_LG2)
#define RECENT_MASK  (RECENT_CAP  - 1)

typedef struct { uint64_t key; int32_t depth; } ShallowSlot;
typedef struct { uint64_t key; int32_t depth; uint64_t clock; } RecentSlot;

static ShallowSlot *g_shallow = NULL;
static RecentSlot  *g_recent  = NULL;
static long long    g_shallow_count = 0;
static long long    g_recent_count  = 0;
static uint64_t     g_recent_clock  = 1;   /* monotonically increasing tick */
static long long    g_evictions_shallow = 0;
static long long    g_evictions_recent  = 0;

static uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint64_t state_key(const BState *s) {
    uint64_t h = splitmix64((uint64_t)s->committed_empty);
    h = splitmix64(h ^ ((uint64_t)(uint8_t)s->player_pos << 8)
                     ^ ((uint64_t)(uint8_t)s->nblocks << 16)
                     ^ ((uint64_t)(uint8_t)s->nholes  << 24));
    for (int i = 0; i < s->nblocks; i++) {
        uint64_t v = ((uint64_t)(uint8_t)s->block_pos[i] << 8) | (uint64_t)s->block_mask[i];
        h = splitmix64(h ^ v);
    }
    for (int i = 0; i < s->nholes; i++) {
        h = splitmix64(h ^ ((uint64_t)(uint8_t)s->hole_pos[i] | 0x100));
    }
    return h ? h : 1;  /* never zero: 0 = empty slot */
}

/* Returns 1 if state should be skipped (already seen at <= this depth).
 * Otherwise inserts/updates and returns 0.  When the table is too full
 * to accept a new entry we set g_dedup_full and return 1 so the caller
 * silently drops this state; the search loop checks g_dedup_full and
 * terminates the current exit's pass gracefully (preserving best). */
static int dedup_check_and_insert(uint64_t key, int depth) {
    /* Hoist the _Thread_local pointer into a local register —
     * gcc/clang otherwise reload it every loop iteration. */
    HashSlot * const tab = g_visited;
    size_t i = (size_t)key & HASH_MASK;
    for (size_t probe = 0; probe < HASH_CAP; probe++) {
        if (tab[i].key == 0) {
            tab[i].key = key;
            tab[i].depth = depth;
            g_visited_count++;
            return 0;
        }
        if (tab[i].key == key) {
            if (tab[i].depth <= depth) return 1;
            tab[i].depth = depth;
            return 0;
        }
        i = (i + 1) & HASH_MASK;
    }
    g_dedup_full = 1;
    return 1;
}

/* -------------------------------------------------------------------------
 * Two-table dedup: lookup, insert, and eviction helpers.
 * ------------------------------------------------------------------------- */

/* Evict the deepest ~half of the shallow table.  Uses a 256-bin histogram
 * to find a depth threshold T such that depth <= T keeps roughly 50%, then
 * sweeps the table copying surviving entries to a fresh allocation. */
static void shallow_evict(void) {
    int hist[256] = {0};
    for (size_t i = 0; i < SHALLOW_CAP; i++)
        if (g_shallow[i].key != 0) {
            int d = g_shallow[i].depth;
            if (d > 255) d = 255;
            hist[d]++;
        }
    long long target = g_shallow_count / 2;
    long long acc = 0;
    int thresh = 255;
    for (int d = 0; d < 256; d++) {
        acc += hist[d];
        if (acc >= target) { thresh = d; break; }
    }
    /* Rehash survivors (depth <= thresh) into a fresh table. */
    ShallowSlot *fresh = calloc(SHALLOW_CAP, sizeof *fresh);
    if (!fresh) { perror("calloc shallow_evict"); exit(1); }
    long long kept = 0;
    for (size_t i = 0; i < SHALLOW_CAP; i++) {
        if (g_shallow[i].key == 0) continue;
        if (g_shallow[i].depth > thresh) continue;
        size_t j = (size_t)g_shallow[i].key & SHALLOW_MASK;
        while (fresh[j].key != 0) j = (j + 1) & SHALLOW_MASK;
        fresh[j] = g_shallow[i];
        kept++;
    }
    free(g_shallow);
    g_shallow = fresh;
    g_shallow_count = kept;
    g_evictions_shallow++;
}

/* Evict the oldest ~half of the recent table by clock.  Histogram bin =
 * (clock - oldest_clock) / step, where step picks ~256 buckets across the
 * range.  Same survivor-copy approach as shallow_evict. */
static void recent_evict(void) {
    /* Find clock range. */
    uint64_t lo = UINT64_MAX, hi = 0;
    for (size_t i = 0; i < RECENT_CAP; i++) {
        if (g_recent[i].key == 0) continue;
        if (g_recent[i].clock < lo) lo = g_recent[i].clock;
        if (g_recent[i].clock > hi) hi = g_recent[i].clock;
    }
    /* If all the same, just halve at random — bin into 256 buckets. */
    uint64_t span = (hi > lo) ? (hi - lo) : 1;
    uint64_t step = (span / 256) + 1;
    int hist[256] = {0};
    for (size_t i = 0; i < RECENT_CAP; i++) {
        if (g_recent[i].key == 0) continue;
        uint64_t b = (g_recent[i].clock - lo) / step;
        if (b > 255) b = 255;
        hist[b]++;
    }
    long long target = g_recent_count / 2;
    long long acc = 0;
    int thresh_bin = 0;
    for (int b = 0; b < 256; b++) {
        acc += hist[b];
        if (acc >= target) { thresh_bin = b; break; }
    }
    uint64_t thresh_clock = lo + (uint64_t)thresh_bin * step;
    /* Rehash survivors (clock >= thresh_clock) into a fresh table. */
    RecentSlot *fresh = calloc(RECENT_CAP, sizeof *fresh);
    if (!fresh) { perror("calloc recent_evict"); exit(1); }
    long long kept = 0;
    for (size_t i = 0; i < RECENT_CAP; i++) {
        if (g_recent[i].key == 0) continue;
        if (g_recent[i].clock < thresh_clock) continue;
        size_t j = (size_t)g_recent[i].key & RECENT_MASK;
        while (fresh[j].key != 0) j = (j + 1) & RECENT_MASK;
        fresh[j] = g_recent[i];
        kept++;
    }
    free(g_recent);
    g_recent = fresh;
    g_recent_count = kept;
    g_evictions_recent++;
}

/* Two-table check-and-insert.  Returns 1 if state was already seen at
 * depth <= the current depth (skip), 0 otherwise (continue exploring).
 * Always inserts new states into both tables (after evicting if needed). */
static int dedup_two_tables(uint64_t key, int depth) {
    int seen_shallow = 0, seen_recent = 0;
    int hit_dup = 0;

    /* Hoist _Thread_local pointers into local registers. */
    ShallowSlot * const stab = g_shallow;
    RecentSlot  * const rtab = g_recent;

    /* --- Probe shallow --- */
    size_t i = (size_t)key & SHALLOW_MASK;
    for (size_t probe = 0; probe < SHALLOW_CAP; probe++) {
        if (stab[i].key == 0) break;
        if (stab[i].key == key) {
            seen_shallow = 1;
            if (stab[i].depth <= depth) hit_dup = 1;
            else stab[i].depth = depth;
            break;
        }
        i = (i + 1) & SHALLOW_MASK;
    }

    /* --- Probe recent --- */
    uint64_t now = ++g_recent_clock;
    i = (size_t)key & RECENT_MASK;
    for (size_t probe = 0; probe < RECENT_CAP; probe++) {
        if (rtab[i].key == 0) break;
        if (rtab[i].key == key) {
            seen_recent = 1;
            rtab[i].clock = now;     /* refresh on hit */
            if (rtab[i].depth <= depth) hit_dup = 1;
            else rtab[i].depth = depth;
            break;
        }
        i = (i + 1) & RECENT_MASK;
    }

    if (hit_dup) return 1;

    /* --- Insert into shallow if missing --- */
    if (!seen_shallow) {
        if (g_shallow_count * 5 >= (long long)SHALLOW_CAP * 4) {
            shallow_evict();
            /* shallow_evict frees the table and reassigns g_shallow, so
             * our hoisted local is now stale; reread for the insert. */
        }
        ShallowSlot * const stab2 = g_shallow;
        size_t j = (size_t)key & SHALLOW_MASK;
        while (stab2[j].key != 0) j = (j + 1) & SHALLOW_MASK;
        stab2[j].key = key;
        stab2[j].depth = depth;
        g_shallow_count++;
    }
    /* --- Insert into recent if missing --- */
    if (!seen_recent) {
        if (g_recent_count * 5 >= (long long)RECENT_CAP * 4) recent_evict();
        RecentSlot * const rtab2 = g_recent;
        size_t j = (size_t)key & RECENT_MASK;
        while (rtab2[j].key != 0) j = (j + 1) & RECENT_MASK;
        rtab2[j].key = key;
        rtab2[j].depth = depth;
        rtab2[j].clock = now;
        g_recent_count++;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Frontier (LIFO stack of BState).
 *
 * We use depth-first instead of breadth-first because BFS-by-depth gets
 * stuck enumerating millions of shallow states once variant 4 (un-consume)
 * lifts the per-state branching factor.  DFS reaches deep states quickly,
 * updates g_best_depth, and lets the shortcut check do its job along the
 * way.  Trade-off: when the search hits the time cap, the result is a
 * lower bound on the true optimum (only the explored subtree is sound).
 * The "exhausted" report still reflects a complete search.
 * ------------------------------------------------------------------------- */

static BState   *g_queue   = NULL;
static size_t    g_q_cap   = 0;
static long long g_q_tail  = 0;        /* stack top                              */
static long long g_q_peak  = 0;        /* high-water mark for stats              */

static void q_push(const BState *s) {
    if ((size_t)g_q_tail == g_q_cap) {
        g_q_cap = g_q_cap ? g_q_cap * 2 : 65536;
        g_queue = realloc(g_queue, g_q_cap * sizeof(BState));
        if (!g_queue) { perror("realloc queue"); exit(1); }
    }
    g_queue[g_q_tail++] = *s;
    if (g_q_tail > g_q_peak) g_q_peak = g_q_tail;
}

static int q_pop(BState *out) {
    if (g_q_tail == 0) return 0;
    *out = g_queue[--g_q_tail];
    return 1;
}

/* -------------------------------------------------------------------------
 * Search bookkeeping
 * ------------------------------------------------------------------------- */

static int      g_best_depth      = 0;
static BState   g_best_state;
static long long g_states_checked = 0;
static long long g_pruned_short   = 0;
static long long g_pruned_dedup   = 0;
static long long g_solver_calls   = 0;

/* Cross-exit overall best (for streaming new bests as they happen). */
static int             g_overall_best_depth = 0;
static BState          g_overall_best_state;
static int             g_overall_best_exit  = -1;
static struct timespec g_t_session_start;

/* New-best stream is debounced: each new-best event starts (or extends)
 * a STREAM_DEBOUNCE_S-second timer anchored to the *first* event since
 * the last print.  When the timer elapses, the latest pending state is
 * printed.  After printing, the timer resets and we wait for the next
 * new-best event before starting another timer. */
#define STREAM_DEBOUNCE_S  1.0
static double  g_first_new_best_time  = -1.0;  /* -1.0 = no timer running */
static int     g_pending_print_depth  = 0;     /* 0 = no pending */
static BState  g_pending_print_state;
static int     g_pending_print_exit   = -1;

static double session_elapsed(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec  - g_t_session_start.tv_sec)
         + (t.tv_nsec - g_t_session_start.tv_nsec) * 1e-9;
}

/* Forward decls — defined after print_puzzle_for_exit. */
static void stream_new_best(const BState *s);
static void flush_pending_new_best(double t);
static void print_puzzle_for_exit(const BState *s, int exit_pos);

/* -------------------------------------------------------------------------
 * Build the partial puzzle for the per-step shortcut check.
 * walls = active region minus committed_empty, plus inactive cells (always walls).
 * ------------------------------------------------------------------------- */
static void build_partial_puzzle(const BState *s, Puzzle *pz) {
    memset(pz, 0, sizeof *pz);
    pz->exit_pos     = g_exit_pos;
    pz->player_start = s->player_pos;
    pz->num_blocks   = s->nblocks;
    for (int i = 0; i < s->nblocks; i++) {
        pz->block_pos[i]      = s->block_pos[i];
        pz->block_pushable[i] = s->block_mask[i];
    }
    /* All tracked holes are active; the forward solver assumes initial-active. */
    pz->num_holes = s->nholes;
    for (int i = 0; i < s->nholes; i++) {
        pz->hole_pos[i] = s->hole_pos[i];
    }
    /* walls: NCELLS cells that are NOT committed_empty.  Inactive cells
     * (outside g_active_mask), block cells, and hole cells are all not-wall
     * (block/hole cells are in committed_empty by construction). */
    pz->walls = ((1u << NCELLS) - 1) & ~s->committed_empty;
}

/* Cutoff-aware shortcut check.  We only care whether a forward path of
 * length < depth exists (which would be a shortcut).  By a parity argument
 * — every move flips the player-cell color, so forward_solve has fixed
 * parity matching the cell-distance to exit — forward_solve at this state
 * has the *opposite* parity from the parent's depth.  Since the parent's
 * depth was d and forward_solve(parent) = d, this state's forward_solve
 * is in {1, 3, 5, ..., d+1}∩{same parity as d+1}.  Anything < d+1 is
 * therefore at most d-1 (one parity step below).  So we only need to
 * search for paths of length <= depth-2 (depth = d+1 here, so d-1 = depth-2).
 *
 * Returns:
 *    >= 0 : a real shortcut of that length exists — caller should prune.
 *    -1   : no shortcut within the cutoff — caller should accept the state.
 *    -2   : solver overflow (treat as prune for safety). */
static int shortcut_check(const BState *s) {
    Puzzle pz;
    build_partial_puzzle(s, &pz);
    g_solver_calls++;
    int max_cost = s->depth - 2;
    return sokoban_solve_cutoff(&pz, NULL, NULL, max_cost);
}

/* -------------------------------------------------------------------------
 * Process one candidate successor: dedup, shortcut, enqueue.
 *
 * Dedup runs first (cheap hash probe).  If the state was already seen at
 * a depth ≤ the current depth, we skip it without paying for sokoban_solve.
 * Only states that survive dedup get the expensive shortcut check.  The
 * dedup table thus holds both shortcut-pass states and shortcut-fail
 * states from prior visits — that's correct: a shortcut-failing state
 * still fails at any *higher* depth, so future revisits should still be
 * pruned.  At a *lower* depth, dedup updates the recorded depth and
 * lets the visit through to re-check the shortcut (since the threshold
 * is lower, it may now pass).
 * ------------------------------------------------------------------------- */
static void try_successor(const BState *s) {
    g_states_checked++;
    /* Periodic debounce check: if a new-best timer is running and
     * STREAM_DEBOUNCE_S has elapsed since it started, fire the print.
     * Cheap unless a timer is active.  Multi-thread-safe via the mutex. */
    if ((g_states_checked & 8191) == 0 && g_first_new_best_time >= 0) {
        double t = session_elapsed();
        if ((t - g_first_new_best_time) >= STREAM_DEBOUNCE_S)
            flush_pending_new_best(t);
    }
    /* --num-walls cap: refuse states whose committed_empty has overgrown
     * the active region beyond the limit set by --num-walls.  Since
     * committed_empty is monotone-growing along any branch, once we
     * cross this threshold there's no way to recover. */
    if (__builtin_popcount(s->committed_empty & g_active_mask) > g_max_committed_in_active) {
        g_pruned_short++;       /* count under shortcut bucket; same effect */
        return;
    }
    /* --max-depth cap: stop exploring past the requested depth.  Used by
     * the wrapper's shallow scan to cover depths < the partition's
     * depth-3 seed floor. */
    if (s->depth > g_max_depth) {
        g_pruned_short++;
        return;
    }
    if (s->depth <= g_dupe_threshold) {
        uint64_t key = state_key(s);
        int dup = g_two_tables ? dedup_two_tables(key, s->depth)
                               : dedup_check_and_insert(key, s->depth);
        if (dup) {
            g_pruned_dedup++;
            return;
        }
    } else {
        g_skipped_dedup++;
    }
    int x = shortcut_check(s);
    /* shortcut_check (cutoff variant) returns:
     *   >= 0 : a shortcut of that length exists (definitely prune).
     *   -1   : no shortcut within the cutoff (accept).
     *   -2   : heap overflow (conservatively prune). */
    if (x >= 0 || x == -2) {
        g_pruned_short++;
        return;
    }
    if (s->depth > g_best_depth) {
        /* Reject candidates with a block sitting on the exit cell.  Those
         * are intermediate (transit) states only; they cannot be the puzzle
         * setup because blocks must not start at the exit.  Harmless when
         * --allow-exit-transit is off (no such state ever arises). */
        int block_on_exit = 0;
        for (int i = 0; i < s->nblocks; i++)
            if (s->block_pos[i] == g_exit_pos) { block_on_exit = 1; break; }
        if (!block_on_exit) {
            g_best_depth = s->depth;
            g_best_state = *s;
            if (s->depth > g_overall_best_depth) {
                g_overall_best_depth = s->depth;
                g_overall_best_state = *s;
                g_overall_best_exit  = g_exit_pos;
                stream_new_best(s);
            }
        }
    }
    q_push(s);
}

/* -------------------------------------------------------------------------
 * Successor enumeration from one state.
 * ------------------------------------------------------------------------- */
static void expand(const BState *s) {
    int P = s->player_pos;
    /* Fast occupancy lookup for the current state's blocks and active holes. */
    uint32_t blk_occ = 0;
    int8_t blk_idx_at[NCELLS];
    for (int i = 0; i < NCELLS; i++) blk_idx_at[i] = -1;
    for (int i = 0; i < s->nblocks; i++) {
        blk_occ |= (1u << s->block_pos[i]);
        blk_idx_at[s->block_pos[i]] = (int8_t)i;
    }
    uint32_t hole_occ = 0;
    for (int i = 0; i < s->nholes; i++) hole_occ |= (1u << s->hole_pos[i]);

    for (int D = 0; D < 4; D++) {
        int C = adj[P][D ^ 2];                   /* player came from C */
        if (C < 0) continue;
        if (C == g_exit_pos) continue;             /* optimal play doesn't revisit exit */
        if (!(g_walkable_mask & (1u << C))) continue;
        if (blk_occ  & (1u << C)) continue;        /* can't walk into a block */
        if (hole_occ & (1u << C)) continue;        /* can't walk into an active hole */

        uint32_t new_E = s->committed_empty | (1u << C);

        /* Successor 1: walk-back. */
        {
            BState ns = *s;
            ns.player_pos      = (int8_t)C;
            ns.committed_empty = new_E;
            ns.depth           = s->depth + 1;
            try_successor(&ns);
        }

        /* Strict-rule fast-out: when transit is disallowed, no push variant
         * may involve the exit cell at all. */
        if (!g_allow_exit_transit && P == g_exit_pos) continue;
        int B = adj[P][D];
        if (B < 0) continue;
        if (!g_allow_exit_transit && B == g_exit_pos) continue;
        if (!(g_walkable_mask & (1u << B))) continue;

        if (blk_idx_at[B] >= 0) {
            /* Successor 2: backward-push existing block from B to P.
             * Mask of that block gets bit D OR'd in. */
            int idx = blk_idx_at[B];
            BState ns = *s;
            ns.block_pos [idx]  = (int8_t)P;
            ns.block_mask[idx] |= (uint8_t)(1 << D);
            sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
            ns.player_pos      = (int8_t)C;
            ns.committed_empty = new_E;
            ns.depth           = s->depth + 1;
            try_successor(&ns);
        } else if (hole_occ & (1u << B)) {
            /* B has an active hole — variants 3 (no consume) is impossible since
             * a block can't sit on an active hole.  Variant 4 (un-consume) is
             * also impossible because B is already an active hole and the
             * un-consume would re-introduce one.  Skip both. */
        } else if (B == g_exit_pos || P == g_exit_pos) {
            /* Variants 3 and 4 introduce a new block at P and (variant 4)
             * a new hole at B.  Even with --allow-exit-transit, blocks and
             * holes must not *originate* at the exit cell — they can only
             * transit through, which is variant 2's job. */
        } else {
            /* B has no block and no active hole.  Two possible variants: */

            /* Successor 3: introduce new block at B (continuously occupying B
             * for the entire backward trace up to now), then push it back to P. */
            if (!(s->committed_empty & (1u << B)) && s->nblocks < g_max_blocks) {
                BState ns = *s;
                ns.block_pos [ns.nblocks] = (int8_t)P;
                ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                ns.nblocks++;
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E | (1u << B);
                ns.depth           = s->depth + 1;
                try_successor(&ns);
            }

            /* Successor 4: backward un-consume.  Reverses a forward push that
             * landed a block onto an active hole, consuming both.  We
             * introduce both: a new block at P (the cell from which it was
             * pushed) with mask D, and a new active hole at B (the cell where
             * it landed and was consumed).  In the puzzle, the hole has been
             * at B since setup; we just hadn't tracked it because it was
             * already inactive when we entered the trace.
             *
             * Skipped entirely under --holeless. */
            /* When --fixedholes is non-empty, restrict variant 4's hole
             * placement: B must be one of the listed cells. */
            if (g_fixed_nholes > 0 && !(g_fixed_holes_mask & (1u << B))) {
                /* fall through — variant 4 is forbidden at this B */
            } else
            if (!g_holeless && s->nblocks < g_max_blocks && s->nholes < g_max_holes) {
                BState ns = *s;
                ns.block_pos [ns.nblocks] = (int8_t)P;
                ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                ns.nblocks++;
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.hole_pos[ns.nholes++] = (int8_t)B;
                sort_holes(ns.hole_pos, ns.nholes);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E | (1u << B);
                ns.depth           = s->depth + 1;
                try_successor(&ns);
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * Final puzzle output
 * ------------------------------------------------------------------------- */
static void print_puzzle_for_exit(const BState *s, int exit_pos) {
    char grid[ROWS][COLS + 1];
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) grid[r][c] = '.';
        grid[r][COLS] = '\0';
    }
    /* Walls = NCELLS cells not in committed_empty.  In the active region,
     * unconstrained cells become walls; inactive cells are also walls. */
    for (int i = 0; i < NCELLS; i++)
        if (!(s->committed_empty & (1u << i))) grid[i / COLS][i % COLS] = '#';
    /* Holes (active in puzzle setup) — drawn before blocks so blocks shadow
     * any cell that was a block in our state but not a hole. */
    for (int i = 0; i < s->nholes; i++) {
        int p = s->hole_pos[i];
        grid[p / COLS][p % COLS] = 'O';
    }
    for (int i = 0; i < s->nblocks; i++) {
        int p = s->block_pos[i];
        grid[p / COLS][p % COLS] = (char)('A' + i);
    }
    grid[exit_pos / COLS][exit_pos % COLS] = '$';
    /* Player drawn last so it overlays whatever cell it stands on. */
    grid[s->player_pos / COLS][s->player_pos % COLS] = '@';

    for (int r = 0; r < g_grid_rows; r++) {
        printf("  ");
        for (int c = 0; c < g_grid_cols; c++) putchar(grid[r][c]);
        if (r < s->nblocks) {
            uint8_t m = s->block_mask[r];
            printf("   %c=[%s%s%s%s]",
                   (char)('A' + r),
                   m & 1 ? "U" : "", m & 2 ? "R" : "",
                   m & 4 ? "D" : "", m & 8 ? "L" : "");
        }
        putchar('\n');
    }
    for (int i = g_grid_rows; i < s->nblocks; i++) {
        uint8_t m = s->block_mask[i];
        printf("            %c=[%s%s%s%s]\n",
               (char)('A' + i),
               m & 1 ? "U" : "", m & 2 ? "R" : "",
               m & 4 ? "D" : "", m & 8 ? "L" : "");
    }
}

/* Print whatever is pending right now and reset the debounce timer.
 * Caller has verified the timer has elapsed and there's something pending. */
static void flush_pending_new_best(double t) {
    if (g_pending_print_depth == 0) return;
    /* Human-readable elapsed: pick largest unit ≥ 1 (s / m / h),
     * 2 decimal places. */
    double val;
    const char *unit;
    if (t < 60.0)        { val = t;            unit = "s"; }
    else if (t < 3600.0) { val = t / 60.0;     unit = "m"; }
    else                 { val = t / 3600.0;   unit = "h"; }
    printf("%d (%.1f%s)\n", g_pending_print_depth, val, unit);
    (void)g_pending_print_exit;
    print_puzzle_for_exit(&g_pending_print_state, g_pending_print_exit);
    putchar('\n');
    fflush(stdout);
    g_pending_print_depth = 0;
    g_first_new_best_time = -1.0;     /* timer reset; next new best will start a fresh one */
}

/* Debounced "new best" event.  Always overwrite the pending slot with
 * the latest state.  If no timer is running, start one; otherwise leave
 * the existing timer alone — the print will fire STREAM_DEBOUNCE_S
 * seconds after the FIRST event in this batch (handled by the periodic
 * check in try_successor). */
static void stream_new_best(const BState *s) {
    g_pending_print_depth = s->depth;
    g_pending_print_state = *s;
    g_pending_print_exit  = g_exit_pos;
    if (g_first_new_best_time < 0)
        g_first_new_best_time = session_elapsed();
}

/* -------------------------------------------------------------------------
 * Task enumeration  (--list-tasks, --task-id N).
 *
 * Generates all valid (1st-backstep, 2nd-backstep) sequences for the
 * current exit and groups the resulting depth-2 states by the user's
 * partition: (transit, hole_loc).
 *
 *   transit  : 1 if a block was moved onto the exit on the 1st backstep
 *              (only possible when --allow-exit-transit is set), else 0.
 *   hole_loc : cell index where a new hole was placed on the 2nd backstep
 *              via variant 4, else -1.
 *
 * Each task ID maps to one (transit, hole_loc) bucket, which holds a
 * list of seed depth-2 states.  When --task-id N is set, run_exit_search
 * pushes only those seeds (instead of the standard depth-0 roots).
 * ------------------------------------------------------------------------- */

#define MAX_TASK_SEEDS    1024
#define MAX_TASKS_PER_EXIT 32

typedef struct {
    BState seeds[MAX_TASK_SEEDS];
    int    n_seeds;
    int    transit;     /* 0 or 1 */
    int    hole_loc;    /* -1 (none) or cell index */
} TaskGroup;

/* Compute all valid backward-step successors of state s into out_buf.
 * Same enumeration logic as expand(), but appends raw successors instead
 * of feeding them through try_successor (no dedup, no shortcut check). */
static int enumerate_successors(const BState *s, BState *out_buf, int max_out) {
    int n = 0;
    int P = s->player_pos;

    uint32_t blk_occ = 0;
    int8_t blk_idx_at[NCELLS];
    for (int i = 0; i < NCELLS; i++) blk_idx_at[i] = -1;
    for (int i = 0; i < s->nblocks; i++) {
        blk_occ |= (1u << s->block_pos[i]);
        blk_idx_at[s->block_pos[i]] = (int8_t)i;
    }
    uint32_t hole_occ = 0;
    for (int i = 0; i < s->nholes; i++) hole_occ |= (1u << s->hole_pos[i]);

    for (int D = 0; D < 4; D++) {
        int C = adj[P][D ^ 2];
        if (C < 0) continue;
        if (C == g_exit_pos) continue;
        if (!(g_walkable_mask & (1u << C))) continue;
        if (blk_occ  & (1u << C)) continue;
        if (hole_occ & (1u << C)) continue;

        uint32_t new_E = s->committed_empty | (1u << C);

        /* Variant 1 — walk-back. */
        if (n < max_out) {
            BState ns = *s;
            ns.player_pos      = (int8_t)C;
            ns.committed_empty = new_E;
            ns.depth           = s->depth + 1;
            out_buf[n++] = ns;
        }

        /* Push variants. */
        if (!g_allow_exit_transit && P == g_exit_pos) continue;
        int B = adj[P][D];
        if (B < 0) continue;
        if (!g_allow_exit_transit && B == g_exit_pos) continue;
        if (!(g_walkable_mask & (1u << B))) continue;

        if (blk_idx_at[B] >= 0) {
            /* Variant 2 — push existing block. */
            if (n < max_out) {
                int idx = blk_idx_at[B];
                BState ns = *s;
                ns.block_pos [idx]  = (int8_t)P;
                ns.block_mask[idx] |= (uint8_t)(1 << D);
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E;
                ns.depth           = s->depth + 1;
                out_buf[n++] = ns;
            }
        } else if (hole_occ & (1u << B)) {
            /* Skip — block can't sit on active hole. */
        } else if (B == g_exit_pos || P == g_exit_pos) {
            /* Skip — variants 3 and 4 forbidden at exit. */
        } else {
            /* Variant 3 — introduce new block at B. */
            if (n < max_out
                && !(s->committed_empty & (1u << B))
                && s->nblocks < g_max_blocks) {
                BState ns = *s;
                ns.block_pos [ns.nblocks] = (int8_t)P;
                ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                ns.nblocks++;
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E | (1u << B);
                ns.depth           = s->depth + 1;
                out_buf[n++] = ns;
            }
            /* Variant 4 — un-consume (introduce block + hole). */
            int v4_allowed = 1;
            if (g_fixed_nholes > 0 && !(g_fixed_holes_mask & (1u << B))) v4_allowed = 0;
            if (g_holeless) v4_allowed = 0;
            if (s->nblocks >= g_max_blocks || s->nholes >= g_max_holes) v4_allowed = 0;
            if (v4_allowed && n < max_out) {
                BState ns = *s;
                ns.block_pos [ns.nblocks] = (int8_t)P;
                ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                ns.nblocks++;
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.hole_pos[ns.nholes++] = (int8_t)B;
                sort_holes(ns.hole_pos, ns.nholes);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E | (1u << B);
                ns.depth           = s->depth + 1;
                out_buf[n++] = ns;
            }
        }
    }

    return n;
}

/* Find or create the task-bucket matching (transit, hole_loc).  Returns
 * -1 if no slot available. */
static int task_bucket_for(TaskGroup *tasks, int *p_n_tasks, int max_tasks,
                           int transit, int hole_loc) {
    for (int t = 0; t < *p_n_tasks; t++) {
        if (tasks[t].transit == transit && tasks[t].hole_loc == hole_loc) return t;
    }
    if (*p_n_tasks >= max_tasks) return -1;
    int t = (*p_n_tasks)++;
    tasks[t].transit  = transit;
    tasks[t].hole_loc = hole_loc;
    tasks[t].n_seeds  = 0;
    return t;
}

/* Enumerate the partition tasks for one exit.  Reads g_exit_pos (must
 * be set by the caller).  Returns the number of non-empty task buckets. */
static int enumerate_tasks_for_exit(int exit_pos, TaskGroup *tasks, int max_tasks) {
    int n_tasks = 0;
    int saved_exit = g_exit_pos;
    g_exit_pos = exit_pos;

    /* Build the depth-0 roots. */
    BState roots[8];
    int n_roots = 0;

    /* Standard root. */
    {
        BState r = {0};
        r.player_pos      = (int8_t)exit_pos;
        r.nblocks         = 0;
        r.nholes          = 0;
        r.committed_empty = 1u << exit_pos;
        r.depth           = 0;
        roots[n_roots++]  = r;
    }
    /* Push-off-exit seeds (only when --allow-exit-transit).  These are
     * depth-1 states representing "right before the push-off win move":
     * player at Y=adj[exit][D'^2], block at exit, mask={D'}.  Seeding
     * here (rather than at depth 0 with player at exit) avoids spurious
     * variant-1 walk-back successors from the exit cell — see the
     * matching comment in run_exit_search.  Note: tasks built from these
     * seeds reach depth-3 in two enumerate_successors steps, while
     * standard-root tasks need three; this is the natural consequence
     * of the push-off win consuming one move. */
    if (g_allow_exit_transit && g_max_blocks >= 1) {
        for (int D = 0; D < 4; D++) {
            int X = adj[exit_pos][D];
            if (X < 0) continue;
            if (!(g_walkable_mask & (1u << X))) continue;
            if (g_fixed_holes_mask & (1u << X)) continue;
            int Y = adj[exit_pos][D ^ 2];
            if (Y < 0) continue;
            if (!(g_walkable_mask & (1u << Y))) continue;
            BState r = {0};
            r.player_pos      = (int8_t)Y;
            r.nblocks         = 1;
            r.nholes          = 0;
            r.committed_empty = (1u << exit_pos) | (1u << X) | (1u << Y);
            r.depth           = 1;
            r.block_pos [0]   = (int8_t)exit_pos;
            r.block_mask[0]   = (uint8_t)(1u << D);
            if (n_roots < (int)(sizeof roots / sizeof roots[0]))
                roots[n_roots++] = r;
        }
    }

    /* For each root, enumerate depth-1, depth-2, depth-3.  Bucket each
     * depth-3 state by:
     *   transit  : did the 1st backstep put a block on the exit?
     *   hole_loc : cell index of a hole on an exit-adjacent cell that
     *              was introduced AFTER the 1st backstep (i.e. on
     *              backstep 2 or 3); -1 if no such hole.
     *
     * We exclude holes already present at depth 1 because those don't
     * distinguish the depth-3 subtrees — they're a property of the root,
     * not of the partitioning backsteps.  In practice variant 4 from
     * depth 0 is always blocked (P=exit), so d1->nholes is always 0,
     * but we code the exclusion explicitly to match the partition spec.
     *
     * Exit-adjacent buckets (one per direction A=U, B=R, C=D, D=L,
     * clipped by off-board/walls) plus the catch-all "no qualifying
     * hole" give up to 5 hole buckets per transit value. */
    uint32_t exit_adj_mask = 0;
    for (int D = 0; D < 4; D++) {
        int X = adj[exit_pos][D];
        if (X < 0) continue;
        exit_adj_mask |= (1u << X);
    }
    for (int r = 0; r < n_roots; r++) {
        BState d1_buf[16];
        int n_d1 = enumerate_successors(&roots[r], d1_buf, 16);
        for (int i = 0; i < n_d1; i++) {
            BState *d1 = &d1_buf[i];
            int transit = 0;
            for (int b = 0; b < d1->nblocks; b++) {
                if (d1->block_pos[b] == exit_pos) { transit = 1; break; }
            }
            uint32_t d1_holes_mask = 0;
            for (int h = 0; h < d1->nholes; h++)
                d1_holes_mask |= (1u << d1->hole_pos[h]);
            BState d2_buf[16];
            int n_d2 = enumerate_successors(d1, d2_buf, 16);
            for (int j = 0; j < n_d2; j++) {
                BState *d2 = &d2_buf[j];
                BState d3_buf[16];
                int n_d3 = enumerate_successors(d2, d3_buf, 16);
                for (int k = 0; k < n_d3; k++) {
                    BState *d3 = &d3_buf[k];
                    int hole_loc = -1;
                    for (int h = 0; h < d3->nholes; h++) {
                        int hp = d3->hole_pos[h];
                        if (!(exit_adj_mask & (1u << hp))) continue;
                        if (d1_holes_mask & (1u << hp)) continue;
                        hole_loc = hp;
                        break;
                    }
                    int t = task_bucket_for(tasks, &n_tasks, max_tasks, transit, hole_loc);
                    if (t >= 0 && tasks[t].n_seeds < MAX_TASK_SEEDS) {
                        tasks[t].seeds[tasks[t].n_seeds++] = *d3;
                    }
                }
            }
        }
    }

    g_exit_pos = saved_exit;
    return n_tasks;
}

/* -------------------------------------------------------------------------
 * Per-exit search.  Returns elapsed seconds; updates the per-exit globals
 * (best, counters) as side effects.  Stops early when the *remaining*
 * shared budget runs out.
 * ------------------------------------------------------------------------- */
static double elapsed_s(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* When non-NULL, run_exit_search uses this list of seeds (depth-2 states
 * from a task) instead of generating depth-0 roots. */
static const BState *g_task_seeds = NULL;
static int           g_task_seed_count = 0;

static double run_exit_search(double remaining_s, int *out_exhausted, int *out_dedup_full) {
    /* Reset per-exit state. */
    if (g_two_tables) {
        memset(g_shallow, 0, SHALLOW_CAP * sizeof *g_shallow);
        memset(g_recent,  0, RECENT_CAP  * sizeof *g_recent);
        g_shallow_count = 0;
        g_recent_count  = 0;
        g_recent_clock  = 1;
        g_evictions_shallow = 0;
        g_evictions_recent  = 0;
    } else {
        memset(g_visited, 0, HASH_CAP * sizeof *g_visited);
    }
    g_visited_count  = 0;
    g_q_tail         = 0;
    g_q_peak         = 0;
    g_best_depth     = 0;
    g_states_checked = 0;
    g_pruned_short   = 0;
    g_pruned_dedup   = 0;
    g_solver_calls   = 0;
    g_dedup_full     = 0;
    g_skipped_dedup  = 0;

    if (g_task_seeds && g_task_seed_count > 0) {
        /* Task mode: skip default depth-0 roots; feed each seed through
         * try_successor.  This is what registers the seed itself as a
         * candidate "best" — bypassing it (a previous bug) caused tight
         * --num-walls runs to miss valid depth-3 puzzles whose only
         * successors were over the popcount cap.  try_successor also
         * handles dedup-insert, the shortcut check, and q_push. */
        g_best_state = g_task_seeds[0];
        for (int i = 0; i < g_task_seed_count; i++) {
            BState s = g_task_seeds[i];
            try_successor(&s);
        }
    } else {
        /* Standard root: winning move was a free walk onto an empty exit. */
        BState init = {
            .player_pos      = (int8_t)g_exit_pos,
            .nblocks         = 0,
            .nholes          = 0,
            .committed_empty = 1u << g_exit_pos,
            .depth           = 0,
        };
        g_best_state = init;
        {
            uint64_t k = state_key(&init);
            if (g_two_tables) dedup_two_tables(k, 0);
            else              dedup_check_and_insert(k, 0);
        }
        q_push(&init);

        /* Push-off-exit seeds (only when --allow-exit-transit).  The win
         * move is "player at Y=adj[exit][D'^2] walked to exit pushing
         * block from exit to X=adj[exit][D']"; we seed the depth-1 state
         * that immediately *precedes* this win move (player at Y, block
         * at exit).  Seeding the post-win state at depth 0 instead would
         * also let expand() generate variant-1 walk-backs from the exit,
         * which are inconsistent with the win move and produce phantom
         * puzzles whose transit block's mask points into a wall in the
         * final layout.  Each seed carries one block, so it's only valid
         * when --num-blocks allows at least one. */
        if (g_allow_exit_transit && g_max_blocks >= 1) {
            for (int D = 0; D < 4; D++) {
                int X = adj[g_exit_pos][D];
                if (X < 0) continue;
                if (!(g_walkable_mask & (1u << X))) continue;
                if (g_fixed_holes_mask & (1u << X)) continue;
                int Y = adj[g_exit_pos][D ^ 2];
                if (Y < 0) continue;
                if (!(g_walkable_mask & (1u << Y))) continue;
                BState seed = {
                    .player_pos      = (int8_t)Y,
                    .nblocks         = 1,
                    .nholes          = 0,
                    .committed_empty = (1u << g_exit_pos) | (1u << X) | (1u << Y),
                    .depth           = 1,
                };
                seed.block_pos [0] = (int8_t)g_exit_pos;
                seed.block_mask[0] = (uint8_t)(1u << D);
                {
                    uint64_t k = state_key(&seed);
                    if (g_two_tables) dedup_two_tables(k, seed.depth);
                    else              dedup_check_and_insert(k, seed.depth);
                }
                q_push(&seed);
            }
        }
    }

    struct timespec t0, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int exhausted = 1;
    long long iter = 0;
    int unlimited = (remaining_s <= 0);
    while (1) {
        BState s;
        if (!q_pop(&s)) break;
        expand(&s);
        if (g_dedup_full) { exhausted = 0; break; }
        if (!unlimited && (++iter & 1023) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            if (elapsed_s(t0, t_now) >= remaining_s) { exhausted = 0; break; }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    *out_exhausted   = exhausted;
    *out_dedup_full  = g_dedup_full;
    return elapsed_s(t0, t_now);
}

/* -------------------------------------------------------------------------
 * CLI
 * ------------------------------------------------------------------------- */
static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--grid RxC] [--time SEC] [--exit N] [--allow-exit-transit]\n"
        "                       [--fixedwalls c,c,...] [--fixedholes c,c,...]\n"
        "  --grid RxC            active region (R,C in [1..5]); cells outside are walls.  Default: 5x5\n"
        "  --time SEC            wall-clock cap in seconds, shared across exits.  Default: %.0f\n"
        "                          0 = no time limit (run until queue is empty for every exit)\n"
        "  --exit N              restrict to one exit cell.  Default: iterate canonical {0,1,2,6,7,12}\n"
        "  --exitloc c,c,...     iterate the listed exit cells (comma-separated).  Overrides the\n"
        "                          canonical default; --exit takes priority if both are given.\n"
        "  --allow-exit-transit  allow blocks to be pushed onto the exit and back off during play\n"
        "                          (block still cannot start at exit at puzzle setup).  Default: off.\n"
        "  --holeless            forbid all holes (disables variant-4 un-consume).  Incompatible\n"
        "                          with --fixedholes.  Default: off (holes allowed).\n"
        "  --two-tables          use a two-table dedup that auto-evicts: a SHALLOW table keeps\n"
        "                          the lowest-depth states (evicts deep ones when full) and a\n"
        "                          RECENT table keeps recently-hit states (evicts old ones when\n"
        "                          full).  No more dedup overflow at the cost of some redundant\n"
        "                          subtree exploration.  Default: off (single table).\n"
        "  --dupe-threshold N    only dedup states at depth <= N; deeper states free-fly without\n"
        "                          dedup tracking.  Trades wall-clock for memory headroom on long\n"
        "                          runs that would otherwise hit the dedup table cap.  Default: no\n"
        "                          threshold (all states deduped).\n"
        "  --fixedwalls c,c,...  cells that must be walls in the reported puzzle setup.\n"
        "                          The search prevents these cells from being walked on or used\n"
        "                          for blocks/holes.  Default: none.\n"
        "  --num-walls N         require at least N cells of the active region to remain walls\n"
        "                          in the reported puzzle (any cells, the search picks).  Counted\n"
        "                          as TOTAL walls including any --fixedwalls.  Default: 0.\n"
        "  --num-blocks N        cap the number of blocks introduced by the search to at most N.\n"
        "                          Default: %d (no extra constraint beyond MAX_BLOCKS).\n"
        "  --num-holes N         cap the number of active holes to at most N (counts --fixedholes).\n"
        "                          Default: %d (no extra constraint beyond MAX_HOLES).\n"
        "  --fixedholes c,c,...  if non-empty, holes are *only* allowed at these cells.  The\n"
        "                          search may use 0..N holes drawn from this list.  Pairs cleanly\n"
        "                          with --num-holes (cap) and --holeless (forbid all).  Default:\n"
        "                          none (holes may appear anywhere in the active region).\n"
        "\nVisited table is %lu MB (2^%d slots × 16 B).  On overflow the search\n"
        "exits the current root gracefully.  To enlarge, recompile with -DHASH_LG2=N.\n"
        "\nNew bests are streamed to stdout as they are found, prefixed with elapsed time.\n",
        prog, DEFAULT_CAP_S, MAX_BLOCKS, MAX_HOLES,
        (unsigned long)((size_t)HASH_CAP * sizeof(HashSlot) / (1024*1024)), HASH_LG2);
}

static int parse_grid(const char *s) {
    int r, c;
    if (sscanf(s, "%dx%d", &r, &c) != 2) return 0;
    if (r < 1 || r > 5 || c < 1 || c > 5) return 0;
    g_grid_rows = r;
    g_grid_cols = c;
    return 1;
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        } else if (strcmp(argv[i], "--grid") == 0) {
            if (++i >= argc || !parse_grid(argv[i])) {
                fprintf(stderr, "error: --grid requires RxC with R,C in [1..5]\n"); return 1;
            }
        } else if (strcmp(argv[i], "--time") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --time requires SEC\n"); return 1; }
            g_time_cap_s = atof(argv[i]);
            if (g_time_cap_s < 0) { fprintf(stderr, "error: --time must be >= 0 (0 = no limit)\n"); return 1; }
        } else if (strcmp(argv[i], "--exit") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --exit requires N\n"); return 1; }
            g_only_exit = atoi(argv[i]);
            if (g_only_exit < 0 || g_only_exit >= NCELLS) {
                fprintf(stderr, "error: --exit must be 0..%d\n", NCELLS - 1); return 1;
            }
        } else if (strcmp(argv[i], "--exitloc") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --exitloc requires c,c,...\n"); return 1; }
            const char *p = argv[i];
            uint32_t seen = 0;
            g_n_only_exits = 0;
            while (*p) {
                char *end;
                long v = strtol(p, &end, 10);
                if (end == p || v < 0 || v >= NCELLS) {
                    fprintf(stderr, "error: invalid cell '%s' in --exitloc\n", p); return 1;
                }
                if (seen & (1u << v)) {
                    fprintf(stderr, "error: duplicate cell %ld in --exitloc\n", v); return 1;
                }
                seen |= (1u << v);
                g_only_exit_list[g_n_only_exits++] = (int)v;
                p = end;
                if (*p == ',') p++;
                else if (*p) { fprintf(stderr, "error: expected ',' in --exitloc\n"); return 1; }
            }
            if (g_n_only_exits == 0) {
                fprintf(stderr, "error: --exitloc requires at least one cell\n"); return 1;
            }
        } else if (strcmp(argv[i], "--allow-exit-transit") == 0) {
            g_allow_exit_transit = 1;
        } else if (strcmp(argv[i], "--holeless") == 0) {
            g_holeless = 1;
        } else if (strcmp(argv[i], "--two-tables") == 0) {
            g_two_tables = 1;
        } else if (strcmp(argv[i], "--list-tasks") == 0) {
            g_list_tasks = 1;
        } else if (strcmp(argv[i], "--task-id") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --task-id requires N\n"); return 1; }
            g_only_task = atoi(argv[i]);
            if (g_only_task < 0) { fprintf(stderr, "error: --task-id must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--num-walls") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --num-walls requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) { fprintf(stderr, "error: --num-walls must be >= 0\n"); return 1; }
            g_min_walls = n;
        } else if (strcmp(argv[i], "--max-depth") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --max-depth requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) { fprintf(stderr, "error: --max-depth must be >= 0\n"); return 1; }
            g_max_depth = n;
        } else if (strcmp(argv[i], "--num-blocks") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --num-blocks requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0 || n > MAX_BLOCKS) {
                fprintf(stderr, "error: --num-blocks must be in [0..%d]\n", MAX_BLOCKS); return 1;
            }
            g_max_blocks = n;
        } else if (strcmp(argv[i], "--num-holes") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --num-holes requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0 || n > MAX_HOLES) {
                fprintf(stderr, "error: --num-holes must be in [0..%d]\n", MAX_HOLES); return 1;
            }
            g_max_holes = n;
        } else if (strcmp(argv[i], "--dupe-threshold") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --dupe-threshold requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) {
                fprintf(stderr, "error: --dupe-threshold must be >= 0\n"); return 1;
            }
            g_dupe_threshold = n;
        } else if (strcmp(argv[i], "--fixedwalls") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --fixedwalls requires c,c,...\n"); return 1; }
            const char *p = argv[i];
            g_fixed_walls_mask = 0;
            while (*p) {
                char *end;
                long v = strtol(p, &end, 10);
                if (end == p || v < 0 || v >= NCELLS) {
                    fprintf(stderr, "error: invalid cell '%s' in --fixedwalls\n", p); return 1;
                }
                if (g_fixed_walls_mask & (1u << v)) {
                    fprintf(stderr, "error: duplicate cell %ld in --fixedwalls\n", v); return 1;
                }
                g_fixed_walls_mask |= (1u << v);
                p = end;
                if (*p == ',') p++;
                else if (*p) { fprintf(stderr, "error: expected ',' in --fixedwalls\n"); return 1; }
            }
        } else if (strcmp(argv[i], "--fixedholes") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --fixedholes requires c,c,...\n"); return 1; }
            const char *p = argv[i];
            g_fixed_nholes = 0;
            g_fixed_holes_mask = 0;
            while (*p) {
                char *end;
                long v = strtol(p, &end, 10);
                if (end == p || v < 0 || v >= NCELLS) {
                    fprintf(stderr, "error: invalid cell '%s' in --fixedholes\n", p); return 1;
                }
                if (g_fixed_holes_mask & (1u << v)) {
                    fprintf(stderr, "error: duplicate cell %ld in --fixedholes\n", v); return 1;
                }
                if (g_fixed_nholes >= MAX_HOLES) {
                    fprintf(stderr, "error: too many fixed holes (max %d)\n", MAX_HOLES); return 1;
                }
                g_fixed_hole_pos[g_fixed_nholes++] = (int)v;
                g_fixed_holes_mask |= (1u << v);
                p = end;
                if (*p == ',') p++;
                else if (*p) { fprintf(stderr, "error: expected ',' in --fixedholes\n"); return 1; }
            }
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    /* Build active mask from grid dimensions. */
    g_active_mask = 0;
    for (int r = 0; r < g_grid_rows; r++)
        for (int c = 0; c < g_grid_cols; c++)
            g_active_mask |= 1u << (r * COLS + c);

    sokoban_init();

    /* Validate fixed holes against the active region. */
    for (int i = 0; i < g_fixed_nholes; i++) {
        int h = g_fixed_hole_pos[i];
        if (!(g_active_mask & (1u << h))) {
            fprintf(stderr, "error: --fixedholes cell %d is outside the %dx%d active region\n",
                    h, g_grid_rows, g_grid_cols);
            return 1;
        }
    }

    /* Validate fixed walls: must be in active region, must not coincide
     * with a fixed hole. */
    if (g_fixed_walls_mask & ~g_active_mask) {
        fprintf(stderr, "error: --fixedwalls includes a cell outside the %dx%d active region\n",
                g_grid_rows, g_grid_cols);
        return 1;
    }
    if (g_fixed_walls_mask & g_fixed_holes_mask) {
        fprintf(stderr, "error: --fixedwalls and --fixedholes overlap\n");
        return 1;
    }
    if (g_holeless && g_max_holes > 0) {
        /* --holeless implies cap=0; just enforce. */
        g_max_holes = 0;
    }
    /* Note: --holeless + --fixedholes is now consistent — both restrict
     * holes; combined they mean "no holes at all" which is fine.
     * --num-holes can be any value <= MAX_HOLES; the combination "at most
     * N holes, only at these cells" simply means up to min(N, len) holes. */

    /* Effective walkable region: active region minus fixed walls. */
    g_walkable_mask = g_active_mask & ~g_fixed_walls_mask;

    /* Translate --num-walls into a popcount cap on committed_empty
     * within the active region. */
    {
        int active_size = __builtin_popcount(g_active_mask);
        if (g_min_walls >= active_size) {
            fprintf(stderr, "error: --num-walls (%d) >= active region size (%d) leaves no room for the puzzle\n",
                    g_min_walls, active_size);
            return 1;
        }
        g_max_committed_in_active = active_size - g_min_walls;
    }

    if (g_two_tables) {
        g_shallow = calloc(SHALLOW_CAP, sizeof *g_shallow);
        g_recent  = calloc(RECENT_CAP,  sizeof *g_recent);
        if (!g_shallow || !g_recent) { perror("calloc two-tables"); return 1; }
    } else {
        g_visited = calloc(HASH_CAP, sizeof *g_visited);
        if (!g_visited) { perror("calloc visited"); return 1; }
    }

    /* Build the list of exits to search.  Sources, in priority order:
     *   1. --exit N             single cell (overrides everything).
     *   2. --exitloc c,c,...    explicit list.
     *   3. default              canonical {0,1,2,6,7,12} in active region.
     * In all cases, cells that conflict with --fixedholes or --fixedwalls
     * are invalid (single-cell forms error; lists silently filter). */
    int exits[NCELLS];
    int n_exits = 0;
    if (g_only_exit >= 0) {
        if (!(g_active_mask & (1u << g_only_exit))) {
            fprintf(stderr, "error: --exit %d is outside the %dx%d active region\n",
                    g_only_exit, g_grid_rows, g_grid_cols);
            return 1;
        }
        if (g_fixed_holes_mask & (1u << g_only_exit)) {
            fprintf(stderr, "error: --exit %d coincides with a fixed hole\n", g_only_exit);
            return 1;
        }
        if (g_fixed_walls_mask & (1u << g_only_exit)) {
            fprintf(stderr, "error: --exit %d coincides with a fixed wall\n", g_only_exit);
            return 1;
        }
        exits[n_exits++] = g_only_exit;
    } else if (g_n_only_exits > 0) {
        for (int i = 0; i < g_n_only_exits; i++) {
            int e = g_only_exit_list[i];
            if (!(g_active_mask & (1u << e))) {
                fprintf(stderr, "error: --exitloc cell %d is outside the %dx%d active region\n",
                        e, g_grid_rows, g_grid_cols);
                return 1;
            }
            if (g_fixed_holes_mask & (1u << e)) {
                fprintf(stderr, "error: --exitloc cell %d coincides with a fixed hole\n", e);
                return 1;
            }
            if (g_fixed_walls_mask & (1u << e)) {
                fprintf(stderr, "error: --exitloc cell %d coincides with a fixed wall\n", e);
                return 1;
            }
            exits[n_exits++] = e;
        }
    } else {
        for (int i = 0; i < NUM_EXIT_CELLS; i++) {
            int e = CANONICAL_EXITS[i];
            if (!(g_active_mask & (1u << e))) continue;
            if (g_fixed_holes_mask & (1u << e)) continue;  /* exit can't be a hole */
            if (g_fixed_walls_mask & (1u << e)) continue;  /* exit can't be a wall */
            exits[n_exits++] = e;
        }
        if (n_exits == 0) {
            fprintf(stderr, "error: no canonical exit fits in the active region (and is neither a fixed hole nor wall)\n");
            return 1;
        }
    }

    /* Initialize cross-exit overall best (streamed during search). */
    g_overall_best_depth = 0;
    g_overall_best_exit  = -1;
    g_first_new_best_time = -1.0;
    g_pending_print_depth = 0;

    long long total_states      = 0;
    long long total_pruned_short= 0;
    long long total_pruned_dedup= 0;
    long long total_solver_calls= 0;
    double   total_elapsed      = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &g_t_session_start);
    struct timespec t_global0 = g_t_session_start, t_global_now;

    /* Handle --list-tasks: enumerate tasks per exit, print total, exit. */
    if (g_list_tasks) {
        int total = 0;
        TaskGroup tasks[MAX_TASKS_PER_EXIT];
        for (int ei = 0; ei < n_exits; ei++) {
            int n = enumerate_tasks_for_exit(exits[ei], tasks, MAX_TASKS_PER_EXIT);
            printf("exit %d: %d task%s\n", exits[ei], n, n == 1 ? "" : "s");
            for (int i = 0; i < n; i++) {
                printf("  task %d (global %d): transit=%d hole_loc=%d seeds=%d\n",
                       i, total + i, tasks[i].transit, tasks[i].hole_loc, tasks[i].n_seeds);
            }
            total += n;
        }
        printf("\ntotal: %d\n", total);
        free(g_visited);
        free(g_queue);
        free(g_shallow);
        free(g_recent);
        return 0;
    }

    /* If --task-id is set, find the (exit, local_task) for that global ID. */
    int task_target_exit = -1;
    TaskGroup task_target = {0};
    if (g_only_task >= 0) {
        int idx = 0;
        TaskGroup tasks[MAX_TASKS_PER_EXIT];
        for (int ei = 0; ei < n_exits; ei++) {
            int n = enumerate_tasks_for_exit(exits[ei], tasks, MAX_TASKS_PER_EXIT);
            if (g_only_task < idx + n) {
                task_target_exit = exits[ei];
                task_target = tasks[g_only_task - idx];
                break;
            }
            idx += n;
        }
        if (task_target_exit < 0) {
            fprintf(stderr, "error: --task-id %d out of range (total tasks = %d)\n",
                    g_only_task, idx);
            return 1;
        }
        printf("[task %d: exit=%d transit=%d hole_loc=%d seeds=%d]\n",
               g_only_task, task_target_exit, task_target.transit,
               task_target.hole_loc, task_target.n_seeds);
    }

    int unlimited = (g_time_cap_s == 0);
    for (int ei = 0; ei < n_exits; ei++) {
        /* In --task-id mode, skip exits that don't host the target task. */
        if (g_only_task >= 0 && exits[ei] != task_target_exit) continue;

        double remain = 0.0;  /* sentinel for unlimited */
        if (!unlimited) {
            clock_gettime(CLOCK_MONOTONIC, &t_global_now);
            double used = elapsed_s(t_global0, t_global_now);
            remain = g_time_cap_s - used;
            if (remain <= 0) {
                printf("[exit %d skipped — time budget exhausted]\n", exits[ei]);
                continue;
            }
        }

        g_exit_pos = exits[ei];

        if (g_only_task >= 0) {
            g_task_seeds = task_target.seeds;
            g_task_seed_count = task_target.n_seeds;
        } else {
            g_task_seeds = NULL;
            g_task_seed_count = 0;
        }

        int exhausted, dedup_full;
        double exit_elapsed = run_exit_search(remain, &exhausted, &dedup_full);

        /* Force-flush any new-best still buffered when the search ended
         * (the periodic timer in try_successor only fires every 8192
         * states, so a best discovered just before the time cap or queue
         * drain could otherwise be missed by the stream). */
        flush_pending_new_best(session_elapsed());

        /* Verify and report this exit. */
        int verify = -1;
        if (g_best_depth > 0) {
            Puzzle pz;
            build_partial_puzzle(&g_best_state, &pz);
            verify = sokoban_solve(&pz, NULL, NULL);
        }

        printf("--- Exit %d (%s) ---\n", g_exit_pos,
               dedup_full ? "dedup table full"
                          : (exhausted ? "exhausted" : "time cap"));
        printf("  elapsed:        %.3f s\n", exit_elapsed);
        printf("  states checked: %lld (shortcut %lld, dedup %lld",
               g_states_checked, g_pruned_short, g_pruned_dedup);
        if (g_dupe_threshold < INT_MAX)
            printf(", free-fly %lld", g_skipped_dedup);
        printf(")\n");
        printf("  solver calls:   %lld\n", g_solver_calls);
        if (g_two_tables) {
            printf("  shallow / recent: %lld / %lld  (evicts %lld / %lld)\n",
                   g_shallow_count, g_recent_count,
                   g_evictions_shallow, g_evictions_recent);
        } else {
            printf("  visited:        %lld\n", g_visited_count);
        }
        printf("  stack peak:     %lld\n", g_q_peak);
        printf("  best depth:     %d", g_best_depth);
        if (g_best_depth > 0)
            printf("  (verify %d %s)", verify, verify == g_best_depth ? "OK" : "MISMATCH");
        printf("\n");

        total_states       += g_states_checked;
        total_pruned_short += g_pruned_short;
        total_pruned_dedup += g_pruned_dedup;
        total_solver_calls += g_solver_calls;
        total_elapsed      += exit_elapsed;

        /* Cross-exit overall best is updated incrementally inside try_successor
         * (so streaming sees it).  Nothing to do here. */
    }

    printf("\n=== Backward search complete ===\n");
    if (g_time_cap_s == 0)
        printf("Total elapsed:     %.2f s (no time limit)\n", total_elapsed);
    else
        printf("Total elapsed:     %.2f s (cap %.2f s)\n", total_elapsed, g_time_cap_s);
    printf("Exits searched:    %d / %d\n", n_exits, n_exits);
    printf("States checked:    %lld\n", total_states);
    printf("  pruned shortcut: %lld\n", total_pruned_short);
    printf("  pruned dedup:    %lld\n", total_pruned_dedup);
    printf("Solver calls:      %lld\n", total_solver_calls);
    printf("Best depth:        %d  (exit %d)\n", g_overall_best_depth, g_overall_best_exit);

    if (g_overall_best_depth > 0) {
        printf("\n");
        print_puzzle_for_exit(&g_overall_best_state, g_overall_best_exit);
    } else {
        printf("(no non-trivial puzzle found)\n");
    }

    free(g_visited);
    free(g_shallow);
    free(g_recent);
    free(g_queue);
    return 0;
}
