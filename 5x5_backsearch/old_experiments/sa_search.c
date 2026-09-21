/*
 * sa_search.c — Simulated annealing over puzzle configurations.
 *
 * Fitness function: sokoban_solve(puzzle) — the actual forward solver.
 * Higher = better.  Unsolvable / invalid → fitness 0.
 *
 * State: a complete puzzle (walls bitmask, player_pos, exit_pos, blocks
 * with their push masks, holes).  Mutations perturb one element at a
 * time, candidate accepted by the Metropolis criterion.
 *
 * No NN, no backsearch.  Just the solver as the oracle.  This is the
 * "does the puzzle-config space have discoverable structure beyond
 * what hand-tuned backsearch finds" experiment.
 *
 * Usage:
 *   sa_search [--grid RxC] [--time SEC] [--steps N] [--restarts N]
 *             [--temp-start T0] [--temp-end T1]
 *             [--init-blocks N] [--init-holes N]
 *             [--max-blocks N] [--max-holes N]
 *             [--seed N] [--seed-puzzle FILE]
 */

#include "sokoban_bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>

/* Our own caps for the SA-side puzzle config; bounded by the solver's
 * MAX_BLOCKS / MAX_HOLES (see sokoban_bfs.h). */
#define SA_MAX_BLOCKS 16
#define SA_MAX_HOLES  64

typedef struct {
    int      player_pos;
    int      exit_pos;
    uint64_t walls;                       /* bit i: cell i is wall */
    int      nblocks;
    int      block_pos[SA_MAX_BLOCKS];
    uint8_t  block_mask[SA_MAX_BLOCKS];   /* non-zero direction bits */
    int      nholes;
    int      hole_pos[SA_MAX_HOLES];
} PC;

static int g_sa_rows = 6, g_sa_cols = 6, g_sa_ncells = 36;
static int g_sa_max_blocks = 12, g_sa_max_holes = 12;

/* Constraint flags (mirror backsearch's naming where possible).
 *
 *   g_sa_fixed_exit: -1 = exit may move; otherwise lock to this cell.
 *   g_sa_fixed_walls: cells that MUST be walls (search may add others
 *                     but cannot remove these).
 *   g_sa_fixed_holes_mask: cells that MUST always be holes.  Additional
 *                          holes elsewhere are still allowed (subject
 *                          to --num-holes).  0 = no required holes.
 *   g_sa_num_blocks_cap / g_sa_num_holes_cap: -1 = unbounded.
 */
static int      g_sa_fixed_exit       = -1;
static uint64_t g_sa_fixed_walls      = 0;
static uint64_t g_sa_fixed_holes_mask = 0;
static int      g_sa_num_blocks_cap   = -1;
static int      g_sa_num_holes_cap    = -1;
/* Fixed blocks: cells locked to a block with a specified direction mask.
 * Stored at the front of every PC's block_pos[] / block_mask[] arrays
 * (indices 0 .. g_sa_n_fixed_blocks-1).  Mutation never touches those
 * slots; ADD/MOVE/REMOVE/CHANGE_MASK only operate on indices ≥ n_fixed. */
static int      g_sa_n_fixed_blocks    = 0;
static int      g_sa_fixed_block_pos [SA_MAX_BLOCKS] = {0};
static uint8_t  g_sa_fixed_block_mask[SA_MAX_BLOCKS] = {0};
/* Fixed-empty: cells that must NOT contain a wall, block, or hole.  The
 * player and exit are still allowed to occupy them — fixed-empty means
 * "always walkable", not "always vacant". */
static uint64_t g_sa_fixed_empty_mask  = 0;
/* Holeboard: every cell must be wall/block/hole/player/exit — no floors.
 * Mutation operators are restricted to those preserving the invariant
 * (block↔hole swaps, player swap with hole, wall↔hole toggle, …).
 * Capped at nh ≤ 31 by the solver's pre-existing hole-mask packing. */
static int      g_sa_holeboard         = 0;
#define SA_HOLEBOARD_MAX_HOLES 31

static inline uint64_t bit(int p) { return 1ULL << p; }

static int cell_kind(const PC *p, int c) {
    if (p->walls & bit(c)) return 1;          /* wall */
    if (c == p->player_pos) return 2;
    if (c == p->exit_pos)   return 3;
    for (int i = 0; i < p->nblocks; i++) if (p->block_pos[i] == c) return 4;
    for (int i = 0; i < p->nholes;  i++) if (p->hole_pos[i]  == c) return 5;
    return 0;                                  /* empty floor */
}

static int valid(const PC *p) {
    if (p->player_pos < 0 || p->player_pos >= g_sa_ncells) return 0;
    if (p->exit_pos   < 0 || p->exit_pos   >= g_sa_ncells) return 0;
    if (p->player_pos == p->exit_pos) return 0;
    if (p->walls & bit(p->player_pos)) return 0;
    if (p->walls & bit(p->exit_pos))   return 0;
    /* Constraint: fixed exit. */
    if (g_sa_fixed_exit >= 0 && p->exit_pos != g_sa_fixed_exit) return 0;
    /* Constraint: fixed walls must all be present. */
    if ((p->walls & g_sa_fixed_walls) != g_sa_fixed_walls) return 0;
    /* Constraint: caps. */
    if (g_sa_num_blocks_cap >= 0 && p->nblocks > g_sa_num_blocks_cap) return 0;
    if (g_sa_num_holes_cap  >= 0 && p->nholes  > g_sa_num_holes_cap)  return 0;
    /* Constraint: fixed holes must all be present. */
    if (g_sa_fixed_holes_mask) {
        uint64_t actual = 0;
        for (int i = 0; i < p->nholes; i++) actual |= bit(p->hole_pos[i]);
        if ((actual & g_sa_fixed_holes_mask) != g_sa_fixed_holes_mask) return 0;
    }
    /* Constraint: fixed blocks must be present in the canonical prefix
     * (block_pos[0..n_fixed-1] match the configured positions and masks). */
    if (g_sa_n_fixed_blocks > 0) {
        if (p->nblocks < g_sa_n_fixed_blocks) return 0;
        for (int i = 0; i < g_sa_n_fixed_blocks; i++) {
            if (p->block_pos [i] != g_sa_fixed_block_pos [i]) return 0;
            if (p->block_mask[i] != g_sa_fixed_block_mask[i]) return 0;
        }
    }
    /* Constraint: fixed-empty cells must not contain a wall, block, or hole. */
    if (g_sa_fixed_empty_mask) {
        if (p->walls & g_sa_fixed_empty_mask) return 0;
        for (int i = 0; i < p->nblocks; i++)
            if (g_sa_fixed_empty_mask & bit(p->block_pos[i])) return 0;
        for (int i = 0; i < p->nholes;  i++)
            if (g_sa_fixed_empty_mask & bit(p->hole_pos [i])) return 0;
    }
    /* Constraint: holeboard mode — every cell must be occupied by a
     * block, hole, player, or exit.  No walls and no floors. */
    if (g_sa_holeboard) {
        if (p->walls != 0) return 0;
        if (p->nholes > SA_HOLEBOARD_MAX_HOLES) return 0;
        uint64_t occupied = bit(p->player_pos) | bit(p->exit_pos);
        for (int i = 0; i < p->nblocks; i++) occupied |= bit(p->block_pos[i]);
        for (int i = 0; i < p->nholes;  i++) occupied |= bit(p->hole_pos [i]);
        uint64_t grid_mask = (g_sa_ncells == 64) ? ~0ULL : ((1ULL << g_sa_ncells) - 1);
        if ((occupied & grid_mask) != grid_mask) return 0;  /* some cell is floor */
    }
    /* Player can't start on a block or hole. */
    for (int i = 0; i < p->nblocks; i++) {
        if (p->block_pos[i] == p->player_pos) return 0;
    }
    for (int i = 0; i < p->nholes; i++) {
        if (p->hole_pos[i] == p->player_pos)  return 0;
    }
    /* Blocks: in grid, not wall, not on exit, distinct, non-zero mask, not on hole. */
    for (int i = 0; i < p->nblocks; i++) {
        int c = p->block_pos[i];
        if (c < 0 || c >= g_sa_ncells)        return 0;
        if (p->walls & bit(c))             return 0;
        if (c == p->exit_pos)              return 0;
        if (p->block_mask[i] == 0)         return 0;
        for (int j = 0; j < i; j++)        if (p->block_pos[j] == c) return 0;
        for (int j = 0; j < p->nholes; j++) if (p->hole_pos[j] == c) return 0;
    }
    /* Holes: in grid, not wall, not on exit, distinct. */
    for (int i = 0; i < p->nholes; i++) {
        int c = p->hole_pos[i];
        if (c < 0 || c >= g_sa_ncells)        return 0;
        if (p->walls & bit(c))             return 0;
        if (c == p->exit_pos)              return 0;
        for (int j = 0; j < i; j++)        if (p->hole_pos[j] == c) return 0;
    }
    return 1;
}

/* ---------- Slow-solve watchdog ----------
 * If a single sokoban_solve() call exceeds g_slow_solve_threshold_s, a
 * background thread prints the puzzle (to stderr) while the solve is
 * still running.  Useful for spotting pathological boards that pin the
 * solver — without that visibility we'd just see the search stall.
 *
 * Concurrency: main thread writes g_solve_pc + start time, then sets
 * g_solve_in_progress via atomic release.  Watchdog reads the flag with
 * acquire ordering, guaranteeing it sees the PC payload.  We snapshot
 * the PC into a watchdog-owned copy before printing so the main thread
 * is free to mutate its own PC after the solve returns. */
static _Atomic int   g_solve_in_progress = 0;
static double        g_slow_solve_threshold_s = 0.0;   /* 0 = disabled */
static PC            g_solve_pc;                       /* read by watchdog */
static struct timespec g_solve_t0;                     /* read by watchdog */
static pthread_t     g_watchdog_thread;
static int           g_watchdog_started = 0;

/* Per-category solver timing — accumulated by evaluate().  Reported at
 * end of run.  Classified by sokoban_solve return code:
 *   ≥0  → solvable (depth)
 *   -1  → unsolvable
 *   -2  → heap overflow (cap or full)
 *   -3  → hash-table probe-saturation
 * Cache hits are not timed (skip the solver entirely). */
static long long g_stat_solvable_count    = 0;
static long long g_stat_unsolvable_count  = 0;
static long long g_stat_aborted_count     = 0;
static double    g_stat_solvable_time_s   = 0.0;
static double    g_stat_unsolvable_time_s = 0.0;
static double    g_stat_aborted_time_s    = 0.0;
/* Forward declarations: defined below. */
static double elapsed_s(struct timespec t0);
static void   fprint_puzzle(FILE *out, const PC *p);

static void *slow_solve_watchdog(void *arg) {
    (void)arg;
    int already_warned_this_solve = 0;
    while (1) {
        struct timespec ts = {.tv_sec = 0, .tv_nsec = 250 * 1000 * 1000}; /* 250 ms */
        nanosleep(&ts, NULL);
        int in_progress = atomic_load_explicit(&g_solve_in_progress, memory_order_acquire);
        if (!in_progress) {
            already_warned_this_solve = 0;
            continue;
        }
        if (already_warned_this_solve) continue;
        double elapsed = elapsed_s(g_solve_t0);
        if (elapsed >= g_slow_solve_threshold_s) {
            PC snap = g_solve_pc;  /* struct copy; cheap */
            fprintf(stderr, "\n[sa] slow solve in progress (%.1fs elapsed, still running):\n",
                    elapsed);
            fprint_puzzle(stderr, &snap);
            fflush(stderr);
            already_warned_this_solve = 1;
        }
    }
    return NULL;
}

/* Solver wrapper.  Returns the optimal solve length, or -1 for
 * "unsolvable" / "solver error" / "invalid puzzle". */
static int evaluate(const PC *p) {
    if (!valid(p)) return -1;
    Puzzle pz;
    memset(&pz, 0, sizeof pz);
    pz.walls        = p->walls;
    pz.exit_pos     = p->exit_pos;
    pz.player_start = p->player_pos;
    pz.num_blocks   = p->nblocks;
    for (int i = 0; i < p->nblocks; i++) {
        pz.block_pos[i]      = p->block_pos[i];
        pz.block_pushable[i] = p->block_mask[i];
    }
    pz.num_holes = p->nholes;
    for (int i = 0; i < p->nholes; i++) {
        pz.hole_pos[i] = p->hole_pos[i];
    }
    struct timespec solve_t0;
    clock_gettime(CLOCK_MONOTONIC, &solve_t0);
    int do_watch = g_watchdog_started && g_slow_solve_threshold_s > 0.0;
    if (do_watch) {
        g_solve_pc = *p;
        g_solve_t0 = solve_t0;
        atomic_store_explicit(&g_solve_in_progress, 1, memory_order_release);
    }
    int rc = sokoban_solve(&pz, NULL, NULL);
    if (do_watch) {
        atomic_store_explicit(&g_solve_in_progress, 0, memory_order_relaxed);
    }
    double solve_dt = elapsed_s(solve_t0);
    if (rc >= 0) {
        g_stat_solvable_count++;
        g_stat_solvable_time_s += solve_dt;
    } else if (rc == -1) {
        g_stat_unsolvable_count++;
        g_stat_unsolvable_time_s += solve_dt;
    } else {
        /* -2 (heap overflow / cap), -3 (probe-saturation) */
        g_stat_aborted_count++;
        g_stat_aborted_time_s += solve_dt;
    }
    return (rc >= 0) ? rc : -1;
}

/* Cheaper fitness: bounded solve.  Used for proposed states where we
 * only care whether they beat the running best. */
static int evaluate_cutoff(const PC *p, int max_cost) {
    if (!valid(p) || max_cost < 0) return -1;
    Puzzle pz;
    memset(&pz, 0, sizeof pz);
    pz.walls        = p->walls;
    pz.exit_pos     = p->exit_pos;
    pz.player_start = p->player_pos;
    pz.num_blocks   = p->nblocks;
    for (int i = 0; i < p->nblocks; i++) {
        pz.block_pos[i]      = p->block_pos[i];
        pz.block_pushable[i] = p->block_mask[i];
    }
    pz.num_holes = p->nholes;
    for (int i = 0; i < p->nholes; i++) {
        pz.hole_pos[i] = p->hole_pos[i];
    }
    int rc = sokoban_solve_cutoff(&pz, NULL, NULL, max_cost);
    return rc;
}

/* ---------- Recent-solve cache ----------
 * N-way set-associative cache: PC hash → solver depth.  Indexed by the
 * low bits of key1 (set index); each set has g_sa_cache_ways entries
 * checked linearly.  Eviction uses round-robin within the set.
 *
 * 128-bit verification (two independent 64-bit FNV-1a hashes).  64-bit
 * alone is unsafe at large eval counts (N²/2^65 ≈ 5% at 10⁹ evals);
 * 128-bit pushes that to ~10⁻²¹ at the same scale.
 *
 * Canonical-ordering mode (default on) sorts blocks and holes by cell
 * position before hashing — so two PCs differing only in array order
 * of the same blocks/holes share a cache entry.  Doubles up nicely with
 * the swap-with-last REMOVE_BLOCK semantics that reorder arrays. */
typedef struct {
    uint64_t key1;    /* primary FNV-1a */
    uint64_t key2;    /* secondary FNV-1a with independent basis+prime */
    int32_t  depth;
    int32_t  _pad;
} SaCacheEntry;

static SaCacheEntry *g_sa_cache           = NULL;
static uint8_t      *g_sa_cache_victim    = NULL;  /* per-set RR pointer */
static uint64_t      g_sa_cache_set_mask  = 0;     /* sets - 1 */
static int           g_sa_cache_ways      = 4;
static int           g_sa_cache_canonical = 1;
static long long     g_sa_cache_hits      = 0;
static long long     g_sa_cache_miss      = 0;

/* Small in-place insertion sort over a paired (pos, mask) array.  n ≤ 16
 * so cost is negligible (~100 ns) vs the solver call we may save. */
static void sort_blocks_canonical(int *pos, uint8_t *mask, int n) {
    for (int i = 1; i < n; i++) {
        int    pp = pos [i];
        uint8_t mm = mask[i];
        int j = i;
        while (j > 0 && pos[j-1] > pp) {
            pos [j] = pos [j-1];
            mask[j] = mask[j-1];
            j--;
        }
        pos [j] = pp;
        mask[j] = mm;
    }
}
static void sort_ints_canonical(int *pos, int n) {
    for (int i = 1; i < n; i++) {
        int pp = pos[i], j = i;
        while (j > 0 && pos[j-1] > pp) { pos[j] = pos[j-1]; j--; }
        pos[j] = pp;
    }
}

static void sa_hash_pc(const PC *p, uint64_t *out1, uint64_t *out2) {
    int     bpos[SA_MAX_BLOCKS];
    uint8_t bmask[SA_MAX_BLOCKS];
    int     hpos[SA_MAX_HOLES];
    for (int i = 0; i < p->nblocks; i++) { bpos[i] = p->block_pos[i]; bmask[i] = p->block_mask[i]; }
    for (int i = 0; i < p->nholes;  i++) { hpos[i] = p->hole_pos [i]; }
    if (g_sa_cache_canonical) {
        sort_blocks_canonical(bpos, bmask, p->nblocks);
        sort_ints_canonical  (hpos, p->nholes);
    }
    uint64_t h1 = 0xcbf29ce484222325ULL;
    uint64_t h2 = 0x9ae16a3b2f90404fULL;
    #define SA_MIX1(x) do { h1 ^= (uint64_t)(x); h1 *= 0x100000001b3ULL; } while (0)
    #define SA_MIX2(x) do { h2 ^= (uint64_t)(x); h2 *= 0xc6a4a7935bd1e995ULL; } while (0)
    #define SA_MIX(x)  do { SA_MIX1(x); SA_MIX2(x); } while (0)
    SA_MIX(p->player_pos);
    SA_MIX(p->exit_pos);
    SA_MIX(p->walls);
    SA_MIX((uint64_t)p->nblocks);
    for (int i = 0; i < p->nblocks; i++) {
        SA_MIX((uint64_t)bpos [i]);
        SA_MIX((uint64_t)bmask[i]);
    }
    SA_MIX((uint64_t)p->nholes);
    for (int i = 0; i < p->nholes; i++) {
        SA_MIX((uint64_t)hpos[i]);
    }
    #undef SA_MIX1
    #undef SA_MIX2
    #undef SA_MIX
    /* Reserve (0,0) as empty-slot marker; bump if both happen to be zero. */
    if (h1 == 0 && h2 == 0) h1 = 1;
    *out1 = h1;
    *out2 = h2;
}

/* Wrapper: try cache, else call evaluate() and store the result.
 * Returns the same value evaluate() would. */
static int evaluate_cached(const PC *p) {
    if (!g_sa_cache) return evaluate(p);
    uint64_t k1, k2;
    sa_hash_pc(p, &k1, &k2);
    uint64_t set = k1 & g_sa_cache_set_mask;
    SaCacheEntry *base = &g_sa_cache[set * (uint64_t)g_sa_cache_ways];
    /* Linear scan within the set. */
    for (int w = 0; w < g_sa_cache_ways; w++) {
        if (base[w].key1 == k1 && base[w].key2 == k2) {
            g_sa_cache_hits++;
            return base[w].depth;
        }
    }
    /* Miss — solve, then place into an empty slot or round-robin evict. */
    g_sa_cache_miss++;
    int d = evaluate(p);
    int target = -1;
    for (int w = 0; w < g_sa_cache_ways; w++) {
        if (base[w].key1 == 0 && base[w].key2 == 0) { target = w; break; }
    }
    if (target < 0) {
        target = g_sa_cache_victim[set];
        g_sa_cache_victim[set] = (uint8_t)((target + 1) % g_sa_cache_ways);
    }
    base[target].key1  = k1;
    base[target].key2  = k2;
    base[target].depth = d;
    return d;
}

/* Pick a floor (kind 0) cell.  If exclude_fixed_empty is nonzero, also
 * skip cells in g_sa_fixed_empty_mask — used for placing obstacles
 * (blocks/holes/walls) that must not land on a fixed-empty cell.  The
 * variants without exclusion are appropriate for placing the player or
 * exit, which are explicitly allowed on fixed-empty cells. */
static int pick_empty_cell_ex(const PC *p, int exclude_fixed_empty) {
    uint64_t excl = exclude_fixed_empty ? g_sa_fixed_empty_mask : 0;
    for (int t = 0; t < 32; t++) {
        int c = rand() % g_sa_ncells;
        if (cell_kind(p, c) == 0 && !(excl & bit(c))) return c;
    }
    int start = rand() % g_sa_ncells;
    for (int i = 0; i < g_sa_ncells; i++) {
        int c = (start + i) % g_sa_ncells;
        if (cell_kind(p, c) == 0 && !(excl & bit(c))) return c;
    }
    return -1;
}
static int pick_empty_cell(const PC *p)          { return pick_empty_cell_ex(p, 0); }
static int pick_empty_for_obstacle(const PC *p)  { return pick_empty_cell_ex(p, 1); }

static int pick_non_player_exit(const PC *p) {
    for (int t = 0; t < 32; t++) {
        int c = rand() % g_sa_ncells;
        if (c != p->player_pos && c != p->exit_pos) return c;
    }
    return -1;
}

enum {
    OP_MOVE_PLAYER,
    OP_MOVE_EXIT,
    OP_TOGGLE_WALL,
    OP_ADD_BLOCK,
    OP_REMOVE_BLOCK,
    OP_MOVE_BLOCK,
    OP_CHANGE_MASK,
    OP_ADD_HOLE,
    OP_REMOVE_HOLE,
    OP_MOVE_HOLE,
    NUM_OPS
};

/* Find an arbitrary hole index that is NOT fixed; -1 if none. */
static int pick_variable_hole_index(const PC *p) {
    if (p->nholes == 0) return -1;
    int start = rand() % p->nholes;
    for (int s = 0; s < p->nholes; s++) {
        int i = (start + s) % p->nholes;
        if (!(g_sa_fixed_holes_mask & bit(p->hole_pos[i]))) return i;
    }
    return -1;
}

/* Holeboard mode rewrites of each operator.  Returns 1 if handled (the
 * caller skips the default), 0 if the default branch should run. */
static int mutate_holeboard(PC *p, int op) {
    if (!g_sa_holeboard) return 0;
    switch (op) {
        case OP_MOVE_PLAYER: {
            /* Swap player with a (variable) hole. */
            int hi = pick_variable_hole_index(p);
            if (hi < 0) return 1;
            int old = p->player_pos;
            p->player_pos    = p->hole_pos[hi];
            p->hole_pos[hi]  = old;
            return 1;
        }
        case OP_MOVE_EXIT: {
            if (g_sa_fixed_exit >= 0) return 1;
            int hi = pick_variable_hole_index(p);
            if (hi < 0) return 1;
            int old = p->exit_pos;
            p->exit_pos      = p->hole_pos[hi];
            p->hole_pos[hi]  = old;
            return 1;
        }
        case OP_TOGGLE_WALL:
            /* Walls are forbidden in holeboard — no-op. */
            return 1;
        case OP_ADD_BLOCK: {
            int cap = g_sa_max_blocks;
            if (g_sa_num_blocks_cap >= 0 && g_sa_num_blocks_cap < cap) cap = g_sa_num_blocks_cap;
            if (p->nblocks >= cap) return 1;
            int hi = pick_variable_hole_index(p);
            if (hi < 0) return 1;
            int c = p->hole_pos[hi];
            /* Remove from hole list, append to block list. */
            p->hole_pos[hi] = p->hole_pos[p->nholes - 1];
            p->nholes--;
            p->block_pos [p->nblocks] = c;
            p->block_mask[p->nblocks] = (uint8_t)(1 + (rand() % 15));
            p->nblocks++;
            return 1;
        }
        case OP_REMOVE_BLOCK: {
            int nvar = p->nblocks - g_sa_n_fixed_blocks;
            if (nvar <= 0) return 1;
            if (p->nholes >= SA_HOLEBOARD_MAX_HOLES) return 1;
            int i = g_sa_n_fixed_blocks + (rand() % nvar);
            int c = p->block_pos[i];
            /* Swap-with-last in block list. */
            p->block_pos [i] = p->block_pos [p->nblocks - 1];
            p->block_mask[i] = p->block_mask[p->nblocks - 1];
            p->nblocks--;
            p->hole_pos[p->nholes++] = c;
            return 1;
        }
        case OP_MOVE_BLOCK: {
            /* Swap a variable block with a variable hole. */
            int nvar = p->nblocks - g_sa_n_fixed_blocks;
            if (nvar <= 0) return 1;
            int hi = pick_variable_hole_index(p);
            if (hi < 0) return 1;
            int bi = g_sa_n_fixed_blocks + (rand() % nvar);
            int old_bp = p->block_pos[bi];
            p->block_pos[bi] = p->hole_pos[hi];
            p->hole_pos[hi]  = old_bp;
            return 1;
        }
        case OP_CHANGE_MASK:
            return 0;  /* same logic; default branch handles it */
        case OP_ADD_HOLE:
        case OP_REMOVE_HOLE:
        case OP_MOVE_HOLE:
            return 1;  /* meaningless in holeboard — no-op */
    }
    return 1;
}

static void mutate(PC *p) {
    int op = rand() % NUM_OPS;
    if (mutate_holeboard(p, op)) return;
    switch (op) {
        case OP_MOVE_PLAYER: {
            int c = pick_empty_cell(p);
            if (c >= 0) p->player_pos = c;
            break;
        }
        case OP_MOVE_EXIT: {
            /* Skip if exit is locked by --exit. */
            if (g_sa_fixed_exit >= 0) break;
            int c = pick_empty_cell(p);
            if (c >= 0) p->exit_pos = c;
            break;
        }
        case OP_TOGGLE_WALL: {
            int c = pick_non_player_exit(p);
            if (c < 0) break;
            /* Can't remove a fixed wall. */
            if (g_sa_fixed_walls & bit(c)) break;
            /* Can't add a wall to a fixed-empty cell. */
            if (g_sa_fixed_empty_mask & bit(c)) {
                int k0 = cell_kind(p, c);
                if (k0 == 0) break;  /* would be a floor→wall toggle; reject */
            }
            int k = cell_kind(p, c);
            if (k == 1) { p->walls &= ~bit(c); break; }
            if (k == 0) { p->walls |=  bit(c); break; }
            break;  /* don't toggle blocks/holes into walls */
        }
        case OP_ADD_BLOCK: {
            int cap = g_sa_max_blocks;
            if (g_sa_num_blocks_cap >= 0 && g_sa_num_blocks_cap < cap) cap = g_sa_num_blocks_cap;
            if (p->nblocks >= cap) break;
            int c = pick_empty_for_obstacle(p);
            if (c < 0) break;
            p->block_pos [p->nblocks] = c;
            p->block_mask[p->nblocks] = (uint8_t)(1 + (rand() % 15));
            p->nblocks++;
            break;
        }
        case OP_REMOVE_BLOCK: {
            /* Only variable blocks (index ≥ n_fixed) can be removed. */
            int nvar = p->nblocks - g_sa_n_fixed_blocks;
            if (nvar <= 0) break;
            int i = g_sa_n_fixed_blocks + (rand() % nvar);
            /* Swap with last (which is variable since nvar > 0). */
            p->block_pos [i] = p->block_pos [p->nblocks - 1];
            p->block_mask[i] = p->block_mask[p->nblocks - 1];
            p->nblocks--;
            break;
        }
        case OP_MOVE_BLOCK: {
            int nvar = p->nblocks - g_sa_n_fixed_blocks;
            if (nvar <= 0) break;
            int i = g_sa_n_fixed_blocks + (rand() % nvar);
            int c = pick_empty_for_obstacle(p);
            if (c < 0) break;
            p->block_pos[i] = c;
            break;
        }
        case OP_CHANGE_MASK: {
            int nvar = p->nblocks - g_sa_n_fixed_blocks;
            if (nvar <= 0) break;
            int i = g_sa_n_fixed_blocks + (rand() % nvar);
            uint8_t new_mask;
            do { new_mask = (uint8_t)(1 + (rand() % 15)); }
            while (new_mask == p->block_mask[i]);
            p->block_mask[i] = new_mask;
            break;
        }
        case OP_ADD_HOLE: {
            int cap = g_sa_max_holes;
            if (g_sa_num_holes_cap >= 0 && g_sa_num_holes_cap < cap) cap = g_sa_num_holes_cap;
            if (p->nholes >= cap) break;
            int c = pick_empty_for_obstacle(p);
            if (c < 0) break;
            p->hole_pos[p->nholes] = c;
            p->nholes++;
            break;
        }
        case OP_REMOVE_HOLE: {
            if (p->nholes == 0) break;
            int i = rand() % p->nholes;
            /* Can't remove a fixed hole. */
            if (g_sa_fixed_holes_mask & bit(p->hole_pos[i])) break;
            p->hole_pos[i] = p->hole_pos[p->nholes - 1];
            p->nholes--;
            break;
        }
        case OP_MOVE_HOLE: {
            if (p->nholes == 0) break;
            int i = rand() % p->nholes;
            /* Can't move a fixed hole. */
            if (g_sa_fixed_holes_mask & bit(p->hole_pos[i])) break;
            int c = pick_empty_for_obstacle(p);
            if (c < 0) break;
            p->hole_pos[i] = c;
            break;
        }
    }
}

static void init_random(PC *p, int init_blocks, int init_holes) {
    memset(p, 0, sizeof *p);
    /* Honor --fixedwalls. */
    p->walls = g_sa_fixed_walls;
    /* Pre-build a mask of fixed-block cells for cheap avoidance below. */
    uint64_t fixed_block_cells = 0;
    for (int i = 0; i < g_sa_n_fixed_blocks; i++)
        fixed_block_cells |= bit(g_sa_fixed_block_pos[i]);
    /* Exit: fixed or random.  Random exits must avoid fixed blocks. */
    if (g_sa_fixed_exit >= 0) {
        p->exit_pos = g_sa_fixed_exit;
    } else {
        do { p->exit_pos = rand() % g_sa_ncells; }
        while ((p->walls & bit(p->exit_pos)) || (fixed_block_cells & bit(p->exit_pos)));
    }
    /* Pre-place fixed blocks at the front of the array. */
    for (int i = 0; i < g_sa_n_fixed_blocks; i++) {
        p->block_pos [i] = g_sa_fixed_block_pos [i];
        p->block_mask[i] = g_sa_fixed_block_mask[i];
        p->nblocks++;
    }
    /* Player: random non-exit non-wall non-fixed-block cell. */
    do {
        p->player_pos = rand() % g_sa_ncells;
    } while (p->player_pos == p->exit_pos
             || (p->walls & bit(p->player_pos))
             || (fixed_block_cells & bit(p->player_pos)));

    /* Caps.  Variable-block count is init_blocks on top of fixed prefix. */
    int blk_cap = g_sa_n_fixed_blocks + init_blocks;
    if (g_sa_num_blocks_cap >= 0 && blk_cap > g_sa_num_blocks_cap) blk_cap = g_sa_num_blocks_cap;
    if (blk_cap > g_sa_max_blocks) blk_cap = g_sa_max_blocks;
    int hol_cap = init_holes;
    if (g_sa_num_holes_cap  >= 0 && hol_cap > g_sa_num_holes_cap)  hol_cap = g_sa_num_holes_cap;

    /* Variable blocks fill from current nblocks up to blk_cap. */
    while (p->nblocks < blk_cap) {
        int c = pick_empty_for_obstacle(p);
        if (c < 0) break;
        p->block_pos[p->nblocks] = c;
        p->block_mask[p->nblocks] = (uint8_t)(1 + (rand() % 15));
        p->nblocks++;
    }
    /* Pre-place all required (fixed) holes first.  If any of them collide
     * with player/exit/walls, we have to shift the player. */
    if (g_sa_fixed_holes_mask) {
        for (int cell = 0; cell < g_sa_ncells; cell++) {
            if (!(g_sa_fixed_holes_mask & bit(cell))) continue;
            /* Move player off this cell if needed. */
            if (p->player_pos == cell) {
                do {
                    p->player_pos = rand() % g_sa_ncells;
                } while (p->player_pos == p->exit_pos
                         || (p->walls & bit(p->player_pos))
                         || (g_sa_fixed_holes_mask & bit(p->player_pos))
                         || (fixed_block_cells & bit(p->player_pos)));
            }
            if (p->nholes >= SA_MAX_HOLES) break;
            p->hole_pos[p->nholes++] = cell;
        }
    }
    /* Then any additional random holes up to hol_cap. */
    for (int k = p->nholes; k < hol_cap; k++) {
        int c = pick_empty_for_obstacle(p);
        if (c < 0) break;
        p->hole_pos[p->nholes++] = c;
    }
    /* Holeboard: fill remaining floor cells.  First with holes up to the
     * solver-imposed cap (31), then with blocks (random masks) to absorb
     * any overflow.  Walls are forbidden in holeboard mode. */
    if (g_sa_holeboard) {
        for (int c = 0; c < g_sa_ncells; c++) {
            if (cell_kind(p, c) != 0) continue;
            if (p->nholes >= SA_HOLEBOARD_MAX_HOLES) break;
            p->hole_pos[p->nholes++] = c;
        }
        for (int c = 0; c < g_sa_ncells; c++) {
            if (cell_kind(p, c) != 0) continue;
            if (p->nblocks >= g_sa_max_blocks) break;
            p->block_pos [p->nblocks] = c;
            p->block_mask[p->nblocks] = (uint8_t)(1 + (rand() % 15));
            p->nblocks++;
        }
    }
}

/* Parse a puzzle from a file in fprint_puzzle format.  Returns 1 on success.
 * Accepts the output of fprint_puzzle / backsearch_worker:
 *   $O....   A=[DL]
 *   ##A.B.   B=[L]
 *   ..C.D.   C=[UR]
 * Lines not starting with a grid char (after stripping leading whitespace)
 * are treated as overflow-legend or skipped.  A leading "depth (time)"
 * header line is tolerated. */
static int parse_puzzle_file(const char *path, PC *out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "error: cannot open seed file '%s'\n", path); return 0; }

    memset(out, 0, sizeof *out);
    char grid[MAX_NCELLS];
    memset(grid, '.', sizeof grid);
    int block_letter[SA_MAX_BLOCKS];
    int nblocks_found = 0;
    uint8_t legend_mask[52] = {0};  /* A-Z = 0..25, a-z = 26..51 */
    int legend_set[52] = {0};
    int grid_rows_read = 0;

    char line[512];
    while (fgets(line, sizeof line, f)) {
        /* Strip leading whitespace. */
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n') continue;

        /* Check if this looks like a grid row (starts with a grid char). */
        int is_grid = 0;
        if (*p == '.' || *p == '#' || *p == '$' || *p == '@' || *p == 'O' ||
            (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z'))
            is_grid = 1;

        if (is_grid && grid_rows_read < g_sa_rows) {
            int r = grid_rows_read;
            for (int c = 0; c < g_sa_cols && *p && *p != ' ' && *p != '\t' && *p != '\n'; c++, p++) {
                int cell = r * g_sa_cols + c;
                grid[cell] = *p;
                if (*p == '@') out->player_pos = cell;
                else if (*p == '$') out->exit_pos = cell;
                else if (*p == '#') out->walls |= bit(cell);
                else if (*p == 'O') {
                    if (out->nholes < SA_MAX_HOLES)
                        out->hole_pos[out->nholes++] = cell;
                }
                else if ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z')) {
                    int li = (*p >= 'a') ? (*p - 'a' + 26) : (*p - 'A');
                    if (nblocks_found < SA_MAX_BLOCKS) {
                        out->block_pos[nblocks_found] = cell;
                        block_letter[nblocks_found] = li;
                        nblocks_found++;
                    }
                }
            }
            grid_rows_read++;
        }

        /* Scan the rest of the line (and overflow-only lines) for legend entries
         * of the form X=[DIRS]. */
        char *scan = line;
        while ((scan = strchr(scan, '=')) != NULL) {
            if (scan > line && scan[1] == '[') {
                char lch = scan[-1];
                int li = -1;
                if (lch >= 'A' && lch <= 'Z') li = lch - 'A';
                else if (lch >= 'a' && lch <= 'z') li = lch - 'a' + 26;
                if (li >= 0) {
                    uint8_t mask = 0;
                    char *q = scan + 2;
                    while (*q && *q != ']') {
                        switch (*q) {
                            case 'U': case 'u': mask |= 1; break;
                            case 'R': case 'r': mask |= 2; break;
                            case 'D': case 'd': mask |= 4; break;
                            case 'L': case 'l': mask |= 8; break;
                        }
                        q++;
                    }
                    if (mask) { legend_mask[li] = mask; legend_set[li] = 1; }
                }
            }
            scan++;
        }
    }
    fclose(f);

    if (grid_rows_read != g_sa_rows) {
        fprintf(stderr, "error: seed file has %d grid rows, expected %d\n",
                grid_rows_read, g_sa_rows);
        return 0;
    }

    out->nblocks = nblocks_found;
    for (int i = 0; i < nblocks_found; i++) {
        int li = block_letter[i];
        if (legend_set[li]) {
            out->block_mask[i] = legend_mask[li];
        } else {
            fprintf(stderr, "error: seed block '%c' has no legend entry\n",
                    li < 26 ? ('A'+li) : ('a'+li-26));
            return 0;
        }
    }
    return 1;
}

/* Print the puzzle in backsearch_worker's exact format:
 *   indented row, optional inline `<letter>=[<dirs>]` legend on the
 *   right of each row.  Block A on row 0, B on row 1, etc.  Overflow
 *   blocks (more blocks than rows) print on continuation lines. */
static void fprint_puzzle(FILE *out, const PC *p) {
    char grid[MAX_NCELLS];
    for (int c = 0; c < g_sa_ncells; c++) grid[c] = '.';
    for (int c = 0; c < g_sa_ncells; c++) {
        if (p->walls & bit(c)) grid[c] = '#';
    }
    /* Holes drawn before blocks so blocks shadow them. */
    for (int i = 0; i < p->nholes; i++) grid[p->hole_pos[i]] = 'O';
    for (int i = 0; i < p->nblocks; i++) {
        char ch = (i < 26) ? ('A' + i) : ('a' + (i - 26));
        grid[p->block_pos[i]] = ch;
    }
    grid[p->exit_pos]   = '$';
    grid[p->player_pos] = '@';

    for (int r = 0; r < g_sa_rows; r++) {
        fputs("  ", out);
        for (int c = 0; c < g_sa_cols; c++) fputc(grid[r * g_sa_cols + c], out);
        if (r < p->nblocks) {
            uint8_t m = p->block_mask[r];
            char ch = (r < 26) ? ('A' + r) : ('a' + (r - 26));
            fprintf(out, "   %c=[%s%s%s%s]", ch,
                    m & 1 ? "U" : "", m & 2 ? "R" : "",
                    m & 4 ? "D" : "", m & 8 ? "L" : "");
        }
        fputc('\n', out);
    }
    /* Overflow legend rows when nblocks > rows. */
    int indent_cols = 2 + g_sa_cols + 3;
    for (int i = g_sa_rows; i < p->nblocks; i++) {
        uint8_t m = p->block_mask[i];
        char ch = (i < 26) ? ('A' + i) : ('a' + (i - 26));
        for (int j = 0; j < indent_cols; j++) fputc(' ', out);
        fprintf(out, "%c=[%s%s%s%s]\n", ch,
                m & 1 ? "U" : "", m & 2 ? "R" : "",
                m & 4 ? "D" : "", m & 8 ? "L" : "");
    }
}

static void print_puzzle(const PC *p, int depth) {
    (void)depth;
    fprint_puzzle(stdout, p);
}

static double elapsed_s(struct timespec t0) {
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static double rand_uniform(void) { return (double)rand() / RAND_MAX; }

/* Live status line on stderr.  Overwritten in place with \r; only active
 * when stderr is a TTY (so redirected logs stay clean) and the user hasn't
 * passed --no-status.  Throttled to ~10 Hz. */
static int    g_status_enabled = 0;
static int    g_status_active  = 0;
static double g_status_last_t  = -1.0;

static void status_clear(void) {
    if (g_status_enabled && g_status_active) {
        fputs("\r\033[K", stderr);
        fflush(stderr);
        g_status_active = 0;
    }
}

static void status_print(int stale_restarts, int global_best, double elapsed,
                         int force) {
    if (!g_status_enabled) return;
    if (!force && g_status_last_t >= 0 && elapsed - g_status_last_t < 0.1) return;
    g_status_last_t = elapsed;
    fprintf(stderr, "\r\033[K[best=%d  stale=%d  t=%.0fs]",
            global_best, stale_restarts, elapsed);
    fflush(stderr);
    g_status_active = 1;
}

int main(int argc, char **argv) {
    int rows = 6, cols = 6;
    /* Default temperature picks: the sweep finding was that this domain
     * rewards sustained hot exploration over annealed cold convergence.
     * Set T0=T1=1.5 by default so cooling is a no-op unless the user
     * explicitly sets --temp-end below T0. */
    double T0 = 1.5, T1 = 1.5;
    long long steps = 10000000;  /* 10M — keeps runs ≥ minutes on 6x6 */
    /* 0 = unlimited restarts (default).  Process termination is driven by
     * --time, --stagnation-time, or Ctrl-C.  Set --restarts N to cap. */
    int restarts = 0;
    double time_cap = 0.0;
    int init_blocks = 2, init_holes = 1;
    int max_blocks = 12, max_holes = 12;
    unsigned seed = (unsigned)(time(NULL) ^ getpid());
    /* --cool-by {steps,time}: cool relative to step count vs wall time.
     * time-based is robust when per-step solver cost is unknown ahead. */
    int cool_by_time = 1;
    /* Reheat: after N stagnant steps, multiply T by REHEAT_FACTOR.  0 disables. */
    long long reheat_after = 0;
    double reheat_factor = 2.0;
    /* Restart on stagnation: after N stagnant steps, abandon the current
     * chain and start a new one (preserves global best).  0 disables.
     * This automates "I notice the search got stuck — start over". */
    /* Default: abandon any chain that goes N steps without improvement.
     * Step-count threshold; chains that solve fast restart quickly, slow
     * chains get more time per attempt.  See also --stagnation-time. */
    long long stagnation_restart_after = 20000;
    /* Time-based alternative: abandon a chain after N seconds without
     * improvement.  Gives consistent chain lifetimes regardless of
     * solver speed.  0 disables — falls back to step-based above. */
    double stagnation_restart_time = 0.0;
    /* Depth-bonus: chains close to global_best get MORE patience before
     * stagnation triggers, on the theory that they may still be climbing.
     * Effective threshold is multiplied by
     *   1 + depth_bonus_factor * max(0, 1 - gap/depth_bonus_window)
     * where gap = global_best - chain_best.  Window=5 means a chain 5+
     * depths below global gets no bonus; a chain matching global gets
     * (1+factor)× patience.
     *
     * Default factor=0 (disabled): a chain matching global is not
     * climbing toward global, it's re-finding the known peak — wasting
     * evals that would be more productive as fresh reseeds.  Set factor
     * >0 to re-enable. */
    double depth_bonus_factor = 0.0;
    double depth_bonus_window = 5.0;
    /* Reseed-from-best: when a restart begins, with this probability
     * start from the current global_best (heavily mutated) instead of
     * pure random.  Lets restarts climb out of the best-known basin
     * while still doing pure random exploration the rest of the time.
     * 0.0 = always random (legacy).  0.5 = half-and-half. */
    double reseed_from_best_prob = 0.5;
    /* Number of random mutations applied to global_best when reseeding.
     * Adaptive: starts at this value, scales up linearly with restarts
     * since the last global-best improvement.  Idea: small perturbations
     * try to refine the current best; if that keeps failing, increase
     * the distance to escape the basin. */
    int    reseed_perturb_count  = 8;
    /* Each restart-since-improvement contributes
     *   step * k^power  extra mutations
     * to the reseed perturbation.  Capped at reseed_perturb_max.
     * Default power=2.0 (quadratic growth) to escape basins decisively. */
    int    reseed_perturb_step   = 3;
    int    reseed_perturb_max    = 200;
    double reseed_perturb_power  = 2.0;
    /* Verbose internal messages (stagnation-restart, reheat). */
    int verbose = 0;
    /* Live status line on stderr (default on when stderr is a TTY). */
    int status_line = isatty(STDERR_FILENO);
    /* Solver cache: 2^cache_bits entries.  Default 20 → 1M × 16 B = 16 MB. */
    int cache_bits = 20;
    /* If a single sokoban_solve takes > N seconds, the watchdog thread
     * prints the in-flight puzzle to stderr.  0 = disabled (no thread). */
    double slow_solve_warn_s = 0.0;
    /* Soft per-solve heap-size cap (forwarded to sokoban_set_heap_cap).
     * 0 = no cap; solves only end at exhaustion / true heap overflow. */
    int heap_cap = 0;
    const char *seed_puzzle_path = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--grid"))         { sscanf(argv[++i], "%dx%d", &rows, &cols); }
        else if (!strcmp(argv[i], "--time"))         time_cap = atof(argv[++i]);
        else if (!strcmp(argv[i], "--steps"))        steps = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--restarts"))     restarts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp-start"))   T0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--temp-end"))     T1 = atof(argv[++i]);
        else if (!strcmp(argv[i], "--init-blocks"))  init_blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--init-holes"))   init_holes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-blocks"))   max_blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-holes"))    max_holes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed"))         seed = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cool-by-steps")) cool_by_time = 0;
        else if (!strcmp(argv[i], "--cool-by-time"))  cool_by_time = 1;
        else if (!strcmp(argv[i], "--no-cool"))       { T1 = T0; cool_by_time = 0; }
        else if (!strcmp(argv[i], "--reheat-after")) reheat_after = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--reheat-factor")) reheat_factor = atof(argv[++i]);
        else if (!strcmp(argv[i], "--restart-after-stale")) stagnation_restart_after = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--stagnation-time"))     stagnation_restart_time  = atof(argv[++i]);
        else if (!strcmp(argv[i], "--depth-bonus"))         depth_bonus_factor       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--depth-bonus-window"))  depth_bonus_window       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--reseed-from-best"))    reseed_from_best_prob    = atof(argv[++i]);
        else if (!strcmp(argv[i], "--reseed-perturb"))      reseed_perturb_count     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reseed-perturb-step")) reseed_perturb_step      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reseed-perturb-max"))  reseed_perturb_max       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reseed-perturb-power")) reseed_perturb_power    = atof(argv[++i]);
        else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) verbose = 1;
        else if (!strcmp(argv[i], "--no-status")) status_line = 0;
        else if (!strcmp(argv[i], "--status"))    status_line = 1;
        else if (!strcmp(argv[i], "--cache-bits")) cache_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-cache"))   cache_bits = 0;
        else if (!strcmp(argv[i], "--cache-ways")) g_sa_cache_ways = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cache-canonical")) g_sa_cache_canonical = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--slow-solve-warn")) slow_solve_warn_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "--heap-cap"))        heap_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed-puzzle"))   seed_puzzle_path = argv[++i];
        /* Constraint flags (match backsearch conventions). */
        else if (!strcmp(argv[i], "--exit"))         g_sa_fixed_exit = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-blocks"))   g_sa_num_blocks_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--num-holes"))    g_sa_num_holes_cap  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fixedwalls") || !strcmp(argv[i], "--fixed-walls")) {
            char *s = argv[++i];
            char *tok = strtok(s, ",");
            while (tok) {
                long v = strtol(tok, NULL, 10);
                if (v < 0 || v >= MAX_NCELLS) {
                    fprintf(stderr, "error: --fixedwalls cell %ld out of range\n", v);
                    return 1;
                }
                g_sa_fixed_walls |= (1ULL << v);
                tok = strtok(NULL, ",");
            }
        }
        else if (!strcmp(argv[i], "--fixedholes") || !strcmp(argv[i], "--fixed-holes")) {
            char *s = argv[++i];
            char *tok = strtok(s, ",");
            while (tok) {
                long v = strtol(tok, NULL, 10);
                if (v < 0 || v >= MAX_NCELLS) {
                    fprintf(stderr, "error: --fixedholes cell %ld out of range\n", v);
                    return 1;
                }
                g_sa_fixed_holes_mask |= (1ULL << v);
                tok = strtok(NULL, ",");
            }
        }
        else if (!strcmp(argv[i], "--holeboard")) g_sa_holeboard = 1;
        else if (!strcmp(argv[i], "--fixedempty") || !strcmp(argv[i], "--fixed-empty")) {
            char *s = argv[++i];
            char *tok = strtok(s, ",");
            while (tok) {
                long v = strtol(tok, NULL, 10);
                if (v < 0 || v >= MAX_NCELLS) {
                    fprintf(stderr, "error: --fixedempty cell %ld out of range\n", v);
                    return 1;
                }
                g_sa_fixed_empty_mask |= (1ULL << v);
                tok = strtok(NULL, ",");
            }
        }
        else if (!strcmp(argv[i], "--fixedblocks") || !strcmp(argv[i], "--fixed-blocks")) {
            /* Format: <cell>=[<dirs>],<cell>=[<dirs>],...
             * Example: --fixedblocks 7=[U],9=[UR]
             * Dirs use U/R/D/L (case-insensitive). */
            char *s = argv[++i];
            char *tok = strtok(s, ",");
            while (tok) {
                char *eq = strchr(tok, '=');
                char *lb = strchr(tok, '[');
                char *rb = strchr(tok, ']');
                if (!eq || !lb || !rb || rb <= lb) {
                    fprintf(stderr, "error: --fixedblocks bad entry '%s' (want CELL=[DIRS])\n", tok);
                    return 1;
                }
                *eq = '\0';
                long cell = strtol(tok, NULL, 10);
                if (cell < 0 || cell >= MAX_NCELLS) {
                    fprintf(stderr, "error: --fixedblocks cell %ld out of range\n", cell);
                    return 1;
                }
                uint8_t mask = 0;
                for (char *q = lb + 1; q < rb; q++) {
                    switch (*q) {
                        case 'U': case 'u': mask |= 1; break;
                        case 'R': case 'r': mask |= 2; break;
                        case 'D': case 'd': mask |= 4; break;
                        case 'L': case 'l': mask |= 8; break;
                        default:
                            fprintf(stderr, "error: --fixedblocks bad dir char '%c'\n", *q);
                            return 1;
                    }
                }
                if (mask == 0) {
                    fprintf(stderr, "error: --fixedblocks empty mask for cell %ld\n", cell);
                    return 1;
                }
                if (g_sa_n_fixed_blocks >= SA_MAX_BLOCKS) {
                    fprintf(stderr, "error: --fixedblocks exceeds SA_MAX_BLOCKS=%d\n", SA_MAX_BLOCKS);
                    return 1;
                }
                /* Reject duplicate cells. */
                for (int j = 0; j < g_sa_n_fixed_blocks; j++) {
                    if (g_sa_fixed_block_pos[j] == cell) {
                        fprintf(stderr, "error: --fixedblocks duplicate cell %ld\n", cell);
                        return 1;
                    }
                }
                g_sa_fixed_block_pos [g_sa_n_fixed_blocks] = (int)cell;
                g_sa_fixed_block_mask[g_sa_n_fixed_blocks] = mask;
                g_sa_n_fixed_blocks++;
                tok = strtok(NULL, ",");
            }
        }
        else {
            fprintf(stderr, "unknown flag: %s\n", argv[i]);
            return 1;
        }
    }

    if (rows < 1 || cols < 1 || rows * cols > MAX_NCELLS) {
        fprintf(stderr, "error: invalid grid %dx%d\n", rows, cols);
        return 1;
    }
    if (max_blocks > SA_MAX_BLOCKS) max_blocks = SA_MAX_BLOCKS;
    if (max_holes  > SA_MAX_HOLES)  max_holes  = SA_MAX_HOLES;

    srand(seed);

    sokoban_set_grid(rows, cols);
    sokoban_init();
    if (heap_cap > 0) {
        sokoban_set_heap_cap(heap_cap);
        fprintf(stderr, "[sa] heap-cap: solver aborts (returns -3) when heap > %d\n",
                heap_cap);
    }
    g_sa_rows = rows; g_sa_cols = cols; g_sa_ncells = rows * cols;
    g_sa_max_blocks = max_blocks; g_sa_max_holes = max_holes;

    /* Validate constraint flags against the actual grid. */
    if (g_sa_fixed_exit >= g_sa_ncells) {
        fprintf(stderr, "error: --exit %d outside grid %dx%d\n", g_sa_fixed_exit, rows, cols);
        return 1;
    }
    uint64_t grid_mask = (g_sa_ncells == 64) ? ~0ULL : ((1ULL << g_sa_ncells) - 1);
    if (g_sa_fixed_walls & ~grid_mask) {
        fprintf(stderr, "error: --fixedwalls includes cells outside grid %dx%d\n", rows, cols);
        return 1;
    }
    if (g_sa_fixed_holes_mask & ~grid_mask) {
        fprintf(stderr, "error: --fixedholes includes cells outside grid %dx%d\n", rows, cols);
        return 1;
    }
    /* --fixedwalls and --exit can't coincide. */
    if (g_sa_fixed_exit >= 0 && (g_sa_fixed_walls & bit(g_sa_fixed_exit))) {
        fprintf(stderr, "error: --exit cell %d also appears in --fixedwalls\n", g_sa_fixed_exit);
        return 1;
    }
    /* --fixedholes and --fixedwalls can't share a cell. */
    if (g_sa_fixed_holes_mask & g_sa_fixed_walls) {
        fprintf(stderr, "error: --fixedholes and --fixedwalls share a cell\n");
        return 1;
    }
    /* --fixedholes and --exit can't share a cell. */
    if (g_sa_fixed_exit >= 0 && (g_sa_fixed_holes_mask & bit(g_sa_fixed_exit))) {
        fprintf(stderr, "error: --exit cell %d also appears in --fixedholes\n", g_sa_fixed_exit);
        return 1;
    }
    /* --holeboard compatibility. */
    if (g_sa_holeboard) {
        if (g_sa_fixed_empty_mask) {
            fprintf(stderr, "error: --holeboard incompatible with --fixedempty\n");
            return 1;
        }
        if (g_sa_fixed_walls) {
            fprintf(stderr, "error: --holeboard incompatible with --fixedwalls (walls forbidden)\n");
            return 1;
        }
        /* Feasibility: ncells - 2 (player + exit) must be coverable by
         * holes (≤ 31) plus blocks (≤ max_blocks or num-blocks). */
        int holes_max  = SA_HOLEBOARD_MAX_HOLES;
        int blocks_max = max_blocks;
        if (g_sa_num_blocks_cap >= 0 && g_sa_num_blocks_cap < blocks_max)
            blocks_max = g_sa_num_blocks_cap;
        int unforced = g_sa_ncells - 2;
        if (unforced > holes_max + blocks_max) {
            fprintf(stderr,
                "error: --holeboard on %dx%d cannot be satisfied — need ≥ %d cells "
                "covered, but holes≤%d + blocks≤%d = %d max.  Raise --max-blocks.\n",
                rows, cols, unforced, holes_max, blocks_max, holes_max + blocks_max);
            return 1;
        }
    }
    /* --fixedempty bounds + collision checks. */
    if (g_sa_fixed_empty_mask & ~grid_mask) {
        fprintf(stderr, "error: --fixedempty includes cells outside grid %dx%d\n", rows, cols);
        return 1;
    }
    if (g_sa_fixed_empty_mask & g_sa_fixed_walls) {
        fprintf(stderr, "error: --fixedempty and --fixedwalls share a cell\n");
        return 1;
    }
    if (g_sa_fixed_empty_mask & g_sa_fixed_holes_mask) {
        fprintf(stderr, "error: --fixedempty and --fixedholes share a cell\n");
        return 1;
    }
    /* --fixedblocks cells must fit the grid and not collide with other locks. */
    if (g_sa_n_fixed_blocks > 0) {
        uint64_t fb_mask = 0;
        for (int i = 0; i < g_sa_n_fixed_blocks; i++) {
            int c = g_sa_fixed_block_pos[i];
            if (c < 0 || c >= g_sa_ncells) {
                fprintf(stderr, "error: --fixedblocks cell %d outside grid %dx%d\n",
                        c, rows, cols);
                return 1;
            }
            fb_mask |= bit(c);
        }
        if (fb_mask & g_sa_fixed_walls) {
            fprintf(stderr, "error: --fixedblocks shares a cell with --fixedwalls\n");
            return 1;
        }
        if (fb_mask & g_sa_fixed_holes_mask) {
            fprintf(stderr, "error: --fixedblocks shares a cell with --fixedholes\n");
            return 1;
        }
        if (g_sa_fixed_exit >= 0 && (fb_mask & bit(g_sa_fixed_exit))) {
            fprintf(stderr, "error: --fixedblocks shares a cell with --exit\n");
            return 1;
        }
        if (g_sa_num_blocks_cap >= 0 && g_sa_num_blocks_cap < g_sa_n_fixed_blocks) {
            fprintf(stderr, "error: --num-blocks %d < %d required by --fixedblocks\n",
                    g_sa_num_blocks_cap, g_sa_n_fixed_blocks);
            return 1;
        }
        if (fb_mask & g_sa_fixed_empty_mask) {
            fprintf(stderr, "error: --fixedblocks shares a cell with --fixedempty\n");
            return 1;
        }
    }
    /* If --num-holes is set, it must be at least the number of fixed holes. */
    if (g_sa_num_holes_cap >= 0) {
        int n_fixed = __builtin_popcountll(g_sa_fixed_holes_mask);
        if (g_sa_num_holes_cap < n_fixed) {
            fprintf(stderr, "error: --num-holes %d < %d required by --fixedholes\n",
                    g_sa_num_holes_cap, n_fixed);
            return 1;
        }
    }

    g_status_enabled = status_line;

    if (cache_bits > 0) {
        if (cache_bits > 28) cache_bits = 28;  /* cap at 4 GB-scale */
        if (g_sa_cache_ways < 1) g_sa_cache_ways = 1;
        /* Cap ways at 8 to keep the round-robin index in a byte and the
         * lookup scan tight; 4 is the empirically motivated default. */
        if (g_sa_cache_ways > 8) g_sa_cache_ways = 8;
        size_t total_entries = (size_t)1 << cache_bits;
        /* Round set count down to ensure total_entries is a multiple of ways. */
        size_t sets = total_entries / (size_t)g_sa_cache_ways;
        if (sets == 0) sets = 1;
        /* Round sets down to a power of two for cheap masking. */
        size_t sets_p2 = 1; while (sets_p2 * 2 <= sets) sets_p2 *= 2;
        sets = sets_p2;
        total_entries = sets * (size_t)g_sa_cache_ways;
        g_sa_cache = (SaCacheEntry *)calloc(total_entries, sizeof(SaCacheEntry));
        g_sa_cache_victim = (uint8_t *)calloc(sets, sizeof(uint8_t));
        if (!g_sa_cache || !g_sa_cache_victim) {
            free(g_sa_cache); free(g_sa_cache_victim);
            g_sa_cache = NULL; g_sa_cache_victim = NULL;
            fprintf(stderr, "[sa] cache calloc failed; running without cache\n");
        } else {
            g_sa_cache_set_mask = (uint64_t)(sets - 1);
            fprintf(stderr,
                    "[sa] solver cache: %zu entries × %d-way (%zu sets, %.1f MB)  canonical=%s\n",
                    total_entries, g_sa_cache_ways, sets,
                    (double)(total_entries * sizeof(SaCacheEntry) + sets) / (1024.0 * 1024.0),
                    g_sa_cache_canonical ? "on" : "off");
        }
    }

    if (slow_solve_warn_s > 0.0) {
        g_slow_solve_threshold_s = slow_solve_warn_s;
        if (pthread_create(&g_watchdog_thread, NULL, slow_solve_watchdog, NULL) == 0) {
            g_watchdog_started = 1;
            fprintf(stderr, "[sa] slow-solve watchdog: warn after %.1fs\n",
                    slow_solve_warn_s);
        } else {
            fprintf(stderr, "[sa] failed to start slow-solve watchdog (disabled)\n");
        }
    }

    char restarts_str[24];
    if (restarts > 0) snprintf(restarts_str, sizeof restarts_str, "%d", restarts);
    else              snprintf(restarts_str, sizeof restarts_str, "unlimited");
    fprintf(stderr,
            "[sa] grid %dx%d  steps=%lld  restarts=%s  T0=%.2f T1=%.2f  "
            "cool-by=%s  reheat-after=%lld factor=%.2f  "
            "restart-after-stale=%lld  "
            "init=(b%d,h%d)  max=(b%d,h%d)  seed=%u\n",
            rows, cols, steps, restarts_str, T0, T1,
            cool_by_time ? "time" : "steps",
            reheat_after, reheat_factor,
            stagnation_restart_after,
            init_blocks, init_holes, max_blocks, max_holes, seed);
    if (g_sa_fixed_exit >= 0 || g_sa_fixed_walls || g_sa_fixed_holes_mask ||
        g_sa_num_blocks_cap >= 0 || g_sa_num_holes_cap >= 0 ||
        g_sa_n_fixed_blocks > 0 || g_sa_fixed_empty_mask || g_sa_holeboard) {
        fprintf(stderr, "[sa] constraints:");
        if (g_sa_holeboard) fprintf(stderr, " holeboard");
        if (g_sa_fixed_exit >= 0) fprintf(stderr, " exit=%d", g_sa_fixed_exit);
        if (g_sa_num_blocks_cap >= 0) fprintf(stderr, " num-blocks≤%d", g_sa_num_blocks_cap);
        if (g_sa_num_holes_cap  >= 0) fprintf(stderr, " num-holes≤%d",  g_sa_num_holes_cap);
        if (g_sa_fixed_walls) {
            fprintf(stderr, " fixedwalls=[");
            int first = 1;
            for (int c = 0; c < g_sa_ncells; c++)
                if (g_sa_fixed_walls & bit(c)) { fprintf(stderr, "%s%d", first?"":",", c); first = 0; }
            fprintf(stderr, "]");
        }
        if (g_sa_fixed_holes_mask) {
            fprintf(stderr, " fixedholes=[");
            int first = 1;
            for (int c = 0; c < g_sa_ncells; c++)
                if (g_sa_fixed_holes_mask & bit(c)) { fprintf(stderr, "%s%d", first?"":",", c); first = 0; }
            fprintf(stderr, "]");
        }
        if (g_sa_n_fixed_blocks > 0) {
            fprintf(stderr, " fixedblocks=[");
            for (int i = 0; i < g_sa_n_fixed_blocks; i++) {
                uint8_t m = g_sa_fixed_block_mask[i];
                fprintf(stderr, "%s%d=[%s%s%s%s]", i?",":"", g_sa_fixed_block_pos[i],
                        m & 1 ? "U" : "", m & 2 ? "R" : "",
                        m & 4 ? "D" : "", m & 8 ? "L" : "");
            }
            fprintf(stderr, "]");
        }
        if (g_sa_fixed_empty_mask) {
            fprintf(stderr, " fixedempty=[");
            int first = 1;
            for (int c = 0; c < g_sa_ncells; c++)
                if (g_sa_fixed_empty_mask & bit(c)) { fprintf(stderr, "%s%d", first?"":",", c); first = 0; }
            fprintf(stderr, "]");
        }
        fprintf(stderr, "\n");
    }

    struct timespec t_start;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int global_best = 0;
    PC  global_best_p;
    memset(&global_best_p, 0, sizeof global_best_p);

    /* Seed puzzle: parse file, evaluate, and use as initial global best. */
    int has_seed_puzzle = 0;
    if (seed_puzzle_path) {
        PC seed_pc;
        if (!parse_puzzle_file(seed_puzzle_path, &seed_pc)) return 1;
        if (!valid(&seed_pc)) {
            fprintf(stderr, "error: seed puzzle fails validation\n");
            return 1;
        }
        int seed_depth = evaluate(&seed_pc);
        if (seed_depth < 0) {
            fprintf(stderr, "error: seed puzzle is unsolvable\n");
            return 1;
        }
        global_best = seed_depth;
        global_best_p = seed_pc;
        has_seed_puzzle = 1;
        fprintf(stderr, "[sa] seed puzzle: depth %d from '%s'\n",
                seed_depth, seed_puzzle_path);
    }

    long long total_evals  = 0;
    long long total_accept = 0;
    /* Count of restarts since the global best was last improved.  Used
     * to scale the reseed perturbation distance: stale chains → bigger
     * kicks → escape the current basin. */
    int restarts_since_improvement = 0;

    /* Debounced streaming output — same shape as backsearch_worker.
     *   - on a new global best, stage the depth + puzzle in pending_*
     *   - after STREAM_DEBOUNCE_S seconds have passed since the first
     *     improvement in the current cluster, flush (print depth + puzzle)
     *   - clusters of rapid improvements coalesce to a single print
     *     showing the *latest* state.
     * STREAM_DEBOUNCE_S matches backsearch's value (1.0 s). */
    const double STREAM_DEBOUNCE_S = 1.0;
    int    pending_best = 0;
    double pending_first_time = -1.0;
    double pending_at_time = 0.0;
#define FLUSH_BEST_NOW()                                                     \
    do {                                                                     \
        if (pending_first_time >= 0) {                                       \
            double _t = pending_at_time;                                     \
            const char *_unit = "s";                                         \
            double _val = _t;                                                \
            if (_t >= 3600.0)      { _val = _t / 3600.0; _unit = "h"; }      \
            else if (_t >= 60.0)   { _val = _t / 60.0;   _unit = "m"; }      \
            status_clear();                                                  \
            printf("%d (%.1f%s)\n", pending_best, _val, _unit);              \
            print_puzzle(&global_best_p, pending_best);                      \
            putchar('\n');                                                   \
            fflush(stdout);                                                  \
            pending_first_time = -1.0;                                       \
        }                                                                    \
    } while (0)
#define MAYBE_FLUSH_BEST()                                                   \
    do {                                                                     \
        if (pending_first_time >= 0) {                                       \
            double _now = elapsed_s(t_start);                                \
            if (_now - pending_first_time >= STREAM_DEBOUNCE_S)              \
                FLUSH_BEST_NOW();                                            \
        }                                                                    \
    } while (0)

    for (int restart = 0; restarts == 0 || restart < restarts; restart++) {
        if (time_cap > 0 && elapsed_s(t_start) > time_cap) break;
        status_print(restarts_since_improvement, global_best,
                     elapsed_s(t_start), 1);

        PC cur;
        /* Seed choice: with reseed_from_best_prob probability, perturb
         * the current global best instead of random init.  The first
         * restart (global_best=0) and restarts where the dice roll says
         * no use pure random.  Reseeding lets us climb out of the
         * best-known basin without losing pure-exploration variance. */
        /* Reseed probability ramps with restarts_since_improvement.
         * Early restarts (count = 0..warmup-1) → pure random (sample
         * basin distribution).  After that, prob rises linearly to its
         * configured max as we get more confident we're stuck. */
        const int reseed_warmup = 10;
        double eff_reseed_prob = 0.0;
        if (global_best > 0 && reseed_from_best_prob > 0.0
            && restarts_since_improvement >= reseed_warmup) {
            double t = (double)(restarts_since_improvement - reseed_warmup) / 10.0;
            if (t > 1.0) t = 1.0;
            eff_reseed_prob = reseed_from_best_prob * t;
        }
        int use_seed_direct = (has_seed_puzzle && restart == 0);
        int reseed = use_seed_direct
                     || (eff_reseed_prob > 0.0 && rand_uniform() < eff_reseed_prob);
        if (reseed) {
            cur = global_best_p;
            /* Deterministic, monotone perturbation distance.  Grows as
             *   base + step * k^power
             * where k = restarts_since_improvement.  Rationale: re-trying
             * small perturbations of the same global best is provably futile
             * after they've already failed, so we commit to escaping the
             * basin by widening the radius without backtracking. */
            int eff_perturb = 0;
            if (!use_seed_direct) {
                int k = restarts_since_improvement;
                double growth = (k > 0) ? pow((double)k, reseed_perturb_power) : 0.0;
                eff_perturb = reseed_perturb_count
                              + (int)(reseed_perturb_step * growth);
                if (eff_perturb > reseed_perturb_max) eff_perturb = reseed_perturb_max;
                if (eff_perturb < reseed_perturb_count) eff_perturb = reseed_perturb_count;
                for (int m = 0; m < eff_perturb; m++) mutate(&cur);
            }
            if (verbose) {
                status_clear();
                fprintf(stderr, "[reseed] restart %d: perturb=%d "
                                "(since-improvement=%d, power=%.2f)\n",
                        restart, eff_perturb,
                        restarts_since_improvement, reseed_perturb_power);
            }
        } else {
            init_random(&cur, init_blocks, init_holes);
        }
        int cur_depth = evaluate_cached(&cur);
        total_evals++;
        if (cur_depth < 0) cur_depth = 0;

        PC best = cur;
        int best_depth = cur_depth;

        double T = T0;
        /* Cooling schedule: T(t_in_phase) = T_phase0 * (T1/T_phase0)^(t/phase_budget).
         * On reheat, phase resets: T_phase0 ← reheat_T, phase_t0 ← now,
         * phase_budget shrinks proportionally to remaining time. */
        double phase_t0     = elapsed_s(t_start);
        double T_phase0     = T0;
        double restart_t0   = phase_t0;
        /* restart_budget=0 means "no per-restart time budget; loop on step
         * count and stagnation only".  This is the right policy when
         * stagnation-restart is on — chains run until they actually get
         * stuck, not until an arbitrary time slice runs out.
         *
         * When stagnation-restart is off and --time is set, we slice the
         * remaining time evenly across remaining restarts. */
        double restart_budget = 0.0;
        if (time_cap > 0 && stagnation_restart_after == 0 && restarts > 0) {
            restart_budget = (time_cap - restart_t0) / (double)(restarts - restart);
            if (restart_budget < 1.0) restart_budget = 1.0;
        }
        double phase_budget = restart_budget;
        long long phase_step0 = 0;  /* step at start of current phase */

        long long stale_steps = 0;
        double    last_improvement_time = elapsed_s(t_start);

        for (long long step = 0; step < steps; step++) {
            double now = elapsed_s(t_start);
            double t_in_phase = now - phase_t0;
            double t_in_restart = now - restart_t0;
            if (time_cap > 0 && (step & 1023) == 0 && now > time_cap) break;
            if (cool_by_time && restart_budget > 0 && t_in_restart > restart_budget) break;
            if ((step & 1023) == 0) {
                MAYBE_FLUSH_BEST();
                status_print(restarts_since_improvement, global_best, now, 0);
            }

            PC prop = cur;
            mutate(&prop);
            int prop_depth = evaluate_cached(&prop);
            total_evals++;
            if (prop_depth < 0) prop_depth = 0;

            int delta = prop_depth - cur_depth;
            int accepted = (delta >= 0) || (rand_uniform() < exp((double)delta / T));
            if (accepted) {
                cur = prop;
                cur_depth = prop_depth;
                total_accept++;
                if (cur_depth > best_depth) {
                    best = cur;
                    best_depth = cur_depth;
                    stale_steps = 0;
                    last_improvement_time = now;
                    if (best_depth > global_best) {
                        global_best = best_depth;
                        global_best_p = best;
                        restarts_since_improvement = 0;
                        /* Paranoid recheck: re-evaluate from a clean PC via
                         * the uncached solver path.  If this disagrees with
                         * best_depth, something between mutate/evaluate and
                         * the global-best assignment has gone wrong (stale
                         * cache, struct corruption, solver non-determinism).
                         * Costs one BFS per improvement — rare enough to be
                         * effectively free. */
                        int verify_depth = evaluate(&global_best_p);
                        if (verify_depth != global_best) {
                            status_clear();
                            fprintf(stderr,
                                "[sa] WARNING new-best mismatch: reported %d, "
                                "re-eval %d.  Using re-eval value.\n",
                                global_best, verify_depth);
                            if (verify_depth < 0) verify_depth = 0;
                            global_best = verify_depth;
                            best_depth  = verify_depth;
                            /* Don't update pending_* — the print will use the
                             * corrected global_best on the next staged event. */
                        }
                        /* Stage for debounced print; MAYBE_FLUSH_BEST emits. */
                        if (pending_first_time < 0) pending_first_time = now;
                        pending_best = global_best;
                        pending_at_time = now;
                    }
                } else {
                    stale_steps++;
                }
            } else {
                stale_steps++;
            }

            /* Restart on stagnation: bail out of this chain, outer loop
             * will spin up a new random init.  Trigger on whichever of
             * step-based or time-based threshold fires first.
             *
             * Threshold is SCALED by gap-to-global: a chain within
             * depth_bonus_window of global_best gets
             *   eff = base * (1 + factor * (1 - gap/window))
             * with gap = global_best - chain_best, clamped ≥ 0.
             * So a chain matching global gets (1+factor)× patience, a
             * chain `window` below gets 1× (no bonus).  Sharper than a
             * ratio formulation: chains stuck a few below the ceiling
             * lose their bonus quickly. */
            int depth_gap = global_best - best_depth;
            if (depth_gap < 0) depth_gap = 0;
            double bonus_weight = 0.0;
            if (global_best > 0 && depth_bonus_window > 0.0
                && depth_gap < depth_bonus_window) {
                bonus_weight = 1.0 - (double)depth_gap / depth_bonus_window;
            }
            double bonus_mult = 1.0 + depth_bonus_factor * bonus_weight;
            long long eff_step_thresh = (long long)((double)stagnation_restart_after * bonus_mult);
            double    eff_time_thresh = stagnation_restart_time * bonus_mult;
            int stale_by_steps = (stagnation_restart_after > 0
                                  && stale_steps >= eff_step_thresh);
            int stale_by_time  = (stagnation_restart_time > 0.0
                                  && (now - last_improvement_time) >= eff_time_thresh);
            if (stale_by_steps || stale_by_time) {
                if (verbose) {
                    status_clear();
                    fprintf(stderr,
                            "[stagnation-restart] chain done at step %lld (%.1fs in restart, "
                            "best=%d, trigger=%s)\n",
                            step, t_in_restart, best_depth,
                            stale_by_time ? "time" : "steps");
                }
                break;
            }
            /* Reheat when stagnant: bump T, reset phase. */
            if (reheat_after > 0 && stale_steps >= reheat_after) {
                T_phase0 = T * reheat_factor;
                if (T_phase0 > T0) T_phase0 = T0;
                T = T_phase0;
                phase_t0 = now;
                phase_step0 = step;
                phase_budget = restart_budget - t_in_restart;
                if (phase_budget < 1.0) phase_budget = 1.0;
                stale_steps = 0;
                if (verbose) {
                    status_clear();
                    fprintf(stderr, "[reheat] T<-%.3f at step %lld (%.1fs in restart)\n",
                            T_phase0, step, t_in_restart);
                }
            }

            /* Cool. */
            if (cool_by_time && phase_budget > 0) {
                double frac = (now - phase_t0) / phase_budget;
                if (frac > 1.0) frac = 1.0;
                T = T_phase0 * pow(T1 / T_phase0, frac);
            } else {
                /* Step-based cooling.  cool_per_step is set so step-based budget
                 * matches the total steps in the current phase. */
                long long remaining = steps - phase_step0;
                if (remaining < 1) remaining = 1;
                double cool_per_step = pow(T1 / T_phase0, 1.0 / (double)remaining);
                T *= cool_per_step;
            }
        }

        /* Per-restart line suppressed by default — too spammy with
         * stagnation-restart firing often.  Final summary printed below. */

        /* Bump the stale-restarts counter; reset on improvement happens
         * inside the inner loop on global_best update. */
        restarts_since_improvement++;
    }

    /* Final flush of any pending debounced update. */
    FLUSH_BEST_NOW();
    status_clear();

    /* Verify the global best (paranoid recheck). */
    int verified = evaluate(&global_best_p);
    printf("\nBest depth:        %d  (verify %d)\n", global_best, verified);
    print_puzzle(&global_best_p, global_best);
    fprintf(stderr, "[sa] total evals: %lld  total elapsed: %.1fs\n",
            total_evals, elapsed_s(t_start));
    if (g_sa_cache) {
        long long lookups = g_sa_cache_hits + g_sa_cache_miss;
        double rate = lookups ? 100.0 * (double)g_sa_cache_hits / (double)lookups : 0.0;
        fprintf(stderr, "[sa] cache: %lld hits / %lld lookups (%.1f%%)\n",
                g_sa_cache_hits, lookups, rate);
    }
    {
        long long total = g_stat_solvable_count + g_stat_unsolvable_count + g_stat_aborted_count;
        double    total_t = g_stat_solvable_time_s + g_stat_unsolvable_time_s + g_stat_aborted_time_s;
        if (total > 0) {
            fprintf(stderr,
                "[sa] solver-time breakdown: solvable=%lld in %.2fs (%.0f%%)  "
                "unsolvable=%lld in %.2fs (%.0f%%)  "
                "aborted=%lld in %.2fs (%.0f%%)\n",
                g_stat_solvable_count,   g_stat_solvable_time_s,
                total_t > 0 ? 100.0 * g_stat_solvable_time_s   / total_t : 0.0,
                g_stat_unsolvable_count, g_stat_unsolvable_time_s,
                total_t > 0 ? 100.0 * g_stat_unsolvable_time_s / total_t : 0.0,
                g_stat_aborted_count,    g_stat_aborted_time_s,
                total_t > 0 ? 100.0 * g_stat_aborted_time_s    / total_t : 0.0);
        }
    }
    return 0;
}
