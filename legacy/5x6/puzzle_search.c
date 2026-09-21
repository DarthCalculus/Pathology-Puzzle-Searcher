/*
 * puzzle_search.c
 *
 * Brute-force enumeration of Sokoban-variant puzzles.  Searches all
 * configurations with a given total number of blocks and reports the one
 * with the longest BFS solution.
 *
 * Usage:  ./puzzle_search <num_blocks>
 *
 * No separate wall enumeration is performed.  A block assigned mask 0
 * (immovable) is functionally identical to a wall, so the bitmask search
 * already covers every wall configuration without redundancy: the position
 * combination "blocks at A and B" subsumes both "wall at A, block at B" and
 * "wall at B, block at A" from a traditional split enumeration.
 *
 * Symmetry reduction: only the 6 exit cells in the top-left quadrant are
 * tried (the rest are equivalent by reflection).
 *
 * Bitmask pruning: enumerates mask vectors top-down (most pushable first).
 * An unsolvable mask prunes all its subsets via Fact 2 (fewer directions →
 * at least as hard).  This avoids expensive unsolvable BFS calls for
 * subsets, since unsolvable BFS must exhaust all states before returning.
 *
 * Parallelism: a work queue of (exit, hole-configuration) pairs is consumed
 * by NUM_THREADS worker threads.  Each nh level is processed in full before
 * the next begins, maximising walk-distance pruning (approach C) for large
 * hole counts.  sokoban_solve() is thread-safe (per-thread BFS state);
 * g_best/g_best_pz are protected by g_best_mutex.
 */

#include "sokoban_bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

#define NUM_THREADS 14   /* default worker threads; overridden at runtime by --nthreads */
static int g_num_threads = NUM_THREADS;

/*
 * Nine canonical exit positions — the D2 fundamental domain for a 5×6 grid.
 * The symmetry group of a rectangle (D2, order 4: identity + 180° rotation +
 * horizontal flip + vertical flip) maps any exit cell to one of these nine
 * representatives (rows 0-2, cols 0-2):
 *
 *   (0,0)=0   (0,1)=1   (0,2)=2
 *   (1,0)=6   (1,1)=7   (1,2)=8
 *   (2,0)=12  (2,1)=13  (2,2)=14
 *
 * Symmetry reduction is applied to hole combinations: for each canonical exit,
 * the stabilizer subgroup (transforms fixing that exit) is used to keep only
 * the lexicographically-minimum hole combination in each orbit.
 *
 * Stabilizers (non-identity elements only):
 *   exits in row 2: (2,0), (2,1), (2,2)  — flip_v: (r,c)↔(4-r,c)
 *   all other exits                       — trivial (no check needed)
 *
 * Player starts are NOT restricted by symmetry; Approaches C (walk-distance
 * bounding) and A (free-component antichain sharing) handle pruning dynamically.
 */
#define NUM_EXIT_CELLS 9
static const int EXIT_CELLS[NUM_EXIT_CELLS] = { 0, 1, 2, 6, 7, 8, 12, 13, 14 };

/*
 * D2 cell transform tables — initialised in precompute_tables().
 * Each maps cell index p=(r*6+c) to its image under the named symmetry.
 *   rot180: (r,c) → (4-r, 5-c)   [180° rotation]
 *   flip_h: (r,c) → (r,   5-c)   [horizontal flip]
 *   flip_v: (r,c) → (4-r, c  )   [vertical flip]
 * (rot90/rot270/flip_d/flip_a do not apply to a non-square grid.)
 */
static int8_t t_rot180[NCELLS];
static int8_t t_flip_h[NCELLS];
static int8_t t_flip_v[NCELLS];

/* -------------------------------------------------------------------------
 * Combination iterator
 * Iterates over all ways to choose k items from a pool of n items (by index).
 * Usage:
 *   Comb c; comb_init(&c, n, k);
 *   do { ... use c.idx[0..k-1] ... } while (comb_next(&c));
 * ------------------------------------------------------------------------- */

typedef struct {
    int n, k;
    int idx[NCELLS];
} Comb;

static void comb_init(Comb *c, int n, int k) {
    c->n = n;
    c->k = k;
    for (int i = 0; i < k; i++) c->idx[i] = i;
}

/* Advance to the next combination.  Returns 0 when all combinations are
 * exhausted (including the k=0 case, which has exactly one combination). */
static int comb_next(Comb *c) {
    int i = c->k - 1;
    while (i >= 0 && c->idx[i] == c->n - c->k + i) i--;
    if (i < 0) return 0;
    c->idx[i]++;
    for (int j = i + 1; j < c->k; j++) c->idx[j] = c->idx[j-1] + 1;
    return 1;
}

/* -------------------------------------------------------------------------
 * Grid helpers
 * ------------------------------------------------------------------------- */

/*
 * Returns the bitmask of push directions that are geometrically possible
 * for a block at cell p.  A block on the top or bottom edge cannot be
 * pushed perpendicular to that edge: pushing it up/down would either place
 * it out of bounds or require the player to stand out of bounds.  The same
 * logic applies to left/right edges.
 */
static uint8_t valid_push_mask(int p) {
    int r = p / COLS, c = p % COLS;
    uint8_t m = 15;
    if (r == 0       || r == ROWS-1) m &= ~(1u | 4u); /* remove U and D */
    if (c == 0       || c == COLS-1) m &= ~(2u | 8u); /* remove R and L */
    return m;
}

/* ---- Precomputed neighbor table (duplicate from sokoban_bfs.c) ----
 * adj[cell][dir] = neighbor cell index, or -1 if out of bounds.
 * Directions: 0=Up, 1=Right, 2=Down, 3=Left
 */
static const int8_t adj[NCELLS][4] = {
    /* row 0 */
    {-1,  1,  6, -1}, {-1,  2,  7,  0}, {-1,  3,  8,  1},
    {-1,  4,  9,  2}, {-1,  5, 10,  3}, {-1, -1, 11,  4},
    /* row 1 */
    { 0,  7, 12, -1}, { 1,  8, 13,  6}, { 2,  9, 14,  7},
    { 3, 10, 15,  8}, { 4, 11, 16,  9}, { 5, -1, 17, 10},
    /* row 2 */
    { 6, 13, 18, -1}, { 7, 14, 19, 12}, { 8, 15, 20, 13},
    { 9, 16, 21, 14}, {10, 17, 22, 15}, {11, -1, 23, 16},
    /* row 3 */
    {12, 19, 24, -1}, {13, 20, 25, 18}, {14, 21, 26, 19},
    {15, 22, 27, 20}, {16, 23, 28, 21}, {17, -1, 29, 22},
    /* row 4 */
    {18, 25, -1, -1}, {19, 26, -1, 24}, {20, 27, -1, 25},
    {21, 28, -1, 26}, {22, 29, -1, 27}, {23, -1, -1, 28},
};

/* Walk BFS from 'start', filling dist_out[cell] with distance to each cell.
 * dist_out[cell] = -1 if unreachable.  dist_out[start] = 0. */
static void walk_all_distances(uint32_t blocked, int start, int8_t dist_out[NCELLS]) {
    for (int i = 0; i < NCELLS; i++) dist_out[i] = -1;
    if (blocked & (1u << start)) return;
    dist_out[start] = 0;
    uint32_t visited = (1u << start) | blocked;
    uint8_t q[NCELLS];
    int qh = 0, qt = 0, ql = 0, dist = 0;
    q[qt++] = (uint8_t)start;
    ql = qt;
    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        int p = q[qh++];
        for (int d = 0; d < 4; d++) {
            int np = adj[p][d];
            if (np < 0) continue;
            uint32_t bit = 1u << np;
            if (visited & bit) continue;
            visited |= bit;
            dist_out[np] = (int8_t)(dist + 1);
            q[qt++] = (uint8_t)np;
        }
    }
}

/*
 * Returns 1 if hp[0..nh-1] (sorted ascending) is the lex-minimum among itself
 * and its image under transform t.  Returns 0 if the image is strictly smaller
 * (meaning hp is not canonical and should be skipped).
 *
 * Used to enforce hole-combination canonicality under the stabilizer of each
 * exit cell.  Only one call per non-identity stabilizer element is needed.
 */
static int holes_lex_min_under(const int *hp, int nh, const int8_t t[NCELLS]) {
    /* Apply transform, sort the result (insertion sort on small array). */
    int8_t th[MAX_HOLES];
    for (int i = 0; i < nh; i++) th[i] = t[hp[i]];
    for (int i = 1; i < nh; i++) {
        int8_t x = th[i]; int j = i - 1;
        while (j >= 0 && th[j] > x) { th[j+1] = th[j]; j--; }
        th[j+1] = x;
    }
    /* Lex compare: if transformed < original, original is not canonical. */
    for (int i = 0; i < nh; i++) {
        if (th[i] < hp[i]) return 0;
        if (th[i] > hp[i]) return 1;
    }
    return 1; /* equal: canonical */
}

/*
 * Returns 1 if transform t maps hp (sorted ascending) to itself as a set.
 * Used to determine which stabilizer elements fix the hole configuration,
 * so those same elements can be applied to block combinations.
 */
static int transform_fixes_holes(const int *hp, int nh, const int8_t t[NCELLS]) {
    int8_t th[MAX_HOLES];
    for (int i = 0; i < nh; i++) th[i] = t[hp[i]];
    for (int i = 1; i < nh; i++) {
        int8_t x = th[i]; int j = i - 1;
        while (j >= 0 && th[j] > x) { th[j+1] = th[j]; j--; }
        th[j+1] = x;
    }
    for (int i = 0; i < nh; i++)
        if (th[i] != (int8_t)hp[i]) return 0;
    return 1;
}

/* Fast reachability check using bitmask flood fill on the 20-cell grid.
 * Returns shortest distance from start to target, or -1 if unreachable.
 * blocked: bitmask of cells the player cannot enter. */
static int fast_reachable(uint32_t blocked, int start, int target) {
    if (blocked & (1u << target)) return -1;
    uint32_t visited = (1u << start) | blocked;
    uint8_t q[NCELLS];
    int qh = 0, qt = 0, ql, dist = 0;
    q[qt++] = (uint8_t)start;
    ql = qt;
    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        int p = q[qh++];
        for (int d = 0; d < 4; d++) {
            int np = adj[p][d];
            if (np < 0) continue;
            uint32_t bit = 1u << np;
            if (visited & bit) continue;
            visited |= bit;
            if (np == target) return dist + 1;
            q[qt++] = (uint8_t)np;
        }
    }
    return -1;
}

/* -------------------------------------------------------------------------
 * Global best tracking  (protected by g_best_mutex)
 * ------------------------------------------------------------------------- */

static int    g_best   = -1;
static Puzzle g_best_pz;

static int    g_skipped    = 0;
static Puzzle g_skipped_pz;

/* Fixed cells: specified by the user, present in every configuration. */
static uint32_t g_fixed_walls      = 0;
static uint32_t g_fixed_holes_mask = 0;
static int      g_fixed_hole_pos[MAX_HOLES];
static int      g_fixed_nholes     = 0;
static uint32_t g_fixed_empty_mask  = 0;   /* always empty — no block/wall/hole here */
static uint32_t g_fixed_blocks_mask = 0;   /* always has a movable block */
static int      g_fixed_block_pos[MAX_BLOCKS];
static int      g_fixed_nblocks     = 0;
static uint8_t  g_fixed_mask[NCELLS];      /* per-cell geo_mask ceiling (default 0xF) */

static pthread_mutex_t g_best_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_print_mutex = PTHREAD_MUTEX_INITIALIZER;
static char            g_last_progress[256] = "";

static void print_block_info(int i, uint8_t m) {
    printf("block %c push=%x [%s%s%s%s]", 'A'+i, m,
           m&1?"U":"", m&2?"R":"", m&4?"D":"", m&8?"L":"");
}

static void print_puzzle(const Puzzle *pz) {
    /* Build an ASCII grid. */
    char grid[ROWS][COLS + 1];
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) grid[r][c] = '.';
        grid[r][COLS] = '\0';
    }
    for (int i = 0; i < NCELLS; i++)
        if (pz->walls & (1u << i)) grid[i/COLS][i%COLS] = '#';
    for (int i = 0; i < pz->num_holes; i++)
        grid[pz->hole_pos[i]/COLS][pz->hole_pos[i]%COLS] = 'O';
    for (int i = 0; i < pz->num_blocks; i++) {
        int p = pz->block_pos[i];
        if (p < NCELLS) grid[p/COLS][p%COLS] = 'A' + i;
    }
    grid[pz->exit_pos    / COLS][pz->exit_pos    % COLS] = 'E';
    grid[pz->player_start/ COLS][pz->player_start% COLS] = '@';

    for (int r = 0; r < ROWS; r++) {
        printf("  %s", grid[r]);
        if (r < pz->num_blocks) {
            printf("   "); print_block_info(r, pz->block_pushable[r]);
        }
        printf("\n");
    }
    for (int i = ROWS; i < pz->num_blocks; i++) {
        printf("             "); print_block_info(i, pz->block_pushable[i]); printf("\n");
    }
    printf("  exit=%d  player_start=%d  walls=%05x\n",
           pz->exit_pos, pz->player_start, pz->walls);
}

static void record_skip(const Puzzle *pz) {
    pthread_mutex_lock(&g_best_mutex);
    if (g_skipped == 0) g_skipped_pz = *pz;
    g_skipped++;
    pthread_mutex_unlock(&g_best_mutex);
}

static void update_best(int d, const Puzzle *pz) {
    if (d <= g_best) return;
    pthread_mutex_lock(&g_best_mutex);
    if (d > g_best) {
        g_best    = d;
        g_best_pz = *pz;
        pthread_mutex_lock(&g_print_mutex);
        if (g_last_progress[0]) {       /* erase progress bar if visible */
            fprintf(stderr, "\r%80s\r", "");
            fflush(stderr);
        }
        printf("New best: %d moves\n", d);
        print_puzzle(pz);
        fflush(stdout);
        if (g_last_progress[0]) {       /* reprint progress bar */
            fprintf(stderr, "%s", g_last_progress);
            fflush(stderr);
        }
        pthread_mutex_unlock(&g_print_mutex);
    }
    pthread_mutex_unlock(&g_best_mutex);
}

/* -------------------------------------------------------------------------
 * Per-thread call counter and BFS profiling (--profile)
 * ------------------------------------------------------------------------- */

static _Thread_local long long tl_ncalls = 0;

#define PROF_TBUCKETS 34   /* 2^0 ns .. 2^33 ns (~8 s) */
#define PROF_HBUCKETS 21   /* 2^0 .. 2^20 entries (1M cap) */

typedef struct {
    long long time_cnt[PROF_TBUCKETS][2];  /* [log2(ns)][solvable] */
    long long heap_cnt[PROF_HBUCKETS][2];  /* [log2(peak_heap)][solvable] */
    double    ps_time[NCELLS];             /* total try_bitmasks time per player start (s) */
    long long ps_calls[NCELLS];            /* BFS calls issued per player start */
} ProfData;

static int           g_profile_mode = 0;
static _Thread_local ProfData tl_prof;

/* -------------------------------------------------------------------------
 * Bitmask enumeration
 *
 * Exploits two facts about the partial order on mask vectors (A ⊆ B means
 * A has fewer pushable directions, i.e. A is at least as hard as B):
 *
 *   Fact 1: If A is solvable, steps(A) ≥ steps(B) for every B ⊇ A.
 *   Fact 2: If B is unsolvable, every A ⊆ B is also unsolvable.
 *
 * Enumeration order: MOST bits first (easiest → hardest).
 *
 *   • Easier masks tend to be solvable.  Solvable BFS calls are cheap
 *     because the solver terminates as soon as the exit is reached.
 *     Unsolvable BFS calls are expensive: the solver must exhaust the
 *     entire reachable state space before returning -1.
 *
 *   • When a mask m returns -1 (no solvable completion exists), Fact 2
 *     guarantees the same for every m' ⊆ m.  We record m in an
 *     "unsolvable antichain" and skip all future subsets.
 *
 * Cross-call antichain propagation:
 *
 *   When we move from mask m to a harder mask m' ⊆ m for block bi, any
 *   mask found unsolvable for block bi+1 under bi=m is also unsolvable
 *   under bi=m' (Fact 2: the full vector only got harder).  Rather than
 *   rebuilding bi+1's antichain from scratch each time, we maintain a
 *   cross-call store of (thresh, mask) pairs at each level: thresh is the
 *   bi mask under which the bi+1 mask was discovered unsolvable.  Before
 *   each call to the next level we seed its antichain with every entry
 *   where current_m ⊆ thresh.  After the call we record any newly
 *   discovered entries with thresh = current_m for future iterations.
 *
 * Interface:
 *   unsolv[0..*pn-1]  on entry : inherited unsolvable masks for block bi.
 *   unsolv[*pn..]     on exit  : newly discovered unsolvable masks appended.
 *   Buffer capacity must be at least MAX_UNSOLV.
 *
 * Returns the best step count found in this subtree, or -1 if none.
 * ------------------------------------------------------------------------- */

#define MAX_UNSOLV 30   /* safe upper bound: inherited(≤15) + new(≤15) */
#define MAX_GA    512   /* global antichain capacity                    */

/* Precomputed tables (initialised once in puzzle_search):
 *   cell_masks[p]    — valid masks for cell p in popcount-descending order
 *   cell_nmasks[p]   — number of valid masks for cell p
 *   superset_list[m] — all 4-bit masks t such that m ⊆ t (i.e. (m&t)==m)
 *   superset_cnt[m]  — number of such supersets (2^(4-popcount(m)))
 *
 * superset_list lets the XC seeding loop skip directly to the relevant
 * thresh buckets instead of scanning all entries linearly. */
static uint8_t cell_masks   [NCELLS][16];
static int     cell_nmasks  [NCELLS];
static uint8_t superset_list[16][16];
static int     superset_cnt [16];

static void precompute_tables(void) {
    /* Per-cell valid-mask lists (descending popcount) */
    for (int p = 0; p < NCELLS; p++) {
        uint8_t valid = valid_push_mask(p);
        int n = 0;
        for (int pc = 4; pc >= 0; pc--)
            for (int m = 15; m >= 0; m--)
                if (__builtin_popcount(m) == pc && (m & valid) == m)
                    cell_masks[p][n++] = (uint8_t)m;
        cell_nmasks[p] = n;
    }
    /* Superset lists for all 4-bit masks.
     * Supersets of m = { m | s : s is a subset of (~m & 0xF) }. */
    for (int m = 0; m < 16; m++) {
        int n = 0, complement = (~m) & 0xF;
        int s = complement;
        do {
            superset_list[m][n++] = (uint8_t)(m | s);
            if (s == 0) break;
            s = (s - 1) & complement;
        } while (1);
        superset_cnt[m] = n;
    }

    /* D2 cell transform tables.  All transforms act on a 5×6 grid where
     * cell p encodes (r,c) as p = r*COLS + c with r,c in [0, ROWS-1/COLS-1]. */
    for (int p = 0; p < NCELLS; p++) {
        int r = p / COLS, c = p % COLS;
        t_rot180[p] = (int8_t)((ROWS-1-r) * COLS + (COLS-1-c));
        t_flip_h[p] = (int8_t)(r          * COLS + (COLS-1-c));
        t_flip_v[p] = (int8_t)((ROWS-1-r) * COLS + c       );
    }
}

/* ---- Global antichain of full mask vectors ----
 *
 * Stores maximal unsolvable full mask vectors (one mask per block).
 * Before any BFS call we check whether the current vector is dominated
 * (component-wise ⊆) by any stored entry; if so it is provably unsolvable
 * and we skip the call entirely.  After BFS returns -1 we add the vector,
 * evicting any existing entries it supersedes.
 *
 * Scoped per board configuration (player_start + block positions + holes +
 * exit): a fresh GlobalAntichain is created for each player-start iteration
 * so no cross-board contamination occurs.
 */
typedef struct { uint8_t m[MAX_BLOCKS]; } GAEntry;
typedef struct { GAEntry entries[MAX_GA]; int count; } GlobalAntichain;

/* KnownSolvable: per-board table of (packed_used, dist) pairs from previous
 * BFS calls on this board.  Before a leaf BFS, if any stored entry's used
 * bitmask is a componentwise subset of the current block_pushable masks, the
 * same solution path is still valid under the current (more restrictive) masks
 * and we return the stored distance without calling sokoban_solve.
 *
 * Pack: bits [i*4+3 : i*4] = used directions for block i (4 bits each).
 * Subset check: (cur_packed & stored_packed) == stored_packed — one AND+CMP. */
#define MAX_KS 128
typedef struct {
    uint32_t packed[MAX_KS];
    int      dist  [MAX_KS];
    int      count;
} KnownSolvable;

/* 1 if masks[] is dominated by (⊆) any stored entry G (G ⊇ masks). */
static int ga_dominated(const GlobalAntichain *ga, const uint8_t *masks, int nb) {
    for (int j = 0; j < ga->count; j++) {
        int dom = 1;
        for (int i = 0; i < nb; i++)
            if ((ga->entries[j].m[i] & masks[i]) != masks[i]) { dom = 0; break; }
        if (dom) return 1;
    }
    return 0;
}

/* Add masks[] to GA, maintaining the antichain of maximal unsolvable vectors. */
static void ga_add(GlobalAntichain *ga, const uint8_t *masks, int nb) {
    if (ga_dominated(ga, masks, nb)) return;   /* already covered, skip */
    /* Remove existing entries that new vector supersedes (new ⊇ old). */
    int out = 0;
    for (int j = 0; j < ga->count; j++) {
        int nd = 1;
        for (int i = 0; i < nb; i++)
            if ((masks[i] & ga->entries[j].m[i]) != ga->entries[j].m[i]) { nd = 0; break; }
        if (!nd) ga->entries[out++] = ga->entries[j];
    }
    ga->count = out;
    if (ga->count < MAX_GA) {
        memcpy(ga->entries[ga->count].m, masks, nb);
        ga->count++;
    }
}

/*
 * Computes per-block maximum pushable masks, tightened beyond valid_push_mask
 * to account for frozen-block patterns and propagation.
 *
 * Three dead-block patterns are detected:
 *   1. Corners: valid_push_mask already returns 0 for these.
 *   2. Two adjacent movable blocks on the same edge row/column: each blocks
 *      the other's landing cell and push-from cell for all remaining push
 *      directions, so both are permanently immovable.
 *   3. Four movable blocks forming a 2x2 square: each has its two remaining
 *      landing cells and push-from cells occupied by the others.
 *
 * After detecting these patterns, the frozen set propagates: any movable
 * block whose landing cell or push-from cell for direction d is occupied by
 * a frozen block loses direction d, and may itself become frozen.
 */
static void compute_geo_masks(const int *mbp, int nb_mov, uint32_t wall_mask,
                               uint8_t *geo_mask)
{
    for (int i = 0; i < nb_mov; i++)
        geo_mask[i] = valid_push_mask(mbp[i]);

    uint32_t mob_occ = 0;
    for (int i = 0; i < nb_mov; i++) mob_occ |= (1u << mbp[i]);

    /* frozen: cells permanently occupied (walls + frozen movable blocks) */
    uint32_t frozen = wall_mask;

    /* Pattern 1: corners (geo_mask already 0 from valid_push_mask) */
    for (int i = 0; i < nb_mov; i++)
        if (geo_mask[i] == 0) frozen |= (1u << mbp[i]);

    /* Pattern 2: two adjacent movable blocks on the same edge */
    for (int i = 0; i < nb_mov; i++) {
        if (geo_mask[i] == 0) continue;
        int r = mbp[i] / COLS, c = mbp[i] % COLS;
        if ((r == 0 || r == ROWS-1) &&
            ((c > 0      && (mob_occ & (1u << pos(r, c-1)))) ||
             (c < COLS-1 && (mob_occ & (1u << pos(r, c+1)))))) {
            geo_mask[i] = 0; frozen |= (1u << mbp[i]); continue;
        }
        if ((c == 0 || c == COLS-1) &&
            ((r > 0      && (mob_occ & (1u << pos(r-1, c)))) ||
             (r < ROWS-1 && (mob_occ & (1u << pos(r+1, c)))))) {
            geo_mask[i] = 0; frozen |= (1u << mbp[i]);
        }
    }

    /* Pattern 3: 2x2 squares of movable blocks */
    for (int r = 0; r < ROWS-1; r++) {
        for (int c = 0; c < COLS-1; c++) {
            if (!((mob_occ >> pos(r,   c  )) & 1)) continue;
            if (!((mob_occ >> pos(r,   c+1)) & 1)) continue;
            if (!((mob_occ >> pos(r+1, c  )) & 1)) continue;
            if (!((mob_occ >> pos(r+1, c+1)) & 1)) continue;
            int ps[4] = { pos(r,c), pos(r,c+1), pos(r+1,c), pos(r+1,c+1) };
            for (int i = 0; i < nb_mov; i++)
                for (int k = 0; k < 4; k++)
                    if (mbp[i] == ps[k])
                        { geo_mask[i] = 0; frozen |= (1u << mbp[i]); }
        }
    }

    /* Propagate: a block is frozen only when ALL its remaining directions are
     * blocked by frozen neighbours.  Partial restrictions are not applied —
     * the pushable mask is global, and a direction blocked from the initial
     * position may become reachable once the block moves elsewhere. */
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < nb_mov; i++) {
            if (geo_mask[i] == 0) continue;
            int p = mbp[i];
            uint8_t rem = geo_mask[i];
            for (int d = 0; d < 4; d++) {
                if (!(rem & (1u << d))) continue;
                int lnd = adj[p][d], pfr = adj[p][d^2];
                if ((lnd >= 0 && (frozen & (1u << lnd))) ||
                    (pfr >= 0 && (frozen & (1u << pfr))))
                    rem &= ~(1u << d);
            }
            if (rem == 0) { geo_mask[i] = 0; frozen |= (1u << p); changed = 1; }
        }
    }
}

/*
 * Multi-start try_bitmasks.  Instead of one player_start, evaluates all
 * starts in a walkable component at each bitmask leaf.  Per-mask walk-
 * distance bounds let most starts be skipped after one BFS call.
 *
 * ps_cells[0..n_ps-1]: cell indices of player starts needing bitmask search
 * ps_vis[0..n_ps-1]:   corresponding vi indices into vwalk[]
 * vwalk[vi][cell]:      walk distance from valid start vi to cell
 * ks_arr[0..n_ps-1]:   per-start KnownSolvable caches
 */
static int try_bitmasks_ms(Puzzle *pz, int bi, uint8_t *unsolv, int *pn,
                           GlobalAntichain *ga, const uint8_t *geo_mask,
                           const int *ps_cells, const int *ps_vis, int n_ps,
                           int8_t (*vwalk)[NCELLS], KnownSolvable *ks_arr) {
    if (bi == pz->num_blocks) {
        if (ga_dominated(ga, pz->block_pushable, pz->num_blocks)) return -1;
        int nb = pz->num_blocks;

        uint32_t cur_packed = 0;
        for (int i = 0; i < nb; i++)
            cur_packed |= ((uint32_t)pz->block_pushable[i] << (i*4));

        int leaf_best = -1;
        int leaf_d[NCELLS];
        int had_skip = 0;

        for (int si = 0; si < n_ps; si++) {
            int ps = ps_cells[si];

            /* Per-mask walk-distance bound: if any earlier start sj got d,
             * from ps the result is at most d + walk(sj→ps).  If that
             * cannot beat g_best, skip this start for this mask. */
            int skip = 0;
            for (int sj = 0; sj < si && !skip; sj++) {
                if (leaf_d[sj] < 0) continue;
                int8_t w = vwalk[ps_vis[sj]][ps];
                if (w < 0) continue;
                if (leaf_d[sj] + (int)w <= g_best) skip = 1;
            }
            if (skip) { leaf_d[si] = -3; continue; }

            /* KS lookup for this start */
            KnownSolvable *ks = &ks_arr[si];
            int ks_hit = 0;
            for (int k = 0; k < ks->count; k++) {
                if ((cur_packed & ks->packed[k]) == ks->packed[k]) {
                    int d = ks->dist[k];
                    if (d > g_best) { pz->player_start = ps; update_best(d, pz); }
                    leaf_d[si] = d;
                    if (d > leaf_best) leaf_best = d;
                    ks_hit = 1;
                    break;
                }
            }
            if (ks_hit) continue;

            /* BFS call */
            pz->player_start = ps;
            tl_ncalls++;
            uint8_t used[MAX_BLOCKS] = {0};
            int d;
            if (g_profile_mode) {
                struct timespec _t0, _t1;
                BfsProfile bprof = {0};
                clock_gettime(CLOCK_MONOTONIC, &_t0);
                d = sokoban_solve(pz, used, &bprof);
                clock_gettime(CLOCK_MONOTONIC, &_t1);
                long long ns = (_t1.tv_sec - _t0.tv_sec) * 1000000000LL
                             + (_t1.tv_nsec - _t0.tv_nsec);
                if (ns < 1) ns = 1;
                int sol = (d >= 0) ? 1 : 0;
                int tb = 63 - __builtin_clzll((unsigned long long)ns);
                if (tb < 0) tb = 0;
                if (tb >= PROF_TBUCKETS) tb = PROF_TBUCKETS - 1;
                tl_prof.time_cnt[tb][sol]++;
                int hp = bprof.peak_heap_sz < 1 ? 1 : bprof.peak_heap_sz;
                int hb = 63 - __builtin_clzll((unsigned long long)hp);
                if (hb < 0) hb = 0;
                if (hb >= PROF_HBUCKETS) hb = PROF_HBUCKETS - 1;
                tl_prof.heap_cnt[hb][sol]++;
                tl_prof.ps_calls[ps]++;
            } else {
                d = sokoban_solve(pz, used, NULL);
            }
            if (d == -2) {
                record_skip(pz);
                leaf_d[si] = -2;
                had_skip = 1;
                continue;
            }
            leaf_d[si] = d;
            if (d > g_best) update_best(d, pz);
            if (d >= 0 && ks->count < MAX_KS) {
                uint32_t u_packed = 0;
                for (int i = 0; i < nb; i++) u_packed |= ((uint32_t)used[i] << (i*4));
                ks->packed[ks->count] = u_packed;
                ks->dist  [ks->count] = d;
                ks->count++;
            }
            if (d == -1) {
                /* Unsolvable from one start → unsolvable from all starts
                 * in the same walkable component (player can walk between). */
                ga_add(ga, pz->block_pushable, nb);
                return -1;
            }
            if (d > leaf_best) leaf_best = d;
        }
        return (leaf_best == -1 && had_skip) ? -2 : leaf_best;
    }

    int            nm   = cell_nmasks[pz->block_pos[bi]];
    const uint8_t *ms   = cell_masks [pz->block_pos[bi]];
    uint8_t        cap  = geo_mask[bi];   /* tighter ceiling from freeze analysis */
    int            best = -1;
    int            had_skip = 0;

    uint8_t xc_masks[16][16];
    int     xc_cnt[16];
    memset(xc_cnt, 0, sizeof xc_cnt);

    for (int mi = 0; mi < nm; mi++) {
        uint8_t m = ms[mi];
        if (m & ~cap) continue;

        int skip = 0;
        for (int j = 0; j < *pn && !skip; j++)
            if ((m & unsolv[j]) == m) skip = 1;
        if (skip) continue;

        uint8_t b_unsolv[MAX_UNSOLV]; int nbx = 0;
        int ns = superset_cnt[m];
        for (int si = 0; si < ns; si++) {
            int t = superset_list[m][si];
            for (int xi = 0; xi < xc_cnt[t]; xi++) {
                uint8_t x = xc_masks[t][xi];
                int dominated = 0;
                for (int k = 0; k < nbx && !dominated; k++)
                    if ((x & b_unsolv[k]) == x) dominated = 1;
                if (!dominated) b_unsolv[nbx++] = x;
            }
        }
        int nb_seed = nbx;

        pz->block_pushable[bi] = m;
        int d = try_bitmasks_ms(pz, bi + 1, b_unsolv, &nbx, ga, geo_mask,
                                ps_cells, ps_vis, n_ps, vwalk, ks_arr);

        for (int j = nb_seed; j < nbx; j++)
            if (xc_cnt[m] < 16) xc_masks[m][xc_cnt[m]++] = b_unsolv[j];

        if (d == -1) unsolv[(*pn)++] = m;
        if (d == -2) had_skip = 1;
        if (d > best) best = d;
    }
    return (best == -1 && had_skip) ? -2 : best;
}

/* -------------------------------------------------------------------------
 * Per-(exit, hole-configuration) work item
 *
 * Processes all block placements and player starts for one specific
 * (exit cell, hole positions) pair.  Called by worker threads.
 * ------------------------------------------------------------------------- */

/* Returns 1 if transform t maps the bitmask to itself (i.e. preserves the
 * set of cells in mask).  Used to check whether a symmetry is still valid
 * when fixed walls or holes are present. */
static int transform_fixes_mask(uint32_t mask, const int8_t *t) {
    uint32_t nm = 0;
    for (int c = 0; c < NCELLS; c++)
        if (mask & (1u << c)) nm |= (1u << t[c]);
    return nm == mask;
}

/* Returns 1 if transform t is compatible with the fixed walls and holes
 * (i.e. applying t would not change the fixed layout). */
static int transform_ok_fixed(const int8_t *t) {
    return (g_fixed_walls == 0        || transform_fixes_mask(g_fixed_walls,        t))
        && (g_fixed_nholes == 0       || transform_fixes_holes(g_fixed_hole_pos, g_fixed_nholes, t))
        && (g_fixed_empty_mask == 0   || transform_fixes_mask(g_fixed_empty_mask,   t))
        && (g_fixed_nblocks == 0      || transform_fixes_mask(g_fixed_blocks_mask,  t));
}

static void process_hole_config(int ei, int nw, int nh, const int *hp, int total) {
    int ep = EXIT_CELLS[ei];

    uint32_t holes_mask = g_fixed_holes_mask;
    for (int i = 0; i < nh; i++) holes_mask |= (1u << hp[i]);

    /* Pool for blocks: not exit, not a hole/wall/fixed-block/fixed-empty. */
    int bpool[NCELLS], nbpool = 0;
    for (int c = 0; c < NCELLS; c++)
        if (c != ep && !(holes_mask & (1u << c)) && !(g_fixed_walls & (1u << c))
                    && !(g_fixed_blocks_mask & (1u << c)) && !(g_fixed_empty_mask & (1u << c)))
            bpool[nbpool++] = c;

    if (total > nbpool) return;

    /* Block canonicality transforms: the subset of the exit's stabilizer
     * elements that also fix this hole configuration (as a set).
     * For nh=0 the full stabilizer applies; for nh>0 only those elements
     * σ with σ(holes) = holes (checked via transform_fixes_holes).
     * These are used inside the block loop to skip non-canonical combos,
     * reducing block enumeration work analogously to hole canonicality. */
    const int8_t *bt[7]; int nbt = 0;
    {
        /* All transforms that are in the exit stabilizer and fix hp */
        #define ADD_BT(t) if (transform_ok_fixed(t) && (nh == 0 || transform_fixes_holes(hp, nh, t))) bt[nbt++] = (t)
        switch (ei) {
        case 6: case 7: case 8: ADD_BT(t_flip_v); break; /* row 2 exits */
        default: break; /* trivial stabilizer */
        }
        #undef ADD_BT
    }

    /* --- Enumerate block placements --- */
    Comb bc;
    comb_init(&bc, nbpool, total);
    do {
        int bp[MAX_BLOCKS];
        for (int i = 0; i < total; i++) bp[i] = bpool[bc.idx[i]];

        /* Block canonicality: skip if any applicable transform maps bp
         * to a lex-smaller combo (that combo is or will be enumerated). */
        {
            int skip = 0;
            for (int si = 0; si < nbt && !skip; si++)
                if (!holes_lex_min_under(bp, total, bt[si])) skip = 1;
            if (skip) continue;
        }

        /* Occupied mask for player-start filtering.
         * Holes are included: the player cannot start on one.
         * All block positions (both wall-designated and movable) are included
         * so that occ correctly blocks the player in all wall subsets. */
        uint32_t occ = (1u << ep) | holes_mask | g_fixed_walls | g_fixed_blocks_mask;
        for (int i = 0; i < total; i++) occ |= (1u << bp[i]);

        /* Precompute walk distances from each valid player start to all
         * cells.  Used to bound the best reachable solution distance for
         * later player starts and skip them when they can't beat g_best.
         * walk_blocked excludes the exit (player can walk onto it).
         *
         * All non-occupied cells are tried as player starts; symmetry
         * reduction is applied to hole placements instead (see above). */
        uint32_t walk_blocked = occ & ~(1u << ep);
        int8_t  vwalk[NCELLS][NCELLS]; /* vwalk[vi][cell]: dist from valid start vi */
        int     vpi  [NCELLS];         /* vpi[vi]: cell index of valid start vi     */
        int     n_valid = 0;
        for (int cell = 0; cell < NCELLS; cell++) {
            if (occ & (1u << cell)) continue;
            vpi[n_valid] = cell;
            walk_all_distances(walk_blocked, cell, vwalk[n_valid]);
            n_valid++;
        }

        /* Free-walking component assignment.
         * comp_id is fixed per block combo (it depends only on occ, which
         * is the same for all wall subsets of a given block placement).
         * comp_ga is shared across player starts within a component but
         * reset for each wall subset: different wall subsets yield different
         * movable-block sets, so their antichains have different dimensions
         * and cannot be shared across subsets. */
        int comp_id[NCELLS];
        for (int vi = 0; vi < n_valid; vi++) {
            comp_id[vi] = vi;
            for (int vj = 0; vj < vi; vj++) {
                if (vwalk[vj][vpi[vi]] >= 0) {
                    comp_id[vi] = comp_id[vj];
                    break;
                }
            }
        }
        GlobalAntichain comp_ga[NCELLS];

        /* --- Enumerate wall subsets of size nw from the total block positions ---
         * For nw=0 comb_next returns 0 immediately after one iteration with an
         * empty index set, giving wall_mask=0 and mbp=bp — identical to the
         * previous (no-wall) behaviour. */
        Comb wsc; comb_init(&wsc, total, nw);
        do {
            /* Partition bp[] into wall cells and movable block cells. */
            int is_wall[MAX_BLOCKS] = {0};
            for (int j = 0; j < nw; j++) is_wall[wsc.idx[j]] = 1;
            uint32_t wall_mask = g_fixed_walls;
            int mbp[MAX_BLOCKS], nb_mov = 0;
            for (int i = 0; i < total; i++) {
                if (is_wall[i]) wall_mask |= (1u << bp[i]);
                else            mbp[nb_mov++] = bp[i];
            }
            for (int i = 0; i < g_fixed_nblocks; i++) mbp[nb_mov++] = g_fixed_block_pos[i];

            /* Tighter per-block push ceilings: corners + edge-pairs + 2x2 freeze. */
            uint8_t geo_mask[MAX_BLOCKS];
            compute_geo_masks(mbp, nb_mov, wall_mask, geo_mask);
            for (int i = 0; i < nb_mov; i++) geo_mask[i] &= g_fixed_mask[mbp[i]];

            /* Reset component antichains for this wall subset. */
            for (int vi = 0; vi < n_valid; vi++)
                if (comp_id[vi] == vi) comp_ga[vi].count = 0;

            /* Phase 1: all-immovable check for every valid start. */
            uint32_t imm_blocked = occ & ~(1u << ep);
            int imm_d[NCELLS];
            for (int vi = 0; vi < n_valid; vi++) {
                int ps = vpi[vi];
                tl_ncalls++;
                imm_d[vi] = fast_reachable(imm_blocked, ps, ep);
                if (imm_d[vi] > g_best) {
                    Puzzle pz; memset(&pz, 0, sizeof pz);
                    pz.exit_pos = ep; pz.player_start = ps; pz.walls = wall_mask;
                    pz.num_blocks = nb_mov;
                    pz.num_holes = g_fixed_nholes + nh;
                    for (int i = 0; i < g_fixed_nholes; i++) pz.hole_pos[i] = g_fixed_hole_pos[i];
                    for (int i = 0; i < nh; i++) pz.hole_pos[g_fixed_nholes + i] = hp[i];
                    for (int i = 0; i < nb_mov; i++) pz.block_pos[i] = mbp[i];
                    for (int i = 0; i < nb_mov; i++) pz.block_pushable[i] = 0;
                    update_best(imm_d[vi], &pz);
                }
            }

            /* Phase 2: per-component bitmask search. */
            for (int ci = 0; ci < n_valid; ci++) {
                if (comp_id[ci] != ci) continue;

                int ps_cells[NCELLS], ps_vis[NCELLS], n_ps = 0;
                for (int vj = 0; vj < n_valid; vj++) {
                    if (comp_id[vj] != ci) continue;
                    if (imm_d[vj] >= 0) continue;
                    int bounded = 0;
                    for (int vk = 0; vk < n_valid && !bounded; vk++) {
                        if (comp_id[vk] != ci || imm_d[vk] < 0) continue;
                        int8_t w = vwalk[vk][vpi[vj]];
                        if (w >= 0 && imm_d[vk] + (int)w <= g_best) bounded = 1;
                    }
                    if (bounded) continue;
                    ps_cells[n_ps] = vpi[vj];
                    ps_vis[n_ps]   = vj;
                    n_ps++;
                }
                if (n_ps == 0) continue;

                Puzzle pz; memset(&pz, 0, sizeof pz);
                pz.exit_pos   = ep;
                pz.walls      = wall_mask;
                pz.num_blocks = nb_mov;
                pz.num_holes  = g_fixed_nholes + nh;
                for (int i = 0; i < g_fixed_nholes; i++) pz.hole_pos[i] = g_fixed_hole_pos[i];
                for (int i = 0; i < nh; i++) pz.hole_pos[g_fixed_nholes + i] = hp[i];
                for (int i = 0; i < nb_mov; i++) pz.block_pos[i] = mbp[i];

                KnownSolvable ks_arr[NCELLS];
                for (int s = 0; s < n_ps; s++) ks_arr[s].count = 0;

                uint8_t unsolv0[MAX_UNSOLV]; int n0 = 0;
                struct timespec _ps_t0, _ps_t1;
                if (g_profile_mode) clock_gettime(CLOCK_MONOTONIC, &_ps_t0);

                try_bitmasks_ms(&pz, 0, unsolv0, &n0, &comp_ga[ci], geo_mask,
                                ps_cells, ps_vis, n_ps, vwalk, ks_arr);

                if (g_profile_mode) {
                    clock_gettime(CLOCK_MONOTONIC, &_ps_t1);
                    double dt = (_ps_t1.tv_sec - _ps_t0.tv_sec)
                              + (_ps_t1.tv_nsec - _ps_t0.tv_nsec) * 1e-9;
                    for (int s = 0; s < n_ps; s++)
                        tl_prof.ps_time[ps_cells[s]] += dt / n_ps;
                }
            }
        } while (comb_next(&wsc));
    } while (comb_next(&bc));
}

/* -------------------------------------------------------------------------
 * Timing helpers
 * ------------------------------------------------------------------------- */

static double elapsed_s(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

static void print_time_calls(double elapsed, long long ncalls) {
    printf("Time:               %.3f s\n", elapsed);
    printf("Total solver calls: %lld\n",   ncalls);
}

static void print_profile(const ProfData *p) {
    /* Merge is already done by caller; just print. */
    printf("\n=== BFS Profile ===\n");

    /* Time distribution */
    printf("\nCall time distribution:\n");
    printf("  %-18s  %12s  %12s\n", "Range", "Solvable", "Unsolvable");
    for (int b = 0; b < PROF_TBUCKETS; b++) {
        long long s = p->time_cnt[b][1], u = p->time_cnt[b][0];
        if (s + u == 0) continue;
        long long lo_ns = 1LL << b, hi_ns = 1LL << (b + 1);
        char lo_buf[24], hi_buf[24];
        /* Format in most readable unit */
        #define FMT_NS(buf, ns) do { \
            if ((ns) >= 1000000000LL) snprintf(buf, sizeof(buf), "%.3g s",  (ns)/1e9); \
            else if ((ns) >= 1000000LL) snprintf(buf, sizeof(buf), "%.3g ms", (ns)/1e6); \
            else if ((ns) >= 1000LL)    snprintf(buf, sizeof(buf), "%.3g µs", (ns)/1e3); \
            else                        snprintf(buf, sizeof(buf), "%lld ns", (long long)(ns)); \
        } while(0)
        FMT_NS(lo_buf, lo_ns);
        FMT_NS(hi_buf, hi_ns);
        #undef FMT_NS
        char range[40];
        snprintf(range, sizeof(range), "%s – %s", lo_buf, hi_buf);
        printf("  %-18s  %12lld  %12lld\n", range, s, u);
    }

    /* Heap distribution */
    printf("\nPeak heap distribution (cap = %d entries = 2^20):\n", 1 << 20);
    printf("  %-18s  %12s  %12s  %8s\n", "Range", "Solvable", "Unsolvable", "% of cap");
    for (int b = 0; b < PROF_HBUCKETS; b++) {
        long long s = p->heap_cnt[b][1], u = p->heap_cnt[b][0];
        if (s + u == 0) continue;
        int lo = 1 << b, hi = (b + 1 < PROF_HBUCKETS) ? (1 << (b + 1)) - 1 : (1 << b);
        double pct_lo = 100.0 * lo / (1 << 20);
        double pct_hi = 100.0 * hi / (1 << 20);
        char range[40];
        if (b == PROF_HBUCKETS - 1)
            snprintf(range, sizeof(range), "= %d (cap)", 1 << 20);
        else
            snprintf(range, sizeof(range), "%d – %d", lo, hi);
        char pct_buf[16];
        snprintf(pct_buf, sizeof(pct_buf), "%.2f–%.2f%%", pct_lo, pct_hi);
        printf("  %-18s  %12lld  %12lld  %8s\n", range, s, u, pct_buf);
    }

    /* Per-player-start breakdown */
    printf("\nTime per player start (sorted by time):\n");
    printf("  %4s  %4s  %4s  %12s  %12s\n", "cell", "row", "col", "time (s)", "BFS calls");
    /* collect non-zero entries and sort by time descending */
    int order[NCELLS];
    int n_ps = 0;
    for (int c = 0; c < NCELLS; c++)
        if (p->ps_calls[c] > 0) order[n_ps++] = c;
    /* insertion sort descending by time */
    for (int i = 1; i < n_ps; i++) {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && p->ps_time[order[j]] < p->ps_time[key]) {
            order[j + 1] = order[j]; j--;
        }
        order[j + 1] = key;
    }
    for (int i = 0; i < n_ps; i++) {
        int c = order[i];
        printf("  %4d  %4d  %4d  %12.3f  %12lld\n",
               c, c / COLS, c % COLS, p->ps_time[c], p->ps_calls[c]);
    }
}

/* -------------------------------------------------------------------------
 * Work queue and thread pool
 *
 * Each item is one canonical (exit, hole-configuration) pair.  The queue
 * is rebuilt for each nh level so that all nh=k items complete before any
 * nh=k+1 items begin — this keeps g_best as high as possible before the
 * expensive large-nh work starts, maximising walk-distance pruning.
 * ------------------------------------------------------------------------- */

typedef struct { int ei, nh; int hp[MAX_HOLES]; int bp[MAX_BLOCKS]; } WorkItem;

static WorkItem       *g_wq       = NULL;
static int             g_wq_cap   = 0;
static int             g_wq_count = 0;
static int             g_wq_next  = 0;
static pthread_mutex_t g_wq_mutex = PTHREAD_MUTEX_INITIALIZER;

static _Atomic int     g_items_done    = 0;
static int             g_items_total   = 0;
static _Atomic int     g_progress_stop = 0;

typedef struct {
    int        total;
    int        nw;
    long long  ncalls;
    ProfData   prof;
} ThreadArg;

/* -------------------------------------------------------------------------
 * Per-(exit, hole-config, block-combo) work item — nh=0 and nh=1.
 *
 * For nh=0: equivalent to the inner block-combo iteration of
 * process_hole_config with holes_mask=0, giving C(pool,total)/symmetry
 * items per exit instead of just 1.
 * For nh=1: same idea but with a fixed hole baked into the item, giving
 * C(pool,total)/symmetry items per (exit, hole) pair, which distributes
 * hard hole placements (near the exit) across all threads evenly.
 * ------------------------------------------------------------------------- */
static void process_block_combo(int ei, int nw, int nh, const int *hp,
                                const int *bp, int total) {
    int ep = EXIT_CELLS[ei];

    uint32_t holes_mask = g_fixed_holes_mask;
    for (int i = 0; i < nh; i++) holes_mask |= (1u << hp[i]);

    uint32_t occ = (1u << ep) | holes_mask | g_fixed_walls | g_fixed_blocks_mask;
    for (int i = 0; i < total; i++) occ |= (1u << bp[i]);

    uint32_t walk_blocked = occ & ~(1u << ep);
    int8_t  vwalk[NCELLS][NCELLS];
    int     vpi  [NCELLS];
    int     n_valid = 0;
    for (int cell = 0; cell < NCELLS; cell++) {
        if (occ & (1u << cell)) continue;
        vpi[n_valid] = cell;
        walk_all_distances(walk_blocked, cell, vwalk[n_valid]);
        n_valid++;
    }

    int comp_id[NCELLS];
    for (int vi = 0; vi < n_valid; vi++) {
        comp_id[vi] = vi;
        for (int vj = 0; vj < vi; vj++) {
            if (vwalk[vj][vpi[vi]] >= 0) { comp_id[vi] = comp_id[vj]; break; }
        }
    }
    GlobalAntichain comp_ga[NCELLS];

    Comb wsc; comb_init(&wsc, total, nw);
    do {
        int is_wall[MAX_BLOCKS] = {0};
        for (int j = 0; j < nw; j++) is_wall[wsc.idx[j]] = 1;
        uint32_t wall_mask = g_fixed_walls;
        int mbp[MAX_BLOCKS], nb_mov = 0;
        for (int i = 0; i < total; i++) {
            if (is_wall[i]) wall_mask |= (1u << bp[i]);
            else            mbp[nb_mov++] = bp[i];
        }
        for (int i = 0; i < g_fixed_nblocks; i++) mbp[nb_mov++] = g_fixed_block_pos[i];

        uint8_t geo_mask[MAX_BLOCKS];
        compute_geo_masks(mbp, nb_mov, wall_mask, geo_mask);
        for (int i = 0; i < nb_mov; i++) geo_mask[i] &= g_fixed_mask[mbp[i]];

        for (int vi = 0; vi < n_valid; vi++)
            if (comp_id[vi] == vi) comp_ga[vi].count = 0;

        /* Phase 1: all-immovable check for every valid start. */
        uint32_t imm_blocked = occ & ~(1u << ep);
        int imm_d[NCELLS];
        for (int vi = 0; vi < n_valid; vi++) {
            int ps = vpi[vi];
            tl_ncalls++;
            imm_d[vi] = fast_reachable(imm_blocked, ps, ep);
            if (imm_d[vi] > g_best) {
                Puzzle pz; memset(&pz, 0, sizeof pz);
                pz.exit_pos = ep; pz.player_start = ps; pz.walls = wall_mask;
                pz.num_blocks = nb_mov;
                pz.num_holes = g_fixed_nholes + nh;
                for (int i = 0; i < g_fixed_nholes; i++) pz.hole_pos[i] = g_fixed_hole_pos[i];
                for (int i = 0; i < nh; i++) pz.hole_pos[g_fixed_nholes + i] = hp[i];
                for (int i = 0; i < nb_mov; i++) pz.block_pos[i] = mbp[i];
                for (int i = 0; i < nb_mov; i++) pz.block_pushable[i] = 0;
                update_best(imm_d[vi], &pz);
            }
        }

        /* Phase 2: per-component bitmask search.
         * Collect starts where fast_reachable failed, bounded away by neither
         * the immovable result nor the walk-distance bound from other starts. */
        for (int ci = 0; ci < n_valid; ci++) {
            if (comp_id[ci] != ci) continue; /* not a component root */

            int ps_cells[NCELLS], ps_vis[NCELLS], n_ps = 0;
            for (int vj = 0; vj < n_valid; vj++) {
                if (comp_id[vj] != ci) continue;
                if (imm_d[vj] >= 0) continue; /* fast_reachable solved it */
                /* Walk-distance bound from immovable results */
                int bounded = 0;
                for (int vk = 0; vk < n_valid && !bounded; vk++) {
                    if (comp_id[vk] != ci || imm_d[vk] < 0) continue;
                    int8_t w = vwalk[vk][vpi[vj]];
                    if (w >= 0 && imm_d[vk] + (int)w <= g_best) bounded = 1;
                }
                if (bounded) continue;
                ps_cells[n_ps] = vpi[vj];
                ps_vis[n_ps]   = vj;
                n_ps++;
            }
            if (n_ps == 0) continue;

            Puzzle pz; memset(&pz, 0, sizeof pz);
            pz.exit_pos   = ep;
            pz.walls      = wall_mask;
            pz.num_blocks = nb_mov;
            pz.num_holes  = g_fixed_nholes + nh;
            for (int i = 0; i < g_fixed_nholes; i++) pz.hole_pos[i] = g_fixed_hole_pos[i];
            for (int i = 0; i < nh; i++) pz.hole_pos[g_fixed_nholes + i] = hp[i];
            for (int i = 0; i < nb_mov; i++) pz.block_pos[i] = mbp[i];

            KnownSolvable ks_arr[NCELLS];
            for (int s = 0; s < n_ps; s++) ks_arr[s].count = 0;

            uint8_t unsolv0[MAX_UNSOLV]; int n0 = 0;
            long long ncalls_before = tl_ncalls;
            struct timespec _ps_t0, _ps_t1;
            if (g_profile_mode) clock_gettime(CLOCK_MONOTONIC, &_ps_t0);

            try_bitmasks_ms(&pz, 0, unsolv0, &n0, &comp_ga[ci], geo_mask,
                            ps_cells, ps_vis, n_ps, vwalk, ks_arr);

            if (g_profile_mode) {
                clock_gettime(CLOCK_MONOTONIC, &_ps_t1);
                double dt = (_ps_t1.tv_sec - _ps_t0.tv_sec)
                          + (_ps_t1.tv_nsec - _ps_t0.tv_nsec) * 1e-9;
                for (int s = 0; s < n_ps; s++)
                    tl_prof.ps_time[ps_cells[s]] += dt / n_ps;
            }
        }
    } while (comb_next(&wsc));
}

static void *worker_thread(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;

    while (1) {
        pthread_mutex_lock(&g_wq_mutex);
        if (g_wq_next >= g_wq_count) {
            pthread_mutex_unlock(&g_wq_mutex);
            break;
        }
        WorkItem item = g_wq[g_wq_next++];
        pthread_mutex_unlock(&g_wq_mutex);

        if (item.nh <= 1)
            process_block_combo(item.ei, a->nw, item.nh, item.hp, item.bp, a->total);
        else
            process_hole_config(item.ei, a->nw, item.nh, item.hp, a->total);
        atomic_fetch_add(&g_items_done, 1);
    }

    a->ncalls = tl_ncalls;
    if (g_profile_mode) a->prof = tl_prof;
    return NULL;
}

/* -------------------------------------------------------------------------
 * Progress bar thread: wakes every 10 s, prints % + ETA to stderr,
 * overwriting the previous line with \r.
 * ------------------------------------------------------------------------- */
typedef struct { struct timespec t0; int nb; int nh; int nh_hi; } ProgressArg;

static void fmt_time(char *buf, int sz, int secs) {
    if (secs < 3600) snprintf(buf, sz, "%d:%02d",   secs / 60,   secs % 60);
    else             snprintf(buf, sz, "%dh%02dm", secs / 3600, (secs % 3600) / 60);
}

static void *progress_thread_func(void *arg) {
    ProgressArg *pa = (ProgressArg *)arg;
    int ticks = 0;
    while (!atomic_load(&g_progress_stop)) {
        struct timespec req = {1, 0};
        nanosleep(&req, NULL);
        if (++ticks % 10 != 0) continue;

        int done  = atomic_load(&g_items_done);
        int total = g_items_total;

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = elapsed_s(pa->t0, now);

        double pct = total > 0 ? 100.0 * done / total : 0.0;

        char elap_buf[32], eta_buf[32];
        fmt_time(elap_buf, sizeof(elap_buf), (int)elapsed);
        if (done > 0 && done < total)
            fmt_time(eta_buf, sizeof(eta_buf), (int)(elapsed / done * (total - done)));
        else
            snprintf(eta_buf, sizeof(eta_buf), done >= total ? "done" : "---");

        snprintf(g_last_progress, sizeof(g_last_progress),
                 "\r  [blocks=%d  holes=%d/%d]  %d/%d  %.1f%%  elapsed %s  ETA %s       ",
                 pa->nb, pa->nh, pa->nh_hi, done, total, pct, elap_buf, eta_buf);
        pthread_mutex_lock(&g_print_mutex);
        fprintf(stderr, "%s", g_last_progress);
        fflush(stderr);
        pthread_mutex_unlock(&g_print_mutex);
    }
    /* Clear the progress line when done (only if we printed one) */
    pthread_mutex_lock(&g_print_mutex);
    if (g_last_progress[0]) {
        g_last_progress[0] = '\0';
        fprintf(stderr, "\r%80s\r", "");
        fflush(stderr);
    }
    pthread_mutex_unlock(&g_print_mutex);
    return NULL;
}

/* -------------------------------------------------------------------------
 * Main search
 * ------------------------------------------------------------------------- */

/* nw:      number of block positions designated as walls per block combo.
 * only_nh: restrict to a single hole count (-1 = all counts).
 * only_ei: restrict to a single exit index into EXIT_CELLS (-1 = all exits). */
static void puzzle_search(int total, int nw, int nh_lo_arg, int nh_hi_arg, int only_ei, int print_summary) {
    if (total < 0 || total > MAX_BLOCKS) {
        fprintf(stderr, "num_blocks must be between 0 and %d\n", MAX_BLOCKS);
        return;
    }
    precompute_tables();
    sokoban_init();
    printf("Searching: blocks = %d,  walls = %d,  grid = %d×%d,  threads = %d",
           total - nw, nw, ROWS, COLS, g_num_threads);
    if (g_fixed_walls) {
        printf("  fixed-walls =");
        for (int c = 0; c < NCELLS; c++) if (g_fixed_walls & (1u<<c)) printf(" %d", c);
    }
    if (g_fixed_nholes) {
        printf("  fixed-holes =");
        for (int i = 0; i < g_fixed_nholes; i++) printf(" %d", g_fixed_hole_pos[i]);
    }
    printf("\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_t threads[g_num_threads];
    ThreadArg args[g_num_threads];
    long long total_ncalls = 0;

    /* Process hole counts one level at a time.  Building the queue per-level
     * ensures g_best is as high as possible before large nh items begin,
     * maximising the walk-distance pruning (Approach C) on those items. */
    int fixed_occupied = g_fixed_nblocks + g_fixed_nholes
                       + __builtin_popcount(g_fixed_walls)
                       + __builtin_popcount(g_fixed_empty_mask);
    int grid_cap = NCELLS - 1 - total - fixed_occupied;
    int max_nh = (nh_hi_arg >= 0) ? grid_cap
                                   : (total - nw + g_fixed_nblocks < grid_cap
                                      ? total - nw + g_fixed_nblocks : grid_cap);
    int nh_lo = (nh_lo_arg >= 0) ? nh_lo_arg : 0;
    int nh_hi = (nh_hi_arg >= 0) ? (nh_hi_arg < max_nh ? nh_hi_arg : max_nh) : max_nh;
    int ei_lo = (only_ei >= 0) ? only_ei : 0;
    int ei_hi = (only_ei >= 0) ? only_ei : NUM_EXIT_CELLS - 1;

    for (int nh = nh_lo; nh <= nh_hi && nh <= MAX_HOLES; nh++) {

        /* Build work queue: one item per canonical (exit, hole-config) pair.
         * Hole canonicality checks (stabilizer of each exit cell) are applied
         * here so workers receive only non-redundant configurations. */
        g_wq_count = 0;
        g_wq_next  = 0;

        if (nh == 0) {
            /* Fine-grained parallelism: one work item per canonical block combo.
             * With nh=0 the old (exit, hole-config) granularity yields at most 6
             * items (one per exit), starving threads when --exitloc is set.
             * Enumerating block combos here gives C(pool,total)/~symmetry items,
             * keeping all threads busy regardless of --exitloc or --nholes 0. */
            for (int ei = ei_lo; ei <= ei_hi; ei++) {
                int ep = EXIT_CELLS[ei];

                int bpool[NCELLS], nbpool = 0;
                for (int c = 0; c < NCELLS; c++)
                    if (c != ep && !(g_fixed_walls & (1u<<c)) && !(g_fixed_holes_mask & (1u<<c))
                                && !(g_fixed_blocks_mask & (1u<<c)) && !(g_fixed_empty_mask & (1u<<c)))
                        bpool[nbpool++] = c;
                if (total > nbpool) continue;

                /* Block canonicality transforms: gate on fixed-layout compatibility. */
                const int8_t *bt[7]; int nbt = 0;
                #define ADD_BT0(t) if (transform_ok_fixed(t)) bt[nbt++] = (t)
                switch (ei) {
                case 6: case 7: case 8: ADD_BT0(t_flip_v); break;
                default: break;
                }
                #undef ADD_BT0

                Comb bc; comb_init(&bc, nbpool, total);
                do {
                    int bp[MAX_BLOCKS];
                    for (int i = 0; i < total; i++) bp[i] = bpool[bc.idx[i]];

                    int skip = 0;
                    for (int si = 0; si < nbt && !skip; si++)
                        if (!holes_lex_min_under(bp, total, bt[si])) skip = 1;
                    if (skip) continue;

                    if (g_wq_count >= g_wq_cap) {
                        g_wq_cap = g_wq_cap ? g_wq_cap * 2 : 65536;
                        g_wq = realloc(g_wq, (size_t)g_wq_cap * sizeof(WorkItem));
                        if (!g_wq) { perror("realloc g_wq"); exit(1); }
                    }
                    g_wq[g_wq_count].ei = ei;
                    g_wq[g_wq_count].nh = 0;
                    memcpy(g_wq[g_wq_count].bp, bp, total * sizeof(int));
                    g_wq_count++;
                } while (comb_next(&bc));
            }
        } else if (nh == 1) {
            /* Fine-grained path for nh=1: one item per canonical (exit, hole, block-combo).
             * Like the nh=0 path, this distributes work at block-combo granularity so that
             * expensive hole placements (near the exit) don't bottleneck a single thread. */
            for (int ei = ei_lo; ei <= ei_hi; ei++) {
                int ep = EXIT_CELLS[ei];

                int hpool[NCELLS], nhpool = 0;
                for (int c = 0; c < NCELLS; c++)
                    if (c != ep && !(g_fixed_walls & (1u<<c)) && !(g_fixed_holes_mask & (1u<<c))
                                && !(g_fixed_blocks_mask & (1u<<c)) && !(g_fixed_empty_mask & (1u<<c)))
                        hpool[nhpool++] = c;

                Comb hc; comb_init(&hc, nhpool, 1);
                do {
                    int hp[1]; hp[0] = hpool[hc.idx[0]];

                    /* Hole canonicality: only apply transforms compatible with fixed layout. */
                    switch (ei) {
                    case 6: case 7: case 8:
                        if (transform_ok_fixed(t_flip_v) && !holes_lex_min_under(hp, 1, t_flip_v)) continue;
                        break;
                    default: break;
                    }

                    /* Block pool: not exit, not the hole, not fixed walls/holes/blocks/empty. */
                    int bpool[NCELLS], nbpool = 0;
                    for (int c = 0; c < NCELLS; c++)
                        if (c != ep && c != hp[0] && !(g_fixed_walls & (1u<<c)) && !(g_fixed_holes_mask & (1u<<c))
                                                  && !(g_fixed_blocks_mask & (1u<<c)) && !(g_fixed_empty_mask & (1u<<c)))
                            bpool[nbpool++] = c;
                    if (total > nbpool) continue;

                    /* Filtered stabilizer for block canonicality: exit stabilizer
                     * elements that also fix this hole position. */
                    const int8_t *bt[7]; int nbt = 0;
                    #define ADD_BT(t) if (transform_ok_fixed(t) && transform_fixes_holes(hp, 1, t)) bt[nbt++] = (t)
                    switch (ei) {
                    case 6: case 7: case 8: ADD_BT(t_flip_v); break;
                    default: break;
                    }
                    #undef ADD_BT

                    /* Enumerate canonical block combos for this (exit, hole) pair. */
                    Comb bc; comb_init(&bc, nbpool, total);
                    do {
                        int bp[MAX_BLOCKS];
                        for (int i = 0; i < total; i++) bp[i] = bpool[bc.idx[i]];

                        int skip = 0;
                        for (int si = 0; si < nbt && !skip; si++)
                            if (!holes_lex_min_under(bp, total, bt[si])) skip = 1;
                        if (skip) continue;

                        if (g_wq_count >= g_wq_cap) {
                            g_wq_cap = g_wq_cap ? g_wq_cap * 2 : 65536;
                            g_wq = realloc(g_wq, (size_t)g_wq_cap * sizeof(WorkItem));
                            if (!g_wq) { perror("realloc g_wq"); exit(1); }
                        }
                        g_wq[g_wq_count].ei    = ei;
                        g_wq[g_wq_count].nh    = 1;
                        g_wq[g_wq_count].hp[0] = hp[0];
                        memcpy(g_wq[g_wq_count].bp, bp, total * sizeof(int));
                        g_wq_count++;
                    } while (comb_next(&bc));
                } while (comb_next(&hc));
            }
        } else {
            /* Coarse-grained path (nh>1): one item per canonical (exit, hole-config). */
            for (int ei = ei_lo; ei <= ei_hi; ei++) {
                int ep = EXIT_CELLS[ei];

                int hpool[NCELLS], nhpool = 0;
                for (int c = 0; c < NCELLS; c++)
                    if (c != ep && !(g_fixed_walls & (1u<<c)) && !(g_fixed_holes_mask & (1u<<c))
                                && !(g_fixed_blocks_mask & (1u<<c)) && !(g_fixed_empty_mask & (1u<<c)))
                        hpool[nhpool++] = c;
                if (nh > nhpool) continue;

                Comb hc;
                comb_init(&hc, nhpool, nh);
                do {
                    int hp[MAX_HOLES];
                    for (int i = 0; i < nh; i++) hp[i] = hpool[hc.idx[i]];

                    switch (ei) {
                    case 6: case 7: case 8:
                        if (transform_ok_fixed(t_flip_v) && !holes_lex_min_under(hp, nh, t_flip_v)) continue;
                        break;
                    default: break;
                    }

                    if (g_wq_count >= g_wq_cap) {
                        g_wq_cap = g_wq_cap ? g_wq_cap * 2 : 65536;
                        g_wq = realloc(g_wq, (size_t)g_wq_cap * sizeof(WorkItem));
                        if (!g_wq) { perror("realloc g_wq"); exit(1); }
                    }
                    g_wq[g_wq_count].ei = ei;
                    g_wq[g_wq_count].nh = nh;
                    memcpy(g_wq[g_wq_count].hp, hp, nh * sizeof(int));
                    g_wq_count++;
                } while (comb_next(&hc));
            }
        }

        atomic_store(&g_items_done, 0);
        atomic_store(&g_progress_stop, 0);
        g_items_total = g_wq_count;

        pthread_t prog_thread;
        struct timespec t_level;
        clock_gettime(CLOCK_MONOTONIC, &t_level);
        ProgressArg prog_arg = { .t0 = t_level, .nb = total - nw, .nh = nh, .nh_hi = nh_hi };
        pthread_create(&prog_thread, NULL, progress_thread_func, &prog_arg);

        for (int t = 0; t < g_num_threads; t++) {
            args[t].total  = total;
            args[t].nw     = nw;
            args[t].ncalls = 0;
            memset(&args[t].prof, 0, sizeof args[t].prof);
            pthread_create(&threads[t], NULL, worker_thread, &args[t]);
        }
        for (int t = 0; t < g_num_threads; t++) {
            pthread_join(threads[t], NULL);
            total_ncalls += args[t].ncalls;
        }

        atomic_store(&g_progress_stop, 1);
        pthread_join(prog_thread, NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    if (print_summary) {
        printf("\n=== Search complete ===\n");
        print_time_calls(elapsed_s(t0, t1), total_ncalls);
        if (g_skipped > 0) {
            printf("Skipped: %d puzzles (heap overflow — state space too large)\n", g_skipped);
            printf("Example skipped puzzle:\n");
            print_puzzle(&g_skipped_pz);
        }
        if (g_best >= 0) {
            printf("Best solution: %d moves\n", g_best);
            print_puzzle(&g_best_pz);
        } else {
            printf("No solvable puzzles found.\n");
        }
        if (g_profile_mode) {
            ProfData merged = {0};
            for (int t = 0; t < g_num_threads; t++)
                for (int b = 0; b < PROF_TBUCKETS; b++) {
                    merged.time_cnt[b][0] += args[t].prof.time_cnt[b][0];
                    merged.time_cnt[b][1] += args[t].prof.time_cnt[b][1];
                }
            for (int t = 0; t < g_num_threads; t++)
                for (int b = 0; b < PROF_HBUCKETS; b++) {
                    merged.heap_cnt[b][0] += args[t].prof.heap_cnt[b][0];
                    merged.heap_cnt[b][1] += args[t].prof.heap_cnt[b][1];
                }
            for (int t = 0; t < g_num_threads; t++)
                for (int c = 0; c < NCELLS; c++) {
                    merged.ps_time[c]  += args[t].prof.ps_time[c];
                    merged.ps_calls[c] += args[t].prof.ps_calls[c];
                }
            print_profile(&merged);
        }
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "       %s              (no arguments: interactive parameter entry)\n"
        "\n"
        "Options:\n"
        "  --nblocks    <n|lo-hi>      movable blocks (1-%d); omit to search all counts\n"
        "  --nwalls     <n|lo-hi>      additional random wall blocks (default: 0)\n"
        "  --nholes     <n|lo-hi>      additional random holes; omit to search all counts\n"
        "                               (default range: 0 to min(nblocks, 23-nblocks-nwalls))\n"
        "  --fixedwalls  <c,c,...>     cells that are always walls (e.g. 5,10,15)\n"
        "  --fixedholes  <c,c,...>     cells that are always holes (e.g. 3,8)\n"
        "  --fixedblocks <c,c,...>     cells that always have a movable block\n"
        "  --fixedempty  <c,c,...>     cells that are always empty (no block/wall/hole)\n"
        "  --fixedmask   <c=m,...>     per-cell pushability ceiling, ANDed into geo_mask\n"
        "                             (e.g. 5=3 limits cell 5 to UD only; 12=0 forces immovable)\n"
        "  --exitloc    <cell>         restrict to one exit cell in {0,1,2,6,7,12} (default: all)\n"
        "  --nthreads   <n>            number of worker threads (default: %d)\n"
        "  --profile                   print BFS call-time and peak-heap distributions\n"
        "  --help, -h                  show this help message\n"
        "\n"
        "Exit cell layout (cell numbers on 5x6 grid, row-major):\n"
        "  0  1  2  3  4  5\n"
        "  6  7  8  9 10 11\n"
        " 12 13 14 15 16 17\n"
        " 18 19 20 21 22 23\n"
        " 24 25 26 27 28 29\n"
        "Valid exit cells (top-left quadrant, up to symmetry): 0 1 2 6 7 8 12 13 14\n",
        prog, prog, MAX_BLOCKS, NUM_THREADS);
}

/* Parse comma-separated cell indices (e.g. "5,10,15") into a bitmask and
 * optionally an array.  Returns 1 on success, 0 on error. */
static int parse_cell_list(const char *s, uint32_t *mask, int *pos, int *count) {
    *mask = 0;
    if (count) *count = 0;
    if (!*s) return 1;   /* empty = no cells */
    while (*s) {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s || v < 0 || v >= NCELLS) return 0;
        int cell = (int)v;
        if (*mask & (1u << cell)) return 0;  /* duplicate */
        *mask |= (1u << cell);
        if (pos && count) pos[(*count)++] = cell;
        s = end;
        if (*s == ',') s++;
        else if (*s) return 0;
    }
    return 1;
}

/* Parse "cell=mask,..." pairs (e.g. "5=3,12=0") into g_fixed_mask[].
 * mask must be 0–15.  Returns 1 on success, 0 on error. */
static int parse_cell_mask_list(const char *s) {
    if (!*s) return 1;
    while (*s) {
        char *end;
        long cell = strtol(s, &end, 10);
        if (end == s || cell < 0 || cell >= NCELLS || *end != '=') return 0;
        s = end + 1;
        long mask = strtol(s, &end, 10);
        if (end == s || mask < 0 || mask > 0xF) return 0;
        g_fixed_mask[cell] = (uint8_t)mask;
        s = end;
        if (*s == ',') s++;
        else if (*s) return 0;
    }
    return 1;
}

/* Parse "N" or "lo-hi" into *lo and *hi.  Returns 1 on success, 0 on error. */
static int parse_range(const char *s, int *lo, int *hi) {
    const char *dash = strchr(s, '-');
    if (dash && dash != s) {
        *lo = atoi(s);
        *hi = atoi(dash + 1);
        return (*lo >= 0 && *hi >= *lo);
    }
    *lo = *hi = atoi(s);
    return (*lo >= 0);
}

/* Read a trimmed line from stdin into buf.  Returns 1 if non-empty, 0 if
 * the user just pressed Enter (use default). */
static int prompt(const char *msg, char *buf, int sz) {
    printf("%s", msg);
    fflush(stdout);
    if (!fgets(buf, sz, stdin)) { buf[0] = '\0'; return 0; }
    /* trim trailing newline/whitespace */
    int len = (int)strlen(buf);
    while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r' || buf[len-1] == ' '))
        buf[--len] = '\0';
    return len > 0;
}

int main(int argc, char **argv) {
    memset(g_fixed_mask, 0xF, sizeof g_fixed_mask);
    int nb_lo = -1, nb_hi = -1;   /* -1 = search all block counts */
    int nw_lo =  0, nw_hi =  0;
    int nh_lo = -1, nh_hi = -1;   /* -1 = search all hole counts  */
    int only_ei = -1;

    if (argc == 1) {
        /* Interactive parameter entry */
        char buf[64];
        printf("=== Puzzle Search — interactive setup ===\n");

        if (prompt("Number of movable blocks (e.g. 3 or 1-5, Enter = all): ", buf, sizeof buf)) {
            if (!parse_range(buf, &nb_lo, &nb_hi)) {
                fprintf(stderr, "error: invalid range '%s'\n", buf); return 1;
            }
        }

        if (prompt("Number of walls (e.g. 0 or 0-2) [0]: ", buf, sizeof buf)) {
            if (!parse_range(buf, &nw_lo, &nw_hi)) {
                fprintf(stderr, "error: invalid range '%s'\n", buf); return 1;
            }
        }

        if (prompt("Number of holes (e.g. 3 or 0-3, Enter = all): ", buf, sizeof buf)) {
            if (!parse_range(buf, &nh_lo, &nh_hi)) {
                fprintf(stderr, "error: invalid range '%s'\n", buf); return 1;
            }
        }

        if (prompt("Fixed wall cells (e.g. 5,10,15, Enter = none): ", buf, sizeof buf)) {
            if (!parse_cell_list(buf, &g_fixed_walls, NULL, NULL)) {
                fprintf(stderr, "error: invalid cell list '%s'\n", buf); return 1;
            }
        }

        if (prompt("Fixed hole cells (e.g. 3,8, Enter = none): ", buf, sizeof buf)) {
            if (!parse_cell_list(buf, &g_fixed_holes_mask, g_fixed_hole_pos, &g_fixed_nholes)) {
                fprintf(stderr, "error: invalid cell list '%s'\n", buf); return 1;
            }
        }

        if (prompt("Fixed block cells (e.g. 6,12, Enter = none): ", buf, sizeof buf)) {
            if (!parse_cell_list(buf, &g_fixed_blocks_mask, g_fixed_block_pos, &g_fixed_nblocks)) {
                fprintf(stderr, "error: invalid cell list '%s'\n", buf); return 1;
            }
        }

        if (prompt("Fixed empty cells (e.g. 4,9, Enter = none): ", buf, sizeof buf)) {
            if (!parse_cell_list(buf, &g_fixed_empty_mask, NULL, NULL)) {
                fprintf(stderr, "error: invalid cell list '%s'\n", buf); return 1;
            }
        }

        if (prompt("Exit cell (Enter = all valid) [0 1 2 6 7 8 12 13 14]: ", buf, sizeof buf)) {
            int exit_cell = atoi(buf);
            for (int j = 0; j < NUM_EXIT_CELLS; j++)
                if (EXIT_CELLS[j] == exit_cell) { only_ei = j; break; }
            if (only_ei < 0) {
                fprintf(stderr, "error: %d is not a valid exit cell\n", exit_cell);
                return 1;
            }
        }

        if (prompt("Number of threads [%d]: ", buf, sizeof buf)) {
            int n = atoi(buf);
            if (n >= 1 && n <= 1024) g_num_threads = n;
            else { fprintf(stderr, "error: threads must be between 1 and 1024\n"); return 1; }
        }
        printf("\n");

    } else {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            } else if (strcmp(argv[i], "--nblocks") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --nblocks requires a value\n"); return 1; }
                if (!parse_range(argv[i], &nb_lo, &nb_hi) || nb_lo < 0 || nb_hi > MAX_BLOCKS) {
                    fprintf(stderr, "error: --nblocks must be between 0 and %d\n", MAX_BLOCKS);
                    return 1;
                }
            } else if (strcmp(argv[i], "--nholes") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --nholes requires a value\n"); return 1; }
                if (!parse_range(argv[i], &nh_lo, &nh_hi)) {
                    fprintf(stderr, "error: --nholes must be non-negative\n"); return 1;
                }
            } else if (strcmp(argv[i], "--exitloc") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --exitloc requires a value\n"); return 1; }
                int exit_cell = atoi(argv[i]);
                for (int j = 0; j < NUM_EXIT_CELLS; j++)
                    if (EXIT_CELLS[j] == exit_cell) { only_ei = j; break; }
                if (only_ei < 0) {
                    fprintf(stderr,
                            "error: %d is not a valid exit cell (valid: 0 1 2 6 7 8 12 13 14)\n",
                            exit_cell);
                    return 1;
                }
            } else if (strcmp(argv[i], "--nwalls") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --nwalls requires a value\n"); return 1; }
                if (!parse_range(argv[i], &nw_lo, &nw_hi)) {
                    fprintf(stderr, "error: --nwalls must be non-negative\n"); return 1;
                }
            } else if (strcmp(argv[i], "--fixedwalls") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --fixedwalls requires a value\n"); return 1; }
                if (!parse_cell_list(argv[i], &g_fixed_walls, NULL, NULL)) {
                    fprintf(stderr, "error: invalid cell list for --fixedwalls: '%s'\n", argv[i]); return 1;
                }
            } else if (strcmp(argv[i], "--fixedholes") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --fixedholes requires a value\n"); return 1; }
                if (!parse_cell_list(argv[i], &g_fixed_holes_mask, g_fixed_hole_pos, &g_fixed_nholes)) {
                    fprintf(stderr, "error: invalid cell list for --fixedholes: '%s'\n", argv[i]); return 1;
                }
            } else if (strcmp(argv[i], "--fixedblocks") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --fixedblocks requires a value\n"); return 1; }
                if (!parse_cell_list(argv[i], &g_fixed_blocks_mask, g_fixed_block_pos, &g_fixed_nblocks)) {
                    fprintf(stderr, "error: invalid cell list for --fixedblocks: '%s'\n", argv[i]); return 1;
                }
            } else if (strcmp(argv[i], "--fixedempty") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --fixedempty requires a value\n"); return 1; }
                if (!parse_cell_list(argv[i], &g_fixed_empty_mask, NULL, NULL)) {
                    fprintf(stderr, "error: invalid cell list for --fixedempty: '%s'\n", argv[i]); return 1;
                }
            } else if (strcmp(argv[i], "--fixedmask") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --fixedmask requires a value\n"); return 1; }
                if (!parse_cell_mask_list(argv[i])) {
                    fprintf(stderr, "error: invalid cell=mask list for --fixedmask: '%s'\n", argv[i]); return 1;
                }
            } else if (strcmp(argv[i], "--nthreads") == 0) {
                if (++i >= argc) { fprintf(stderr, "error: --nthreads requires a value\n"); return 1; }
                g_num_threads = atoi(argv[i]);
                if (g_num_threads < 1 || g_num_threads > 1024) {
                    fprintf(stderr, "error: --nthreads must be between 1 and 1024\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--profile") == 0) {
                g_profile_mode = 1;
            } else {
                fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }
    }

    /* Validate fixed cells — all four masks must be mutually disjoint */
    {
        uint32_t all = g_fixed_walls | g_fixed_holes_mask | g_fixed_blocks_mask | g_fixed_empty_mask;
        if (__builtin_popcount(all) !=
            __builtin_popcount(g_fixed_walls) + __builtin_popcount(g_fixed_holes_mask) +
            __builtin_popcount(g_fixed_blocks_mask) + __builtin_popcount(g_fixed_empty_mask)) {
            fprintf(stderr, "error: fixed cell lists overlap\n"); return 1;
        }
    }
    {
        int j0 = (only_ei >= 0) ? only_ei : 0;
        int j1 = (only_ei >= 0) ? only_ei : NUM_EXIT_CELLS - 1;
        for (int j = j0; j <= j1; j++) {
            uint32_t eb = 1u << EXIT_CELLS[j];
            uint32_t conflict = (g_fixed_walls | g_fixed_holes_mask | g_fixed_blocks_mask | g_fixed_empty_mask) & eb;
            if (conflict) {
                fprintf(stderr, "error: fixed cell %d conflicts with a valid exit cell\n", EXIT_CELLS[j]); return 1;
            }
        }
    }

    /* Validate nwalls upper bound */
    if (nw_hi >= MAX_BLOCKS) {
        fprintf(stderr, "error: --nwalls (%d) leaves no room for movable blocks\n", nw_hi);
        return 1;
    }

    /* Resolve unrestrained ranges */
    int real_nb_lo = (nb_lo >= 0) ? nb_lo : (g_fixed_nblocks > 0 ? 0 : 1);
    int pool_avail = NCELLS - 1 - g_fixed_nblocks - g_fixed_nholes
                   - __builtin_popcount(g_fixed_walls)
                   - __builtin_popcount(g_fixed_empty_mask);
    if (pool_avail < 0) pool_avail = 0;
    int real_nb_hi = (nb_hi >= 0) ? nb_hi : (pool_avail < MAX_BLOCKS ? pool_avail : MAX_BLOCKS);

    /* Pre-compute last valid (nw, nb) combo so we know when to print summary */
    int last_nw = nw_lo, last_nb = real_nb_lo;
    for (int cw = nw_lo; cw <= nw_hi; cw++) {
        for (int cb = real_nb_lo; cb <= real_nb_hi; cb++) {
            int total = cb + cw;
            if (total < 0 || total > MAX_BLOCKS) continue;
            if (nh_lo >= 0 && nh_lo > NCELLS - 2 - total) continue;
            last_nw = cw; last_nb = cb;
        }
    }

    for (int cur_nw = nw_lo; cur_nw <= nw_hi; cur_nw++) {
        for (int cur_nb = real_nb_lo; cur_nb <= real_nb_hi; cur_nb++) {
            int total = cur_nb + cur_nw;
            if (total < 0 || total > MAX_BLOCKS) continue;
            if (nh_lo >= 0 && nh_lo > NCELLS - 2 - total) {
                if (nb_lo >= 0 && nb_lo == nb_hi && nw_lo == nw_hi) {
                    fprintf(stderr,
                            "error: --nholes lo (%d) cannot exceed 23 - (nblocks+nwalls) = %d\n",
                            nh_lo, NCELLS - 2 - total);
                    return 1;
                }
                continue;
            }
            int is_last = (cur_nw == last_nw && cur_nb == last_nb);
            puzzle_search(total, cur_nw, nh_lo, nh_hi, only_ei, is_last);
        }
    }

    return 0;
}
