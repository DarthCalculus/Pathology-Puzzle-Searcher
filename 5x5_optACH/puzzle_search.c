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

#define NUM_THREADS 6   /* worker threads; can be raised independently of exit count */

/*
 * Six canonical exit positions — the D4 fundamental domain for a 5×5 grid.
 * The full symmetry group of the square (D4, order 8: 4 rotations + 4
 * reflections) maps any exit cell to one of these six representatives:
 *
 *   (0,0)=0   (0,1)=1   (0,2)=2
 *             (1,1)=6   (1,2)=7
 *                       (2,2)=12
 *
 * Symmetry reduction is applied to hole combinations: for each canonical exit,
 * the stabilizer subgroup (transforms fixing that exit) is used to keep only
 * the lexicographically-minimum hole combination in each orbit.
 *
 * Stabilizers (non-identity elements only):
 *   exit (0,0), (1,1)  — flip_d: (r,c)↔(c,r)
 *   exit (0,2), (1,2)  — flip_h: (r,c)↔(r,4-c)
 *   exit (0,1)         — trivial (no check needed)
 *   exit (2,2)         — full D4 (7 non-identity transforms checked)
 *
 * Player starts are NOT restricted by symmetry; Approaches C (walk-distance
 * bounding) and A (free-component antichain sharing) handle pruning dynamically.
 */
#define NUM_EXIT_CELLS 6
static const int EXIT_CELLS[NUM_EXIT_CELLS] = { 0, 1, 2, 6, 7, 12 };

/*
 * D4 cell transform tables — initialised in precompute_tables().
 * Each maps cell index p=(r*5+c) to its image under the named symmetry.
 *   rot90 : (r,c) → (c,   4-r)   [90° clockwise]
 *   rot180: (r,c) → (4-r, 4-c)
 *   rot270: (r,c) → (4-c, r  )   [270° clockwise]
 *   flip_h: (r,c) → (r,   4-c)   [horizontal flip]
 *   flip_v: (r,c) → (4-r, c  )   [vertical flip]
 *   flip_d: (r,c) → (c,   r  )   [main-diagonal flip]
 *   flip_a: (r,c) → (4-c, 4-r)   [anti-diagonal flip]
 */
static int8_t t_rot90 [NCELLS];
static int8_t t_rot180[NCELLS];
static int8_t t_rot270[NCELLS];
static int8_t t_flip_h[NCELLS];
static int8_t t_flip_v[NCELLS];
static int8_t t_flip_d[NCELLS];
static int8_t t_flip_a[NCELLS];

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
    {-1,  1,  5, -1}, {-1,  2,  6,  0}, {-1,  3,  7,  1},
    {-1,  4,  8,  2}, {-1, -1,  9,  3},
    /* row 1 */
    { 0,  6, 10, -1}, { 1,  7, 11,  5}, { 2,  8, 12,  6},
    { 3,  9, 13,  7}, { 4, -1, 14,  8},
    /* row 2 */
    { 5, 11, 15, -1}, { 6, 12, 16, 10}, { 7, 13, 17, 11},
    { 8, 14, 18, 12}, { 9, -1, 19, 13},
    /* row 3 */
    {10, 16, 20, -1}, {11, 17, 21, 15}, {12, 18, 22, 16},
    {13, 19, 23, 17}, {14, -1, 24, 18},
    /* row 4 */
    {15, 21, -1, -1}, {16, 22, -1, 20}, {17, 23, -1, 21},
    {18, 24, -1, 22}, {19, -1, -1, 23},
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

static pthread_mutex_t g_best_mutex = PTHREAD_MUTEX_INITIALIZER;

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

static void update_best(int d, const Puzzle *pz) {
    if (d <= g_best) return;
    pthread_mutex_lock(&g_best_mutex);
    if (d > g_best) {
        g_best    = d;
        g_best_pz = *pz;
        printf("New best: %d moves\n", d);
        print_puzzle(pz);
        fflush(stdout);
    }
    pthread_mutex_unlock(&g_best_mutex);
}

/* -------------------------------------------------------------------------
 * Per-thread call counter
 * ------------------------------------------------------------------------- */

static _Thread_local long long tl_ncalls = 0;

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

    /* D4 cell transform tables.  All transforms act on a 5×5 grid where
     * cell p encodes (r,c) as p = r*COLS + c with r,c in [0, ROWS-1]. */
    for (int p = 0; p < NCELLS; p++) {
        int r = p / COLS, c = p % COLS;
        t_rot90 [p] = (int8_t)(c        * COLS + (COLS-1-r));
        t_rot180[p] = (int8_t)((ROWS-1-r) * COLS + (COLS-1-c));
        t_rot270[p] = (int8_t)((COLS-1-c) * COLS + r       );
        t_flip_h[p] = (int8_t)(r          * COLS + (COLS-1-c));
        t_flip_v[p] = (int8_t)((ROWS-1-r) * COLS + c       );
        t_flip_d[p] = (int8_t)(c          * COLS + r       );
        t_flip_a[p] = (int8_t)((COLS-1-c) * COLS + (ROWS-1-r));
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

static int try_bitmasks(Puzzle *pz, int bi, uint8_t *unsolv, int *pn,
                        GlobalAntichain *ga, KnownSolvable *ks) {
    if (bi == pz->num_blocks) {
        if (ga_dominated(ga, pz->block_pushable, pz->num_blocks)) return -1;
        int nb = pz->num_blocks;

        /* KS lookup: pack current masks; if any stored used-dirs is a subset,
         * the stored solution works here too — same distance, no BFS needed. */
        uint32_t cur_packed = 0;
        for (int i = 0; i < nb; i++)
            cur_packed |= ((uint32_t)pz->block_pushable[i] << (i*4));
        for (int k = 0; k < ks->count; k++) {
            if ((cur_packed & ks->packed[k]) == ks->packed[k]) {
                int d = ks->dist[k];
                if (d > g_best) update_best(d, pz);
                return d;
            }
        }

        tl_ncalls++;
        uint8_t used[MAX_BLOCKS] = {0};
        int d = sokoban_solve(pz, used);
        if (d == -2) {
            fprintf(stderr, "fatal: BFS queue overflow (out of memory) — "
                    "search results are incomplete. Increase QSZ in sokoban_bfs.c.\n");
            exit(1);
        }
        if (d > g_best) update_best(d, pz);
        if (d >= 0 && ks->count < MAX_KS) {
            uint32_t u_packed = 0;
            for (int i = 0; i < nb; i++) u_packed |= ((uint32_t)used[i] << (i*4));
            ks->packed[ks->count] = u_packed;
            ks->dist  [ks->count] = d;
            ks->count++;
        }
        if (d == -1) ga_add(ga, pz->block_pushable, nb);
        return d;
    }

    int            nm   = cell_nmasks[pz->block_pos[bi]];
    const uint8_t *ms   = cell_masks [pz->block_pos[bi]];
    int            best = -1;

    /* XC table indexed by thresh (4-bit mask, 0-15).
     * xc_masks[t] holds unsolvable masks for block bi+1 when block bi = t.
     * Seeding uses superset_list[m] to visit only relevant buckets:
     * for m ⊆ t, the bi+1 mask was also unsolvable when bi=m (Fact 2).
     * With 16 buckets of ≤16 entries each, the hot high-popcount masks
     * (1-2 supersets) check 16-32 entries instead of scanning all 256. */
    uint8_t xc_masks[16][16];
    int     xc_cnt[16];
    memset(xc_cnt, 0, sizeof xc_cnt);

    for (int mi = 0; mi < nm; mi++) {
        uint8_t m = ms[mi];

        /* Skip if m is dominated by the current level's unsolvable antichain */
        int skip = 0;
        for (int j = 0; j < *pn && !skip; j++)
            if ((m & unsolv[j]) == m) skip = 1;
        if (skip) continue;

        /* Seed next level's antichain from XC buckets at thresh ⊇ m */
        uint8_t b_unsolv[MAX_UNSOLV]; int nb = 0;
        int ns = superset_cnt[m];
        for (int si = 0; si < ns; si++) {
            int t = superset_list[m][si];
            for (int xi = 0; xi < xc_cnt[t]; xi++) {
                uint8_t x = xc_masks[t][xi];
                int dominated = 0;
                for (int k = 0; k < nb && !dominated; k++)
                    if ((x & b_unsolv[k]) == x) dominated = 1;
                if (!dominated) b_unsolv[nb++] = x;
            }
        }
        int nb_seed = nb;

        pz->block_pushable[bi] = m;
        int d = try_bitmasks(pz, bi + 1, b_unsolv, &nb, ga, ks);

        /* Record newly discovered unsolvable masks under thresh=m */
        for (int j = nb_seed; j < nb; j++)
            if (xc_cnt[m] < 16) xc_masks[m][xc_cnt[m]++] = b_unsolv[j];

        if (d < 0) unsolv[(*pn)++] = m;
        if (d > best) best = d;
    }
    return best;
}

/* -------------------------------------------------------------------------
 * Per-(exit, hole-configuration) work item
 *
 * Processes all block placements and player starts for one specific
 * (exit cell, hole positions) pair.  Called by worker threads.
 * ------------------------------------------------------------------------- */

static void process_hole_config(int ei, int nw, int nh, const int *hp, int total) {
    int ep = EXIT_CELLS[ei];

    uint32_t holes_mask = 0;
    for (int i = 0; i < nh; i++) holes_mask |= (1u << hp[i]);

    /* Pool for blocks: not exit, not a hole. */
    int bpool[NCELLS], nbpool = 0;
    for (int c = 0; c < NCELLS; c++)
        if (c != ep && !(holes_mask & (1u << c)))
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
        #define ADD_BT(t) if (nh == 0 || transform_fixes_holes(hp, nh, t)) bt[nbt++] = (t)
        switch (ei) {
        case 0: case 3: ADD_BT(t_flip_d); break;
        case 2: case 4: ADD_BT(t_flip_h); break;
        case 5:
            ADD_BT(t_rot90); ADD_BT(t_rot180); ADD_BT(t_rot270);
            ADD_BT(t_flip_h); ADD_BT(t_flip_v);
            ADD_BT(t_flip_d); ADD_BT(t_flip_a);
            break;
        default: break; /* ei=1: trivial stabilizer */
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
        uint32_t occ = (1u << ep) | holes_mask;
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
            uint32_t wall_mask = 0;
            int mbp[MAX_BLOCKS], nb_mov = 0;
            for (int i = 0; i < total; i++) {
                if (is_wall[i]) wall_mask |= (1u << bp[i]);
                else            mbp[nb_mov++] = bp[i];
            }

            /* Reset component antichains for this wall subset. */
            for (int vi = 0; vi < n_valid; vi++)
                if (comp_id[vi] == vi) comp_ga[vi].count = 0;

            /* proc_K[vi] = best solution distance found for valid start vi.
             * -1 means unsolvable (no bitmask yielded a solution). */
            int proc_K[NCELLS];
            int n_proc = 0;

            /* --- Enumerate player starting positions --- */
            for (int vi = 0; vi < n_valid; vi++) {
                int ps = vpi[vi];

                /* Walk-distance upper bound: if every previously processed
                 * player start ps0 with result K0 gives K0 + walk(ps0→ps)
                 * ≤ g_best, then no bitmask can yield a better result from
                 * ps (since from ps you can walk to ps0 in walk(ps0→ps)
                 * steps and follow ps0's solution). Skip ps in that case. */
                int bound = INT_MAX;
                for (int pv = 0; pv < n_proc; pv++) {
                    if (proc_K[pv] < 0) continue;   /* unsolvable: no bound */
                    int8_t d = vwalk[pv][ps];
                    if (d < 0) continue;             /* ps unreachable from pv */
                    int b = proc_K[pv] + (int)d;
                    if (b < bound) bound = b;
                }
                if (bound <= g_best) continue;       /* cannot beat global best */

                Puzzle pz;
                memset(&pz, 0, sizeof pz);
                pz.exit_pos     = ep;
                pz.player_start = ps;
                pz.walls        = wall_mask;
                pz.num_blocks   = nb_mov;
                pz.num_holes    = nh;
                for (int i = 0; i < nh; i++) pz.hole_pos[i] = hp[i];
                for (int i = 0; i < nb_mov; i++)
                    pz.block_pos[i] = mbp[i];

                /* All-immovable check: mask=0 is the lattice bottom,
                 * a subset of every mask vector.  By Fact 1, if it is
                 * solvable with d steps, no other mask vector can give
                 * more than d steps — d is the maximum for this position.
                 * imm_blocked uses occ which covers all block positions
                 * (both wall-designated and movable), so wall cells are
                 * correctly treated as obstacles here. */
                for (int i = 0; i < nb_mov; i++)
                    pz.block_pushable[i] = 0;
                tl_ncalls++;
                uint32_t imm_blocked = occ & ~(1u << ep);
                int d0 = fast_reachable(imm_blocked, ps, ep);
                if (d0 > g_best) update_best(d0, &pz);
                if (d0 >= 0) { proc_K[n_proc++] = d0; continue; }

                /* All-immovable unsolvable: full bitmask search.
                 * try_bitmasks starts top-down (max valid first).
                 * Uses the shared component GlobalAntichain so that
                 * unsolvable bitmask vectors found for any prior player
                 * start in the same free-walking component are immediately
                 * pruned without a BFS call. KnownSolvable is kept
                 * per-player-start as distances differ across starts. */
                KnownSolvable   ks; ks.count = 0;
                uint8_t unsolv0[MAX_UNSOLV]; int n0 = 0;
                int k_bitmask = try_bitmasks(&pz, 0, unsolv0, &n0,
                                             &comp_ga[comp_id[vi]], &ks);
                proc_K[n_proc++] = k_bitmask;
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

/* -------------------------------------------------------------------------
 * Work queue and thread pool
 *
 * Each item is one canonical (exit, hole-configuration) pair.  The queue
 * is rebuilt for each nh level so that all nh=k items complete before any
 * nh=k+1 items begin — this keeps g_best as high as possible before the
 * expensive large-nh work starts, maximising walk-distance pruning.
 * ------------------------------------------------------------------------- */

typedef struct { int ei, nh; int hp[MAX_HOLES]; } WorkItem;

static WorkItem       *g_wq       = NULL;
static int             g_wq_cap   = 0;
static int             g_wq_count = 0;
static int             g_wq_next  = 0;
static pthread_mutex_t g_wq_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int        total;
    int        nw;
    long long  ncalls;
} ThreadArg;

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

        process_hole_config(item.ei, a->nw, item.nh, item.hp, a->total);
    }

    a->ncalls = tl_ncalls;
    return NULL;
}

/* -------------------------------------------------------------------------
 * Main search
 * ------------------------------------------------------------------------- */

/* nw:      number of block positions designated as walls per block combo.
 * only_nh: restrict to a single hole count (-1 = all counts).
 * only_ei: restrict to a single exit index into EXIT_CELLS (-1 = all exits). */
static void puzzle_search(int total, int nw, int only_nh, int only_ei) {
    if (total < 1 || total > MAX_BLOCKS) {
        fprintf(stderr, "num_blocks must be between 1 and %d\n", MAX_BLOCKS);
        return;
    }
    precompute_tables();
    printf("Searching: blocks = %d,  walls = %d,  grid = %d×%d,  threads = %d\n\n",
           total, nw, ROWS, COLS, NUM_THREADS);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    long long total_ncalls = 0;

    /* Process hole counts one level at a time.  Building the queue per-level
     * ensures g_best is as high as possible before large nh items begin,
     * maximising the walk-distance pruning (Approach C) on those items. */
    int nh_lo = (only_nh >= 0) ? only_nh : 0;
    int nh_hi = (only_nh >= 0) ? only_nh : total;
    int ei_lo = (only_ei >= 0) ? only_ei : 0;
    int ei_hi = (only_ei >= 0) ? only_ei : NUM_EXIT_CELLS - 1;

    for (int nh = nh_lo; nh <= nh_hi && nh <= MAX_HOLES; nh++) {

        /* Build work queue: one item per canonical (exit, hole-config) pair.
         * Hole canonicality checks (stabilizer of each exit cell) are applied
         * here so workers receive only non-redundant configurations. */
        g_wq_count = 0;
        g_wq_next  = 0;

        for (int ei = ei_lo; ei <= ei_hi; ei++) {
            int ep = EXIT_CELLS[ei];

            int hpool[NCELLS], nhpool = 0;
            for (int c = 0; c < NCELLS; c++)
                if (c != ep) hpool[nhpool++] = c;
            if (nh > nhpool) continue;

            Comb hc;
            comb_init(&hc, nhpool, nh);
            do {
                int hp[MAX_HOLES];
                for (int i = 0; i < nh; i++) hp[i] = hpool[hc.idx[i]];

                /*
                 * Hole canonicality check: keep only the lex-min hole combo
                 * under the stabilizer of this exit cell.  hpool is built in
                 * ascending order so hc produces sorted combos — satisfying
                 * holes_lex_min_under's precondition.
                 *
                 * Stabilizer per exit index:
                 *   ei=0,3  — flip_d
                 *   ei=2,4  — flip_h
                 *   ei=1    — trivial (no check)
                 *   ei=5    — full D4 (7 non-identity elements)
                 */
                if (nh > 0) {
                    switch (ei) {
                    case 0: case 3:
                        if (!holes_lex_min_under(hp, nh, t_flip_d)) continue;
                        break;
                    case 2: case 4:
                        if (!holes_lex_min_under(hp, nh, t_flip_h)) continue;
                        break;
                    case 5:
                        if (!holes_lex_min_under(hp, nh, t_rot90 )) continue;
                        if (!holes_lex_min_under(hp, nh, t_rot180)) continue;
                        if (!holes_lex_min_under(hp, nh, t_rot270)) continue;
                        if (!holes_lex_min_under(hp, nh, t_flip_h)) continue;
                        if (!holes_lex_min_under(hp, nh, t_flip_v)) continue;
                        if (!holes_lex_min_under(hp, nh, t_flip_d)) continue;
                        if (!holes_lex_min_under(hp, nh, t_flip_a)) continue;
                        break;
                    default: break; /* ei=1: no check */
                    }
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

        for (int t = 0; t < NUM_THREADS; t++) {
            args[t].total  = total;
            args[t].nw     = nw;
            args[t].ncalls = 0;
            pthread_create(&threads[t], NULL, worker_thread, &args[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
            total_ncalls += args[t].ncalls;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("\n=== Search complete ===\n");
    print_time_calls(elapsed_s(t0, t1), total_ncalls);
    if (g_best >= 0) {
        printf("Best solution: %d moves\n", g_best);
        print_puzzle(&g_best_pz);
    } else {
        printf("No solvable puzzles found.\n");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <num_blocks> [--nholes <n>] [--exitloc <cell>] [--nwalls <n>]\n"
                "  --nholes  <n>   : restrict search to exactly n holes (default: all)\n"
                "  --exitloc <cell>: restrict to one exit cell in {0,1,2,6,7,12} (default: all)\n"
                "  --nwalls  <n>   : designate n of the block positions as walls (default: 0)\n",
                argv[0]);
        return 1;
    }

    int total   = atoi(argv[1]);
    int only_nh = -1;
    int only_ei = -1;
    int nw      = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--nholes") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nholes requires a value\n"); return 1; }
            only_nh = atoi(argv[i]);
            if (only_nh < 0 || only_nh > MAX_HOLES) {
                fprintf(stderr, "error: --nholes must be between 0 and %d\n", MAX_HOLES);
                return 1;
            }
        } else if (strcmp(argv[i], "--exitloc") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --exitloc requires a value\n"); return 1; }
            int exit_cell = atoi(argv[i]);
            only_ei = -1;
            for (int j = 0; j < NUM_EXIT_CELLS; j++) {
                if (EXIT_CELLS[j] == exit_cell) { only_ei = j; break; }
            }
            if (only_ei < 0) {
                fprintf(stderr,
                        "error: %d is not a valid exit cell (valid: 0 1 2 6 7 12)\n",
                        exit_cell);
                return 1;
            }
        } else if (strcmp(argv[i], "--nwalls") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nwalls requires a value\n"); return 1; }
            nw = atoi(argv[i]);
            if (nw < 0) {
                fprintf(stderr, "error: --nwalls must be non-negative\n");
                return 1;
            }
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            return 1;
        }
    }

    if (nw > total) {
        fprintf(stderr, "error: --nwalls (%d) cannot exceed num_blocks (%d)\n", nw, total);
        return 1;
    }

    puzzle_search(total, nw, only_nh, only_ei);
    return 0;
}
