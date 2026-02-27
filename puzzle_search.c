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
 * Parallelism: the (exit, hole-count) pairs form a work queue consumed by
 * NUM_THREADS worker threads.  sokoban_solve() is thread-safe (per-thread
 * BFS state); g_best/g_best_pz are protected by g_best_mutex.
 */

#include "sokoban_bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

#define NUM_THREADS 8

/* Six canonical exit positions (top-left 2×3 sub-grid). */
static const int EXIT_CELLS[6] = { 0, 1, 2, 5, 6, 7 };

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
    {-1,  1,  5, -1}, {-1,  2,  6,  0}, {-1,  3,  7,  1},
    {-1,  4,  8,  2}, {-1, -1,  9,  3},
    { 0,  6, 10, -1}, { 1,  7, 11,  5}, { 2,  8, 12,  6},
    { 3,  9, 13,  7}, { 4, -1, 14,  8},
    { 5, 11, 15, -1}, { 6, 12, 16, 10}, { 7, 13, 17, 11},
    { 8, 14, 18, 12}, { 9, -1, 19, 13},
    {10, 16, -1, -1}, {11, 17, -1, 15}, {12, 18, -1, 16},
    {13, 19, -1, 17}, {14, -1, -1, 18},
};

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

/* Shard mode: one thread per exit, per-shard record tracking. */
static int g_shard_mode = 0;
static _Thread_local int tl_shard_best      = -1;
static _Thread_local int tl_shard_exit_cell = -1;

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

/* Call under g_best_mutex or with the double-check idiom below.
 * In shard mode printing is suppressed; the global best is still tracked
 * so puzzle_search_shard can report it in the final summary. */
static void update_best(int d, const Puzzle *pz) {
    if (d <= g_best) return;
    pthread_mutex_lock(&g_best_mutex);
    if (d > g_best) {
        g_best    = d;
        g_best_pz = *pz;
        if (!g_shard_mode) {
            printf("New best: %d moves\n", d);
            print_puzzle(pz);
            fflush(stdout);
        }
    }
    pthread_mutex_unlock(&g_best_mutex);
}

/* Shard-mode output: called after every solvable BFS result.
 * Prints when d beats this shard's record, or when d > 60. */
static void shard_update(int d, const Puzzle *pz) {
    if (d < 0) return;
    int is_record = (d > tl_shard_best);
    if (is_record) tl_shard_best = d;
    if (is_record || d > 60) {
        pthread_mutex_lock(&g_best_mutex);
        printf(is_record ? "New record (exit %d): %d moves\n"
                         : "Notable   (exit %d): %d moves\n",
               tl_shard_exit_cell, d);
        print_puzzle(pz);
        fflush(stdout);
        pthread_mutex_unlock(&g_best_mutex);
    }
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
    int      max_d; /* highest step count reported via shard_update for this board */
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
        if (d > g_best) update_best(d, pz);
        if (g_shard_mode && d > ks->max_d) { ks->max_d = d; shard_update(d, pz); }
        if (d >= 0 && ks->count < MAX_KS) {
            uint32_t u_packed = 0;
            for (int i = 0; i < nb; i++) u_packed |= ((uint32_t)used[i] << (i*4));
            ks->packed[ks->count] = u_packed;
            ks->dist  [ks->count] = d;
            ks->count++;
        }
        if (d < 0) ga_add(ga, pz->block_pushable, nb);
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
 * Per-(exit, hole-count) work item
 * ------------------------------------------------------------------------- */

static void process_exit_nh(int ei, int nh, int total) {
    int ep = EXIT_CELLS[ei];

    /* Candidate player-start positions after symmetry reduction.
     * Depends only on ep, so computed once here rather than checked
     * inside every (hole_combo × block_combo × player) iteration. */
    int ps_cands[NCELLS], n_ps = 0;
    for (int ps = 0; ps < NCELLS; ps++) {
        if (ps == ep) continue;
        if ((ep == 2 || ep == 7) && col_(ps) > 2) continue;
        ps_cands[n_ps++] = ps;
    }

    /* Pool of cells for holes and blocks: everything except the exit. */
    int hpool[NCELLS], nhpool = 0;
    for (int c = 0; c < NCELLS; c++)
        if (c != ep) hpool[nhpool++] = c;

    if (nh > nhpool) return;

    Comb hc;
    comb_init(&hc, nhpool, nh);
    do {
        uint32_t holes_mask = 0;
        int hp[MAX_HOLES];
        for (int i = 0; i < nh; i++) {
            hp[i] = hpool[hc.idx[i]];
            holes_mask |= (1u << hp[i]);
        }

        /* Pool for blocks: not exit, not a hole. */
        int bpool[NCELLS], nbpool = 0;
        for (int c = 0; c < NCELLS; c++)
            if (c != ep && !(holes_mask & (1u << c)))
                bpool[nbpool++] = c;

        if (total > nbpool) continue;

        /* --- Enumerate block placements --- */
        Comb bc;
        comb_init(&bc, nbpool, total);
        do {
            int bp[MAX_BLOCKS];
            for (int i = 0; i < total; i++) bp[i] = bpool[bc.idx[i]];

            /* Occupied mask for player-start filtering.
             * Holes are included: the player cannot start on one. */
            uint32_t occ = (1u << ep) | holes_mask;
            for (int i = 0; i < total; i++) occ |= (1u << bp[i]);

            /* --- Enumerate player starting positions --- */
            for (int pi = 0; pi < n_ps; pi++) {
                int ps = ps_cands[pi];
                if (occ & (1u << ps)) continue;

                Puzzle pz;
                memset(&pz, 0, sizeof pz);
                pz.exit_pos     = ep;
                pz.player_start = ps;
                pz.num_blocks   = total;
                pz.num_holes    = nh;
                for (int i = 0; i < nh; i++) pz.hole_pos[i] = hp[i];
                for (int i = 0; i < total; i++)
                    pz.block_pos[i] = bp[i];

                /* All-immovable check: mask=0 is the lattice bottom,
                 * a subset of every mask vector.  By Fact 1, if it is
                 * solvable with d steps, no other mask vector can give
                 * more than d steps — d is the maximum for this
                 * position.  Fast bitmask flood fill on 20-cell grid:
                 * blocks never move, so it's pure pathfinding. */
                for (int i = 0; i < total; i++)
                    pz.block_pushable[i] = 0;
                tl_ncalls++;
                uint32_t imm_blocked = occ & ~(1u << ep);
                int d0 = fast_reachable(imm_blocked, ps, ep);
                if (d0 > g_best) update_best(d0, &pz);
                if (g_shard_mode) shard_update(d0, &pz);
                if (d0 >= 0) continue;

                /* All-immovable unsolvable: full bitmask search.
                 * try_bitmasks starts top-down (max valid first).
                 * Fresh GlobalAntichain per board: scoped to this exact
                 * (player_start, block positions, holes, exit) combo. */
                GlobalAntichain ga; ga.count = 0;
                KnownSolvable   ks; ks.count = 0; ks.max_d = -1;
                uint8_t unsolv0[MAX_UNSOLV]; int n0 = 0;
                try_bitmasks(&pz, 0, unsolv0, &n0, &ga, &ks);
            }
        } while (comb_next(&bc));

    } while (comb_next(&hc));
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
 * ------------------------------------------------------------------------- */

typedef struct { int ei, nh; } WorkItem;

/* Populated once before threads start; read-only during search. */
static WorkItem g_wq[6 * (MAX_HOLES + 1)];
static int      g_wq_count = 0;

/* Next unclaimed item index — protected by g_wq_mutex. */
static int             g_wq_next  = 0;
static pthread_mutex_t g_wq_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    int        total;
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

        process_exit_nh(item.ei, item.nh, a->total);
    }

    a->ncalls = tl_ncalls;
    return NULL;
}

/* -------------------------------------------------------------------------
 * Main search
 * ------------------------------------------------------------------------- */

static void puzzle_search(int total) {
    if (total < 1 || total > MAX_BLOCKS) {
        fprintf(stderr, "num_blocks must be between 1 and %d\n", MAX_BLOCKS);
        return;
    }
    precompute_tables();
    printf("Searching: blocks = %d,  grid = %d×%d,  threads = %d\n\n",
           total, ROWS, COLS, NUM_THREADS);

    /* Build the work queue: one item per (exit, hole-count) pair. */
    g_wq_count = 0;
    g_wq_next  = 0;
    for (int ei = 0; ei < 6; ei++)
        for (int nh = 0; nh <= total && nh <= MAX_HOLES; nh++)
            g_wq[g_wq_count++] = (WorkItem){ ei, nh };

    /* Launch worker threads. */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].total  = total;
        args[t].ncalls = 0;
        pthread_create(&threads[t], NULL, worker_thread, &args[t]);
    }

    /* Wait for all threads and collect call counts. */
    long long total_ncalls = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
        total_ncalls += args[t].ncalls;
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

/* -------------------------------------------------------------------------
 * Shard mode: one thread per exit cell, no shared work queue.
 * Each thread processes all hole-counts for its exit independently.
 * Output: per-shard new records and any result > 60.
 * ------------------------------------------------------------------------- */

typedef struct {
    int       exit_idx;   /* index into EXIT_CELLS (0-5) */
    int       total;
    int       shard_best;
    long long ncalls;
} ShardArg;

static void *shard_worker(void *arg) {
    ShardArg *sa = (ShardArg *)arg;
    tl_shard_exit_cell = EXIT_CELLS[sa->exit_idx];
    tl_shard_best      = -1;

    for (int nh = 0; nh <= sa->total && nh <= MAX_HOLES; nh++)
        process_exit_nh(sa->exit_idx, nh, sa->total);

    sa->shard_best = tl_shard_best;
    sa->ncalls     = tl_ncalls;
    return NULL;
}

static void puzzle_search_shard(int total) {
    if (total < 1 || total > MAX_BLOCKS) {
        fprintf(stderr, "num_blocks must be between 1 and %d\n", MAX_BLOCKS);
        return;
    }
    g_shard_mode = 1;
    precompute_tables();
    printf("Shard mode: blocks = %d,  grid = %d×%d,  shards = 6\n\n",
           total, ROWS, COLS);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_t threads[6];
    ShardArg  args[6];
    for (int ei = 0; ei < 6; ei++) {
        args[ei] = (ShardArg){ ei, total, -1, 0 };
        pthread_create(&threads[ei], NULL, shard_worker, &args[ei]);
    }

    long long total_ncalls = 0;
    for (int ei = 0; ei < 6; ei++) {
        pthread_join(threads[ei], NULL);
        total_ncalls += args[ei].ncalls;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("\n=== Shard search complete ===\n");
    print_time_calls(elapsed_s(t0, t1), total_ncalls);
    printf("\nPer-shard bests:\n");
    for (int ei = 0; ei < 6; ei++)
        printf("  exit cell %d: %d moves  (%lld calls)\n",
               EXIT_CELLS[ei], args[ei].shard_best, args[ei].ncalls);
    if (g_best >= 0) {
        printf("\nGlobal best: %d moves\n", g_best);
        print_puzzle(&g_best_pz);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <num_blocks> [--shard]\n", argv[0]);
        return 1;
    }
    int total = atoi(argv[1]);
    if (argc >= 3 && strcmp(argv[2], "--shard") == 0)
        puzzle_search_shard(total);
    else
        puzzle_search(total);
    return 0;
}
