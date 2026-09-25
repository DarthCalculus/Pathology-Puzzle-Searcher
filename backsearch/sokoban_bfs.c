/* sokoban_bfs.c — the forward solver.
 *
 * This file includes itself: the normal pass (SOKW undefined) is everything up
 * to the "#else" near the end; the push solvers are written once, as the PUSH
 * SOLVER TEMPLATE after that "#else", and the normal pass includes this file
 * again once per packed-state width with SOKW = 1, 2, 4 (PUSH SOLVER INSTANCES)
 * to compile them.  Keeping the template here keeps the solver in the three
 * files SRC_HASH covers (build_pgo.sh). */
#ifndef SOKW
#include "sokoban_bfs.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#ifdef FWPROF
/* Forward-solver phase profiler (compile with -DFWPROF for a throwaway
 * profiling binary; zero cost otherwise).  Times the per-pop phases of
 * solve_push_cutoff (every width) and dumps a cumulative + delta breakdown to stderr
 * every FWP_DUMP_EVERY solve calls, so a steady-state window can be read. */
#include <mach/mach_time.h>
enum { FWP_SETUP = 0, FWP_UNPACK, FWP_PASS1, FWP_WALK, FWP_PASS2, FWP_NPH };
static const char *g_fwp_names[FWP_NPH] = {
    "setup", "unpack", "pass1(cand+mand)", "walk_dists", "pass2(enqueue)" };
static uint64_t g_fwp[FWP_NPH];       /* accumulated mach ticks per phase   */
static uint64_t g_fwp_prev[FWP_NPH];  /* snapshot at last dump              */
static uint64_t g_fwp_solves;
/* Deadlock-computation sub-timers (overlap the buckets above):
 *  g_msetup_ticks  = time inside mand_setup()  (subset of FWP_SETUP)
 *  g_mdead_ticks   = time inside mand_dead()   (subset of FWP_PASS1)
 * Both are measured with a mach_absolute_time() PAIR around the call, so the
 * per-call timer overhead is charged to them; fwp_dump() calibrates that
 * overhead and reports an overhead-corrected figure for the high-frequency
 * mand_dead timer. */
static uint64_t g_msetup_ticks;
static uint64_t g_mctx_ticks;    /* per-pop mand_pop_ctx precompute (subset of FWP_UNPACK) */
static uint64_t g_mdead_ticks;
static uint64_t g_mdead_calls;
#define FWP_DUMP_EVERY 2000000ULL
#define FWP_DECL()   uint64_t _tp = mach_absolute_time()
#define FWP_LAP(idx) do { uint64_t _n = mach_absolute_time(); \
                          g_fwp[idx] += _n - _tp; _tp = _n; } while (0)
/* Min cost of a back-to-back mach_absolute_time() pair, in ticks. */
static uint64_t fwp_timer_overhead(void) {
    uint64_t best = UINT64_MAX;
    for (int i = 0; i < 200000; i++) {
        uint64_t a = mach_absolute_time();
        uint64_t b = mach_absolute_time();
        if (b - a < best) best = b - a;
    }
    return best;
}
static void fwp_dump(void) {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    uint64_t tot = 0, dtot = 0;
    for (int i = 0; i < FWP_NPH; i++) { tot += g_fwp[i]; dtot += g_fwp[i] - g_fwp_prev[i]; }
    if (tot == 0) return;
    fprintf(stderr, "[FWPROF] solves=%llu  phase breakdown (cumulative %% | delta %%):\n",
            (unsigned long long)g_fwp_solves);
    for (int i = 0; i < FWP_NPH; i++) {
        uint64_t d = g_fwp[i] - g_fwp_prev[i];
        double cms = (double)g_fwp[i] * tb.numer / tb.denom / 1e6;
        fprintf(stderr, "    %-18s %5.1f%% | %5.1f%%   (%.0f ms cum)\n",
                g_fwp_names[i], tot ? 100.0 * g_fwp[i] / tot : 0.0,
                dtot ? 100.0 * d / dtot : 0.0, cms);
        g_fwp_prev[i] = g_fwp[i];
    }
    /* Deadlock-computation summary (overlapping subsets of the buckets). */
    uint64_t ovh   = fwp_timer_overhead();
    uint64_t raw_d = g_mdead_ticks;
    uint64_t corr  = ovh * g_mdead_calls;                 /* timer overhead in the mdead pair */
    uint64_t net_d = raw_d > corr ? raw_d - corr : 0;     /* overhead-corrected mand_dead time */
    double ms = 1.0 * tb.numer / tb.denom / 1e6;
    fprintf(stderr, "  -- deadlock computation (of %.0f ms solver total) --\n",
            (double)tot * ms);
    fprintf(stderr, "    mand_setup        %5.1f%%   (%.0f ms cum)\n",
            100.0 * g_msetup_ticks / tot, (double)g_msetup_ticks * ms);
    fprintf(stderr, "    mand_pop_ctx      %5.1f%%   (%.0f ms cum, per-pop precompute)\n",
            100.0 * g_mctx_ticks / tot, (double)g_mctx_ticks * ms);
    fprintf(stderr, "    mand_dead raw     %5.1f%%   (%.0f ms cum, %llu calls)\n",
            100.0 * raw_d / tot, (double)raw_d * ms, (unsigned long long)g_mdead_calls);
    fprintf(stderr, "    mand_dead corr'd  %5.1f%%   (%.0f ms, minus %.0f ms timer overhead)\n",
            100.0 * net_d / tot, (double)net_d * ms, (double)corr * ms);
    fprintf(stderr, "    DEADLOCK TOTAL    %5.1f%%   (setup + ctx + corrected mand_dead)\n",
            100.0 * (g_msetup_ticks + g_mctx_ticks + net_d) / tot);
    fflush(stderr);
}
#else
#define FWP_DECL()   ((void)0)
#define FWP_LAP(idx) ((void)0)
#endif

/* Runtime grid dimensions (defined here, declared extern in the header). */
int     g_rows           = 0;
int     g_cols           = 0;
int     g_ncells         = 0;
int     g_consumed       = 0;     /* sentinel for "block fell in hole"; == g_ncells */
int8_t  g_adj[MAX_NCELLS][4];

/* Bits-per-cell for state encoding (5 for ≤32 cells, 6 for ≤64). */
static int      g_bits_per_cell = 0;
static uint64_t g_cell_mask     = 0;   /* = (1<<g_bits_per_cell) - 1 */

/* Bitmask edge constants — recomputed in sokoban_set_grid(). */
static uint64_t g_col0_mask    = 0;
static uint64_t g_collast_mask = 0;
static uint64_t g_all_cells    = 0;

static int bits_needed(int n) {
    int b = 1;
    while ((1 << b) < n) b++;
    return b;
}

/* Build g_adj[][] for the current g_rows x g_cols grid. */
static void build_adj(void) {
    for (int i = 0; i < MAX_NCELLS; i++)
        for (int d = 0; d < 4; d++) g_adj[i][d] = -1;
    for (int r = 0; r < g_rows; r++) {
        for (int c = 0; c < g_cols; c++) {
            int p = r * g_cols + c;
            g_adj[p][0] = (r > 0)          ? (int8_t)((r - 1) * g_cols + c)     : -1;
            g_adj[p][1] = (c < g_cols - 1) ? (int8_t)( r      * g_cols + c + 1) : -1;
            g_adj[p][2] = (r < g_rows - 1) ? (int8_t)((r + 1) * g_cols + c)     : -1;
            g_adj[p][3] = (c > 0)          ? (int8_t)( r      * g_cols + c - 1) : -1;
        }
    }
}

void sokoban_set_grid(int rows, int cols) {
    if (rows < 1 || cols < 1 || rows > MAX_ROWS || cols > MAX_COLS
        || rows * cols > MAX_NCELLS) {
        fprintf(stderr, "sokoban_set_grid: %dx%d out of range "
                        "(need R>=1, C>=1, R*C<=%d)\n",
                rows, cols, MAX_NCELLS);
        abort();
    }
    g_rows   = rows;
    g_cols   = cols;
    g_ncells = rows * cols;
    g_consumed = g_ncells;                          /* sentinel just above max valid cell */
    g_bits_per_cell = bits_needed(g_ncells + 1);    /* must encode 0..g_ncells (CONSUMED) */
    if (g_bits_per_cell < 5) g_bits_per_cell = 5;   /* historical minimum; keeps small grids in known territory */
    g_cell_mask = (g_bits_per_cell >= 64) ? ~0ULL : ((1ULL << g_bits_per_cell) - 1);

    g_all_cells    = (g_ncells == 64) ? ~0ULL : ((1ULL << g_ncells) - 1);
    g_col0_mask    = 0;
    g_collast_mask = 0;
    for (int r = 0; r < g_rows; r++) {
        g_col0_mask    |= 1ULL << (r * g_cols + 0);
        g_collast_mask |= 1ULL << (r * g_cols + (g_cols - 1));
    }

    build_adj();
}

/* ========================================================================
 * ZOBRIST HASHING  (comment out USE_ZOBRIST below to revert to pack5 keys)
 *
 * Assigns a random 64-bit value to each (pushability_mask, cell) pair.
 * XOR-ing those values together gives an order-independent state key:
 * two blocks with identical pushability masks at swapped positions produce
 * the same key, collapsing duplicate states without sorting.
 *
 * Collision probability ≈ states_visited / 2^64 — negligible in practice.
 *
 * Keys the push solvers' tables (every state width).
 * ======================================================================== */
#define USE_ZOBRIST

#ifdef USE_ZOBRIST
/* z_block[mask][cell]: mask in [0,15], cell in [0, MAX_NCELLS]
 * Allocated for the worst case; only [0, g_ncells] are populated. */
static uint64_t z_player[MAX_NCELLS];
static uint64_t z_block [16][MAX_NCELLS + 1];
static uint64_t z_hole  [MAX_NCELLS];   /* indexed by the hole's CELL: keys of states with an extra (filled) hole then equal the keys without it */
/* A second, independent key set (review M7): the multi-start label table
 * checks it on every key match, so two different states whose primary keys
 * collide can never share a label slot and its cached expansion (which could
 * invent a shortcut and prune a valid level).  Generated by splitmix64, not by
 * the xorshift stream above, so no XOR relation among the primary keys carries
 * over to these. */
static uint64_t z2_player[MAX_NCELLS];
static uint64_t z2_block [16][MAX_NCELLS + 1];
static uint64_t z2_hole  [MAX_NCELLS];

static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void zobrist_init(void) {
    /* xorshift64 with fixed seed for reproducibility */
    uint64_t s = 0x9E3779B97F4A7C15ULL;
#define ZN() (s ^= s << 13, s ^= s >> 7, s ^= s << 17, s)
    for (int c = 0; c < g_ncells;    c++) z_player[c]       = ZN();
    for (int m = 0; m < 16;          m++)
        for (int c = 0; c <= g_ncells; c++) z_block[m][c]   = ZN();
    for (int h = 0; h < MAX_NCELLS;  h++) z_hole[h]          = ZN();
#undef ZN
    uint64_t x = 0x243F6A8885A308D3ULL;   /* pi */
    for (int c = 0; c < MAX_NCELLS;  c++) z2_player[c] = splitmix64_next(&x);
    for (int m = 0; m < 16;          m++)
        for (int c = 0; c <= MAX_NCELLS; c++) z2_block[m][c] = splitmix64_next(&x);
    for (int h = 0; h < MAX_NCELLS;  h++) z2_hole[h] = splitmix64_next(&x);
}

/* Order-independent 64-bit state key.  Two blocks with equal pushability
 * masks contribute the same XOR regardless of which index they occupy. */
static inline uint64_t zpack64(int pl, const int *bp,
                                const uint8_t *masks, int nb,
                                int hm, int nh, const int *hole_pos) {
    uint64_t h = z_player[pl];
    for (int i = 0; i < nb; i++) h ^= z_block[masks[i] & 0xF][bp[i]];
    for (int j = 0; j < nh; j++) if (hm & (1 << j)) h ^= z_hole[hole_pos[j]];
    return h;
}
uint64_t sokoban_zobrist_block(int mask, int cell) { return z_block[mask & 0xF][cell]; }
/* The same key over the second set (the multi-start collision check). */
static inline uint64_t zpack64_2(int pl, const int *bp,
                                 const uint8_t *masks, int nb,
                                 int hm, int nh, const int *hole_pos) {
    uint64_t h = z2_player[pl];
    for (int i = 0; i < nb; i++) h ^= z2_block[masks[i] & 0xF][bp[i]];
    for (int j = 0; j < nh; j++) if (hm & (1 << j)) h ^= z2_hole[hole_pos[j]];
    return h;
}
#endif /* USE_ZOBRIST */

void sokoban_init(void) {
#ifdef USE_ZOBRIST
    zobrist_init();
#endif
}

/* Soft per-solve heap-size cap; 0 = disabled.  Solvers return -3 if
 * heap_sz grows past this value.  See sokoban_set_heap_cap() in header. */
static int g_heap_cap = 0;

void sokoban_set_heap_cap(int n) { g_heap_cap = (n > 0) ? n : 0; }

/* Fatal errors (allocation failure): never a silent NULL dereference.  The
 * default handler prints the message and exits with code 5 (the protocol's
 * `error` status); backsearch installs its own that also prints a SUMMARY line
 * with status "error" before exiting 5.  The handler must not return. */
static void sok_default_fatal(const char *msg) { fprintf(stderr, "fatal: %s\n", msg); fflush(stderr); exit(5); }
static void (*g_sok_fatal)(const char *) = sok_default_fatal;
void sokoban_set_fatal_handler(void (*fn)(const char *msg)) { g_sok_fatal = fn ? fn : sok_default_fatal; }
static void sok_oom(const char *what, size_t bytes) {
    char msg[160];
    snprintf(msg, sizeof msg, "out of memory: %s (%zu bytes)", what, bytes);
    g_sok_fatal(msg);
    abort();   /* a handler that returns is a bug */
}
static void *sok_xcalloc(size_t n, size_t sz, const char *what) {
    void *p = calloc(n, sz);
    if (!p && n && sz) sok_oom(what, n * sz);
    return p;
}
static void *sok_xmalloc(size_t sz, const char *what) {
    void *p = malloc(sz);
    if (!p && sz) sok_oom(what, sz);
    return p;
}
static void *sok_xrealloc(void *old, size_t sz, const char *what) {
    void *p = realloc(old, sz);
    if (!p && sz) sok_oom(what, sz);   /* old stays valid, but we exit anyway */
    return p;
}

/* State space size: g_ncells * (g_ncells+1)^nb * 2^nh (as int64_t to detect overflow) */
static int64_t state_space_size(int nb, int nh) {
    int64_t s = g_ncells;
    for (int i = 0; i < nb; i++) s *= g_ncells + 1;
    for (int i = 0; i < nh; i++) s *= 2;
    return s;
}

/* ---- Threshold: direct indexing vs hash table ----
 * Direct indexing is used when the state space fits comfortably in memory.
 * 8M states = 16 MB visited array — fits well within realistic working sets.
 * Above this, the original hash table approach is used.
 */
#ifndef DIRECT_LIMIT
#define DIRECT_LIMIT (1 << 23)   /* 8 M states */
#endif

/* ========================================================================
 * DIRECT-INDEXED SOLVER  (small state spaces, nb <= ~3)
 *
 * Mixed-radix encoding maps each state to a unique array index.
 * uint16_t visited array with generation counter — fits in L2 cache
 * for 3 blocks (~2.8 MB).  No hashing, no collisions.
 * ======================================================================== */

typedef struct {
    uint16_t *visited;  /* generation-counter visited array */
    uint32_t *qs;       /* BFS queue: packed states */
    uint32_t *qused;    /* BFS queue: used push direction bitmasks */
    uint16_t  gen;      /* current generation counter */
    int       vis_cap;  /* allocated visited array capacity */
} DirectState;

static _Thread_local DirectState *ds_tls;

static DirectState *ds_get(int state_space) {
    if (!ds_tls) {
        ds_tls = sok_xcalloc(1, sizeof(DirectState), "direct solver state");
        ds_tls->gen = 1;
    }
    if (state_space > ds_tls->vis_cap) {
        free(ds_tls->visited);
        free(ds_tls->qs);
        free(ds_tls->qused);
        ds_tls->visited = sok_xcalloc(state_space, sizeof(uint16_t), "direct solver visited");
        ds_tls->qs      = sok_xmalloc(state_space * sizeof(uint32_t), "direct solver queue");
        ds_tls->qused   = sok_xmalloc(state_space * sizeof(uint32_t), "direct solver queue");
        ds_tls->vis_cap = state_space;
    }
    return ds_tls;
}

static void ds_clear(DirectState *ds) {
    if (++ds->gen == 0) {
        memset(ds->visited, 0, ds->vis_cap * sizeof(uint16_t));
        ds->gen = 1;
    }
}

static inline int ds_mark(DirectState *ds, uint32_t state) {
    if (ds->visited[state] == ds->gen) return 0;
    ds->visited[state] = ds->gen;
    return 1;
}

/* Mixed-radix packing: state = player + g_ncells*(block[0] + (g_ncells+1)*(block[1] + ...)) */
static inline uint32_t mr_pack(int pl, const int *bp, int nb, int hm) {
    uint32_t s = (uint32_t)hm;
    for (int i = nb - 1; i >= 0; i--)
        s = s * (g_ncells + 1) + (uint32_t)bp[i];
    return s * g_ncells + (uint32_t)pl;
}

static inline void mr_unpack(uint32_t s, int *pl, int *bp, int nb, int *hm) {
    *pl = (int)(s % g_ncells);
    s /= g_ncells;
    for (int i = 0; i < nb; i++) {
        bp[i] = (int)(s % (g_ncells + 1));
        s /= (g_ncells + 1);
    }
    *hm = (int)s;
}

static int solve_direct(const Puzzle *pz, uint8_t *used_dirs, int ss_size) {
    int nb = pz->num_blocks;
    int nh = pz->num_holes;

    /* Precompute strides for incremental state updates.
     * stride[i] = g_ncells * (g_ncells+1)^i  (coefficient of block i in packed state)
     * hm_stride = g_ncells * (g_ncells+1)^nb (coefficient of hole_mask)
     */
    int stride[MAX_BLOCKS];
    if (nb > 0) {
        stride[0] = g_ncells;
        for (int i = 1; i < nb; i++) stride[i] = stride[i - 1] * (g_ncells + 1);
    }
    int hm_stride = (nb > 0) ? stride[nb - 1] * (g_ncells + 1) : g_ncells;

    DirectState *ds = ds_get(ss_size);
    ds_clear(ds);

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    uint32_t st = mr_pack(pz->player_start, ib, nb, ihm);
    ds_mark(ds, st);

    int qh = 0, qt = 0, ql = 0, dist = -1;
    ds->qs[qt] = st;
    if (used_dirs) ds->qused[qt] = 0;
    qt++;

    const int      exit_pos = pz->exit_pos;
    const uint64_t walls    = pz->walls;

    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        uint32_t cur = ds->qs[qh];
        uint32_t cur_used = used_dirs ? ds->qused[qh] : 0;
        qh++;

        int pl, bp[MAX_BLOCKS], hm;
        mr_unpack(cur, &pl, bp, nb, &hm);

        uint64_t block_occ = 0;
        for (int i = 0; i < nb; i++)
            if (bp[i] < g_ncells) block_occ |= (1ULL << bp[i]);

        uint64_t active_holes = 0;
        for (int h = 0; h < nh; h++)
            if (hm & (1 << h)) active_holes |= (1ULL << pz->hole_pos[h]);

        uint64_t blocked = walls | active_holes;

        for (int d = 0; d < 4; d++) {
            int np = g_adj[pl][d];
            if (np < 0) continue;
            if (blocked & (1ULL << np)) continue;

            if (block_occ & (1ULL << np)) {
                /* Push */
                int bi = -1;
                for (int b = 0; b < nb; b++)
                    if (bp[b] == np) { bi = b; break; }
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int bnp = g_adj[np][d];
                if (bnp < 0) continue;
                if (walls     & (1ULL << bnp)) continue;
                if (block_occ & (1ULL << bnp)) continue;

                if (np == exit_pos) {
                    if (used_dirs) {
                        uint32_t u = cur_used | (1u << (bi * 4 + d));
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (u >> (i * 4)) & 0xF;
                    }
                    return dist + 1;
                }

                int delta = (np - pl);
                int ih = 0;
                if (active_holes & (1ULL << bnp)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == bnp && (hm & (1 << h))) {
                            ih = 1;
                            delta += (g_consumed - bp[bi]) * stride[bi]
                                   - (1 << h) * hm_stride;
                            break;
                        }
                    }
                }
                if (!ih) delta += (bnp - bp[bi]) * stride[bi];

                uint32_t ns = (uint32_t)((int)cur + delta);
                if (ds_mark(ds, ns)) {
                    ds->qs[qt] = ns;
                    if (used_dirs)
                        ds->qused[qt] = cur_used | (1u << (bi * 4 + d));
                    qt++;
                }

            } else {
                /* Free move */
                if (np == exit_pos) {
                    if (used_dirs) {
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (cur_used >> (i * 4)) & 0xF;
                    }
                    return dist + 1;
                }

                uint32_t ns = (uint32_t)((int)cur + (np - pl));
                if (ds_mark(ds, ns)) {
                    ds->qs[qt] = ns;
                    if (used_dirs) ds->qused[qt] = cur_used;
                    qt++;
                }
            }
        }
    }
    return -1;
}

/* ========================================================================
 * HASH TABLE SOLVER  (large state spaces, nb >= 4)
 *
 * Original 5-bit packing with splitmix64 hash + linear probing.
 * 384 MB per thread but handles arbitrarily large state spaces.
 * Still benefits from adj table and merged blocked mask.
 * ======================================================================== */

/* Every table-size macro below can be overridden with -D for a test build
 * ("knob build").  build_pgo.sh folds such overrides into SRC_HASH (KNOBS=...),
 * and `--version` prints the effective values (KNOBS line), so a knob build can
 * never pass for the release binary. */
#ifndef HT_SIZE
#define HT_SIZE  (1 << 24)          /* 16 M slots   */
#endif
#define HT_MASK  (HT_SIZE - 1)
#ifndef QSZ
#define QSZ      (1 << 24)          /* 16 M entries */
#endif

/* Linear-probing scan depth for the dedup hash tables.  Bumped from the
 * original 128 after observing depth over-counts caused by silent state-
 * drops when a probe chain saturated mid-BFS — a 9-block 6x6 reachable
 * via SA needed > 4K probes to terminate correctly; at the old limit it
 * returned 275 (wrong), at 4K it bailed via -3, at 64K it returns 191.
 *
 * Tables are 4–16 M slots, so 64K is still well-bounded.  Probe-failure
 * is now flagged via HashState*::probe_failed and propagated as -3, so
 * even if a future puzzle exceeds this limit the answer can never be
 * silently wrong — the solver bails instead. */
#ifndef HT_PROBE_LIMIT
#define HT_PROBE_LIMIT 65536
#endif

typedef struct {
    uint64_t htk   [HT_SIZE];   /* stored keys        — 128 MB */
    uint32_t ht_gen[HT_SIZE];   /* generation stamps  —  64 MB */
    uint32_t ht_seq;
    uint64_t qs    [QSZ];       /* BFS queue: states  — 128 MB */
    uint32_t qused [QSZ];       /* BFS queue: used push dirs —  64 MB */
} HashState;

static _Thread_local HashState *hs_tls;

static HashState *hs_get(void) {
    if (!hs_tls) { hs_tls = sok_xcalloc(1, sizeof *hs_tls, "hash solver state"); hs_tls->ht_seq = 1; }
    return hs_tls;
}

static void hs_clear(HashState *hs) {
    if (++hs->ht_seq == 0) {
        memset(hs->ht_gen, 0, sizeof(hs->ht_gen));
        hs->ht_seq = 1;
    }
}

static inline uint64_t h64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline int hs_mark(HashState *hs, uint64_t k) {
    uint64_t h = h64(k) & HT_MASK;
    for (int i = 0; i < HT_PROBE_LIMIT; i++) {
        uint32_t idx = (uint32_t)((h + i) & HT_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) {
            hs->ht_gen[idx] = hs->ht_seq; hs->htk[idx] = k; return 1;
        }
        if (hs->htk[idx] == k) return 0;
    }
    return 0;  /* table full — treat as already visited */
}

/* g_bits_per_cell-wide packing: player[low] block0 block1 ... hm[top] */
static inline uint64_t pack5(int pl, const int *bp, int nb, int hm) {
    uint64_t s = (uint64_t)pl;
    int sh = g_bits_per_cell;
    for (int i = 0; i < nb; i++) { s |= ((uint64_t)bp[i] << sh); sh += g_bits_per_cell; }
    s |= ((uint64_t)hm << sh);
    return s;
}

static int solve_hash(const Puzzle *pz, uint8_t *used_dirs) {
    HashState *hs = hs_get();
    hs_clear(hs);

    int nb = pz->num_blocks;
    int nh = pz->num_holes;

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    uint64_t st = pack5(pz->player_start, ib, nb, ihm);
    hs_mark(hs, st);

    int qh = 0, qt = 0, ql = 0, dist = -1;
    hs->qs[qt] = st;
    if (used_dirs) hs->qused[qt] = 0;
    qt++;

    const int      exit_pos = pz->exit_pos;
    const uint64_t walls    = pz->walls;

    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        uint64_t cur = hs->qs[qh];
        uint32_t cur_used = used_dirs ? hs->qused[qh] : 0;
        qh++;

        /* Unpack */
        int pl = (int)(cur & g_cell_mask), sh = g_bits_per_cell, bp[MAX_BLOCKS];
        for (int i = 0; i < nb; i++) { bp[i] = (int)((cur >> sh) & g_cell_mask); sh += g_bits_per_cell; }
        int hm = (int)(cur >> sh);   /* hm is at the top — no mask needed */

        uint64_t block_occ = 0;
        for (int i = 0; i < nb; i++)
            if (bp[i] < g_ncells) block_occ |= (1ULL << bp[i]);

        uint64_t active_holes = 0;
        for (int h = 0; h < nh; h++)
            if (hm & (1 << h)) active_holes |= (1ULL << pz->hole_pos[h]);

        uint64_t blocked = walls | active_holes;

        for (int d = 0; d < 4; d++) {
            int np = g_adj[pl][d];
            if (np < 0) continue;
            if (blocked & (1ULL << np)) continue;

            if (block_occ & (1ULL << np)) {
                /* Push */
                int bi = -1;
                for (int b = 0; b < nb; b++)
                    if (bp[b] == np) { bi = b; break; }
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int bnp = g_adj[np][d];
                if (bnp < 0) continue;
                if (walls     & (1ULL << bnp)) continue;
                if (block_occ & (1ULL << bnp)) continue;

                if (np == exit_pos) {
                    if (used_dirs) {
                        uint32_t u = cur_used | (1u << (bi * 4 + d));
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (u >> (i * 4)) & 0xF;
                    }
                    return dist + 1;
                }

                /* Build new state via in-place bit modification */
                int new_bpos = bnp;
                int nhm = hm;
                if (active_holes & (1ULL << bnp)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == bnp && (hm & (1 << h))) {
                            new_bpos = g_consumed;
                            nhm &= ~(1 << h);
                            break;
                        }
                    }
                }
                uint64_t ns = (cur & ~g_cell_mask) | (uint64_t)np;
                int bsh = g_bits_per_cell * (bi + 1);
                ns = (ns & ~(g_cell_mask << bsh)) | ((uint64_t)new_bpos << bsh);
                if (nhm != hm) {
                    int hmsh = g_bits_per_cell * (nb + 1);
                    uint64_t hm_mask = ((uint64_t)((1 << nh) - 1)) << hmsh;
                    ns = (ns & ~hm_mask) | ((uint64_t)nhm << hmsh);
                }

                if (hs_mark(hs, ns)) {
                    if (qt >= QSZ) return -2;
                    hs->qs[qt] = ns;
                    if (used_dirs)
                        hs->qused[qt] = cur_used | (1u << (bi * 4 + d));
                    qt++;
                }

            } else {
                /* Free move */
                if (np == exit_pos) {
                    if (used_dirs) {
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (cur_used >> (i * 4)) & 0xF;
                    }
                    return dist + 1;
                }

                uint64_t ns = (cur & ~g_cell_mask) | (uint64_t)np;
                if (hs_mark(hs, ns)) {
                    if (qt >= QSZ) return -2;
                    hs->qs[qt] = ns;
                    if (used_dirs) hs->qused[qt] = cur_used;
                    qt++;
                }
            }
        }
    }
    return -1;
}

/* ========================================================================
 * HELPERS FOR PUSH-BASED DIJKSTRA SOLVERS
 *
 * Bitmask BFS: all g_ncells cells fit in a uint64_t (capped at 64 = 8x8),
 * so each level of BFS expands with 4 shift operations rather than
 * iterating cell-by-cell through the adj table.  Directions:
 *   Up    — shift right by g_cols  (high bits dropped naturally)
 *   Down  — shift left  by g_cols  (out-of-region bits masked by g_all_cells)
 *   Right — shift left  by 1, masking last column to prevent row wraparound
 *   Left  — shift right by 1, masking first column to prevent row wraparound
 * ======================================================================== */

/* walk_dists_from: BFS distances from start, recorded only for cells in
 * `interesting` (the only cells the caller will query).  out[i] = distance,
 * or -1 if unreachable / not interesting.  Terminates as soon as every
 * reachable interesting cell has settled. */
static void walk_dists_from(uint64_t blocked, int start, uint64_t interesting,
                            int8_t *out) {
    memset(out, -1, (size_t)g_ncells);
    if ((blocked >> start) & 1) return;
    const uint64_t col0    = g_col0_mask;
    const uint64_t collast = g_collast_mask;
    const int      shc     = g_cols;
    uint64_t free_mask = ~blocked & g_all_cells;
    uint64_t reached   = 1ULL << start;
    uint64_t frontier  = reached;
    out[start] = 0;
    uint64_t remaining = interesting & free_mask & ~reached;
    int8_t dist = 0;
    while (frontier && remaining) {
        dist++;
        uint64_t nxt = (frontier >> shc)
                     | (frontier << shc)
                     | ((frontier & ~collast) << 1)
                     | ((frontier & ~col0)    >> 1);
        frontier = nxt & free_mask & ~reached;
        reached |= frontier;
        for (uint64_t tmp = frontier & remaining; tmp; tmp &= tmp - 1)
            out[__builtin_ctzll(tmp)] = dist;
        remaining &= ~frontier;
    }
}

/* Same as walk_dists_from() but stops after `lim` steps: cells farther than
 * lim (or unreachable) stay -1.  Used by the cutoff solver, where a cell that
 * is more than `slack` moves away can never lead to a state within the cutoff. */
static void walk_dists_from_lim(uint64_t blocked, int start, uint64_t interesting,
                                int8_t *out, int lim) {
    memset(out, -1, (size_t)g_ncells);
    if ((blocked >> start) & 1) return;
    const uint64_t col0    = g_col0_mask;
    const uint64_t collast = g_collast_mask;
    const int      shc     = g_cols;
    uint64_t free_mask = ~blocked & g_all_cells;
    uint64_t reached   = 1ULL << start;
    uint64_t frontier  = reached;
    out[start] = 0;
    uint64_t remaining = interesting & free_mask & ~reached;
    int8_t dist = 0;
    while (frontier && remaining && dist < lim) {
        dist++;
        uint64_t nxt = (frontier >> shc)
                     | (frontier << shc)
                     | ((frontier & ~collast) << 1)
                     | ((frontier & ~col0)    >> 1);
        frontier = nxt & free_mask & ~reached;
        reached |= frontier;
        for (uint64_t tmp = frontier & remaining; tmp; tmp &= tmp - 1)
            out[__builtin_ctzll(tmp)] = dist;
        remaining &= ~frontier;
    }
}

/* Neighbor mask: all cells orthogonally adjacent to a set bit of m. */
static inline uint64_t nbr_mask(uint64_t m) {
    return ((m >> g_cols)
          | (m << g_cols)
          | ((m & ~g_collast_mask) << 1)
          | ((m & ~g_col0_mask)    >> 1)) & g_all_cells;
}

/* ========================================================================
 * MANDATORY-HOLE PRUNE  (ported from ~/solver.c "solver6")
 *
 * A hole is MANDATORY if the player cannot reach the exit while it stays
 * open (flood from player over walls+this-hole only; blocks and all other
 * holes treated as passable — the most permissive test, so it can only
 * under-report, never falsely claim a hole mandatory).  Such a hole must
 * be filled by a block in any solution.  A successor is provably dead when
 * the number of blocks that could still ever reach an unfilled mandatory
 * hole is smaller than the number of such holes.
 *
 * "Block can still reach hole" uses the relaxed push closure obstructed by
 * walls only (block_closure), a superset of the cells the block can ever
 * occupy.  Precomputed per (mask, cell) into closure_tab so the per-state
 * test is a table lookup and an AND.  Both are relaxations in the direction
 * that only drops prunes, so the reported optimum is exact.
 *
 * State lives thread-local: setup runs once per solve call (cheap — only
 * builds closure_tab when at least one mandatory hole exists).
 * ======================================================================== */

static int g_prune_mand = 0;   /* runtime toggle (see sokoban_set_hole_prune); default OFF */

static int g_decision_only = 0;
void sokoban_set_decision_only(int on) { g_decision_only = on; }

void sokoban_set_hole_prune(int on) { g_prune_mand = on ? 1 : 0; }
/* Cells whose holes are forced mandatory without a reachability check (union
 * with the check).  Set via sokoban_set_forced_mandatory; 0 = none. */
static uint64_t g_forced_mand_cells = 0;
void sokoban_set_forced_mandatory(uint64_t cell_mask) { g_forced_mand_cells = cell_mask; }

static _Thread_local uint32_t g_mand_hmask;                    /* hole-idx mask of mandatory holes */
static _Thread_local uint64_t g_closure_tab[16][MAX_NCELLS];  /* [mask][cell] relaxed push closure */
/* Cache validity for g_closure_tab.  The table depends only on the wall set
 * (block positions never affect it), so it can be reused across the many
 * consecutive backward-search solves whose walls are unchanged.  g_closure_walls
 * is the wall set the table was built for; g_closure_masks is the set of block
 * push-masks currently populated for that wall set.  A NULL grid (walls all-ones
 * sentinel is impossible since walls come from g_active_mask) can't collide because
 * we also gate on the grid dims via sokoban_set_grid resetting the cache. */
static _Thread_local uint64_t g_closure_walls;
static _Thread_local uint16_t g_closure_masks;
static _Thread_local int      g_closure_valid;   /* 0 until first build this grid */
static _Thread_local int      g_closure_ncells;  /* grid the cache was built for */

/* Reachable-cell flood from `start`, treating `blocked` cells as impassable. */
static uint64_t reach_flood(uint64_t blocked, int start) {
    uint64_t freem = ~blocked & g_all_cells;
    if (!((freem >> start) & 1)) return 0;
    uint64_t comp = 1ULL << start;
    for (;;) {
        uint64_t nx = (comp | nbr_mask(comp)) & freem;
        if (nx == comp) return comp;
        comp = nx;
    }
}

/* Relaxed push closure of a block at `start` with push-mask `mask`: the set
 * of cells it could ever occupy if only walls obstructed (other blocks
 * transparent).  A push in direction d needs an open stand cell behind and
 * an open landing cell ahead. */
static uint64_t block_closure(int start, int mask, uint64_t walls) {
    uint64_t set = 1ULL << start;
    for (;;) {
        uint64_t add = 0;
        for (uint64_t t = set; t; t &= t - 1) {
            int c = __builtin_ctzll(t);
            for (int d = 0; d < 4; d++) {
                if (!(mask & (1 << d))) continue;
                int stand = g_adj[c][d ^ 2];   /* player stands opposite the push */
                if (stand < 0 || (walls & (1ULL << stand))) continue;
                int lnd = g_adj[c][d];          /* block lands ahead */
                if (lnd < 0 || (walls & (1ULL << lnd))) continue;
                add |= 1ULL << lnd;
            }
        }
        uint64_t ns = set | add;
        if (ns == set) return set;
        set = ns;
    }
}

/* Per-solve setup: classify mandatory holes and, if any exist, build the
 * push-closure table for the block masks actually present. */
static void mand_setup(const Puzzle *pz) {
    g_mand_hmask = 0;
    if (!g_prune_mand || pz->num_holes == 0) return;

    const uint64_t walls = pz->walls;
    uint32_t mh = 0;
    for (int h = 0; h < pz->num_holes; h++) {
        int hc = pz->hole_pos[h];
        if (hc == pz->player_start || hc == pz->exit_pos) continue;
        if ((g_forced_mand_cells >> hc) & 1) { mh |= 1u << h; continue; }  /* forced: skip the reach check */
        uint64_t r = reach_flood(walls | (1ULL << hc), pz->player_start);
        if (!((r >> pz->exit_pos) & 1)) mh |= 1u << h;
    }
    g_mand_hmask = mh;
    if (!mh) return;

    uint16_t masks_present = 0;
    for (int i = 0; i < pz->num_blocks; i++)
        masks_present |= 1u << (pz->block_pushable[i] & 0xF);

    /* Reuse the closure table when the walls are unchanged from the previous
     * solve; block moves never invalidate it.  On a wall change, rebuild every
     * present mask; on the same walls but a newly-seen mask, build only that. */
    uint16_t todo;
    if (!g_closure_valid || walls != g_closure_walls || g_closure_ncells != g_ncells) {
        g_closure_walls  = walls;
        g_closure_masks  = masks_present;
        g_closure_ncells = g_ncells;
        g_closure_valid  = 1;
        todo = masks_present;
    } else {
        todo = masks_present & ~g_closure_masks;
        g_closure_masks |= masks_present;
    }
    for (int m = 1; m < 16; m++) {
        if (!(todo & (1u << m))) continue;
        for (int c = 0; c < g_ncells; c++)
            g_closure_tab[m][c] = ((walls >> c) & 1) ? 0 : block_closure(c, m, walls);
    }
}

/* Returns 1 if the successor in which block `bi` moves to `new_bpos` (or is
 * consumed, new_bpos >= g_ncells) leaving hole mask `nhm` is provably dead
 * because too few blocks can still reach the unfilled mandatory holes. */
static inline int mand_dead(const Puzzle *pz, const int *bp, int nb,
                            int bi, int new_bpos, uint32_t nhm) {
    if (!g_mand_hmask) return 0;
    uint32_t mleft = g_mand_hmask & nhm;
    if (!mleft) return 0;

    uint64_t mcells = 0;
    for (uint32_t t = mleft; t; t &= t - 1)
        mcells |= 1ULL << pz->hole_pos[__builtin_ctz(t)];

    int need = __builtin_popcount(mleft), have = 0;
    for (int j = 0; j < nb && have < need; j++) {
        int p = (j == bi) ? new_bpos : bp[j];
        if (p >= g_ncells) continue;             /* consumed */
        uint8_t m = pz->block_pushable[j] & 0xF;
        if (m == 0) continue;                    /* immovable: reaches nothing */
        if (g_closure_tab[m][p] & mcells) have++;
    }
    return have < need;
}

/* Per-POP deadlock context, computed once for a popped state and reused across
 * all of its candidate pushes.  Captures the invariants that a non-hole-filling
 * push cannot change: the unfilled-mandatory cell set (mcells), how many must be
 * filled (need), and which/how-many blocks can currently reach them.  hm is the
 * popped state's hole mask. */
typedef struct {
    uint64_t mcells;    /* cells of unfilled mandatory holes in this state */
    uint32_t canreach;  /* bit j set: block j's closure hits mcells        */
    int      need;      /* # unfilled mandatory holes                      */
    int      have;      /* # blocks reaching mcells (exact)                */
} MandCtx;

static inline void mand_pop_ctx(const Puzzle *pz, const int *bp, int nb,
                                int hm, MandCtx *mc) {
    mc->mcells = 0; mc->canreach = 0; mc->need = 0; mc->have = 0;
    if (!g_mand_hmask) return;
    uint32_t mleft = g_mand_hmask & (uint32_t)hm;
    if (!mleft) return;
    uint64_t mcells = 0;
    for (uint32_t t = mleft; t; t &= t - 1)
        mcells |= 1ULL << pz->hole_pos[__builtin_ctz(t)];
    mc->mcells = mcells;
    mc->need   = __builtin_popcount(mleft);
    int have = 0; uint32_t cr = 0;
    for (int j = 0; j < nb; j++) {
        int p = bp[j];
        if (p >= g_ncells) continue;
        uint8_t m = pz->block_pushable[j] & 0xF;
        if (m == 0) continue;
        if (g_closure_tab[m][p] & mcells) { cr |= 1u << j; have++; }
    }
    mc->have = have; mc->canreach = cr;
}

/* O(1) successor deadlock test using the per-pop context.  Exact: returns the
 * same verdict as mand_dead().  Correctness rests on closure monotonicity — a
 * real push satisfies the relaxed edge condition, so new_bpos in closure(bpos)
 * and hence closure(new_bpos) subset closure(bpos): a push can only shrink a
 * block's reach, never grow it, and only the moved block bi changes.  When the
 * push fills a MANDATORY hole (need and mcells shrink) we fall back to the full
 * recompute — the rare productive case.  ch = filled hole index, or -1. */
static inline int mand_dead_ctx(const Puzzle *pz, const MandCtx *mc,
                                const int *bp, int nb, int bi,
                                int new_bpos, uint32_t nhm, int ch) {
#ifdef MAND_OLD
    return mand_dead(pz, bp, nb, bi, new_bpos, nhm);   /* A/B baseline: force old path */
#endif
    int r;
    if (!g_mand_hmask) {
        r = 0;
    } else if (ch >= 0 && ((g_mand_hmask >> ch) & 1)) {
        r = mand_dead(pz, bp, nb, bi, new_bpos, nhm);      /* mandatory fill: recount */
    } else if (mc->need == 0) {
        r = 0;
    } else if (mc->have > mc->need) {
        r = 0;                                             /* slack: no push can kill it */
    } else {
        int old_r = (mc->canreach >> bi) & 1;
        int new_r = 0;
        if (old_r && new_bpos < g_ncells) {
            uint8_t m = pz->block_pushable[bi] & 0xF;
            new_r = (g_closure_tab[m][new_bpos] & mc->mcells) != 0;
        }
        r = (mc->have - old_r + new_r) < mc->need;
    }
#ifdef MAND_ASSERT
    int slow = mand_dead(pz, bp, nb, bi, new_bpos, nhm);
    if (r != slow) {
        fprintf(stderr, "[MAND_ASSERT] mismatch fast=%d slow=%d  bi=%d new_bpos=%d "
                "ch=%d need=%d have=%d\n", r, slow, bi, new_bpos, ch, mc->need, mc->have);
        abort();
    }
#endif
    return r;
}

/* ========================================================================
 * PUSH-BASED DIJKSTRA SOLVER  (one implementation for every state width)
 *
 * State: (player cell, block cells, hole mask).  The exact player cell is
 * tracked per heap entry; the packed state holds the blocks and the hole
 * mask only, and is used for nothing but unpacking.  The table key is the
 * 64-bit Zobrist key of (player cell, blocks with their push masks, open
 * holes): it is independent of the packing, so the hash tables, the
 * reference tables (parent-table reuse, XOR translation) and the multi-start
 * labels below are shared by all widths, and a table exported by a solve of
 * one width serves a child of another.
 *
 * Edge cost = walk_distance(player_pos, push_from_cell) + 1.
 * Dijkstra with lazy deletion — edge weights in [1, g_ncells].
 *
 * Priority queue: Dial's bucket queue.  Dijkstra pops priorities in
 * non-decreasing order and every edge weight is < PQ64_NBUCKETS, so all
 * pending priorities fit in a circular window of PQ64_NBUCKETS buckets
 * indexed by (prio & PQ64_BMASK).  O(1) push/pop vs O(log n) heap sifts.
 *
 * Hash table: one 16-byte slot per state (key + generation + cost) so a
 * probe touches a single cache line instead of three parallel arrays.
 * Queue entries carry their slot index, making the stale-entry check on
 * pop a single load instead of a re-probe.
 *
 * The solvers themselves (full solve, cutoff solve, multi-start solve) are
 * written once, in the PUSH SOLVER TEMPLATE at the end of this file, and
 * instantiated for three packed-state widths (see PUSH SOLVER INSTANCES):
 *   w1  one uint64_t: block i at bits [b*i, b*i+b) (b = g_bits_per_cell),
 *       the hole mask above the last block; used when b*nb + nh <= 64
 *       (5x5: nb <= 12 with nh <= 4; 6x6: nb <= 10 with no holes);
 *   w2  16 block bytes + a 32-bit hole mask (nb <= 16);
 *   w4  32 block bytes + a 32-bit hole mask (nb <= 32 = MAX_BLOCKS).
 * Every state on a grid of <= MAX_NCELLS cells with <= MAX_BLOCKS blocks and
 * <= 30 holes fits one of them (sok_fits); nothing else is ever solved.
 * ======================================================================== */

#ifndef USE_ZOBRIST
#error "the push solvers key their tables by the Zobrist key (USE_ZOBRIST)"
#endif

#ifndef HTP_SIZE
#define HTP_SIZE  (1 << 24)    /* 16 M hash-table slots.  Doubled from 8M
                                 * after probe-saturation was observed on
                                 * 11-block 6x6 boards even at 65K-probe
                                 * limit — load factor was still high
                                 * enough for clusters to exceed the cap.
                                 * 16M halves the load factor again.
                                 * Filling it to 85% ends a solve with -5
                                 * (SOK_TABLE_FULL: undecided, never a prune). */
#endif
#define HTP_MASK  (HTP_SIZE - 1)
#ifndef HP64_SIZE
#define HP64_SIZE (1 << 20)    /* pending-entry cap, every width (-2 on overflow: undecided) */
#endif

#define PQ64_NBUCKETS 256      /* power of two > max f-step (2 x max edge weight, edge <= 64) */
#define PQ64_BMASK    (PQ64_NBUCKETS - 1)

/* Width selection (test knobs: smaller values push states into the wider
 * instances, so the w2/w4 code can be cross-checked on 5x5 puzzles). */
#ifndef SOK_W1_BITS
#define SOK_W1_BITS 64         /* a state goes to w1 iff b*nb + nh <= SOK_W1_BITS */
#endif
#ifndef SOK_W2_BLOCKS
#define SOK_W2_BLOCKS 16       /* ... else to w2 iff nb <= SOK_W2_BLOCKS, else to w4 */
#endif
#if SOK_W1_BITS > 64 || SOK_W1_BITS < 0
#error "SOK_W1_BITS must be 0..64"
#endif
#if SOK_W2_BLOCKS > 16 || SOK_W2_BLOCKS < 0
#error "SOK_W2_BLOCKS must be 0..16 (w2 holds 16 block cells)"
#endif
#if MAX_BLOCKS > 32
#error "w4 holds 32 block cells"
#endif

typedef struct {
    uint64_t key;
    uint32_t gen;
    int32_t  cost;
} HSlot64;                /* 16 B — one cache-line touch per probe */

/* A Dial bucket.  items points to heap entries of the width that owns the
 * bucket array (HashStatePush64.bq[width index]). */
typedef struct {
    void *items;
    int len, cap;
} BucketV;

/* Two backing tables.  The SMALL one (64K slots, 1 MB) is L2-resident and
 * serves the overwhelming majority of shortcut checks, which touch ~100
 * states.  If a solve inserts more than half of it, the solver aborts and
 * re-runs on the BIG table; the wasted work is bounded by that threshold
 * and the case is rare, so probe saturation (-3) becomes unreachable in
 * practice. */
#ifndef HTP_SMALL_LG2
#define HTP_SMALL_LG2 16
#endif
#define HTP_SMALL_SIZE (1u << HTP_SMALL_LG2)

typedef struct {
    HSlot64 *slot;               /* active table (== slot_small or slot_big) */
    uint32_t ht_mask;            /* active table size - 1                     */
    uint32_t ht_limit;           /* insert count that triggers overflow       */
    uint32_t ht_count;           /* inserts this solve                        */
    int      overflow;           /* set when ht_count reached ht_limit        */
    HSlot64 *slot_small;         /* HTP_SMALL_SIZE slots, 1 MB                */
    HSlot64 *slot_big;           /* HTP_SIZE slots, 256 MB, lazily allocated  */
    uint32_t ht_seq;
    BucketV  bq[3][PQ64_NBUCKETS];   /* Dial bucket queues of the w1 / w2 / w4 solvers (lazy alloc); a solve resets its own */
    int      pq_count;           /* pending entries across buckets */
    int      probe_failed;       /* set when probe limit hit */
    void    *ms_lab;             /* multi-start label vectors, one per SMALL slot (lazy) */
    size_t   ms_lab_bytes;       /* its size, for the generation-wrap clear in hsp64_clear */
    uint32_t *touched;           /* slots inserted this solve (small table only) */
    uint32_t  ntouched;
    int       last_exhaustive;   /* last cutoff solve on the small table ran to completion with no win */
    int       last_ref_used;     /* ... with a reference table installed (its own labels alone are then incomplete) */
    int       last_maxcost;
} HashStatePush64;

static _Thread_local HashStatePush64 *hsp64_tls;

static HashStatePush64 *hsp64_get(void) {
    if (!hsp64_tls) {
        hsp64_tls = sok_xcalloc(1, sizeof *hsp64_tls, "push-solver state");
        hsp64_tls->ht_seq = 1;
        hsp64_tls->slot_small = sok_xcalloc(HTP_SMALL_SIZE, sizeof(HSlot64), "small push table");
        hsp64_tls->touched = sok_xmalloc(HTP_SMALL_SIZE * sizeof(uint32_t), "small push table index");
    }
    return hsp64_tls;
}

/* Select the active table.  big=0 -> small L2 table; big=1 -> 16M table. */
static void hsp64_select(HashStatePush64 *hs, int big) {
    if (big) {
        if (!hs->slot_big) hs->slot_big = sok_xcalloc(HTP_SIZE, sizeof(HSlot64), "big push table");
        hs->slot = hs->slot_big;   hs->ht_mask = HTP_MASK;
        hs->ht_limit = (uint32_t)(HTP_SIZE * 0.85);
    } else {
        hs->slot = hs->slot_small; hs->ht_mask = HTP_SMALL_SIZE - 1;
        hs->ht_limit = HTP_SMALL_SIZE / 2;
    }
}

/* Start a solve: a new generation for every table keyed by ht_seq.  The
 * caller also resets its own width's bucket queue (bq_reset in the template). */
static void hsp64_clear(HashStatePush64 *hs) {
    if (++hs->ht_seq == 0) {
        /* Generation wrap: every table keyed by ht_seq must forget its stamps,
         * the multi-start labels included (they share the counter). */
        memset(hs->slot_small, 0, HTP_SMALL_SIZE * sizeof(HSlot64));
        if (hs->slot_big) memset(hs->slot_big, 0, (size_t)HTP_SIZE * sizeof(HSlot64));
        if (hs->ms_lab) memset(hs->ms_lab, 0, hs->ms_lab_bytes);
        hs->ht_seq = 1;
    }
    hs->pq_count = 0;
    hs->probe_failed = 0;
    hs->ht_count = 0;
    hs->overflow = 0;
    hs->ntouched = 0;
    hs->last_exhaustive = 0;
}

/* Update stored cost for key k.
 * Returns the slot index if new_cost is an improvement (caller should
 * enqueue), or -1 otherwise.  Sets hs->overflow (and returns -1) when the
 * active table has reached its insert limit; the caller must then retry
 * the solve on the big table. */
static inline int hsp64_update(HashStatePush64 *hs, uint64_t k, int new_cost) {
    const uint32_t mask = hs->ht_mask;
    uint64_t h = h64(k) & mask;
    for (int i = 0; i < HT_PROBE_LIMIT; i++) {
        uint32_t idx = (uint32_t)((h + i) & mask);
        HSlot64 *s = &hs->slot[idx];
        if (s->gen != hs->ht_seq) {
            if (hs->ht_count >= hs->ht_limit) { hs->overflow = 1; return -1; }
            hs->ht_count++;
            s->gen  = hs->ht_seq;
            s->key  = k;
            s->cost = new_cost;
            if (hs->slot == hs->slot_small) hs->touched[hs->ntouched++] = idx;
            return (int)idx;
        }
        if (s->key == k) {
            if (new_cost < s->cost) { s->cost = new_cost; return (int)idx; }
            return -1;
        }
    }
    /* Probe limit exceeded — would silently drop the state, which can
     * make BFS over-count depth.  Flag for the solver to bail with -3. */
    hs->probe_failed = 1;
    return -1;
}

/* Candidate push collected in pass 1 (before walk distances are known).  The
 * successor's packed state is built only in pass 2, for the candidates that
 * survive: block bi moves to cell nbp (g_consumed when it fills hole ch). */
typedef struct {
    uint64_t key;     /* successor's Zobrist key                 */
    int16_t  pfr;     /* push-from cell — needs wdist[pfr] >= 0  */
    int16_t  bpos;    /* block's old cell == player landing cell */
    int8_t   bi;      /* the pushed block                        */
    int8_t   nbp;     /* its new cell, or g_consumed             */
    int8_t   ch;      /* hole index it fills, or -1              */
    int8_t   pad;
} PushCand;           /* 16 B */

/* used_dirs is no longer tracked (no caller asks for it, and the old 32-bit
 * mask overflowed for blocks >= 8).  A caller that still passes a buffer gets
 * the conservative superset "every direction may be used". */
static inline void fill_used_dirs(uint8_t *used_dirs, int nb) {
    if (used_dirs) for (int i = 0; i < nb; i++) used_dirs[i] = 0xF;
}

/* Zero the near-goal frontier window and record the cutoff ceiling.  Called
 * by the cutoff solvers before their main loop (and on early return). */
static inline void bfs_tail_reset(BfsProfile *prof, int max_cost) {
    if (!prof) return;
    prof->max_cost_seen = max_cost;
    for (int i = 0; i < BFS_TAIL_W; i++) prof->tail_width[i] = 0;
}
/* ========================================================================
 * HOLE-CROSSING LOWER BOUND
 *
 * The A* term hdist[] is the player's walk distance to the exit over walls
 * only; it ignores that an unfilled hole cannot be stepped on.  When the
 * player cannot reach the exit at all while the unfilled holes stand (flood
 * over walls + holes, blocks passable), every solution must first fill some
 * hole H with some block b, which costs at least:
 *     D(p, b) - 1        walk to a cell next to b        (D: walls-only distance)
 *   + |b - H|_1          pushes moving b to H            (one move each)
 *   + D(H, exit) - 1     from the cell next to H on to the exit
 * These three move sets are disjoint in time, so the minimum over all
 * (unfilled H, unconsumed b that can move toward H) is admissible; if no pair
 * qualifies the state is dead.  D() is computed once per distinct wall set
 * (siblings share it) and the bound is evaluated only when the direct walk
 * check already failed.  MEASURED 2026-09-18: exact, but it prunes only 1-2%
 * of pops and the per-wall-set D() table plus the per-pop flood cost ~25% on
 * the replay corpus, so it is compiled OUT unless -DHOLEBOUND.
 * ======================================================================== */
static _Thread_local uint64_t g_hb_walls = ~0ULL;
static _Thread_local int      g_hb_valid = 0;
static _Thread_local int8_t   g_hb_D[MAX_NCELLS][MAX_NCELLS];

static void hb_prepare(uint64_t walls) {
    if (g_hb_valid && walls == g_hb_walls) return;
    for (int c = 0; c < g_ncells; c++) {
        if (walls >> c & 1) { memset(g_hb_D[c], -1, (size_t)g_ncells); continue; }
        walk_dists_from(walls, c, g_all_cells, g_hb_D[c]);
    }
    g_hb_walls = walls; g_hb_valid = 1;
}

/* Can the player at start reach target over cells not in blocked? */
static inline int hb_reaches(uint64_t blocked, int start, int target) {
    if ((blocked >> start) & 1) return 0;
    uint64_t free_mask = ~blocked & g_all_cells, reached = 1ULL << start, frontier = reached, tgt = 1ULL << target;
    while (frontier) {
        if (reached & tgt) return 1;
        uint64_t nxt = (frontier >> g_cols) | (frontier << g_cols)
                     | ((frontier & ~g_collast_mask) << 1) | ((frontier & ~g_col0_mask) >> 1);
        frontier = nxt & free_mask & ~reached;
        reached |= frontier;
    }
    return (reached & tgt) != 0;
}

#define HB_DEAD 1000
/* Lower bound on remaining moves given the player must cross a hole. */
static inline int hb_bound(const Puzzle *pz, int p, const int *bp, int nb, int hm, int exit_pos) {
    int best = HB_DEAD;
    for (int h = 0; h < pz->num_holes; h++) {
        if (!(hm & (1 << h))) continue;
        int hx = pz->hole_pos[h], dHE = g_hb_D[hx][exit_pos];
        if (dHE < 0) continue;
        int hr = hx / g_cols, hc = hx % g_cols;
        for (int b = 0; b < nb; b++) {
            int bc = bp[b]; if (bc >= g_ncells) continue;
            int mb = pz->block_pushable[b] & 0xF;
            int dr = hr - bc / g_cols, dc = hc - bc % g_cols;
            if (dr < 0 && !(mb & 1)) continue;      /* needs U */
            if (dr > 0 && !(mb & 4)) continue;      /* needs D */
            if (dc > 0 && !(mb & 2)) continue;      /* needs R */
            if (dc < 0 && !(mb & 8)) continue;      /* needs L */
            int dPb = g_hb_D[p][bc]; if (dPb < 0) continue;
            int v = (dPb > 0 ? dPb - 1 : 0) + (dr < 0 ? -dr : dr) + (dc < 0 ? -dc : dc) + (dHE > 0 ? dHE - 1 : 0);
            if (v < best) best = v;
        }
    }
    return best;
}

/* ========================================================================
 * REFERENCE TABLE  (parent's settled labels reused by a child's check)
 *
 * When a generator state P (depth d) is accepted, its shortcut check has run
 * to exhaustion: every forward state Y with f <= d-2 is settled at its optimal
 * cost g_P(Y), and no solution of length <= d-2 passes through any of them,
 * i.e. rest(Y) >= d - g_P(Y) (parity).  A child of P whose forward puzzle is
 * IDENTICAL (same walls, blocks-as-a-set with the same masks, holes) and whose
 * cutoff is d-1+k' ... in general, a child at chain offset k (cutoff d-2+k)
 * can only use Y if g_child(Y) + rest(Y) <= d-2+k, i.e. g_child(Y) <= g_P(Y)
 * + k - 2 ... so with k = 1 (an immediate push-back child) only states it
 * reaches STRICTLY cheaper than P did can matter; everything P already
 * covered is skipped.  Keys are Zobrist over (player, blocks+masks, holes)
 * and independent of the walls, so a lookup is direct.
 * ======================================================================== */
#ifndef REF_LG2
#define REF_LG2 16
#endif
static _Thread_local uint64_t *g_ref_keys;   /* REF_SIZE slots, 0 = empty */
static _Thread_local int32_t  *g_ref_cost;
static _Thread_local int       g_ref_n = 0;   /* live entries (0 = no reference) */
static _Thread_local int       g_ref_k = 0;   /* chain offset k of the child being solved (cut = cut_P + k) */
/* Puzzle delta between the table's owner and the state being solved: cells that
 * are floor now but were walls for the owner.  A continuation that never uses a
 * delta cell is an owner-puzzle path (bounded by the table); one that does must
 * bring the player next to that cell and then to the exit. */
static _Thread_local uint64_t g_ref_delta = 0;
static _Thread_local int8_t   g_ref_dexit[MAX_NCELLS];   /* D(c, exit) for delta cells, walls-only in the current puzzle */
static _Thread_local int      g_ref_cut = 0;             /* the child's cutoff */
static _Thread_local uint32_t g_ref_mask = 0;            /* current table size - 1: sized to the entry count so lookups stay cache-resident */
static _Thread_local int      g_ref_on = 0;              /* prune enabled (0 while a REF_CHECK re-run is in progress) */
static _Thread_local uint64_t g_ref_xor = 0;             /* key translation child -> owner: the child's states that coincide with the owner's
                                                          * differ by a constant (an extra block sitting consumed, an extra hole filled) */
#define REF_SIZE (1u << REF_LG2)
#define REF_MASK g_ref_mask

long long g_refstat[8];   /* 0 lookups, 1 hits, 2 prunes, 3 delta-blocked, 4 builds, 5 build entries, 6 clears, 7 clear entries */
/* The reference is an attached, caller-owned hash table (SokRefTable), built
 * once when a state's table is exported and attached by pointer for every
 * expansion that uses it -- no per-use copying. */
static _Thread_local SokRefTable g_ref_own;   /* storage for sokoban_set_reference (copying API) */
void sokoban_clear_reference(void) {
    g_refstat[6]++; g_refstat[7] += g_ref_n;
    g_ref_keys = NULL; g_ref_cost = NULL; g_ref_mask = 0;
    g_ref_n = 0; g_ref_delta = 0; g_ref_on = 0; g_ref_xor = 0;
}
int sokoban_ref_build(const uint64_t *keys, const int32_t *costs, int n, SokRefTable *t) {
    uint32_t sz = 256; while (sz < 2u * (uint32_t)n) sz <<= 1;
    if (t->cap < sz) { free(t->hk); free(t->hc); t->hk = sok_xmalloc(sz * sizeof(uint64_t), "reference table keys"); t->hc = sok_xmalloc(sz * sizeof(int32_t), "reference table costs"); t->cap = sz; }
    memset(t->hk, 0, sz * sizeof(uint64_t));
    t->mask = sz - 1; t->n = 0;
    g_refstat[4]++; g_refstat[5] += n;
    for (int i = 0; i < n; i++) {
        uint64_t key = keys[i] ? keys[i] : 1;
        uint32_t h = (uint32_t)(h64(key) & t->mask);
        while (t->hk[h] && t->hk[h] != key) h = (h + 1) & t->mask;
        if (!t->hk[h]) { t->hk[h] = key; t->hc[h] = costs[i]; t->n++; }
        else if (costs[i] < t->hc[h]) t->hc[h] = costs[i];
    }
    return t->n;
}
void sokoban_ref_attach(const SokRefTable *t) {
    g_ref_keys = t->hk; g_ref_cost = t->hc; g_ref_mask = t->mask; g_ref_n = t->n;
    g_ref_k = 0; g_ref_delta = 0; g_ref_xor = 0; g_ref_on = g_ref_n > 0;
}
void sokoban_ref_set_k(int k) { g_ref_k = k; g_ref_delta = 0; g_ref_xor = 0; }
void sokoban_ref_set_xor(uint64_t x) { g_ref_xor = x; }
void sokoban_ref_suspend(int suspend) { g_ref_on = suspend ? 0 : (g_ref_n > 0); }
int  sokoban_ref_active(void) { return g_ref_on; }
/* Declare the puzzle delta for the reference just installed (call after
 * sokoban_set_reference).  walls: the child's walls; exit_pos: its exit. */
void sokoban_set_reference_delta(uint64_t delta, uint64_t walls, int exit_pos) {
    g_ref_delta = delta & ~walls;
    if (!g_ref_delta) return;
    hb_prepare(walls);
    for (uint64_t t = g_ref_delta; t; t &= t - 1) { int c = __builtin_ctzll(t); g_ref_dexit[c] = g_hb_D[c][exit_pos]; }
}
void sokoban_ref_load(const uint64_t *keys, const int32_t *costs, int n) {
    if (g_ref_n) sokoban_clear_reference();
    if (n > (int)(REF_SIZE / 2)) return;                 /* too big: no reference */
    sokoban_ref_build(keys, costs, n, &g_ref_own);
    sokoban_ref_attach(&g_ref_own);
}
void sokoban_set_reference(const uint64_t *keys, const int32_t *costs, int n, int k) { sokoban_ref_load(keys, costs, n); g_ref_k = k; }
static inline int ref_lookup(uint64_t key) {
    key ^= g_ref_xor;
    if (!key) key = 1;
    uint32_t h = (uint32_t)(h64(key) & REF_MASK);
    while (g_ref_keys[h]) { if (g_ref_keys[h] == key) return g_ref_cost[h]; h = (h + 1) & REF_MASK; }
    return -1;
}
/* Prune test: child at chain offset k reaching Y (player landing at cell p)
 * at cost nc.  Without a delta: nc + rest <= cut_P + k with rest >= cut_P + 2
 * - g_P  =>  prune iff nc > g_P + k - 2.  With a delta the owner's bound only
 * covers continuations avoiding the delta cells, so additionally every delta
 * cell must be too far: nc + D(p, c) - 1 + D(c, exit) - 1 > cut for all c. */
/* Same test with the lookup already done (gp = ref_lookup(key), or -1). */
static inline int ref_prunes_gp(int gp, int nc, int p, int cut, int k) {
    if (gp < 0 || nc <= gp + k - 2) return 0;
    if (!g_ref_delta) { g_refstat[2]++; return 1; }
    for (uint64_t t = g_ref_delta; t; t &= t - 1) {
        int c = __builtin_ctzll(t);
        int dpc = g_hb_D[p][c], dce = g_ref_dexit[c];
        if (dpc < 0 || dce < 0) continue;
        if (nc + (dpc > 0 ? dpc - 1 : 0) + (dce > 0 ? dce - 1 : 0) <= cut) { g_refstat[3]++; return 0; }
    }
    g_refstat[2]++;
    return 1;
}
static inline int ref_prunes_at(uint64_t key, int nc, int p, int cut) {
    if (!g_ref_on) return 0;
    g_refstat[0]++;
    int gp = ref_lookup(key);
    if (gp >= 0) g_refstat[1]++;
    if (gp < 0 || nc <= gp + g_ref_k - 2) return 0;
    if (!g_ref_delta) { g_refstat[2]++; return 1; }
    for (uint64_t t = g_ref_delta; t; t &= t - 1) {
        int c = __builtin_ctzll(t);
        int dpc = g_hb_D[p][c], dce = g_ref_dexit[c];
        if (dpc < 0 || dce < 0) continue;                 /* cannot use this cell at all */
        if (nc + (dpc > 0 ? dpc - 1 : 0) + (dce > 0 ? dce - 1 : 0) <= cut) { g_refstat[3]++; return 0; }
    }
    g_refstat[2]++;
    return 1;
}
static inline int ref_prunes(uint64_t key, int nc) { return ref_prunes_at(key, nc, 0, 0); }   /* delta-free callers */

/* Export the labels of the last exhaustive small-table solve as a table of
 * upper bounds on the solved state's forward distances.  Every label in the
 * hash is the cost of a real path, hence an upper bound.  If the solve used a
 * reference (owner A at chain offset k), states pruned by it are missing from
 * the hash; A reaches them at g_A and the solved state reaches A's start in k
 * moves, so g_A + k is an upper bound for them: the union is a complete table
 * (exact wherever it matters -- a pruned state's true distance IS g_A + k by
 * parity) and the next child can use it with k = 1.  Duplicates are resolved
 * to the minimum when the table is loaded. */
int sokoban_export_settled(uint64_t *keys, int32_t *costs, int max) {
    HashStatePush64 *hs = hsp64_get();
    if (!hs->last_exhaustive || hs->slot != hs->slot_small) return -1;
    int n = 0;
    for (uint32_t i = 0; i < hs->ntouched && n < max; i++) {
        HSlot64 *sl = &hs->slot_small[hs->touched[i]];
        if (sl->gen != hs->ht_seq) continue;
        keys[n] = sl->key; costs[n] = sl->cost; n++;
    }
    if (hs->last_ref_used) {
        if (!g_ref_keys) return -1;      /* the reference was cleared since: its entries are gone */
        if (n + g_ref_n >= max) return -1;
        for (uint32_t h = 0; h <= g_ref_mask; h++) if (g_ref_keys[h]) { keys[n] = g_ref_keys[h] ^ g_ref_xor; costs[n] = g_ref_cost[h] + g_ref_k; n++; }
    }
    return (n < max) ? n : -1;
}

/* ========================================================================
 * MULTI-START DECISION SOLVE
 *
 * Every walk-back descendant of a generator state that stays on already
 * committed cells shares the same forward puzzle (walls, blocks, masks,
 * holes) and differs only in the player's start cell.  In the push-macro
 * formulation the start cell only affects the cost of reaching the FIRST
 * push, so one Dijkstra can serve all of them: every forward state carries
 * a vector of labels, one per start cell.  Per start i the question is "does
 * a solution of length <= cut[i] exist"; the answer is exact and identical to
 * running sokoban_solve_cutoff() from starts[i] with max_cost cut[i].
 *
 * Lockstep Dijkstra: an entry is pushed at prio = some start's label; when
 * popped, exactly the starts whose label equals the prio and are not yet
 * relaxed from this state are final (positive edge weights) and get relaxed.
 * The EXPANSION of a state (unpack, walk BFS, push candidates) does not
 * depend on the start or the cost, so it is done once and its successor list
 * is cached in an arena; later pops of the same state at other labels only
 * replay the list with the new offsets.
 *
 * pred[i] (optional) is a bitmask of starts adjacent to start i on a shortest
 * path back to the parent cell.  A shortcut at such a predecessor implies one
 * at i (walk one step, then the predecessor's shortcut), so starts behind a
 * decided shortcut are decided without any search.
 *
 * The same relation prunes the search itself.  Let j be a chain ancestor of i
 * at chain distance k (so cut[i] = cut[j] + k).  If i reaches forward state X
 * at cost g_i and j at cost g_j with g_i - g_j >= k - 1, then i need not
 * explore X: a solution S for i through X (|S| <= cut[i]) gives j the solution
 * "j's prefix to X, then S's suffix" of length g_j + |S| - g_i <= cut[j] + 1,
 * hence <= cut[j] by parity (all solutions from one start have one parity), so
 * j has a shortcut and i is decided by propagation anyway.  Labels only
 * decrease, so j's current label is a valid upper bound on g_j.  For the
 * adjacent ancestor (k = 1) this means i only explores states it reaches
 * STRICTLY cheaper than its predecessor.
 * ======================================================================== */
#define MS_MAX   24                        /* starts per solve: bitmask-bound (uint32 alive/done), not a knob */
#define MS_INF   255
#ifndef MS_LG2
#define MS_LG2   16
#endif
#define MS_SIZE  (1u << MS_LG2)
#define MS_MASK  (MS_SIZE - 1)
#ifndef MS_ARENA
#define MS_ARENA (1u << 20)                 /* cached successors per solve (w1 32 MB, w2 48 MB, w4 64 MB, one arena per width in use; touched lazily) */
#endif
typedef struct {
    uint64_t key; uint32_t gen; uint32_t done;
    uint64_t key2;                          /* the state's second key (z2_*): checked on every key match (M7) */
    uint8_t  lab[MS_MAX];
    uint32_t soff; uint8_t scount; int8_t wexit; uint8_t cached; uint8_t hb;   /* hb: hole-crossing bound at this state (0 = not applicable) */
} MSLab;                                    /* 56 B */

/* Test knob: -DMS_KEY_TEST_BITS=N keys the label table by the low N bits of
 * the primary key only, so collisions are frequent and the key2 check (or,
 * with -DMS_NO_KEY2, its absence) is exercised. */
#ifdef MS_KEY_TEST_BITS
#define MS_KEY(k) ((k) & ((1ULL << (MS_KEY_TEST_BITS)) - 1))
#else
#define MS_KEY(k) (k)
#endif
static long long g_ms_collisions;   /* label-table key collisions caught by the key2 check (each ends its multi solve with -2) */
long long sokoban_ms_collisions(void) { return g_ms_collisions; }

/* The label slot of the state with keys (k, k2).  -1: table full, probe limit,
 * or a primary-key collision (same k, different k2): the caller ends the multi
 * solve with -2 and the starts are decided one by one, which is exact. */
static inline int ms_slot(HashStatePush64 *hs, MSLab *L, uint64_t k, uint64_t k2, int *is_new) {
    k = MS_KEY(k);
    uint64_t h = h64(k) & MS_MASK;
    for (int i = 0; i < HT_PROBE_LIMIT; i++) {
        uint32_t idx = (uint32_t)((h + i) & MS_MASK);
        MSLab *sl = &L[idx];
        if (sl->gen != hs->ht_seq) {
            if (hs->ht_count >= MS_SIZE / 2) { hs->overflow = 1; return -1; }
            hs->ht_count++; sl->gen = hs->ht_seq; sl->key = k; sl->key2 = k2; sl->done = 0; sl->cached = 0;
            memset(sl->lab, MS_INF, MS_MAX); *is_new = 1;
            return (int)idx;
        }
        if (sl->key == k) {
#ifndef MS_NO_KEY2
            if (sl->key2 != k2) { g_ms_collisions++; return -1; }
#endif
            *is_new = 0; return (int)idx;
        }
    }
    hs->probe_failed = 1;
    return -1;
}

/* ========================================================================
 * PUSH SOLVER INSTANCES
 *
 * The PUSH SOLVER TEMPLATE at the end of this file, compiled once per packed
 * state width: solve_push_w<W>, solve_push_cutoff_w<W> and solve_multi_w<W>
 * for W = 1, 2, 4 (see the PUSH-BASED DIJKSTRA SOLVER comment).
 * ======================================================================== */
#define SOKW 1
#include "sokoban_bfs.c"
#undef SOKW
#define SOKW 2
#include "sokoban_bfs.c"
#undef SOKW
#define SOKW 4
#include "sokoban_bfs.c"
#undef SOKW

/* ========================================================================
 * DISPATCHER
 * ======================================================================== */

/* A state the solvers can represent: at most MAX_BLOCKS blocks (w4 holds 32
 * block cells) and at most 30 holes (int hole masks).  Every state the
 * generator can build on a grid of <= MAX_NCELLS cells fits; one that does not
 * is SOK_NO_FIT: never solved on a truncated state, never a prune, and the
 * caller ends the run with an error.  (-DSOK_STATE_BITS_MAX=N below the
 * default 261 is a test knob that makes wider states SOK_NO_FIT.) */
static inline int sok_fits(int nb, int nh) {
    return nb >= 0 && nb <= MAX_BLOCKS && nh >= 0 && nh <= 30 && g_bits_per_cell * (1 + nb) + nh <= SOK_STATE_BITS_MAX;
}
/* The packed-state width for a state that fits: 1, 2 or 4 (see the PUSH-BASED
 * DIJKSTRA SOLVER comment).  w1 needs b*nb + nh <= 64 and b*nb < 64 (the hole
 * mask's shift). */
static inline int sok_width(int nb, int nh) {
    const int bb = g_bits_per_cell * nb;
    if (bb + nh <= SOK_W1_BITS && bb < 64) return 1;
    return nb <= SOK_W2_BLOCKS ? 2 : 4;
}
int sokoban_state_bits_max(void) { return SOK_STATE_BITS_MAX; }
int sokoban_state_width(int nb, int nh) { return sok_fits(nb, nh) ? sok_width(nb, nh) : 0; }

int sokoban_solve(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof) {
    int nb = pz->num_blocks, nh = pz->num_holes;
    hsp64_get()->last_exhaustive = 0;          /* a full solve never leaves an exportable cutoff table behind */
    if (!sok_fits(nb, nh)) return SOK_NO_FIT;
    switch (sok_width(nb, nh)) {
    case 1:  return solve_push_w1(pz, used_dirs, prof);
    case 2:  return solve_push_w2(pz, used_dirs, prof);
    default: return solve_push_w4(pz, used_dirs, prof);
    }
}

int sokoban_solve_cutoff(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof, int max_cost) {
    int nb = pz->num_blocks, nh = pz->num_holes;
    /* Only an exhaustive small-table solve made by THIS call may be exported
     * afterwards (sokoban_export_settled): clear the flag first, so no early
     * return (max_cost < 0, no fit) can leave the previous solve's table
     * looking exportable. */
    hsp64_get()->last_exhaustive = 0;
    if (!sok_fits(nb, nh)) return SOK_NO_FIT;
    switch (sok_width(nb, nh)) {
    case 1:  return solve_push_cutoff_w1(pz, used_dirs, prof, max_cost);
    case 2:  return solve_push_cutoff_w2(pz, used_dirs, prof, max_cost);
    default: return solve_push_cutoff_w4(pz, used_dirs, prof, max_cost);
    }
}

int sokoban_solve_multi(const Puzzle *pz, const int8_t *starts, const int16_t *cut, const uint32_t *pred,
                        const uint8_t *refk, int n, uint8_t *out, BfsProfile *prof) {
    const int nb = pz->num_blocks, nh = pz->num_holes;
    hsp64_get()->last_exhaustive = 0;
    if (n <= 0 || n > MS_MAX) return -2;
    if (!sok_fits(nb, nh)) return -2;          /* the caller's per-start fallback then reports SOK_NO_FIT */
    switch (sok_width(nb, nh)) {
    case 1:  return solve_multi_w1(pz, starts, cut, pred, refk, n, out, prof);
    case 2:  return solve_multi_w2(pz, starts, cut, pred, refk, n, out, prof);
    default: return solve_multi_w4(pz, starts, cut, pred, refk, n, out, prof);
    }
}

/* Effective table-size knobs of this build, as a JSON object body (for the
 * worker's --version KNOBS line). */
const char *sokoban_knobs_json(void) {
    static const char defs[] = ""   /* compile-time solver switches (outside the snprintf: no directives in macro arguments) */
#ifdef NO_ASTAR
        "NO_ASTAR "
#endif
#ifdef HOLEBOUND
        "HOLEBOUND "
#endif
#ifdef MS_NO_DOM
        "MS_NO_DOM "
#endif
#ifdef MS_NO_KEY2
        "MS_NO_KEY2 "
#endif
#ifdef MS_KEY_TEST_BITS
        "MS_KEY_TEST_BITS "
#endif
#ifdef MAND_OLD
        "MAND_OLD "
#endif
#ifdef MAND_ASSERT
        "MAND_ASSERT "
#endif
#ifdef FWPROF
        "FWPROF "
#endif
        ;
    static char buf[768];
    snprintf(buf, sizeof buf,
             "\"HTP_SIZE\":%lld,\"HTP_SMALL_LG2\":%d,\"HP64_SIZE\":%lld,\"HT_PROBE_LIMIT\":%lld,"
             "\"REF_LG2\":%d,\"MS_LG2\":%d,\"MS_ARENA\":%lld,\"SOK_W1_BITS\":%d,\"SOK_W2_BLOCKS\":%d,"
             "\"SOK_STATE_BITS_MAX\":%d,\"DIRECT_LIMIT\":%lld,\"HT_SIZE\":%lld,\"QSZ\":%lld,\"solver_defs\":\"%s\"",
             (long long)(HTP_SIZE), (int)(HTP_SMALL_LG2), (long long)(HP64_SIZE), (long long)(HT_PROBE_LIMIT),
             (int)(REF_LG2), (int)(MS_LG2), (long long)(MS_ARENA), (int)(SOK_W1_BITS), (int)(SOK_W2_BLOCKS),
             (int)(SOK_STATE_BITS_MAX), (long long)(DIRECT_LIMIT), (long long)(HT_SIZE), (long long)(QSZ), defs);
    return buf;
}
#else /* SOKW defined: the PUSH SOLVER TEMPLATE pass */
/* ========================================================================
 * PUSH SOLVER TEMPLATE
 *
 * Included by this very file once per state width (SOKW = 1, 2, 4; see
 * PUSH SOLVER INSTANCES).  Everything here depends on the width only through
 * the packed state type SOKN(PSt) and its three operations: pack, unpack and
 * move (one block to a new cell, optionally filling a hole).  Every name gets
 * the suffix _w<SOKW>.
 * ======================================================================== */
#define SOKN__(n, w) n##_w##w
#define SOKN_(n, w)  SOKN__(n, w)
#define SOKN(n)      SOKN_(n, SOKW)

#if SOKW == 1
#define SOKW_IX 0
/* One word: block i at bits [b*i, b*i+b), hole mask from bit b*nb.  The
 * dispatcher guarantees b*nb + nh <= 64 and b*nb < 64. */
typedef uint64_t SOKN(PSt);
static inline void SOKN(pst_pack)(SOKN(PSt) *s, const int *bp, int nb, int hm) {
    uint64_t v = 0; int sh = 0;
    for (int i = 0; i < nb; i++) { v |= (uint64_t)bp[i] << sh; sh += g_bits_per_cell; }
    *s = v | ((uint64_t)(uint32_t)hm << sh);
}
static inline int SOKN(pst_unpack)(const SOKN(PSt) *s, int *bp, int nb) {
    const uint64_t v = *s; int sh = 0;
    for (int i = 0; i < nb; i++) { bp[i] = (int)((v >> sh) & g_cell_mask); sh += g_bits_per_cell; }
    return (int)(v >> sh);   /* the hole mask is at the top: no mask needed */
}
/* *d = *s with block bi on cell c and the hole-mask bits hx flipped. */
static inline void SOKN(pst_move)(SOKN(PSt) *d, const SOKN(PSt) *s, int nb, int bi, int c, uint32_t hx) {
    const int bsh = g_bits_per_cell * bi;
    uint64_t v = (*s & ~(g_cell_mask << bsh)) | ((uint64_t)c << bsh);
    if (hx) v ^= (uint64_t)hx << (g_bits_per_cell * nb);
    *d = v;
}
#else
#define SOKW_IX (SOKW == 2 ? 1 : 2)
/* One byte per block cell (8*SOKW of them), the hole mask in its own word. */
typedef struct { uint8_t c[8 * SOKW]; uint32_t hm; } SOKN(PSt);
static inline void SOKN(pst_pack)(SOKN(PSt) *s, const int *bp, int nb, int hm) {
    memset(s, 0, sizeof *s);
    for (int i = 0; i < nb; i++) s->c[i] = (uint8_t)bp[i];
    s->hm = (uint32_t)hm;
}
static inline int SOKN(pst_unpack)(const SOKN(PSt) *s, int *bp, int nb) {
    for (int i = 0; i < nb; i++) bp[i] = s->c[i];
    return (int)s->hm;
}
static inline void SOKN(pst_move)(SOKN(PSt) *d, const SOKN(PSt) *s, int nb, int bi, int c, uint32_t hx) {
    (void)nb;
    *d = *s;
    d->c[bi] = (uint8_t)c;
    d->hm ^= hx;
}
#endif

/* No per-entry used-direction bits (see fill_used_dirs). */
typedef struct {
    int       prio;        /* bucket priority: g, or f = g + h in the cutoff solver */
    int       g;           /* exact cost so far (walks + pushes)      */
    int       player_pos;  /* exact player cell (not in the packed state) */
    uint32_t  slot;        /* hash-table slot holding this state      */
    uint64_t  key;         /* Zobrist key used for the table          */
    SOKN(PSt) state;       /* packed blocks + hole mask (for unpacking) */
} SOKN(HeapE);             /* w1 32 B, w2 48 B, w4 64 B */

static inline void SOKN(bq_reset)(BucketV *bq) {
    for (int i = 0; i < PQ64_NBUCKETS; i++) bq[i].len = 0;
}

/* Bucket queue: push entry into its priority's bucket. */
static inline void SOKN(bq_push)(HashStatePush64 *hs, BucketV *bq, const SOKN(HeapE) *e) {
    BucketV *b = &bq[e->prio & PQ64_BMASK];
    if (b->len == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 1024;
        b->items = sok_xrealloc(b->items, (size_t)b->cap * sizeof(SOKN(HeapE)), "push-solver bucket");
    }
    ((SOKN(HeapE) *)b->items)[b->len++] = *e;
    hs->pq_count++;
}

/* Full (uncapped) solve: the shortest forward solution, -1, or an undecided code. */
static int SOKN(solve_push)(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof) {
    HashStatePush64 *hs = hsp64_get();
    BucketV *bq = hs->bq[SOKW_IX];
    int use_big = 0;
retry:
    hsp64_select(hs, use_big);
    hsp64_clear(hs);
    SOKN(bq_reset)(bq);
    mand_setup(pz);
    int peak_heap = 1;
    int n_popped  = 0;

    const int      nb       = pz->num_blocks;
    const int      nh       = pz->num_holes;
    const int      exit_pos = pz->exit_pos;
    const uint64_t walls    = pz->walls;

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    /* Enqueue initial state at cost 0. */
    {
        SOKN(HeapE) e0;
        e0.prio = 0; e0.g = 0; e0.player_pos = pz->player_start;
        SOKN(pst_pack)(&e0.state, ib, nb, ihm);
        e0.key = zpack64(pz->player_start, ib, pz->block_pushable, nb, ihm, nh, pz->hole_pos);
        e0.slot = (uint32_t)hsp64_update(hs, e0.key, 0);
        SOKN(bq_push)(hs, bq, &e0);
    }

    int best_win = INT_MAX;

    int pq_min = 0;
    while (hs->pq_count > 0) {
        BucketV *bkt = &bq[pq_min & PQ64_BMASK];
        if (bkt->len == 0) { pq_min++; continue; }
        SOKN(HeapE) e = ((SOKN(HeapE) *)bkt->items)[--bkt->len];
        hs->pq_count--;
        if (e.prio >= best_win) break; /* Dijkstra: can't improve */

        /* Lazy deletion: skip if a shorter path was already found. */
        if (hs->slot[e.slot].cost < e.prio) continue;
        n_popped++;

        int bp[MAX_BLOCKS];
        int hm = SOKN(pst_unpack)(&e.state, bp, nb);

        uint64_t blk_occ = 0;
        for (int i = 0; i < nb; i++) if (bp[i] < g_ncells) blk_occ |= (1ULL << bp[i]);
        uint64_t cur_holes = 0;
        for (int h = 0; h < nh; h++) if (hm & (1 << h)) cur_holes |= (1ULL << pz->hole_pos[h]);

        /* Pass 1: enumerate candidate pushes (everything except player
         * reachability), compute their key, and prefetch each one's hash
         * slot.  The walk-distance BFS below then overlaps with the table's
         * cold-miss latency. */
        PushCand cand[MAX_BLOCKS * 4];
        int ncand = 0;
        for (int bi = 0; bi < nb; bi++) {
            if (bp[bi] >= g_ncells) continue; /* block consumed */
            const int bpos = bp[bi];
            const int mb   = pz->block_pushable[bi] & 0xF;

            for (int d = 0; d < 4; d++) {
                if (!(mb & (1 << d))) continue;

                int pfr = g_adj[bpos][d ^ 2]; /* push-from cell (player must be here) */
                if (pfr < 0) continue;

                int lnd = g_adj[bpos][d]; /* landing cell for block */
                if (lnd < 0 || (walls & (1ULL << lnd)) || (blk_occ & (1ULL << lnd))) continue;

                /* Compute new block position (handle hole consumption). */
                int new_bpos = lnd, nhm = hm, ch = -1;
                if (cur_holes & (1ULL << lnd)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == lnd && (hm & (1 << h))) {
                            new_bpos = g_consumed;
                            nhm &= ~(1 << h);
                            ch = h;
                            break;
                        }
                    }
                }

#ifdef FWPROF
                { uint64_t _d0 = mach_absolute_time();
                  int _dead = mand_dead(pz, bp, nb, bi, new_bpos, (uint32_t)nhm);
                  g_mdead_ticks += mach_absolute_time() - _d0; g_mdead_calls++;
                  if (_dead) continue; }
#else
                if (mand_dead(pz, bp, nb, bi, new_bpos, (uint32_t)nhm)) continue;
#endif

                /* Incremental Zobrist: XOR out moved pieces, XOR in new. */
                uint64_t nk = e.key ^ z_player[e.player_pos] ^ z_player[bpos]
                            ^ z_block[mb][bpos] ^ z_block[mb][new_bpos];
                if (ch >= 0) nk ^= z_hole[pz->hole_pos[ch]];
                __builtin_prefetch(&hs->slot[h64(nk) & hs->ht_mask], 1, 1);
                cand[ncand].key  = nk;
                cand[ncand].pfr  = (int16_t)pfr;
                cand[ncand].bpos = (int16_t)bpos;
                cand[ncand].bi   = (int8_t)bi;
                cand[ncand].nbp  = (int8_t)new_bpos;
                cand[ncand].ch   = (int8_t)ch;
                ncand++;
            }
        }

        /* Walk distances from exact player position (only block-adjacent
         * cells and the exit are ever queried). */
        int8_t wdist[MAX_NCELLS];
        walk_dists_from(walls | cur_holes | blk_occ, e.player_pos,
                        nbr_mask(blk_occ) | (1ULL << exit_pos), wdist);

        /* Win check: can player walk to exit? */
        if (wdist[exit_pos] >= 0) {
            int wc = e.prio + (int)wdist[exit_pos];
            if (wc < best_win) best_win = wc;
        }

        /* Pass 2: keep candidates whose push-from cell is reachable. */
        for (int ci = 0; ci < ncand; ci++) {
            if (wdist[cand[ci].pfr] < 0) continue;

            int      nc = e.prio + (int)wdist[cand[ci].pfr] + 1;
            uint64_t nk = cand[ci].key;
            int slot = hsp64_update(hs, nk, nc);
            if (slot >= 0) {
                if (hs->pq_count >= HP64_SIZE) {
                    if (prof) { prof->peak_heap_sz = HP64_SIZE; prof->states_popped = n_popped; }
                    return SOK_PENDING_CAP;
                }
                SOKN(HeapE) ne;
                ne.prio = nc; ne.g = nc; ne.player_pos = cand[ci].bpos;
                ne.slot = (uint32_t)slot; ne.key = nk;
                SOKN(pst_move)(&ne.state, &e.state, nb, cand[ci].bi, cand[ci].nbp, cand[ci].ch >= 0 ? 1u << cand[ci].ch : 0u);
                SOKN(bq_push)(hs, bq, &ne);
                if (hs->pq_count > peak_heap) peak_heap = hs->pq_count;
                if (g_heap_cap > 0 && hs->pq_count > g_heap_cap) {
                    if (prof) { prof->peak_heap_sz = peak_heap; prof->states_popped = n_popped; }
                    return SOK_PROBE_LIMIT;
                }
            } else if (hs->overflow) {
                /* Table saturated.  The small one: rerun this solve once on the
                 * big one.  The big one: the identical rerun would overflow at
                 * the same insert forever, so the solve is undecided (-5). */
                if (use_big) {
                    if (prof) { prof->peak_heap_sz = peak_heap; prof->states_popped = n_popped; }
                    return SOK_TABLE_FULL;
                }
                use_big = 1;
                goto retry;
            }
        }
    }

    if (prof) { prof->peak_heap_sz = peak_heap; prof->states_popped = n_popped; }
    if (hs->probe_failed) return SOK_PROBE_LIMIT;
    if (best_win == INT_MAX) return -1;
    fill_used_dirs(used_dirs, nb);
    return best_win;
}

/* ========================================================================
 * CUTOFF VARIANT  —  solve_push with a max_cost ceiling.
 *
 * Returns:
 *   x >= 0   a forward solve of length x <= max_cost (the shortest one unless
 *            decision mode is on: then the first one found).
 *   -1       no path of length <= max_cost exists (or unsolvable).
 *   -2/-3/-5 undecided (pending cap / probe limit or heap cap / big table full).
 *
 * Internally, Dijkstra terminates as soon as the priority queue's head
 * exceeds max_cost (no future entry can beat it), and skips enqueuing
 * any successor whose cost would exceed max_cost.  This prunes the
 * "outer shell" of states above the cutoff that the standard solver
 * would otherwise explore.
 * ======================================================================== */
static int SOKN(solve_push_cutoff)(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof, int max_cost) {
    if (max_cost < 0) {
        /* No table is built: whatever the previous solve left must not be
         * exported as this state's parent table (the dispatcher already
         * cleared last_exhaustive; kept here for direct callers). */
        hsp64_get()->last_exhaustive = 0;
        if (prof) { prof->peak_heap_sz = 1; prof->states_popped = 0; bfs_tail_reset(prof, max_cost); }
        return -1;
    }

#ifdef FWPROF
    if (++g_fwp_solves % FWP_DUMP_EVERY == 0) fwp_dump();
    uint64_t _ts = mach_absolute_time();
#endif
    HashStatePush64 *hs = hsp64_get();
    BucketV *bq = hs->bq[SOKW_IX];
    int use_big = 0;
retry:
    hsp64_select(hs, use_big);
    hsp64_clear(hs);
    SOKN(bq_reset)(bq);
#ifdef FWPROF
    { uint64_t _m0 = mach_absolute_time(); mand_setup(pz);
      g_msetup_ticks += mach_absolute_time() - _m0; }
#else
    mand_setup(pz);
#endif
    bfs_tail_reset(prof, max_cost);
    int peak_heap = 1;
    int n_popped  = 0;

    const int      nb       = pz->num_blocks;
    const int      nh       = pz->num_holes;
    const int      exit_pos = pz->exit_pos;
    const uint64_t walls    = pz->walls;

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

#ifndef NO_ASTAR
    /* Admissible lower bound on remaining moves: grid distance from each
     * cell to the exit treating only walls as obstacles (blocks and holes
     * are passable in this relaxation, so it never over-estimates).  Walls
     * are fixed for the whole solve, so this is computed once. */
    int8_t hdist[MAX_NCELLS];
    walk_dists_from(walls, exit_pos, g_all_cells, hdist);
    if (hdist[pz->player_start] < 0 || hdist[pz->player_start] > max_cost) {
        if (prof) { prof->peak_heap_sz = 1; prof->states_popped = 0; }
        return -1;
    }
#ifdef HOLEBOUND
    hb_prepare(walls);
#endif
#endif

    {
        SOKN(HeapE) e0;
        e0.g = 0; e0.player_pos = pz->player_start;
#ifndef NO_ASTAR
        e0.prio = g_decision_only ? hdist[pz->player_start] : 0;   /* f = g + h; h consistent (1 cell per unit cost) */
#else
        e0.prio = 0;
#endif
        SOKN(pst_pack)(&e0.state, ib, nb, ihm);
        e0.key = zpack64(pz->player_start, ib, pz->block_pushable, nb, ihm, nh, pz->hole_pos);
        e0.slot = (uint32_t)hsp64_update(hs, e0.key, 0);
        SOKN(bq_push)(hs, bq, &e0);
    }

    int best_win = INT_MAX;

#ifdef FWPROF
    g_fwp[FWP_SETUP] += mach_absolute_time() - _ts;
#endif
    int pq_min = 0;
    while (hs->pq_count > 0) {
        BucketV *bkt = &bq[pq_min & PQ64_BMASK];
        if (bkt->len == 0) { pq_min++; continue; }
        SOKN(HeapE) e = ((SOKN(HeapE) *)bkt->items)[--bkt->len];
        hs->pq_count--;
        if (e.prio > max_cost) break;          /* CUTOFF (prio = g, or f >= g) */
        if (e.prio >= best_win) break;         /* f is a lower bound through this state */

        if (hs->slot[e.slot].cost < e.g) continue;
        n_popped++;
        if (prof) {
            int twi = e.g - max_cost + (BFS_TAIL_W - 1);
            if (twi >= 0 && twi < BFS_TAIL_W) prof->tail_width[twi]++;
        }
        const int slack = max_cost - e.g;      /* >= 0 here */
        if (slack == 0) {
            /* No push child can cost <= max_cost, and a walk-win needs
             * wdist[exit] == 0, i.e. the player already stands on the exit. */
            if (e.player_pos == exit_pos && e.g < best_win) {
                best_win = e.g;
                if (g_decision_only) goto done;
            }
            continue;
        }
        FWP_DECL();

        int bp[MAX_BLOCKS];
        int hm = SOKN(pst_unpack)(&e.state, bp, nb);

        uint64_t blk_occ = 0;
        for (int i = 0; i < nb; i++) if (bp[i] < g_ncells) blk_occ |= (1ULL << bp[i]);
        uint64_t cur_holes = 0;
        for (int h = 0; h < nh; h++) if (hm & (1 << h)) cur_holes |= (1ULL << pz->hole_pos[h]);
        MandCtx mc;
#ifdef FWPROF
        { uint64_t _c0 = mach_absolute_time();
          mand_pop_ctx(pz, bp, nb, hm, &mc);
          g_mctx_ticks += mach_absolute_time() - _c0; }
#else
        mand_pop_ctx(pz, bp, nb, hm, &mc);
#endif
        FWP_LAP(FWP_UNPACK);

        /* Pass 1: candidates + slot prefetch (see solve_push). */
        PushCand cand[MAX_BLOCKS * 4];
        int ncand = 0;
        for (int bi = 0; bi < nb; bi++) {
            if (bp[bi] >= g_ncells) continue;
            const int bpos = bp[bi];
            const int mb   = pz->block_pushable[bi] & 0xF;

            for (int d = 0; d < 4; d++) {
                if (!(mb & (1 << d))) continue;

                int pfr = g_adj[bpos][d ^ 2];
                if (pfr < 0) continue;

                int lnd = g_adj[bpos][d];
                if (lnd < 0 || (walls & (1ULL << lnd)) || (blk_occ & (1ULL << lnd))) continue;

                int new_bpos = lnd, nhm = hm, ch = -1;
                if (cur_holes & (1ULL << lnd)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == lnd && (hm & (1 << h))) {
                            new_bpos = g_consumed;
                            nhm &= ~(1 << h);
                            ch = h;
                            break;
                        }
                    }
                }

#ifdef FWPROF
                { uint64_t _d0 = mach_absolute_time();
                  int _dead = mand_dead_ctx(pz, &mc, bp, nb, bi, new_bpos, (uint32_t)nhm, ch);
                  g_mdead_ticks += mach_absolute_time() - _d0; g_mdead_calls++;
                  if (_dead) continue; }
#else
                if (mand_dead_ctx(pz, &mc, bp, nb, bi, new_bpos, (uint32_t)nhm, ch)) continue;
#endif

                uint64_t nk = e.key ^ z_player[e.player_pos] ^ z_player[bpos]
                            ^ z_block[mb][bpos] ^ z_block[mb][new_bpos];
                if (ch >= 0) nk ^= z_hole[pz->hole_pos[ch]];
                __builtin_prefetch(&hs->slot[h64(nk) & hs->ht_mask], 1, 1);
                cand[ncand].key  = nk;
                cand[ncand].pfr  = (int16_t)pfr;
                cand[ncand].bpos = (int16_t)bpos;
                cand[ncand].bi   = (int8_t)bi;
                cand[ncand].nbp  = (int8_t)new_bpos;
                cand[ncand].ch   = (int8_t)ch;
                ncand++;
            }
        }

        FWP_LAP(FWP_PASS1);
        int8_t wdist[MAX_NCELLS];
        walk_dists_from_lim(walls | cur_holes | blk_occ, e.player_pos,
                            nbr_mask(blk_occ) | (1ULL << exit_pos), wdist, slack);
        FWP_LAP(FWP_WALK);

        if (wdist[exit_pos] >= 0) {
            int wc = e.g + (int)wdist[exit_pos];
            if (wc <= max_cost && wc < best_win) {
                best_win = wc;
                if (g_decision_only) goto done;   /* any solution within the cutoff decides the check */
            }
        }
#if !defined(NO_ASTAR) && defined(HOLEBOUND)
        else if (hm && !hb_reaches(walls | cur_holes, e.player_pos, exit_pos)) {
            /* No walk to the exit even through blocks: some hole must be filled first. */
            if (e.g + hb_bound(pz, e.player_pos, bp, nb, hm, exit_pos) > max_cost) continue;
        }
#endif

        /* Pass 2: reachable push-from cells, costs within the cutoff. */
        for (int ci = 0; ci < ncand; ci++) {
            if (wdist[cand[ci].pfr] < 0) continue;

            int nc = e.g + (int)wdist[cand[ci].pfr] + 1;
            if (nc > max_cost) continue;       /* CUTOFF: don't enqueue */
            int nprio = nc;
#ifndef NO_ASTAR
            {   /* A*: the player lands on the block's old cell; it still
                 * needs at least hdist[] more moves to stand on the exit. */
                int hd = hdist[cand[ci].bpos];
                if (hd < 0 || nc + hd > max_cost) continue;
                if (g_decision_only) nprio = nc + hd;
            }
#endif

            uint64_t nk = cand[ci].key;
            int slot = hsp64_update(hs, nk, nc);
            if (slot >= 0) {
                /* New or improved label: an ancestor's table may still prove the state useless
                 * at this cost (the label stays as an upper bound, the state is not enqueued). */
                if (g_ref_on && ref_prunes_at(nk, nc, cand[ci].bpos, max_cost)) continue;
                if (hs->pq_count >= HP64_SIZE) {
                    if (prof) { prof->peak_heap_sz = HP64_SIZE; prof->states_popped = n_popped; }
                    return SOK_PENDING_CAP;
                }
                SOKN(HeapE) ne;
                ne.prio = nprio; ne.g = nc; ne.player_pos = cand[ci].bpos;
                ne.slot = (uint32_t)slot; ne.key = nk;
                SOKN(pst_move)(&ne.state, &e.state, nb, cand[ci].bi, cand[ci].nbp, cand[ci].ch >= 0 ? 1u << cand[ci].ch : 0u);
                SOKN(bq_push)(hs, bq, &ne);
                if (hs->pq_count > peak_heap) peak_heap = hs->pq_count;
                if (g_heap_cap > 0 && hs->pq_count > g_heap_cap) {
                    if (prof) { prof->peak_heap_sz = peak_heap; prof->states_popped = n_popped; }
                    return SOK_PROBE_LIMIT;
                }
            } else if (hs->overflow) {
                /* Small table saturated: rerun this solve once on the big one.
                 * The big one saturated: undecided (-5), never a retry loop. */
                if (use_big) {
                    if (prof) { prof->peak_heap_sz = peak_heap; prof->states_popped = n_popped; }
                    return SOK_TABLE_FULL;
                }
                use_big = 1;
                goto retry;
            }
        }
        FWP_LAP(FWP_PASS2);
    }

done:
    if (prof) { prof->peak_heap_sz = peak_heap; prof->states_popped = n_popped; }
    /* A dropped state (probe limit) makes "no solution" unproven, but a solution
     * that was found is real: in decision mode it still decides the check. */
    if (hs->probe_failed && !(g_decision_only && best_win != INT_MAX)) return SOK_PROBE_LIMIT;
    if (best_win == INT_MAX) { hs->last_exhaustive = 1; hs->last_ref_used = g_ref_on; hs->last_maxcost = max_cost; return -1; }
    fill_used_dirs(used_dirs, nb);
    return best_win;
}

/* ========================================================================
 * MULTI-START DECISION SOLVE  (see the comment above MS_MAX)
 * ======================================================================== */
/* A cached successor: its packed state is the one built from the FIRST pop of
 * its parent's label slot (a later pop of that slot may carry an equivalent
 * state with same-mask blocks permuted, so moves are never re-applied to it). */
typedef struct {
    uint64_t  key, key2;             /* the successor's primary and second key */
    SOKN(PSt) state;
    int32_t   slot;                  /* table index once known, else -1 */
    int8_t    bpos, w, hd, pad;
} SOKN(MSSucc);                      /* w1 32 B, w2 48 B, w4 64 B */

static _Thread_local SOKN(MSSucc) *SOKN(ms_arena_tls);

static int SOKN(solve_multi)(const Puzzle *pz, const int8_t *starts, const int16_t *cut, const uint32_t *pred,
                             const uint8_t *refk, int n, uint8_t *out, BfsProfile *prof) {
    const int nb = pz->num_blocks, nh = pz->num_holes;
    HashStatePush64 *hs = hsp64_get();
    if (!hs->ms_lab) { hs->ms_lab = sok_xcalloc(MS_SIZE, sizeof(MSLab), "multi-start labels"); hs->ms_lab_bytes = (size_t)MS_SIZE * sizeof(MSLab); }
    if (!SOKN(ms_arena_tls)) SOKN(ms_arena_tls) = sok_xmalloc((size_t)MS_ARENA * sizeof(SOKN(MSSucc)), "multi-start arena");
    MSLab *L = (MSLab *)hs->ms_lab; SOKN(MSSucc) *AR = SOKN(ms_arena_tls); uint32_t arena_n = 0;
    BucketV *bq = hs->bq[SOKW_IX];
    hsp64_select(hs, 0);
    hsp64_clear(hs);   /* a generation wrap clears L too (see hsp64_clear) */
    SOKN(bq_reset)(bq);
    mand_setup(pz);

    const int      exit_pos = pz->exit_pos;
    const uint64_t walls    = pz->walls;
    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    const int ihm = (1 << nh) - 1;
    int n_popped = 0, peak = 0, n_expanded = 0;   /* prof: states_popped = (state,label) pops, peak_heap_sz = distinct expansions */

    int8_t hdist[MAX_NCELLS];
    walk_dists_from(walls, exit_pos, g_all_cells, hdist);
#ifdef HOLEBOUND
    hb_prepare(walls);
#endif

    uint32_t alive = 0;
    for (int i = 0; i < n; i++) {
        out[i] = 0;
        if (cut[i] < 0 || hdist[starts[i]] < 0 || hdist[starts[i]] > cut[i]) continue;
        alive |= 1u << i;
    }
    /* Decide start i as "shortcut" and everything behind it. */
#define MS_DECIDE_SHORT(i0) do { uint32_t _q = 1u << (i0); \
        while (_q) { int _j = __builtin_ctz(_q); _q &= _q - 1; \
            if (out[_j]) continue; out[_j] = 1; alive &= ~(1u << _j); \
            if (pred) for (int _k = 0; _k < n; _k++) if ((alive >> _k & 1) && (pred[_k] >> _j & 1)) _q |= 1u << _k; } } while (0)
    if (!alive) goto finish;
    /* dom[i]: every start on a shortest chain between the parent cell and
     * start i; kd[i][j]: its chain distance (levels of the pred closure). */
    uint32_t dom[MS_MAX]; uint8_t kd[MS_MAX][MS_MAX];
    for (int i = 0; i < n; i++) {
        uint32_t d = 0, frontier = pred ? pred[i] : 0; int k = 1;
        while (frontier) {
            for (uint32_t t = frontier; t; t &= t - 1) kd[i][__builtin_ctz(t)] = (uint8_t)k;
            d |= frontier;
            uint32_t nxt = 0;
            for (uint32_t t = frontier; t; t &= t - 1) nxt |= pred[__builtin_ctz(t)];
            frontier = nxt & ~d; k++;
        }
        dom[i] = d;
    }

    uint64_t blk_occ0 = 0; for (int i = 0; i < nb; i++) blk_occ0 |= 1ULL << ib[i];
    uint64_t holes0 = 0;   for (int h = 0; h < nh; h++) holes0 |= 1ULL << pz->hole_pos[h];
    const uint64_t interesting0 = nbr_mask(blk_occ0) | (1ULL << exit_pos);
    int8_t wd0[MS_MAX][MAX_NCELLS];
    for (int i = 0; i < n; i++) {
        if (!(alive >> i & 1)) continue;
        walk_dists_from_lim(walls | holes0 | blk_occ0, starts[i], interesting0, wd0[i], cut[i]);
        if (wd0[i][exit_pos] >= 0 && wd0[i][exit_pos] <= cut[i]) MS_DECIDE_SHORT(i);
    }
    if (!alive) goto finish;
    int maxcut = 0;
    for (int i = 0; i < n; i++) if ((alive >> i & 1) && cut[i] > maxcut) maxcut = cut[i];
    if (maxcut >= MS_INF - 1) return -2;        /* labels are uint8: deeper searches fall back */

    /* First pushes: labels = per-start cost; one queue entry per distinct cost. */
    {
    MandCtx mc0; mand_pop_ctx(pz, ib, nb, ihm, &mc0);
    SOKN(PSt) base_state; SOKN(pst_pack)(&base_state, ib, nb, ihm);
    const uint64_t base_key = zpack64(0, ib, pz->block_pushable, nb, ihm, nh, pz->hole_pos) ^ z_player[0];
    const uint64_t base_key2 = zpack64_2(0, ib, pz->block_pushable, nb, ihm, nh, pz->hole_pos) ^ z2_player[0];
    for (int bi = 0; bi < nb; bi++) {
        const int bpos = ib[bi], mb = pz->block_pushable[bi] & 0xF;
        for (int d = 0; d < 4; d++) {
            if (!(mb & (1 << d))) continue;
            int pfr = g_adj[bpos][d ^ 2]; if (pfr < 0) continue;
            int lnd = g_adj[bpos][d];
            if (lnd < 0 || (walls & (1ULL << lnd)) || (blk_occ0 & (1ULL << lnd))) continue;
            int new_bpos = lnd, nhm = ihm, ch = -1;
            if (holes0 & (1ULL << lnd))
                for (int h = 0; h < nh; h++) if (pz->hole_pos[h] == lnd) { new_bpos = g_consumed; nhm &= ~(1 << h); ch = h; break; }
            if (mand_dead_ctx(pz, &mc0, ib, nb, bi, new_bpos, (uint32_t)nhm, ch)) continue;
            SOKN(PSt) ns; SOKN(pst_move)(&ns, &base_state, nb, bi, new_bpos, (uint32_t)(ihm ^ nhm));
            uint64_t nk = base_key ^ z_player[bpos] ^ z_block[mb][bpos] ^ z_block[mb][new_bpos];
            uint64_t nk2 = base_key2 ^ z2_player[bpos] ^ z2_block[mb][bpos] ^ z2_block[mb][new_bpos];
            if (ch >= 0) { nk ^= z_hole[pz->hole_pos[ch]]; nk2 ^= z2_hole[pz->hole_pos[ch]]; }
            int hd = hdist[bpos]; if (hd < 0) continue;
            int is_new = 0, slot = -1; uint64_t seen = 0;
            int gp = -1; if (g_ref_on && refk) { g_refstat[0]++; gp = ref_lookup(nk); if (gp >= 0) g_refstat[1]++; }
            for (uint32_t t = alive; t; t &= t - 1) {
                int i = __builtin_ctz(t);
                if (wd0[i][pfr] < 0) continue;
                int nc = wd0[i][pfr] + 1;
                if (nc + hd > cut[i]) continue;
                if (gp >= 0 && ref_prunes_gp(gp, nc, bpos, cut[i], refk[i])) continue;
                if (slot < 0) { slot = ms_slot(hs, L, nk, nk2, &is_new); if (slot < 0) return -2; }
#ifndef MS_NO_DOM
                { int dominated = 0;   /* an ancestor j already here with g_i - g_j >= k - 1: i cannot need X */
                  for (uint32_t u = dom[i]; u; u &= u - 1) { int j = __builtin_ctz(u); if (L[slot].lab[j] + kd[i][j] - 1 <= nc) { dominated = 1; break; } }
                  if (dominated) continue; }
#endif
                if (nc < L[slot].lab[i]) L[slot].lab[i] = (uint8_t)nc;
                if (nc < 64 && (seen >> nc & 1)) continue;
                if (nc < 64) seen |= 1ULL << nc;
                SOKN(HeapE) e; e.prio = nc; e.g = nc; e.player_pos = bpos; e.slot = (uint32_t)slot; e.key = nk; e.state = ns;
                SOKN(bq_push)(hs, bq, &e);
            }
        }
    }
    }
    if (hs->pq_count > peak) peak = hs->pq_count;

    int pq_min = 0;
    while (hs->pq_count > 0 && alive) {
        BucketV *bkt = &bq[pq_min & PQ64_BMASK];
        if (bkt->len == 0) { pq_min++; continue; }
        SOKN(HeapE) e = ((SOKN(HeapE) *)bkt->items)[--bkt->len];
        hs->pq_count--;
        if (e.prio > maxcut) break;
        MSLab *lb = &L[e.slot];
        uint32_t R = 0;
        for (uint32_t t = alive & ~lb->done; t; t &= t - 1) { int i = __builtin_ctz(t); if (lb->lab[i] == e.prio) R |= 1u << i; }
        if (!R) continue;
        lb->done |= R;
        n_popped++;

        if (!lb->cached) {
            /* Expand once: everything here is independent of start and cost. */
            int bp[MAX_BLOCKS];
            int hm = SOKN(pst_unpack)(&e.state, bp, nb);
            uint64_t blk_occ = 0;
            for (int i = 0; i < nb; i++) if (bp[i] < g_ncells) blk_occ |= 1ULL << bp[i];
            uint64_t cur_holes = 0;
            for (int h = 0; h < nh; h++) if (hm & (1 << h)) cur_holes |= 1ULL << pz->hole_pos[h];
            MandCtx mc; mand_pop_ctx(pz, bp, nb, hm, &mc);
            int8_t wdist[MAX_NCELLS];
            /* First pop of this state has its smallest label, so this slack is
             * the largest any later pop can need. */
            walk_dists_from_lim(walls | cur_holes | blk_occ, e.player_pos, nbr_mask(blk_occ) | (1ULL << exit_pos), wdist, maxcut - e.prio);
            lb->wexit = wdist[exit_pos];
            lb->hb = 0;
#ifdef HOLEBOUND
            if (wdist[exit_pos] < 0 && hm && !hb_reaches(walls | cur_holes, e.player_pos, exit_pos)) {
                int v = hb_bound(pz, e.player_pos, bp, nb, hm, exit_pos); lb->hb = (uint8_t)(v > 254 ? 254 : v);
            }
#endif
            lb->soff = arena_n; int cnt = 0;
            for (int bi = 0; bi < nb; bi++) {
                if (bp[bi] >= g_ncells) continue;
                const int bpos = bp[bi], mb = pz->block_pushable[bi] & 0xF;
                for (int d = 0; d < 4; d++) {
                    if (!(mb & (1 << d))) continue;
                    int pfr = g_adj[bpos][d ^ 2]; if (pfr < 0 || wdist[pfr] < 0) continue;
                    int lnd = g_adj[bpos][d];
                    if (lnd < 0 || (walls & (1ULL << lnd)) || (blk_occ & (1ULL << lnd))) continue;
                    int hd = hdist[bpos]; if (hd < 0) continue;
                    int new_bpos = lnd, nhm = hm, ch = -1;
                    if (cur_holes & (1ULL << lnd))
                        for (int h = 0; h < nh; h++) if (pz->hole_pos[h] == lnd && (hm & (1 << h))) { new_bpos = g_consumed; nhm &= ~(1 << h); ch = h; break; }
                    if (mand_dead_ctx(pz, &mc, bp, nb, bi, new_bpos, (uint32_t)nhm, ch)) continue;
                    uint64_t nk = e.key ^ z_player[e.player_pos] ^ z_player[bpos] ^ z_block[mb][bpos] ^ z_block[mb][new_bpos];
                    uint64_t nk2 = lb->key2 ^ z2_player[e.player_pos] ^ z2_player[bpos] ^ z2_block[mb][bpos] ^ z2_block[mb][new_bpos];
                    if (ch >= 0) { nk ^= z_hole[pz->hole_pos[ch]]; nk2 ^= z2_hole[pz->hole_pos[ch]]; }
                    if (arena_n >= MS_ARENA || cnt >= 255) return -2;
                    SOKN(MSSucc) *sc = &AR[arena_n++]; cnt++;
                    SOKN(pst_move)(&sc->state, &e.state, nb, bi, new_bpos, (uint32_t)(hm ^ nhm));
                    sc->key = nk; sc->key2 = nk2; sc->slot = -1; sc->bpos = (int8_t)bpos; sc->w = (int8_t)(wdist[pfr] + 1); sc->hd = (int8_t)hd;
                }
            }
            lb->scount = (uint8_t)cnt; lb->cached = 1; n_expanded++;
        }

        /* Replay the cached expansion for the starts in R at this cost. */
        if (lb->hb) {   /* hole-crossing bound: drop starts whose cutoff this state can no longer meet */
            for (uint32_t t = R; t; t &= t - 1) { int i = __builtin_ctz(t); if (e.prio + lb->hb > cut[i]) R &= ~(1u << i); }
            if (!R) continue;
        }
        if (lb->wexit >= 0) {
            int wc = e.prio + lb->wexit;
            for (uint32_t t = R; t; t &= t - 1) { int i = __builtin_ctz(t); if (wc <= cut[i]) MS_DECIDE_SHORT(i); }
            R &= alive;
            if (!R) continue;
        }
        SOKN(MSSucc) *sl = &AR[lb->soff];
        for (int k = 0; k < lb->scount; k++) {
            SOKN(MSSucc) *sc = &sl[k];
            int nc = e.prio + sc->w, fmin = nc + sc->hd;
            int slot = sc->slot, is_new = 0, improved = 0;   /* slots never move within a solve: probe once, reuse on every replay */
            int gp = -1; if (g_ref_on && refk) { g_refstat[0]++; gp = ref_lookup(sc->key); if (gp >= 0) g_refstat[1]++; }
            for (uint32_t t = R; t; t &= t - 1) {
                int i = __builtin_ctz(t);
                if (fmin > cut[i]) continue;
                if (gp >= 0 && ref_prunes_gp(gp, nc, sc->bpos, cut[i], refk[i])) continue;
                if (slot < 0) { slot = ms_slot(hs, L, sc->key, sc->key2, &is_new); if (slot < 0) return -2; sc->slot = slot; }
#ifndef MS_NO_DOM
                { int dominated = 0;
                  for (uint32_t u = dom[i]; u; u &= u - 1) { int j = __builtin_ctz(u); if (L[slot].lab[j] + kd[i][j] - 1 <= nc) { dominated = 1; break; } }
                  if (dominated) continue; }
#endif
                if (nc < L[slot].lab[i]) { L[slot].lab[i] = (uint8_t)nc; improved = 1; }
            }
            if (slot < 0 || !improved) continue;
            if (hs->pq_count >= HP64_SIZE || (g_heap_cap > 0 && hs->pq_count > g_heap_cap)) return -2;
            SOKN(HeapE) ne; ne.prio = nc; ne.g = nc; ne.player_pos = sc->bpos; ne.slot = (uint32_t)slot; ne.key = sc->key; ne.state = sc->state;
            SOKN(bq_push)(hs, bq, &ne);
            if (hs->pq_count > peak) peak = hs->pq_count;
        }
    }
finish:
    if (hs->probe_failed) return -2;
    (void)peak;
    if (prof) { prof->states_popped = n_popped; prof->peak_heap_sz = n_expanded; }
    return 0;
#undef MS_DECIDE_SHORT
}

#undef SOKW_IX
#undef SOKN
#undef SOKN_
#undef SOKN__
#endif /* SOKW */
