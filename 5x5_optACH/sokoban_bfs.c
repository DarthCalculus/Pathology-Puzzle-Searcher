#include "sokoban_bfs.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

/* ---- Precomputed neighbor table ----
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

/* State space size: NCELLS * (NCELLS+1)^nb * 2^nh (as int64_t to detect overflow) */
static int64_t state_space_size(int nb, int nh) {
    int64_t s = NCELLS;
    for (int i = 0; i < nb; i++) s *= NCELLS + 1;
    for (int i = 0; i < nh; i++) s *= 2;
    return s;
}

/* ---- Threshold: direct indexing vs hash table ----
 * Direct indexing is used when the state space fits comfortably in memory.
 * 8M states = 16 MB visited array — fits well within realistic working sets.
 * Above this, the original hash table approach is used.
 */
#define DIRECT_LIMIT (1 << 23)   /* 8 M states */

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
        ds_tls = calloc(1, sizeof(DirectState));
        ds_tls->gen = 1;
    }
    if (state_space > ds_tls->vis_cap) {
        free(ds_tls->visited);
        free(ds_tls->qs);
        free(ds_tls->qused);
        ds_tls->visited = calloc(state_space, sizeof(uint16_t));
        ds_tls->qs      = malloc(state_space * sizeof(uint32_t));
        ds_tls->qused   = malloc(state_space * sizeof(uint32_t));
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

/* Mixed-radix packing: state = player + NCELLS*(block[0] + (NCELLS+1)*(block[1] + ...)) */
static inline uint32_t mr_pack(int pl, const int *bp, int nb, int hm) {
    uint32_t s = (uint32_t)hm;
    for (int i = nb - 1; i >= 0; i--)
        s = s * (NCELLS + 1) + (uint32_t)bp[i];
    return s * NCELLS + (uint32_t)pl;
}

static inline void mr_unpack(uint32_t s, int *pl, int *bp, int nb, int *hm) {
    *pl = (int)(s % NCELLS);
    s /= NCELLS;
    for (int i = 0; i < nb; i++) {
        bp[i] = (int)(s % (NCELLS + 1));
        s /= (NCELLS + 1);
    }
    *hm = (int)s;
}

static int solve_direct(const Puzzle *pz, uint8_t *used_dirs, int ss_size) {
    int nb = pz->num_blocks;
    int nh = pz->num_holes;

    /* Precompute strides for incremental state updates.
     * stride[i] = NCELLS * (NCELLS+1)^i  (coefficient of block i in packed state)
     * hm_stride = NCELLS * (NCELLS+1)^nb (coefficient of hole_mask)
     */
    int stride[MAX_BLOCKS];
    if (nb > 0) {
        stride[0] = NCELLS;
        for (int i = 1; i < nb; i++) stride[i] = stride[i - 1] * (NCELLS + 1);
    }
    int hm_stride = (nb > 0) ? stride[nb - 1] * (NCELLS + 1) : NCELLS;

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
    const uint32_t walls    = pz->walls;

    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        uint32_t cur = ds->qs[qh];
        uint32_t cur_used = used_dirs ? ds->qused[qh] : 0;
        qh++;

        int pl, bp[MAX_BLOCKS], hm;
        mr_unpack(cur, &pl, bp, nb, &hm);

        uint32_t block_occ = 0;
        for (int i = 0; i < nb; i++)
            if (bp[i] < NCELLS) block_occ |= (1u << bp[i]);

        uint32_t active_holes = 0;
        for (int h = 0; h < nh; h++)
            if (hm & (1 << h)) active_holes |= (1u << pz->hole_pos[h]);

        uint32_t blocked = walls | active_holes;

        for (int d = 0; d < 4; d++) {
            int np = adj[pl][d];
            if (np < 0) continue;
            if (blocked & (1u << np)) continue;

            if (block_occ & (1u << np)) {
                /* Push */
                int bi = -1;
                for (int b = 0; b < nb; b++)
                    if (bp[b] == np) { bi = b; break; }
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int bnp = adj[np][d];
                if (bnp < 0) continue;
                if (walls     & (1u << bnp)) continue;
                if (block_occ & (1u << bnp)) continue;

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
                if (active_holes & (1u << bnp)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == bnp && (hm & (1 << h))) {
                            ih = 1;
                            delta += (CONSUMED - bp[bi]) * stride[bi]
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

#define HT_SIZE  (1 << 24)          /* 16 M slots   */
#define HT_MASK  (HT_SIZE - 1)
#define QSZ      (1 << 24)          /* 16 M entries */

typedef struct {
    uint64_t htk   [HT_SIZE];   /* stored keys        — 128 MB */
    uint32_t ht_gen[HT_SIZE];   /* generation stamps  —  64 MB */
    uint32_t ht_seq;
    uint64_t qs    [QSZ];       /* BFS queue: states  — 128 MB */
    uint32_t qused [QSZ];       /* BFS queue: used push dirs —  64 MB */
} HashState;

static _Thread_local HashState *hs_tls;

static HashState *hs_get(void) {
    if (!hs_tls) { hs_tls = calloc(1, sizeof *hs_tls); hs_tls->ht_seq = 1; }
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
    for (int i = 0; i < 128; i++) {
        uint32_t idx = (uint32_t)((h + i) & HT_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) {
            hs->ht_gen[idx] = hs->ht_seq; hs->htk[idx] = k; return 1;
        }
        if (hs->htk[idx] == k) return 0;
    }
    return 0;  /* table full — treat as already visited */
}

/* 5-bit packing: player[4:0] block0[9:5] block1[14:10] ... hm[last] */
static inline uint64_t pack5(int pl, const int *bp, int nb, int hm) {
    uint64_t s = (uint64_t)pl;
    int sh = 5;
    for (int i = 0; i < nb; i++) { s |= ((uint64_t)bp[i] << sh); sh += 5; }
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
    const uint32_t walls    = pz->walls;

    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        uint64_t cur = hs->qs[qh];
        uint32_t cur_used = used_dirs ? hs->qused[qh] : 0;
        qh++;

        /* Unpack */
        int pl = (int)(cur & 0x1F), sh = 5, bp[MAX_BLOCKS];
        for (int i = 0; i < nb; i++) { bp[i] = (int)((cur >> sh) & 0x1F); sh += 5; }
        int hm = (int)(cur >> sh);   /* hm is at the top — no mask needed */

        uint32_t block_occ = 0;
        for (int i = 0; i < nb; i++)
            if (bp[i] < NCELLS) block_occ |= (1u << bp[i]);

        uint32_t active_holes = 0;
        for (int h = 0; h < nh; h++)
            if (hm & (1 << h)) active_holes |= (1u << pz->hole_pos[h]);

        uint32_t blocked = walls | active_holes;

        for (int d = 0; d < 4; d++) {
            int np = adj[pl][d];
            if (np < 0) continue;
            if (blocked & (1u << np)) continue;

            if (block_occ & (1u << np)) {
                /* Push */
                int bi = -1;
                for (int b = 0; b < nb; b++)
                    if (bp[b] == np) { bi = b; break; }
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int bnp = adj[np][d];
                if (bnp < 0) continue;
                if (walls     & (1u << bnp)) continue;
                if (block_occ & (1u << bnp)) continue;

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
                if (active_holes & (1u << bnp)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == bnp && (hm & (1 << h))) {
                            new_bpos = CONSUMED;
                            nhm &= ~(1 << h);
                            break;
                        }
                    }
                }
                uint64_t ns = (cur & ~0x1FULL) | (uint64_t)np;
                int bsh = 5 * (bi + 1);
                ns = (ns & ~(0x1FULL << bsh)) | ((uint64_t)new_bpos << bsh);
                if (nhm != hm) {
                    int hmsh = 5 * (nb + 1);
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

                uint64_t ns = (cur & ~0x1FULL) | (uint64_t)np;
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
 * Bitmask BFS: all 25 cells fit in a uint32_t, so each level of BFS can
 * be expanded with 4 shift operations rather than iterating cell-by-cell
 * through the adj table.  Directions on the 5×5 grid:
 *   Up    — shift right by 5  (no wrap guard needed; high bits go to 0)
 *   Down  — shift left  by 5  (bits above 24 masked out by free_mask)
 *   Right — shift left  by 1, masking col 4 to prevent row wraparound
 *   Left  — shift right by 1, masking col 0 to prevent row wraparound
 * ======================================================================== */

#define COL0_MASK  0x108421u   /* bits 0,5,10,15,20  — column 0 */
#define COL4_MASK  0x1084210u  /* bits 4,9,14,19,24  — column 4 */
#define ALL_CELLS  0x1FFFFFFu  /* bits 0-24          — all 25 cells */

/* walk_dists_from: BFS distances from start to all reachable cells.
 * out[i] = distance, or -1 if unreachable. */
static void walk_dists_from(uint32_t blocked, int start, int8_t *out) {
    memset(out, -1, NCELLS);
    if ((blocked >> start) & 1) return;
    uint32_t free_mask = ~blocked & ALL_CELLS;
    uint32_t reached   = 1u << start;
    uint32_t frontier  = reached;
    out[start] = 0;
    int8_t dist = 0;
    while (frontier) {
        dist++;
        uint32_t nxt = (frontier >> 5)
                     | (frontier << 5)
                     | ((frontier & ~COL4_MASK) << 1)
                     | ((frontier & ~COL0_MASK) >> 1);
        frontier = nxt & free_mask & ~reached;
        reached |= frontier;
        for (uint32_t tmp = frontier; tmp; tmp &= tmp - 1)
            out[__builtin_ctz(tmp)] = dist;
    }
}

/* ========================================================================
 * PUSH-BASED DIJKSTRA SOLVER — 64-bit state key
 *
 * State key: (canonical_player_cell, block_positions, hole_mask).
 *   canonical_player_cell = lowest-index cell reachable from player.
 *   The exact player position is tracked per heap entry (not in key).
 *
 * Edge cost = walk_distance(player_pos, push_from_cell) + 1.
 * Dijkstra with lazy deletion — edge costs in [1, 25] on a 5x5 grid.
 * ~352 MB per thread, allocated on first use.
 * ======================================================================== */

#define HTP_SIZE  (1 << 22)    /* 4 M hash-table slots               */
#define HTP_MASK  (HTP_SIZE - 1)
#define HP64_SIZE (1 << 20)    /* 1 M heap entries                   */

typedef struct {
    int      prio;        /* priority: total cost (walks + pushes)  */
    int      player_pos;  /* exact player cell (not in state key)   */
    uint32_t used;        /* accumulated used-direction bits         */
    uint32_t _pad;
    uint64_t state;       /* packed state key                        */
} HeapEntry64;            /* 24 bytes                                */

typedef struct {
    uint64_t    htk    [HTP_SIZE];   /* stored keys        — 128 MB */
    uint32_t    ht_gen [HTP_SIZE];   /* generation stamps  —  64 MB */
    int32_t     ht_cost[HTP_SIZE];   /* min cost per state —  64 MB */
    uint32_t    ht_seq;
    HeapEntry64 heap   [HP64_SIZE];  /* min-heap           —  96 MB */
    int         heap_sz;
} HashStatePush64;        /* total ~352 MB                           */

static _Thread_local HashStatePush64 *hsp64_tls;

static HashStatePush64 *hsp64_get(void) {
    if (!hsp64_tls) {
        hsp64_tls = calloc(1, sizeof *hsp64_tls);
        hsp64_tls->ht_seq = 1;
    }
    return hsp64_tls;
}

static void hsp64_clear(HashStatePush64 *hs) {
    if (++hs->ht_seq == 0) {
        memset(hs->ht_gen, 0, sizeof(hs->ht_gen));
        hs->ht_seq = 1;
    }
    hs->heap_sz = 0;
}

/* Update stored cost for key k.
 * Returns 1 if new_cost is an improvement (caller should enqueue). */
static inline int hsp64_update(HashStatePush64 *hs, uint64_t k, int new_cost) {
    uint64_t h = h64(k) & HTP_MASK;
    for (int i = 0; i < 128; i++) {
        uint32_t idx = (uint32_t)((h + i) & HTP_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) {
            hs->ht_gen[idx] = hs->ht_seq;
            hs->htk[idx]    = k;
            hs->ht_cost[idx] = new_cost;
            return 1;
        }
        if (hs->htk[idx] == k) {
            if (new_cost < hs->ht_cost[idx]) {
                hs->ht_cost[idx] = new_cost;
                return 1;
            }
            return 0;
        }
    }
    return 0; /* table full — skip */
}

/* Return stored min cost, or INT_MAX if key not in table. */
static inline int hsp64_get_cost(const HashStatePush64 *hs, uint64_t k) {
    uint64_t h = h64(k) & HTP_MASK;
    for (int i = 0; i < 128; i++) {
        uint32_t idx = (uint32_t)((h + i) & HTP_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) return INT_MAX;
        if (hs->htk[idx] == k) return hs->ht_cost[idx];
    }
    return INT_MAX;
}

/* Min-heap: push entry. */
static inline void heap64_push(HashStatePush64 *hs, HeapEntry64 e) {
    int i = hs->heap_sz++;
    hs->heap[i] = e;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (hs->heap[p].prio <= hs->heap[i].prio) break;
        HeapEntry64 tmp  = hs->heap[p];
        hs->heap[p] = hs->heap[i];
        hs->heap[i] = tmp;
        i = p;
    }
}

/* Min-heap: pop minimum entry. */
static inline HeapEntry64 heap64_pop(HashStatePush64 *hs) {
    HeapEntry64 top = hs->heap[0];
    int sz = --hs->heap_sz;
    hs->heap[0] = hs->heap[sz];
    int i = 0;
    for (;;) {
        int l = (i << 1) | 1, r = l + 1, m = i;
        if (l < sz && hs->heap[l].prio < hs->heap[m].prio) m = l;
        if (r < sz && hs->heap[r].prio < hs->heap[m].prio) m = r;
        if (m == i) break;
        HeapEntry64 tmp  = hs->heap[m];
        hs->heap[m] = hs->heap[i];
        hs->heap[i] = tmp;
        i = m;
    }
    return top;
}

static int solve_push64(const Puzzle *pz, uint8_t *used_dirs) {
    HashStatePush64 *hs = hsp64_get();
    hsp64_clear(hs);

    const int      nb       = pz->num_blocks;
    const int      nh       = pz->num_holes;
    const int      exit_pos = pz->exit_pos;
    const uint32_t walls    = pz->walls;

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    /* Enqueue initial state at cost 0. */
    uint64_t init_st = pack5(pz->player_start, ib, nb, ihm);
    {
        HeapEntry64 e0;
        e0.prio = 0; e0.player_pos = pz->player_start;
        e0.used = 0; e0._pad = 0; e0.state = init_st;
        hsp64_update(hs, init_st, 0);
        heap64_push(hs, e0);
    }

    int      best_win  = INT_MAX;
    uint32_t best_used = 0;

    while (hs->heap_sz > 0) {
        HeapEntry64 e = heap64_pop(hs);
        if (e.prio >= best_win) break; /* Dijkstra: can't improve */

        /* Lazy deletion: skip if a shorter path was already found. */
        if (hsp64_get_cost(hs, e.state) < e.prio) continue;

        /* Unpack blocks and hole mask (skip canonical player cell in bits[4:0]). */
        int sh = 5, bp[MAX_BLOCKS];
        for (int i = 0; i < nb; i++) { bp[i] = (int)((e.state >> sh) & 0x1F); sh += 5; }
        int hm = (int)(e.state >> sh);

        uint32_t blk_occ = 0;
        for (int i = 0; i < nb; i++) if (bp[i] < NCELLS) blk_occ |= (1u << bp[i]);
        uint32_t cur_holes = 0;
        for (int h = 0; h < nh; h++) if (hm & (1 << h)) cur_holes |= (1u << pz->hole_pos[h]);

        /* Walk distances from exact player position. */
        int8_t wdist[NCELLS];
        walk_dists_from(walls | cur_holes | blk_occ, e.player_pos, wdist);

        /* Win check: can player walk to exit? */
        if (wdist[exit_pos] >= 0) {
            int wc = e.prio + (int)wdist[exit_pos];
            if (wc < best_win) { best_win = wc; best_used = e.used; }
        }

        /* Enumerate valid pushes. */
        for (int bi = 0; bi < nb; bi++) {
            if (bp[bi] >= NCELLS) continue; /* block consumed */
            int bpos = bp[bi];

            for (int d = 0; d < 4; d++) {
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int pfr = adj[bpos][d ^ 2]; /* push-from cell (player must be here) */
                if (pfr < 0 || wdist[pfr] < 0) continue;

                int lnd = adj[bpos][d]; /* landing cell for block */
                if (lnd < 0 || (walls & (1u << lnd)) || (blk_occ & (1u << lnd))) continue;

                /* Compute new block position (handle hole consumption). */
                int new_bpos = lnd, nhm = hm;
                if (cur_holes & (1u << lnd)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == lnd && (hm & (1 << h))) {
                            new_bpos = CONSUMED;
                            nhm &= ~(1 << h);
                            break;
                        }
                    }
                }

                int nbp[MAX_BLOCKS];
                for (int i = 0; i < nb; i++) nbp[i] = bp[i];
                nbp[bi] = new_bpos;

                /* Player lands at bpos. */
                uint64_t ns = pack5(bpos, nbp, nb, nhm);
                int      nc = e.prio + (int)wdist[pfr] + 1;
                uint32_t nu = e.used | (1u << (bi * 4 + d));

                if (hsp64_update(hs, ns, nc)) {
                    if (hs->heap_sz >= HP64_SIZE) return -2;
                    HeapEntry64 ne;
                    ne.prio = nc; ne.player_pos = bpos;
                    ne.used = nu; ne._pad = 0; ne.state = ns;
                    heap64_push(hs, ne);
                }
            }
        }
    }

    if (best_win == INT_MAX) return -1;
    if (used_dirs)
        for (int i = 0; i < nb; i++)
            used_dirs[i] = (uint8_t)((best_used >> (i * 4)) & 0xF);
    return best_win;
}

/* ========================================================================
 * WIDE HASH TABLE SOLVER  (5 + 5*nb + nh > 64 bits)
 *
 * Uses __uint128_t for state keys and used-direction bitmask.
 * Smaller table (4 M slots) — crowded boards have tiny reachable spaces.
 * 208 MB per thread, allocated on first use.
 * ======================================================================== */

#define HT128_SIZE  (1 << 22)
#define HT128_MASK  (HT128_SIZE - 1)
#define QSZ128      (1 << 22)

typedef struct {
    __uint128_t htk   [HT128_SIZE];   /* stored keys       —  64 MB */
    uint32_t    ht_gen[HT128_SIZE];   /* generation stamps —  16 MB */
    uint32_t    ht_seq;
    __uint128_t qs    [QSZ128];       /* BFS queue: states —  64 MB */
    __uint128_t qused [QSZ128];       /* BFS queue: dirs   —  64 MB */
} HashState128;

static _Thread_local HashState128 *hs128_tls;

static HashState128 *hs128_get(void) {
    if (!hs128_tls) { hs128_tls = calloc(1, sizeof *hs128_tls); hs128_tls->ht_seq = 1; }
    return hs128_tls;
}

static void hs128_clear(HashState128 *hs) {
    if (++hs->ht_seq == 0) {
        memset(hs->ht_gen, 0, sizeof(hs->ht_gen));
        hs->ht_seq = 1;
    }
}

static inline uint64_t h128(__uint128_t x) {
    return h64((uint64_t)x ^ (uint64_t)(x >> 64));
}

static inline int hs128_mark(HashState128 *hs, __uint128_t k) {
    uint64_t h = h128(k) & HT128_MASK;
    for (int i = 0; i < 128; i++) {
        uint32_t idx = (uint32_t)((h + i) & HT128_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) {
            hs->ht_gen[idx] = hs->ht_seq; hs->htk[idx] = k; return 1;
        }
        if (hs->htk[idx] == k) return 0;
    }
    return 0;
}

static inline __uint128_t pack128(int pl, const int *bp, int nb, int hm) {
    __uint128_t s = (__uint128_t)pl;
    int sh = 5;
    for (int i = 0; i < nb; i++) { s |= ((__uint128_t)bp[i] << sh); sh += 5; }
    s |= ((__uint128_t)hm << sh);
    return s;
}

static int solve_hash128(const Puzzle *pz, uint8_t *used_dirs) {
    HashState128 *hs = hs128_get();
    hs128_clear(hs);

    int nb = pz->num_blocks;
    int nh = pz->num_holes;

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    __uint128_t st = pack128(pz->player_start, ib, nb, ihm);
    hs128_mark(hs, st);

    int qh = 0, qt = 0, ql = 0, dist = -1;
    hs->qs[qt] = st;
    if (used_dirs) hs->qused[qt] = 0;
    qt++;

    const int      exit_pos = pz->exit_pos;
    const uint32_t walls    = pz->walls;

    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        __uint128_t cur      = hs->qs[qh];
        __uint128_t cur_used = used_dirs ? hs->qused[qh] : 0;
        qh++;

        /* Unpack */
        int pl = (int)(cur & 0x1F), sh = 5, bp[MAX_BLOCKS];
        for (int i = 0; i < nb; i++) { bp[i] = (int)((cur >> sh) & 0x1F); sh += 5; }
        int hm = (int)(cur >> sh);   /* hm at the top — no mask needed */

        uint32_t block_occ = 0;
        for (int i = 0; i < nb; i++)
            if (bp[i] < NCELLS) block_occ |= (1u << bp[i]);

        uint32_t active_holes = 0;
        for (int h = 0; h < nh; h++)
            if (hm & (1 << h)) active_holes |= (1u << pz->hole_pos[h]);

        uint32_t blocked = walls | active_holes;

        for (int d = 0; d < 4; d++) {
            int np = adj[pl][d];
            if (np < 0) continue;
            if (blocked & (1u << np)) continue;

            if (block_occ & (1u << np)) {
                /* Push */
                int bi = -1;
                for (int b = 0; b < nb; b++)
                    if (bp[b] == np) { bi = b; break; }
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int bnp = adj[np][d];
                if (bnp < 0) continue;
                if (walls     & (1u << bnp)) continue;
                if (block_occ & (1u << bnp)) continue;

                if (np == exit_pos) {
                    if (used_dirs) {
                        __uint128_t u = cur_used | ((__uint128_t)1 << (bi * 4 + d));
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (uint8_t)((u >> (i * 4)) & 0xF);
                    }
                    return dist + 1;
                }

                /* Build new state via in-place bit modification */
                int new_bpos = bnp;
                int nhm = hm;
                if (active_holes & (1u << bnp)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == bnp && (hm & (1 << h))) {
                            new_bpos = CONSUMED;
                            nhm &= ~(1 << h);
                            break;
                        }
                    }
                }
                __uint128_t ns = (cur & ~(__uint128_t)0x1F) | (__uint128_t)np;
                int bsh = 5 * (bi + 1);
                ns = (ns & ~((__uint128_t)0x1F << bsh)) | ((__uint128_t)new_bpos << bsh);
                if (nhm != hm) {
                    int hmsh = 5 * (nb + 1);
                    __uint128_t hm_mask = ((__uint128_t)((1 << nh) - 1)) << hmsh;
                    ns = (ns & ~hm_mask) | ((__uint128_t)nhm << hmsh);
                }

                if (hs128_mark(hs, ns)) {
                    if (qt >= QSZ128) return -2;
                    hs->qs[qt] = ns;
                    if (used_dirs)
                        hs->qused[qt] = cur_used | ((__uint128_t)1 << (bi * 4 + d));
                    qt++;
                }

            } else {
                /* Free move */
                if (np == exit_pos) {
                    if (used_dirs) {
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (uint8_t)((cur_used >> (i * 4)) & 0xF);
                    }
                    return dist + 1;
                }

                __uint128_t ns = (cur & ~(__uint128_t)0x1F) | (__uint128_t)np;
                if (hs128_mark(hs, ns)) {
                    if (qt >= QSZ128) return -2;
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
 * PUSH-BASED DIJKSTRA SOLVER — 128-bit state key
 *
 * Used when 5 + 5*nb + nh > 64 bits.
 * Smaller heap (1 M entries) — crowded boards have fewer reachable states.
 * ~144 MB per thread, allocated on first use.
 * ======================================================================== */

#define HP128_SIZE (1 << 20)   /* 1 M heap entries */

typedef struct {
    __uint128_t state;
    __uint128_t used;
    int         prio;
    int         player_pos;
} HeapEntry128;   /* ~48 bytes with alignment */

typedef struct {
    __uint128_t  htk    [HT128_SIZE];  /* stored keys        —  64 MB */
    uint32_t     ht_gen [HT128_SIZE];  /* generation stamps  —  16 MB */
    int32_t      ht_cost[HT128_SIZE];  /* min cost per state —  16 MB */
    uint32_t     ht_seq;
    HeapEntry128 heap   [HP128_SIZE];  /* min-heap           —  ~48 MB */
    int          heap_sz;
} HashStatePush128;   /* total ~144 MB */

static _Thread_local HashStatePush128 *hsp128_tls;

static HashStatePush128 *hsp128_get(void) {
    if (!hsp128_tls) {
        hsp128_tls = calloc(1, sizeof *hsp128_tls);
        hsp128_tls->ht_seq = 1;
    }
    return hsp128_tls;
}

static void hsp128_clear(HashStatePush128 *hs) {
    if (++hs->ht_seq == 0) {
        memset(hs->ht_gen, 0, sizeof(hs->ht_gen));
        hs->ht_seq = 1;
    }
    hs->heap_sz = 0;
}

static inline int hsp128_update(HashStatePush128 *hs, __uint128_t k, int new_cost) {
    uint64_t h = h128(k) & HT128_MASK;
    for (int i = 0; i < 128; i++) {
        uint32_t idx = (uint32_t)((h + i) & HT128_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) {
            hs->ht_gen[idx]  = hs->ht_seq;
            hs->htk[idx]     = k;
            hs->ht_cost[idx] = new_cost;
            return 1;
        }
        if (hs->htk[idx] == k) {
            if (new_cost < hs->ht_cost[idx]) { hs->ht_cost[idx] = new_cost; return 1; }
            return 0;
        }
    }
    return 0;
}

static inline int hsp128_get_cost(const HashStatePush128 *hs, __uint128_t k) {
    uint64_t h = h128(k) & HT128_MASK;
    for (int i = 0; i < 128; i++) {
        uint32_t idx = (uint32_t)((h + i) & HT128_MASK);
        if (hs->ht_gen[idx] != hs->ht_seq) return INT_MAX;
        if (hs->htk[idx] == k) return hs->ht_cost[idx];
    }
    return INT_MAX;
}

static inline void heap128_push(HashStatePush128 *hs, HeapEntry128 e) {
    int i = hs->heap_sz++;
    hs->heap[i] = e;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (hs->heap[p].prio <= hs->heap[i].prio) break;
        HeapEntry128 tmp = hs->heap[p];
        hs->heap[p] = hs->heap[i];
        hs->heap[i] = tmp;
        i = p;
    }
}

static inline HeapEntry128 heap128_pop(HashStatePush128 *hs) {
    HeapEntry128 top = hs->heap[0];
    int sz = --hs->heap_sz;
    hs->heap[0] = hs->heap[sz];
    int i = 0;
    for (;;) {
        int l = (i << 1) | 1, r = l + 1, m = i;
        if (l < sz && hs->heap[l].prio < hs->heap[m].prio) m = l;
        if (r < sz && hs->heap[r].prio < hs->heap[m].prio) m = r;
        if (m == i) break;
        HeapEntry128 tmp = hs->heap[m];
        hs->heap[m] = hs->heap[i];
        hs->heap[i] = tmp;
        i = m;
    }
    return top;
}

static int solve_push128(const Puzzle *pz, uint8_t *used_dirs) {
    HashStatePush128 *hs = hsp128_get();
    hsp128_clear(hs);

    const int      nb       = pz->num_blocks;
    const int      nh       = pz->num_holes;
    const int      exit_pos = pz->exit_pos;
    const uint32_t walls    = pz->walls;

    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    __uint128_t init_st = pack128(pz->player_start, ib, nb, ihm);
    {
        HeapEntry128 e0;
        e0.state = init_st; e0.used = 0;
        e0.prio = 0; e0.player_pos = pz->player_start;
        hsp128_update(hs, init_st, 0);
        heap128_push(hs, e0);
    }

    int         best_win  = INT_MAX;
    __uint128_t best_used = 0;

    while (hs->heap_sz > 0) {
        HeapEntry128 e = heap128_pop(hs);
        if (e.prio >= best_win) break;

        if (hsp128_get_cost(hs, e.state) < e.prio) continue;

        int sh = 5, bp[MAX_BLOCKS];
        for (int i = 0; i < nb; i++) { bp[i] = (int)((e.state >> sh) & 0x1F); sh += 5; }
        int hm = (int)(e.state >> sh);

        uint32_t blk_occ = 0;
        for (int i = 0; i < nb; i++) if (bp[i] < NCELLS) blk_occ |= (1u << bp[i]);
        uint32_t cur_holes = 0;
        for (int h = 0; h < nh; h++) if (hm & (1 << h)) cur_holes |= (1u << pz->hole_pos[h]);

        int8_t wdist[NCELLS];
        walk_dists_from(walls | cur_holes | blk_occ, e.player_pos, wdist);

        if (wdist[exit_pos] >= 0) {
            int wc = e.prio + (int)wdist[exit_pos];
            if (wc < best_win) { best_win = wc; best_used = e.used; }
        }

        for (int bi = 0; bi < nb; bi++) {
            if (bp[bi] >= NCELLS) continue;
            int bpos = bp[bi];

            for (int d = 0; d < 4; d++) {
                if (!(pz->block_pushable[bi] & (1 << d))) continue;

                int pfr = adj[bpos][d ^ 2];
                if (pfr < 0 || wdist[pfr] < 0) continue;

                int lnd = adj[bpos][d];
                if (lnd < 0 || (walls & (1u << lnd)) || (blk_occ & (1u << lnd))) continue;

                int new_bpos = lnd, nhm = hm;
                if (cur_holes & (1u << lnd)) {
                    for (int h = 0; h < nh; h++) {
                        if (pz->hole_pos[h] == lnd && (hm & (1 << h))) {
                            new_bpos = CONSUMED;
                            nhm &= ~(1 << h);
                            break;
                        }
                    }
                }

                int nbp[MAX_BLOCKS];
                for (int i = 0; i < nb; i++) nbp[i] = bp[i];
                nbp[bi] = new_bpos;

                /* Player lands at bpos. */
                __uint128_t ns = pack128(bpos, nbp, nb, nhm);
                int         nc = e.prio + (int)wdist[pfr] + 1;
                __uint128_t nu = e.used | ((__uint128_t)1 << (bi * 4 + d));

                if (hsp128_update(hs, ns, nc)) {
                    if (hs->heap_sz >= HP128_SIZE) return -2;
                    HeapEntry128 ne;
                    ne.state = ns; ne.used = nu;
                    ne.prio = nc; ne.player_pos = bpos;
                    heap128_push(hs, ne);
                }
            }
        }
    }

    if (best_win == INT_MAX) return -1;
    if (used_dirs)
        for (int i = 0; i < nb; i++)
            used_dirs[i] = (uint8_t)((best_used >> (i * 4)) & 0xF);
    return best_win;
}

/* ========================================================================
 * DISPATCHER
 * ======================================================================== */

int sokoban_solve(const Puzzle *pz, uint8_t *used_dirs) {
    int nb = pz->num_blocks, nh = pz->num_holes;
    if (5 + 5 * nb + nh > 64)
        return solve_push128(pz, used_dirs);
    return solve_push64(pz, used_dirs);
}
