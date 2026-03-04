#include "sokoban_bfs.h"
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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
 * DISPATCHER
 * ======================================================================== */

int sokoban_solve(const Puzzle *pz, uint8_t *used_dirs) {
    int nb = pz->num_blocks, nh = pz->num_holes;
    if (5 + 5 * nb + nh > 64)
        return solve_hash128(pz, used_dirs);
    int64_t ss = state_space_size(nb, nh);
    if (ss <= DIRECT_LIMIT)
        return solve_direct(pz, used_dirs, (int)ss);
    return solve_hash(pz, used_dirs);
}
