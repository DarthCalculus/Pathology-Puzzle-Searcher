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
    {10, 16, -1, -1}, {11, 17, -1, 15}, {12, 18, -1, 16},
    {13, 19, -1, 17}, {14, -1, -1, 18},
};

/* ---- Per-thread solver state ----
 *
 * Uses a direct-indexed uint16_t visited array with generation counter
 * instead of the old 192 MB hash table.  The mixed-radix state encoding
 * maps each (player, blocks, hole_mask) tuple to a unique index in
 * [0, state_space).
 *
 * Memory per thread (3 blocks, 3 holes):
 *   visited: 2.83 MB   qs: 5.66 MB   qused: 5.66 MB   total: ~14 MB
 * The visited array fits in L2 cache (~3 ns access vs ~100 ns DRAM).
 */
typedef struct {
    uint16_t *visited;  /* generation-counter visited array */
    uint32_t *qs;       /* BFS queue: packed states */
    uint32_t *qused;    /* BFS queue: used push direction bitmasks */
    uint16_t  gen;      /* current generation counter */
    int       vis_cap;  /* allocated visited array capacity */
} SolverState;

static _Thread_local SolverState *ss_tls;

static SolverState *ss_get(int state_space) {
    if (!ss_tls) {
        ss_tls = calloc(1, sizeof(SolverState));
        ss_tls->gen = 1;
    }
    if (state_space > ss_tls->vis_cap) {
        free(ss_tls->visited);
        free(ss_tls->qs);
        free(ss_tls->qused);
        ss_tls->visited = calloc(state_space, sizeof(uint16_t));
        ss_tls->qs      = malloc(state_space * sizeof(uint32_t));
        ss_tls->qused   = malloc(state_space * sizeof(uint32_t));
        ss_tls->vis_cap = state_space;
    }
    return ss_tls;
}

/* Advance generation counter; full reset on 16-bit wrap. */
static void vis_clear(SolverState *ss) {
    if (++ss->gen == 0) {
        memset(ss->visited, 0, ss->vis_cap * sizeof(uint16_t));
        ss->gen = 1;
    }
}

/* Mark state as visited. Returns 1 if newly visited, 0 if already seen. */
static inline int vis_mark(SolverState *ss, uint32_t state) {
    if (ss->visited[state] == ss->gen) return 0;
    ss->visited[state] = ss->gen;
    return 1;
}

/* ---- Mixed-radix state encoding ----
 *
 * state = player + 20 * (block[0] + 21 * (block[1] + 21 * (... + 21 * hole_mask)))
 *
 * Player: 0-19 (20 values), Block: 0-20 (21 values, CONSUMED=20),
 * Hole mask: 0 to 2^nh - 1.
 *
 * 3 blocks, 3 holes: 20 * 21^3 * 8 = 1,481,760 states (2.83 MB visited array).
 */
static inline uint32_t pack(int pl, const int *bp, int nb, int hm) {
    uint32_t s = (uint32_t)hm;
    for (int i = nb - 1; i >= 0; i--)
        s = s * 21 + (uint32_t)bp[i];
    return s * 20 + (uint32_t)pl;
}

static inline void unpack(uint32_t s, int *pl, int *bp, int nb, int *hm) {
    *pl = (int)(s % 20);
    s /= 20;
    for (int i = 0; i < nb; i++) {
        bp[i] = (int)(s % 21);
        s /= 21;
    }
    *hm = (int)s;
}

/* State space size: 20 * 21^nb * 2^nh */
static int state_space_size(int nb, int nh) {
    int s = 20;
    for (int i = 0; i < nb; i++) s *= 21;
    for (int i = 0; i < nh; i++) s *= 2;
    return s;
}

/* ---- Solver ----
 *
 * Optimisations on the hot path:
 *   (a) block_occ bitmask     — O(1) "is there a block here?" check
 *   (b) merged blocked mask   — walls | active_holes, one check per direction
 *   (c) neighbor table        — no division/modulo/bounds arithmetic
 *   (d) incremental updates   — arithmetic delta instead of full pack()
 *   (e) direct-indexed visited — no hash, no collision probing
 */
int sokoban_solve(const Puzzle *pz, uint8_t *used_dirs) {
    int nb = pz->num_blocks;
    int nh = pz->num_holes;
    int ss_size = state_space_size(nb, nh);

    /* Precompute strides for incremental state updates.
     * stride[i] = 20 * 21^i  (coefficient of block i in packed state)
     * hm_stride = 20 * 21^nb (coefficient of hole_mask)
     */
    int stride[MAX_BLOCKS];
    if (nb > 0) {
        stride[0] = 20;
        for (int i = 1; i < nb; i++) stride[i] = stride[i - 1] * 21;
    }
    int hm_stride = (nb > 0) ? stride[nb - 1] * 21 : 20;

    SolverState *ss = ss_get(ss_size);
    vis_clear(ss);

    /* Initial state */
    int ib[MAX_BLOCKS];
    for (int i = 0; i < nb; i++) ib[i] = pz->block_pos[i];
    int ihm = (1 << nh) - 1;

    uint32_t st = pack(pz->player_start, ib, nb, ihm);
    vis_mark(ss, st);

    int qh = 0, qt = 0, ql = 0, dist = -1;
    ss->qs[qt] = st;
    if (used_dirs) ss->qused[qt] = 0;
    qt++;

    const int      exit_pos = pz->exit_pos;
    const uint32_t walls    = pz->walls;

    while (qh < qt) {
        if (qh == ql) { dist++; ql = qt; }
        uint32_t cur = ss->qs[qh];
        uint32_t cur_used = used_dirs ? ss->qused[qh] : 0;
        qh++;

        /* Unpack current state */
        int pl, bp[MAX_BLOCKS], hm;
        unpack(cur, &pl, bp, nb, &hm);

        /* (a) Block occupancy bitmask */
        uint32_t block_occ = 0;
        for (int i = 0; i < nb; i++)
            if (bp[i] < NCELLS) block_occ |= (1u << bp[i]);

        /* Active-hole bitmask */
        uint32_t active_holes = 0;
        for (int h = 0; h < nh; h++)
            if (hm & (1 << h)) active_holes |= (1u << pz->hole_pos[h]);

        /* (b) Merged blocked mask for player movement */
        uint32_t blocked = walls | active_holes;

        for (int d = 0; d < 4; d++) {
            /* (c) Neighbor table lookup — no div/mod/bounds check */
            int np = adj[pl][d];
            if (np < 0) continue;
            if (blocked & (1u << np)) continue;

            if (block_occ & (1u << np)) {
                /* Push — rare path */
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

                /* (d) Incremental state update */
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

                /* (e) Direct-indexed visited check */
                uint32_t ns = (uint32_t)((int)cur + delta);
                if (vis_mark(ss, ns)) {
                    ss->qs[qt] = ns;
                    if (used_dirs)
                        ss->qused[qt] = cur_used | (1u << (bi * 4 + d));
                    qt++;
                }

            } else {
                /* Free move — hot path */
                if (np == exit_pos) {
                    if (used_dirs) {
                        for (int i = 0; i < nb; i++)
                            used_dirs[i] = (cur_used >> (i * 4)) & 0xF;
                    }
                    return dist + 1;
                }

                /* (d) Incremental: just swap player digit */
                uint32_t ns = (uint32_t)((int)cur + (np - pl));
                if (vis_mark(ss, ns)) {
                    ss->qs[qt] = ns;
                    if (used_dirs) ss->qused[qt] = cur_used;
                    qt++;
                }
            }
        }
    }
    return -1;
}
