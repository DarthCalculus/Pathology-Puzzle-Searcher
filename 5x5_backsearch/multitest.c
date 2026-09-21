/* multitest — check sokoban_solve_multi() against sokoban_solve_cutoff() on
 * real generator states from a --harvest file.  For each record, every cell of
 * the player's component (over committed cells, minus blocks/holes/exit) is a
 * start with cutoff depth-2+dist, exactly as the bulk walk-back generation uses
 * it.  Build: cc -O3 -I. -o multitest multitest.c sokoban_bfs.c
 * Run:   ./multitest CORPUS.bin [maxrecords] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "sokoban_bfs.h"
#include "harvest_format.h"
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); HarvestFileHeader h; fread(&h, sizeof h, 1, f); fseek(f, h.argv_blob_len, SEEK_CUR);
    long maxr = argc > 2 ? atol(argv[2]) : 200000;
    int multi_only = argc > 3 && !strcmp(argv[3], "--multi-only");
    int R = h.grid_rows, C = h.grid_cols, N = R * C;
    uint64_t active = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);
    sokoban_set_grid(R, C); sokoban_init(); sokoban_set_decision_only(1);
    HarvestRecord r; long recs = 0, multi_calls = 0, starts = 0, mism = 0, fallback = 0; long long pop_multi = 0, pop_single = 0; long long bd_multi[16] = {0}, bd_single[16] = {0}, bd_calls[16] = {0}, bd_starts[16] = {0}; double t_multi = 0, t_single = 0;
    while (fread(&r, sizeof r, 1, f) == 1 && recs < maxr) {
        if (r.outcome != HARVEST_OUTCOME_ACCEPTED) continue;
        recs++;
        Puzzle pz; memset(&pz, 0, sizeof pz);
        pz.exit_pos = r.exit_pos; pz.player_start = r.player_pos; pz.num_blocks = r.nblocks;
        uint64_t blk = 0, hol = 0;
        for (int b = 0; b < r.nblocks; b++) { pz.block_pos[b] = r.block_pos[b]; pz.block_pushable[b] = r.block_mask[b]; blk |= 1ULL << r.block_pos[b]; }
        pz.num_holes = r.nholes;
        for (int k = 0; k < r.nholes; k++) { pz.hole_pos[k] = r.hole_pos[k]; hol |= 1ULL << r.hole_pos[k]; }
        pz.walls = active & ~r.committed_empty;
        uint64_t passable = r.committed_empty & ~blk & ~hol & ~(1ULL << r.exit_pos) & active;
        int8_t dist[64]; memset(dist, -1, 64);
        if (!(passable >> r.player_pos & 1)) continue;
        int q[64], qh = 0, qt = 0; q[qt++] = r.player_pos; dist[r.player_pos] = 0;
        while (qh < qt) { int c = q[qh++], rr = c / C, cc = c % C; int nb4[4] = { rr > 0 ? c - C : -1, rr < R - 1 ? c + C : -1, cc > 0 ? c - 1 : -1, cc < C - 1 ? c + 1 : -1 };
            for (int k = 0; k < 4; k++) { int nn = nb4[k]; if (nn >= 0 && (passable >> nn & 1) && dist[nn] < 0) { dist[nn] = dist[c] + 1; q[qt++] = nn; } } }
        int8_t st[24]; int16_t cut[24]; uint32_t pred[24]; int n = 0;
        for (int i = 1; i < qt && n < 24; i++) { st[n] = q[i]; cut[n] = r.depth - 2 + dist[q[i]]; n++; }
        for (int j = 0; j < n; j++) { pred[j] = 0; int rj = st[j] / C, cj = st[j] % C;
            for (int k = 0; k < n; k++) if (dist[st[k]] == dist[st[j]] - 1 && abs(st[k] / C - rj) + abs(st[k] % C - cj) == 1) pred[j] |= 1u << k; }
        if (!n) continue;
        uint8_t out[24]; BfsProfile prof = {0};
        double t0 = now(); int rc = sokoban_solve_multi(&pz, st, cut, pred, NULL, n, out, &prof); t_multi += now() - t0;
        multi_calls++; pop_multi += prof.states_popped; int bd = r.depth / 10; if (bd > 15) bd = 15; bd_multi[bd] += prof.states_popped; bd_calls[bd]++;
        if (rc < 0) { fallback++; continue; }
        for (int i = 0; i < n && !multi_only; i++) {
            pz.player_start = st[i]; BfsProfile p2 = {0};
            t0 = now(); int x = sokoban_solve_cutoff(&pz, NULL, &p2, cut[i]); t_single += now() - t0;
            starts++; pop_single += p2.states_popped; bd_single[bd] += p2.states_popped; bd_starts[bd]++;
            int single_short = (x >= 0);
            if (x == -2 || x == -3) continue;
            if (single_short != out[i]) { mism++; if (mism <= 5) fprintf(stderr, "MISMATCH rec %ld start %d cut %d: multi %d single %d\n", recs, st[i], cut[i], out[i], x); }
        }
    }
    printf("records %ld  multi calls %ld (fallback %ld)  starts %ld (avg %.1f/call)  mismatches %ld\n", recs, multi_calls, fallback, starts, multi_calls ? (double)starts / multi_calls : 0, mism);
    printf("time: multi %.3fs (%.1f us/call)  singles %.3fs (%.1f us/start)  -> multi is %.2fx the cost of one single, covers %.1f starts\n",
           t_multi, multi_calls ? t_multi * 1e6 / multi_calls : 0, t_single, starts ? t_single * 1e6 / starts : 0,
           (starts && multi_calls) ? (t_multi / multi_calls) / (t_single / starts) : 0, multi_calls ? (double)starts / multi_calls : 0);
    printf("pops: multi %lld  singles %lld  (ratio %.2f)\n", pop_multi, pop_single, pop_single ? (double)pop_multi / pop_single : 0);
    printf("  depth   calls  starts/call  singles-pops/start  multi-pops/singles-pops\n");
    for (int b = 0; b < 16; b++) if (bd_calls[b]) printf("  %3d-%-3d %6lld  %5.1f  %8.1f  %5.2f\n", b*10, b*10+9, bd_calls[b], (double)bd_starts[b]/bd_calls[b], bd_starts[b] ? (double)bd_single[b]/bd_starts[b] : 0, bd_single[b] ? (double)bd_multi[b]/bd_single[b] : 0);
    return mism ? 3 : 0;
}
