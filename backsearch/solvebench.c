/*
 * solvebench — replay + regression oracle for the forward solver.
 *
 * Reads a backsearch --harvest file, rebuilds the exact Puzzle that
 * shortcut_check() handed to sokoban_solve_cutoff() for every accepted or
 * shortcut-pruned state, re-solves it with the same cutoff (depth-2), and
 * checks the answer against the recorded one.  Reports µs/call, states
 * popped, and a per-depth / per-outcome breakdown.  Any mismatch means the
 * solver change is unsound (or the harvest is from a different rule set).
 *
 * Build:
 *   cc -O3 -I. -o solvebench solvebench.c sokoban_bfs.c
 *   cc -O3 -DNO_ASTAR -I. -o solvebench_noastar solvebench.c sokoban_bfs.c   # A/B the A* prune
 *
 * Make a corpus (5 s of real search, ~350k solver calls, ~55 MB):
 *   ./backsearch_worker --grid 5x5 --exit 0 --time 5 --harvest /tmp/corpus_5x5.bin
 *
 * Run:
 *   ./solvebench /tmp/corpus_5x5.bin            # full table + per-depth profile
 *   ./solvebench /tmp/corpus_5x5.bin --quiet    # one-line summary + A/S split
 *   ./solvebench /tmp/corpus_5x5.bin --mand     # with the mandatory-hole prune enabled
 *   ./solvebench /tmp/corpus_5x5.bin --decision # first-win decision mode (what the generator uses); checks sign only
 *
 * Exit status 3 on any mismatch, so it can gate a build.  Timing noise on a
 * laptop is easily ±15%; run each variant twice and compare minimums.
 *
 * Reference numbers (Apple M-series, 2026-09-14, 5x5 corpus):
 *   original solver         ~13.4-14.5 µs/call
 *   small L2 table          ~13.0
 *   + A* lower bound        ~11.6
 *   + PGO build             ~11.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "sokoban_bfs.h"
#include "harvest_format.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: solvebench CORPUS.bin [maxcalls] [--quiet] [--mand]\n"); return 2; }
    long maxcalls = (argc > 2 && argv[2][0] != '-') ? atol(argv[2]) : -1;
    int quiet = 0, mand = 0, decision = 0;
    for (int i = 2; i < argc; i++) { if (!strcmp(argv[i], "--quiet")) quiet = 1; if (!strcmp(argv[i], "--mand")) mand = 1; if (!strcmp(argv[i], "--decision")) decision = 1; }

    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
    HarvestFileHeader h; if (fread(&h, sizeof h, 1, f) != 1) { fprintf(stderr, "short header\n"); return 1; }
    if (memcmp(h.magic, HARVEST_MAGIC, 4)) { fprintf(stderr, "bad magic\n"); return 1; }
    fseek(f, h.argv_blob_len, SEEK_CUR);
    int R = h.grid_rows, C = h.grid_cols, N = R * C;
    uint64_t active = (N >= 64) ? ~0ULL : ((1ULL << N) - 1);   /* == g_active_mask for a plain --grid run */

    long cap = 1 << 20, n = 0; HarvestRecord *rec = malloc(cap * sizeof *rec);
    while (fread(&rec[n], sizeof *rec, 1, f) == 1) { if (++n == cap) { cap *= 2; rec = realloc(rec, cap * sizeof *rec); } }
    fclose(f);

    sokoban_set_grid(R, C);
    sokoban_init();
    sokoban_set_hole_prune(mand);
    sokoban_set_decision_only(decision);   /* then only the sign of the answer is checked */

    long calls = 0, mism = 0, overflow = 0; long long popped = 0;
    long callsA = 0, callsS = 0; double timeA = 0, timeS = 0; long long popA = 0, popS = 0;
    long by_depth_calls[128] = {0}; double by_depth_time[128] = {0};
    double t_total = 0;
    for (long i = 0; i < n; i++) {
        HarvestRecord *r = &rec[i];
        if (r->outcome != HARVEST_OUTCOME_ACCEPTED && r->outcome != HARVEST_OUTCOME_SHORTCUT) continue;
        if (maxcalls >= 0 && calls >= maxcalls) break;
        /* Mirror build_partial_puzzle(): every cell not committed-empty is a wall. */
        Puzzle pz; memset(&pz, 0, sizeof pz);
        pz.exit_pos = r->exit_pos; pz.player_start = r->player_pos;
        pz.num_blocks = r->nblocks;
        for (int b = 0; b < r->nblocks; b++) { pz.block_pos[b] = r->block_pos[b]; pz.block_pushable[b] = r->block_mask[b]; }
        pz.num_holes = r->nholes;
        for (int k = 0; k < r->nholes && k < HARVEST_MAX_HOLES; k++) pz.hole_pos[k] = r->hole_pos[k];
        pz.walls = active & ~r->committed_empty;
        int max_cost = r->depth - 2;
        BfsProfile prof = {0};
        double t0 = now();
        int rc = sokoban_solve_cutoff(&pz, NULL, &prof, max_cost);
        double dt = now() - t0;
        t_total += dt; calls++; popped += prof.states_popped;
        if (r->outcome == HARVEST_OUTCOME_SHORTCUT) { callsS++; timeS += dt; popS += prof.states_popped; }
        else { callsA++; timeA += dt; popA += prof.states_popped; }
        int d = r->depth < 0 ? 0 : r->depth > 127 ? 127 : r->depth;
        by_depth_calls[d]++; by_depth_time[d] += dt;
        if (rc == -2 || rc == -3) overflow++;
        int expect = (r->outcome == HARVEST_OUTCOME_SHORTCUT) ? r->forward_solve : -1;
        int ok = decision ? ((rc >= 0) == (expect >= 0) && (rc < 0 || rc <= max_cost)) : (rc == expect);
        if (!ok) { mism++; if (mism <= 5 && !quiet) fprintf(stderr, "MISMATCH rec %ld depth %d: got %d expected %d (outcome %c)\n", i, r->depth, rc, expect, r->outcome); }
    }
    printf("%dx%d  calls=%ld  solver_time=%.3fs  us/call=%.2f  states_popped=%lld (avg %.1f)  mismatches=%ld  overflow=%ld\n",
           R, C, calls, t_total, calls ? t_total * 1e6 / calls : 0, popped, calls ? (double)popped / calls : 0, mism, overflow);
    printf("  accept(A): calls=%ld  time=%.3fs (%.0f%%)  us/call=%.2f  avg_states=%.1f\n",
           callsA, timeA, t_total ? 100 * timeA / t_total : 0, callsA ? timeA * 1e6 / callsA : 0, callsA ? (double)popA / callsA : 0);
    printf("  shortcut(S): calls=%ld  time=%.3fs (%.0f%%)  us/call=%.2f  avg_states=%.1f\n",
           callsS, timeS, t_total ? 100 * timeS / t_total : 0, callsS ? timeS * 1e6 / callsS : 0, callsS ? (double)popS / callsS : 0);
    if (!quiet) {
        printf("  depth   calls    us/call\n");
        for (int d = 0; d < 128; d += 5) {
            long c = 0; double t = 0; for (int k = d; k < d + 5 && k < 128; k++) { c += by_depth_calls[k]; t += by_depth_time[k]; }
            if (c) printf("  %3d-%-3d %7ld  %8.2f\n", d, d + 4, c, t * 1e6 / c);
        }
    }
    return mism ? 3 : 0;
}
