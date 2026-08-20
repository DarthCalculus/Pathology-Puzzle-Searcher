/*
 * backsearch.c — backward-search puzzle generator.
 *
 * Constructs hard puzzles by walking backward from the solved state
 * (player on exit cell 0).  Each backward step either undoes a forward
 * walk or undoes a forward push (introducing a new block constraint
 * if needed).  After every step we run sokoban_solve on the partial
 * puzzle as a shortcut check: if the optimal forward solve is shorter
 * than the backward depth, the branch is pruned.
 *
 * The shortcut-check puzzle uses (a) walls = current unconstrained
 * cells (the *most* walls we'll ever have) and (b) masks = current
 * min-mask (the *fewest* push directions we'll ever allow).  These
 * choices maximise forward solve length, so the result is an upper
 * bound on the final puzzle's forward solve.
 *
 * Grid size is configurable at runtime via --grid RxC (R,C in [1..5]).
 * The underlying sokoban_bfs.c stays a 5x5 solver; smaller grids are
 * emulated by treating cells outside the R*C window as permanent walls.
 *
 * Win condition (verified in sokoban_bfs.c:682) is "player ends turn
 * on exit_pos" — the existing header comment about blocks reaching
 * the exit is misleading.
 */

#include "sokoban_bfs.h"
#include "harvest_format.h"
#include "nn_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>
#include <zlib.h>

#define DEFAULT_CAP_S    0.0    /* 0 = no time limit (run until queue is empty) */

/* Canonical exit cells = D4-orbit representatives in row-major order,
 * filled by build_canonical_exits() once the D4 group is built.  For 5x5
 * this reproduces the historical {0,1,2,6,7,12}. */
static int g_canonical_exits[MAX_NCELLS];
static int g_n_canonical_exits = 0;

/* Adjacency table is owned by sokoban_bfs.c (g_adj[MAX_NCELLS][4]) and
 * populated by sokoban_set_grid(). */

/* -------------------------------------------------------------------------
 * Global config
 * ------------------------------------------------------------------------- */

/* g_grid_rows/g_grid_cols are pending --grid parsing.  Once set, they're
 * forwarded to sokoban_set_grid() which populates the solver's g_rows/
 * g_cols/g_ncells/g_adj/edge masks.  Inside this file we read those
 * globals directly via the header. */
static int      g_grid_rows   = 5;
static int      g_grid_cols   = 5;
static uint64_t g_active_mask = 0;          /* set in main() from --grid    */
static double   g_time_cap_s  = DEFAULT_CAP_S;
static int      g_exit_pos    = 0;          /* current exit cell being searched */
static int      g_only_exit   = -1;         /* -1 = iterate canonical exits */
/* --exitloc: explicit list of exit cells to search (overrides default
 * canonical set when non-empty).  --exit takes priority over --exitloc
 * if both are given. */
static int      g_n_only_exits = 0;
static int      g_only_exit_list[MAX_NCELLS];
static int      g_allow_exit_transit = 0;   /* 1 = block may transit through exit */
/* --allow-block-on-exit: permit the accepted puzzle to have a block resting
 * on the exit cell in its start position.  Implies --allow-exit-transit,
 * since a block can only land on the exit via a transit backstep at the
 * root.  Default: 0 (such candidates are rejected as best states). */
static int      g_allow_block_on_exit = 0;
static int      g_holeless           = 0;   /* 1 = forbid all holes (no variant 4) */
static int      g_two_tables         = 0;   /* 1 = use shallow+recent two-table dedup */
static int      g_reverse_order      = 0;   /* 1 = invert expand() emission order (flips DFS priority) */
static int      g_shortcut_state_cap = 0;   /* >0 = treat shortcut_check as a prune when states_popped exceeds N */
static int      g_beam_score_branching = 0; /* 1 = beam_score = -branch_factor (branching-only ranking) */
static int      g_beam_score_branching_mid = 0; /* 1 = beam_score = -|branch_factor - target| per level */
static int      g_branching_target_pct     = 50; /* percentile used by --beam-score-branching-mid (0..100) */
static int      g_use_tailwidth      = 0;   /* 1 = branch_factor = deep-weighted near-goal frontier width */
static int      g_tailwidth_window   = 24;  /* # of deepest cost-levels summed for the tail-width metric */
static int      g_last_peak_heap     = 0;   /* branching signal from the most recent shortcut_check call */

/* Fixed holes: when non-empty, holes are *only* allowed at these cells.
 * Variant 4 (un-consume) skips placing a hole at any cell not in this
 * mask.  No pre-placement — the search may choose to put 0..N holes at
 * any subset of these allowed cells. */
static int      g_fixed_nholes = 0;
static int      g_fixed_hole_pos[MAX_HOLES];
static uint64_t g_fixed_holes_mask = 0;

/* Fixed walls: cells that must be walls in the reported puzzle setup.
 * The search prevents these cells from ever entering committed_empty —
 * the player never walks on them, no block or hole is introduced there. */
static uint64_t g_fixed_walls_mask = 0;

/* Effective walkable region = active region minus fixed walls.  This is
 * what expand() consults to decide if a cell can be entered or have a
 * block/hole placed on it. */
static uint64_t g_walkable_mask = 0;

/* Lower bound on wall count in the reported puzzle.  The search prunes
 * any successor whose committed_empty grows so large that fewer than
 * this many cells of the active region remain walls.  Default: 0 (no
 * constraint). */
static int      g_min_walls         = 0;
static int      g_max_committed_in_active = INT_MAX;  /* derived from g_min_walls */

/* --single-axis-blocks: at most one block may be pushed along both axes
 * (horizontal AND vertical).  Every other block must stay on a single
 * axis.  Any backward step that would make a second block span both axes
 * is pruned.  Default: 0 (off). */
static int      g_single_axis_blocks = 0;

/* --single-axis-strict: implies --single-axis-blocks, and additionally
 * relaxes each block's pushable mask for the shortcut check so that a
 * block pulled along an axis is treated as pushable *both ways* on that
 * axis (vertical bit -> U|D, horizontal bit -> L|R).  This makes the
 * shortcut solver strictly more permissive, so surviving puzzles admit no
 * shortcut even when blocks can be pushed either direction on their axis.
 * Default: 0 (off). */
static int      g_axis_both_ways = 0;

/* Upper bounds on dynamically-introduced blocks/holes (--num-blocks,
 * --num-holes).  Variants 3 and 4 in expand() will not introduce new
 * blocks/holes once the bound is reached.  --num-holes counts the TOTAL
 * (including --fixedholes); --num-blocks counts the total nblocks. */
static int      g_max_blocks        = MAX_BLOCKS;
static int      g_max_holes         = MAX_HOLES;

/* Upper bound on a state's depth.  Successors with depth > this are
 * pruned in try_successor.  Used by the wrapper's "shallow scan" pass
 * (--max-depth 2) to cover depths the depth-3 task partition can't
 * reach when --num-walls is tight.  Default: INT_MAX (no cap). */
static int      g_max_depth         = INT_MAX;

/* Task partitioning.  When --task-id is set, the search starts not from
 * the standard depth-0 roots but from a specific subset of depth-2
 * states grouped by:
 *   (transit, hole_loc)
 * where transit ∈ {0, 1} indicates whether the 1st backstep involved an
 * exit-transit (block moving from adj[exit][D] to exit, only possible
 * when --allow-exit-transit is set), and hole_loc ∈ {-1, cell index}
 * indicates whether the 2nd backstep introduced a new hole (and at
 * which cell).
 *
 * Tasks are auto-enumerated per exit; --list-tasks prints how many
 * tasks each searched exit has, then exits.  --task-id N selects the
 * N-th task (global, across the iterated exits).
 *
 * Use case: shell-level parallelism.
 *     N=$(./backsearch ... --list-tasks)
 *     for i in $(seq 0 $((N-1))); do
 *         ./backsearch ... --task-id $i &
 *     done; wait
 */
static int g_only_task   = -1;   /* -1 = run all tasks merged (default) */
static int g_list_tasks  = 0;

/* --seed-path "D1V1,D2V2,...": apply a specific sequence of backward
 * steps from the standard root, then DFS-explore everything reachable
 * from the resulting state.  Each token is a direction letter (U/R/D/L)
 * followed by a variant digit (1=walk-back, 2=push-back existing block,
 * 3=introduce new block, 4=un-consume = new block + new hole).  Useful
 * for manually seeding the low-depth state that lies on a known optimal
 * trace (so the search doesn't have to discover it on its own).
 * Action digit meanings:
 *   1  walk-back   (player walked, no block movement)
 *   2  push-back   (block moved in direction D — back-search figures out
 *                   whether it's an existing tracked block or a newly
 *                   introduced one; both are the same forward action)
 *   3  un-consume  (introduces a new block AND a new hole simultaneously,
 *                   reversing a forward block-into-hole consumption)
 */
typedef struct { int8_t direction; int8_t variant; } SeedStep;
static SeedStep g_seed_path[1024];
static int      g_seed_path_n = 0;
static int      g_seed_path_overflow = 0;   /* set by parse_seed_path when the token cap is hit */
static int      g_have_seed_path = 0;
static int      g_print_seed_key = 0; /* build the --seed-path state, print its canonical key, exit */
static uint64_t g_watch_key = 0;       /* if nonzero, report when this canonical key shows up in a rollout pool */
static int      g_watch_key_depth = -1;/* only check pools at this depth (-1 = any) */
static uint64_t g_watch_keys[16];      /* multi-key watch: per-gen raw multiplicity + pool presence */
static int      g_watch_nkeys = 0;
static int      g_beam_level_report = 0;/* beam mode: print distinct-state frontier size per depth */

/* Dedup horizon: states at depth <= this value are deduped against the
 * visited table; states deeper than this skip dedup entirely (free-fly).
 * Trade-off: free-flying deep states do redundant subtree exploration
 * (cheap because subtrees are small at high depth) but the dedup table
 * stays small and never overflows.  Default: unlimited (current behaviour). */
static int       g_dupe_threshold = INT_MAX;
static long long g_skipped_dedup  = 0;

/* D4 symmetry exploitation: the (exit_pos, walkable_mask,
 * fixed_holes_mask) configuration has a symmetry subgroup (up to all of
 * D4 for centered exits on square grids, down to just identity for
 * generic edge cells).  Each first-backstep direction and each
 * push-off-exit seed direction sits in an orbit under that subgroup;
 * only one representative per orbit is enumerated.  Up to 8x speedup
 * for centered exits, 2x for corner/edge exits, no impact otherwise.
 *
 * D4 cell + direction permutations.  Built once at startup from the
 * grid dimensions: 4 transforms always (identity + h-reflect + v-reflect
 * + 180°-rotation), 8 when the grid is square (adds 90°/270° rotations
 * and the two diagonal reflections). */
typedef struct {
    int8_t cell[MAX_NCELLS];   /* cell[i] = where active cell i maps; -1 if i is inactive */
    int8_t dir[4];             /* dir[d]  = where direction d maps */
} Sym;

static int g_n_d4 = 0;
static Sym g_d4[8];

/* Per-exit symmetry subgroup (refreshed at the start of each exit's search). */
static int       g_n_exit_syms = 0;
static const Sym *g_exit_syms[8];

/* Bitmask of canonical first-direction representatives at the current
 * exit (one bit per orbit under g_exit_syms). */
static int g_canonical_dir_mask = 0xF;

/* -------------------------------------------------------------------------
 * D4 symmetry helpers (used only when --canonical-roots is set).
 * ------------------------------------------------------------------------- */

/* Build the up-to-eight D4 transforms acting on the active region.  The
 * four "axis-aligned" elements (identity, h-reflect, v-reflect, 180°)
 * always preserve the RxC active region.  The four "diagonal" elements
 * (90° / 270° rotations, main / anti diagonal reflections) only
 * preserve it when R == C — they swap the row/column counts otherwise. */
static void build_d4(void) {
    g_n_d4 = 0;
    int R = g_grid_rows, C = g_grid_cols;
    int square = (R == C);

    /* code: 0=id, 1=rot90 CW, 2=rot180, 3=rot270 CW,
     *       4=h-reflect (flip vertically), 5=v-reflect (flip horizontally),
     *       6=main-diag, 7=anti-diag. */
    static const int diag_codes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    static const int needs_square[8] = {0, 1, 0, 1, 0, 0, 1, 1};

    /* Direction permutations.  Direction encoding: 0=U, 1=R, 2=D, 3=L. */
    static const int8_t dir_perm[8][4] = {
        {0, 1, 2, 3},   /* identity */
        {1, 2, 3, 0},   /* rot90 CW: U->R, R->D, D->L, L->U */
        {2, 3, 0, 1},   /* rot180 */
        {3, 0, 1, 2},   /* rot270 CW */
        {2, 1, 0, 3},   /* h-reflect (flip vertically): U<->D */
        {0, 3, 2, 1},   /* v-reflect (flip horizontally): R<->L */
        {3, 2, 1, 0},   /* main diag: U<->L, R<->D */
        {1, 0, 3, 2},   /* anti diag: U<->R, D<->L */
    };

    for (int it = 0; it < 8; it++) {
        if (needs_square[it] && !square) continue;
        Sym *s = &g_d4[g_n_d4++];
        for (int idx = 0; idx < MAX_NCELLS; idx++) s->cell[idx] = -1;
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                int nr, nc;
                switch (diag_codes[it]) {
                    case 0: nr = r;             nc = c;             break;
                    case 1: nr = c;             nc = R - 1 - r;     break;
                    case 2: nr = R - 1 - r;     nc = C - 1 - c;     break;
                    case 3: nr = C - 1 - c;     nc = r;             break;
                    case 4: nr = R - 1 - r;     nc = c;             break;
                    case 5: nr = r;             nc = C - 1 - c;     break;
                    case 6: nr = c;             nc = r;             break;
                    case 7: nr = C - 1 - c;     nc = R - 1 - r;     break;
                    default: nr = r;            nc = c;             break;
                }
                /* Cell ids are row-major within the RxC grid.  For square
                 * grids, rotations and diagonal reflections may map (r,c)
                 * to (nr,nc) where nc/nr are bounded by R/C respectively
                 * — but R == C in those cases, so a single stride works. */
                s->cell[r * C + c] = (int8_t)(nr * C + nc);
            }
        }
        for (int d = 0; d < 4; d++) s->dir[d] = dir_perm[it][d];
    }
}

static uint64_t apply_sym_mask(const Sym *s, uint64_t m) {
    uint64_t out = 0;
    while (m) {
        int b = __builtin_ctzll(m);
        out |= 1ULL << s->cell[b];
        m &= m - 1;
    }
    return out;
}

/* Filter g_d4 to those that fix (exit_pos, g_walkable_mask, g_fixed_holes_mask). */
static void compute_exit_syms(int exit_pos) {
    g_n_exit_syms = 0;
    for (int i = 0; i < g_n_d4; i++) {
        const Sym *s = &g_d4[i];
        if (s->cell[exit_pos] != exit_pos) continue;
        if (apply_sym_mask(s, g_walkable_mask) != g_walkable_mask) continue;
        if (apply_sym_mask(s, g_fixed_holes_mask) != g_fixed_holes_mask) continue;
        g_exit_syms[g_n_exit_syms++] = s;
    }
}

/* Compute g_canonical_dir_mask: one bit per orbit of {0,1,2,3} under
 * g_exit_syms (smallest D in each orbit).  Defaults to 0xF when not
 * canonicalising. */
static void compute_canonical_dir_mask(void) {
    int orbit_id[4] = {-1, -1, -1, -1};
    int next_orbit = 0;
    for (int D = 0; D < 4; D++) {
        if (orbit_id[D] >= 0) continue;
        orbit_id[D] = next_orbit;
        int frontier[4]; int nf = 0;
        frontier[nf++] = D;
        while (nf > 0) {
            int x = frontier[--nf];
            for (int i = 0; i < g_n_exit_syms; i++) {
                int y = g_exit_syms[i]->dir[x];
                if (orbit_id[y] >= 0) continue;
                orbit_id[y] = next_orbit;
                frontier[nf++] = y;
            }
        }
        next_orbit++;
    }
    int seen_orbit[4] = {0};
    int mask = 0;
    for (int D = 0; D < 4; D++) {
        if (!seen_orbit[orbit_id[D]]) {
            seen_orbit[orbit_id[D]] = 1;
            mask |= 1 << D;
        }
    }
    g_canonical_dir_mask = mask;
}

/* Refresh g_canonical_dir_mask for a new exit. */
static void refresh_canonical_for_exit(int exit_pos) {
    compute_exit_syms(exit_pos);
    compute_canonical_dir_mask();
}

/* Build g_canonical_exits: one representative per D4 orbit.  Iterates
 * cells in row-major order, picks each cell that hasn't been seen as
 * the image of an earlier cell. */
static void build_canonical_exits(void) {
    g_n_canonical_exits = 0;
    int8_t orbit_rep[MAX_NCELLS];
    for (int i = 0; i < g_ncells; i++) orbit_rep[i] = -1;
    for (int p = 0; p < g_ncells; p++) {
        if (orbit_rep[p] >= 0) continue;
        orbit_rep[p] = (int8_t)p;
        g_canonical_exits[g_n_canonical_exits++] = p;
        for (int i = 0; i < g_n_d4; i++) {
            int q = g_d4[i].cell[p];
            if (q >= 0 && q < g_ncells && orbit_rep[q] < 0)
                orbit_rep[q] = (int8_t)p;
        }
    }
}

/* Runtime toggle for state-level canonicalisation in dedup, used for A/B
 * timing.  When 0, canonical_state_key reduces to state_key. */
static int g_state_canon = 1;

/* -------------------------------------------------------------------------
 * Backward state
 * ------------------------------------------------------------------------- */

typedef struct {
    int8_t   player_pos;
    int8_t   nblocks;
    int8_t   nholes;                     /* count of currently-active holes  */
    int8_t   block_pos [MAX_BLOCKS];
    uint8_t  block_mask[MAX_BLOCKS];
    int8_t   hole_pos  [MAX_HOLES];      /* sorted ascending; all active     */
    uint64_t committed_empty;            /* bit i: cell i is known not-wall */
    int32_t  depth;
    int32_t  state_id;                   /* analysis only; -1 when --trace-csv off */
    int8_t   kids_lookahead;             /* beam mode: cached 1-ply child count */
    int8_t   hole_adj_exit;              /* beam mode: cached # holes adjacent to exit */
    int8_t   mask_pop_sum;               /* beam mode: cached sum of popcount(block_mask[i]) */
    int8_t   score_valid;                /* beam mode: 1 iff score_cached has been populated */
    float    score_cached;               /* beam mode: cached beam_score value (NN or hand-tuned) */
    int32_t  branch_factor;              /* peak heap sz of the forward shortcut_check (0 if not run) */
} BState;

/* --trace-csv: dump (state_id, parent_id, features) for every accepted
 * state to FILE.  Used for offline analysis of search-graph structure. */
static FILE     *g_trace_csv         = NULL;
/* --bf-dump: dump (depth,branch_factor) for every shortcut_check eval to FILE.
 * Captures the raw pool distribution the percentile band ranks on. */
static FILE     *g_bf_dump           = NULL;
static long long g_next_state_id     = 0;
static long long g_current_parent_id = -1;

/* --harvest <file>: emit a binary record per *visited* state, including
 * those pruned by shortcut/dedup/caps.  Carries the full state blob
 * (committed_empty bitmask, block_pos/mask pairs, hole_pos) and the
 * forward-solve return value.  Used as the data source for offline
 * value-head and solver-surrogate training.  See harvest_format.h. */
static FILE     *g_harvest_csv  = NULL;
static gzFile    g_harvest_gz   = NULL;  /* non-NULL → output is gzip-compressed */
static char     *g_harvest_buf  = NULL;
static int       g_harvest_have_header = 0;

#define HARVEST_ACTIVE (g_harvest_csv != NULL || g_harvest_gz != NULL)

/* Write a buffer to whichever sink is open.  Returns bytes written on success. */
static size_t harvest_write(const void *ptr, size_t n) {
    if (g_harvest_gz)  return (size_t)gzwrite(g_harvest_gz, ptr, (unsigned)n);
    if (g_harvest_csv) return fwrite(ptr, 1, n, g_harvest_csv);
    return 0;
}

/* Forward decl — canonical_state_key is defined further down. */
static uint64_t canonical_state_key(const BState *s);

/* Encode flags into the HarvestFileHeader bitfield. */
static uint16_t g_harvest_flags = 0;

/* Open the harvest file, set a 4 MB buffer, and write the header +
 * argv blob.  We delay the actual write until sokoban_set_grid() has
 * run (so grid_rows/cols are known) — see harvest_write_header().
 *
 * argv pieces are stored joined by NUL bytes; the resulting blob lets
 * downstream tools recover the exact invocation. */
static char  **g_harvest_argv = NULL;
static int     g_harvest_argc = 0;

static int harvest_open(const char *path, int argc, char **argv) {
    size_t n = strlen(path);
    int want_gz = (n >= 3 && strcmp(path + n - 3, ".gz") == 0);
    if (want_gz) {
        g_harvest_gz = gzopen(path, "wb1");   /* level 1: fast, good ratio for our payload */
        if (!g_harvest_gz) { perror("gzopen --harvest"); return 0; }
        /* zlib's default 32 KB internal buffer is plenty for our write rate. */
    } else {
        g_harvest_csv = fopen(path, "wb");
        if (!g_harvest_csv) { perror("fopen --harvest"); return 0; }
        g_harvest_buf = malloc(4 * 1024 * 1024);
        if (g_harvest_buf) setvbuf(g_harvest_csv, g_harvest_buf, _IOFBF, 4 * 1024 * 1024);
    }
    g_harvest_argv = argv;
    g_harvest_argc = argc;
    g_harvest_have_header = 0;
    return 1;
}

/* Parse a 40-char hex SHA from __GIT_SHA__ (defined at build time) into
 * 20 bytes, zero-padding on parse failure. */
#ifndef GIT_SHA_STR
#define GIT_SHA_STR ""
#endif
static void harvest_fill_code_sha(uint8_t out[20]) {
    memset(out, 0, 20);
    const char *s = GIT_SHA_STR;
    if (!s || !*s) return;
    int n = (int)strlen(s);
    if (n < 40) return;
    for (int i = 0; i < 20; i++) {
        int hi = -1, lo = -1;
        char ch = s[2*i];
        if      (ch >= '0' && ch <= '9') hi = ch - '0';
        else if (ch >= 'a' && ch <= 'f') hi = 10 + (ch - 'a');
        else if (ch >= 'A' && ch <= 'F') hi = 10 + (ch - 'A');
        ch = s[2*i + 1];
        if      (ch >= '0' && ch <= '9') lo = ch - '0';
        else if (ch >= 'a' && ch <= 'f') lo = 10 + (ch - 'a');
        else if (ch >= 'A' && ch <= 'F') lo = 10 + (ch - 'A');
        if (hi < 0 || lo < 0) { memset(out, 0, 20); return; }
        out[i] = (uint8_t)((hi << 4) | lo);
    }
}

static void harvest_write_header(void) {
    if (!(g_harvest_csv || g_harvest_gz) || g_harvest_have_header) return;
    HarvestFileHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, HARVEST_MAGIC, 4);
    h.version          = HARVEST_VERSION;
    h.grid_rows        = (uint8_t)g_rows;
    h.grid_cols        = (uint8_t)g_cols;
    h.exit_pos         = -1;          /* multi-exit by default; updated per exit below */
    h.flags            = g_harvest_flags;
    h.started_at_unix  = (uint64_t)time(NULL);
    harvest_fill_code_sha(h.code_sha);

    /* Build argv blob: NUL-separated argv strings. */
    size_t blob_len = 0;
    for (int i = 0; i < g_harvest_argc; i++) blob_len += strlen(g_harvest_argv[i]) + 1;
    if (blob_len > (size_t)UINT32_MAX) blob_len = 0;
    h.argv_blob_len = (uint32_t)blob_len;

    harvest_write(&h, sizeof h);
    for (int i = 0; i < g_harvest_argc; i++) {
        harvest_write(g_harvest_argv[i], strlen(g_harvest_argv[i]) + 1);
    }
    g_harvest_have_header = 1;
}

/* outcome: HARVEST_OUTCOME_*; forward_solve as documented in harvest_format.h. */
static void harvest_emit(const BState *s, long long sid, char outcome, int forward_solve) {
    if (!g_harvest_csv && !g_harvest_gz) return;
    if (!g_harvest_have_header) harvest_write_header();

    HarvestRecord r;
    memset(&r, 0, sizeof r);
    r.state_id        = (uint64_t)sid;
    r.parent_id       = (int64_t)g_current_parent_id;
    r.canonical_key   = canonical_state_key(s);
    r.depth           = s->depth;
    r.forward_solve   = forward_solve;
    r.outcome         = (uint8_t)outcome;
    r.nblocks         = (uint8_t)s->nblocks;
    r.nholes          = (uint8_t)s->nholes;
    r.player_pos      = (uint8_t)s->player_pos;
    r.exit_pos        = (int16_t)g_exit_pos;
    r.committed_empty = s->committed_empty;
    int nb = s->nblocks < HARVEST_MAX_BLOCKS ? s->nblocks : HARVEST_MAX_BLOCKS;
    for (int i = 0; i < nb; i++) {
        r.block_pos [i] = s->block_pos [i];
        r.block_mask[i] = s->block_mask[i];
    }
    int nh = s->nholes < HARVEST_MAX_HOLES ? s->nholes : HARVEST_MAX_HOLES;
    for (int i = 0; i < nh; i++) {
        r.hole_pos[i] = s->hole_pos[i];
    }
    harvest_write(&r, sizeof r);
}

/* Sort blocks ascending by (pos, mask).  Insertion sort — nblocks is small. */
static void sort_blocks(int8_t *bp, uint8_t *bm, int n) {
    for (int i = 1; i < n; i++) {
        int8_t  p = bp[i];
        uint8_t m = bm[i];
        int j = i - 1;
        while (j >= 0 && (bp[j] > p || (bp[j] == p && bm[j] > m))) {
            bp[j+1] = bp[j];
            bm[j+1] = bm[j];
            j--;
        }
        bp[j+1] = p;
        bm[j+1] = m;
    }
}

/* Sort holes ascending by position.  Holes have no per-hole flag — all
 * tracked holes are active. */
static void sort_holes(int8_t *hp, int n) {
    for (int i = 1; i < n; i++) {
        int8_t p = hp[i];
        int j = i - 1;
        while (j >= 0 && hp[j] > p) { hp[j+1] = hp[j]; j--; }
        hp[j+1] = p;
    }
}

/* -------------------------------------------------------------------------
 * Visited hash set
 *
 * Open addressing, linear probing.  Stores 64-bit hash of canonical state
 * representation plus the depth at which it was first reached.  Two states
 * with the same hash are treated as equal — collision probability ≈
 * states_visited / 2^64, negligible at the scales reachable in 60s.
 * ------------------------------------------------------------------------- */

/* Visited table size — compile-time constant for hot-path speed.
 * Default 24 (16M slots, 256 MB) balances cache locality and per-exit
 * memset cost.  Recompile with -DHASH_LG2=26 (1 GB) for long 5x5 runs
 * that would otherwise hit the graceful-overflow path. */
#ifndef HASH_LG2
#define HASH_LG2 24
#endif
#define HASH_CAP  ((size_t)1 << HASH_LG2)
#define HASH_MASK (HASH_CAP - 1)

typedef struct { uint64_t key; int32_t depth; } HashSlot;
static HashSlot *g_visited = NULL;
static long long g_visited_count = 0;
static int       g_dedup_full   = 0;    /* set when table can't accept more */

/* -------------------------------------------------------------------------
 * Two-table dedup mode (--two-tables).
 *
 * Idea: split the dedup budget between two tables with different eviction
 * policies, both consulted on every lookup, both inserted on every miss:
 *
 *  - SHALLOW table: keeps the lowest-depth states ever seen.  When ~80%
 *    full it evicts entries with depth above a histogram-derived
 *    threshold, halving the table.
 *  - RECENT table: LRU-style.  Each slot stores a 64-bit clock value;
 *    when ~80% full it evicts entries whose clock is below a histogram-
 *    derived threshold, halving the table.
 *
 * The shallow table catches structurally important duplicates (large
 * subtrees below them); the recent table catches the temporally hot
 * frontier of the DFS.  Together they keep dedup useful indefinitely
 * without ever overflowing.
 * ------------------------------------------------------------------------- */
#ifndef SHALLOW_LG2
#define SHALLOW_LG2 22
#endif
#ifndef RECENT_LG2
#define RECENT_LG2  22
#endif
#define SHALLOW_CAP  ((size_t)1 << SHALLOW_LG2)
#define SHALLOW_MASK (SHALLOW_CAP - 1)
#define RECENT_CAP   ((size_t)1 << RECENT_LG2)
#define RECENT_MASK  (RECENT_CAP  - 1)

typedef struct { uint64_t key; int32_t depth; } ShallowSlot;
typedef struct { uint64_t key; int32_t depth; uint64_t clock; } RecentSlot;

static ShallowSlot *g_shallow = NULL;
static RecentSlot  *g_recent  = NULL;
static long long    g_shallow_count = 0;
static long long    g_recent_count  = 0;
static uint64_t     g_recent_clock  = 1;   /* monotonically increasing tick */
static long long    g_evictions_shallow = 0;
static long long    g_evictions_recent  = 0;

static uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint64_t state_key(const BState *s) {
    uint64_t h = splitmix64((uint64_t)s->committed_empty);
    h = splitmix64(h ^ ((uint64_t)(uint8_t)s->player_pos << 8)
                     ^ ((uint64_t)(uint8_t)s->nblocks << 16)
                     ^ ((uint64_t)(uint8_t)s->nholes  << 24));
    for (int i = 0; i < s->nblocks; i++) {
        uint64_t v = ((uint64_t)(uint8_t)s->block_pos[i] << 8) | (uint64_t)s->block_mask[i];
        h = splitmix64(h ^ v);
    }
    for (int i = 0; i < s->nholes; i++) {
        h = splitmix64(h ^ ((uint64_t)(uint8_t)s->hole_pos[i] | 0x100));
    }
    return h ? h : 1;  /* never zero: 0 = empty slot */
}

/* Permute every cell-valued field of s through symmetry sym, then re-sort
 * blocks and holes so the result has canonical sort order.  Used by
 * canonical_state_key to fold sym-equivalent states into one dedup entry. */
static void apply_sym_to_state(const Sym *sym, const BState *in, BState *out) {
    *out = *in;
    out->committed_empty = apply_sym_mask(sym, in->committed_empty);
    out->player_pos = sym->cell[(int)in->player_pos];
    for (int i = 0; i < in->nblocks; i++) {
        out->block_pos[i] = sym->cell[(int)in->block_pos[i]];
        uint8_t old_mask = in->block_mask[i];
        uint8_t new_mask = 0;
        for (int d = 0; d < 4; d++) {
            if (old_mask & (1u << d)) new_mask |= (uint8_t)(1u << sym->dir[d]);
        }
        out->block_mask[i] = new_mask;
    }
    for (int i = 0; i < in->nholes; i++) {
        out->hole_pos[i] = sym->cell[(int)in->hole_pos[i]];
    }
    sort_blocks(out->block_pos, out->block_mask, out->nblocks);
    sort_holes(out->hole_pos, out->nholes);
}

/* Canonical hash under the active symmetry subgroup.  Two states related
 * by some σ ∈ g_exit_syms produce the same key, so their dedup entries
 * collide and the second-arriving state (and its entire subtree) is
 * skipped.  When the subgroup is just identity (|G|=1) this is identical
 * to state_key — no overhead. */
static uint64_t canonical_state_key(const BState *s) {
    if (!g_state_canon) return state_key(s);
    /* g_exit_syms[0] is identity; start with its key and look for smaller. */
    uint64_t best = state_key(s);
    for (int i = 1; i < g_n_exit_syms; i++) {
        BState perm;
        apply_sym_to_state(g_exit_syms[i], s, &perm);
        uint64_t k = state_key(&perm);
        if (k < best) best = k;
    }
    return best;
}

/* Returns 1 if state should be skipped (already seen at <= this depth).
 * Otherwise inserts/updates and returns 0.  When the table is too full
 * to accept a new entry we set g_dedup_full and return 1 so the caller
 * silently drops this state; the search loop checks g_dedup_full and
 * terminates the current exit's pass gracefully (preserving best). */
static int dedup_check_and_insert(uint64_t key, int depth) {
    /* Hoist the _Thread_local pointer into a local register —
     * gcc/clang otherwise reload it every loop iteration. */
    HashSlot * const tab = g_visited;
    size_t i = (size_t)key & HASH_MASK;
    for (size_t probe = 0; probe < HASH_CAP; probe++) {
        if (tab[i].key == 0) {
            tab[i].key = key;
            tab[i].depth = depth;
            g_visited_count++;
            return 0;
        }
        if (tab[i].key == key) {
            if (tab[i].depth <= depth) return 1;
            tab[i].depth = depth;
            return 0;
        }
        i = (i + 1) & HASH_MASK;
    }
    g_dedup_full = 1;
    return 1;
}

/* -------------------------------------------------------------------------
 * Two-table dedup: lookup, insert, and eviction helpers.
 * ------------------------------------------------------------------------- */

/* Evict the deepest ~half of the shallow table.  Uses a 256-bin histogram
 * to find a depth threshold T such that depth <= T keeps roughly 50%, then
 * sweeps the table copying surviving entries to a fresh allocation. */
static void shallow_evict(void) {
    int hist[256] = {0};
    for (size_t i = 0; i < SHALLOW_CAP; i++)
        if (g_shallow[i].key != 0) {
            int d = g_shallow[i].depth;
            if (d > 255) d = 255;
            hist[d]++;
        }
    long long target = g_shallow_count / 2;
    long long acc = 0;
    int thresh = 255;
    for (int d = 0; d < 256; d++) {
        acc += hist[d];
        if (acc >= target) { thresh = d; break; }
    }
    /* Rehash survivors (depth <= thresh) into a fresh table. */
    ShallowSlot *fresh = calloc(SHALLOW_CAP, sizeof *fresh);
    if (!fresh) { perror("calloc shallow_evict"); exit(1); }
    long long kept = 0;
    for (size_t i = 0; i < SHALLOW_CAP; i++) {
        if (g_shallow[i].key == 0) continue;
        if (g_shallow[i].depth > thresh) continue;
        size_t j = (size_t)g_shallow[i].key & SHALLOW_MASK;
        while (fresh[j].key != 0) j = (j + 1) & SHALLOW_MASK;
        fresh[j] = g_shallow[i];
        kept++;
    }
    free(g_shallow);
    g_shallow = fresh;
    g_shallow_count = kept;
    g_evictions_shallow++;
}

/* Evict the oldest ~half of the recent table by clock.  Histogram bin =
 * (clock - oldest_clock) / step, where step picks ~256 buckets across the
 * range.  Same survivor-copy approach as shallow_evict. */
static void recent_evict(void) {
    /* Find clock range. */
    uint64_t lo = UINT64_MAX, hi = 0;
    for (size_t i = 0; i < RECENT_CAP; i++) {
        if (g_recent[i].key == 0) continue;
        if (g_recent[i].clock < lo) lo = g_recent[i].clock;
        if (g_recent[i].clock > hi) hi = g_recent[i].clock;
    }
    /* If all the same, just halve at random — bin into 256 buckets. */
    uint64_t span = (hi > lo) ? (hi - lo) : 1;
    uint64_t step = (span / 256) + 1;
    int hist[256] = {0};
    for (size_t i = 0; i < RECENT_CAP; i++) {
        if (g_recent[i].key == 0) continue;
        uint64_t b = (g_recent[i].clock - lo) / step;
        if (b > 255) b = 255;
        hist[b]++;
    }
    long long target = g_recent_count / 2;
    long long acc = 0;
    int thresh_bin = 0;
    for (int b = 0; b < 256; b++) {
        acc += hist[b];
        if (acc >= target) { thresh_bin = b; break; }
    }
    uint64_t thresh_clock = lo + (uint64_t)thresh_bin * step;
    /* Rehash survivors (clock >= thresh_clock) into a fresh table. */
    RecentSlot *fresh = calloc(RECENT_CAP, sizeof *fresh);
    if (!fresh) { perror("calloc recent_evict"); exit(1); }
    long long kept = 0;
    for (size_t i = 0; i < RECENT_CAP; i++) {
        if (g_recent[i].key == 0) continue;
        if (g_recent[i].clock < thresh_clock) continue;
        size_t j = (size_t)g_recent[i].key & RECENT_MASK;
        while (fresh[j].key != 0) j = (j + 1) & RECENT_MASK;
        fresh[j] = g_recent[i];
        kept++;
    }
    free(g_recent);
    g_recent = fresh;
    g_recent_count = kept;
    g_evictions_recent++;
}

/* Two-table check-and-insert.  Returns 1 if state was already seen at
 * depth <= the current depth (skip), 0 otherwise (continue exploring).
 * Always inserts new states into both tables (after evicting if needed). */
static int dedup_two_tables(uint64_t key, int depth) {
    int seen_shallow = 0, seen_recent = 0;
    int hit_dup = 0;

    /* Hoist _Thread_local pointers into local registers. */
    ShallowSlot * const stab = g_shallow;
    RecentSlot  * const rtab = g_recent;

    /* --- Probe shallow --- */
    size_t i = (size_t)key & SHALLOW_MASK;
    for (size_t probe = 0; probe < SHALLOW_CAP; probe++) {
        if (stab[i].key == 0) break;
        if (stab[i].key == key) {
            seen_shallow = 1;
            if (stab[i].depth <= depth) hit_dup = 1;
            else stab[i].depth = depth;
            break;
        }
        i = (i + 1) & SHALLOW_MASK;
    }

    /* --- Probe recent --- */
    uint64_t now = ++g_recent_clock;
    i = (size_t)key & RECENT_MASK;
    for (size_t probe = 0; probe < RECENT_CAP; probe++) {
        if (rtab[i].key == 0) break;
        if (rtab[i].key == key) {
            seen_recent = 1;
            rtab[i].clock = now;     /* refresh on hit */
            if (rtab[i].depth <= depth) hit_dup = 1;
            else rtab[i].depth = depth;
            break;
        }
        i = (i + 1) & RECENT_MASK;
    }

    if (hit_dup) return 1;

    /* --- Insert into shallow if missing --- */
    if (!seen_shallow) {
        if (g_shallow_count * 5 >= (long long)SHALLOW_CAP * 4) {
            shallow_evict();
            /* shallow_evict frees the table and reassigns g_shallow, so
             * our hoisted local is now stale; reread for the insert. */
        }
        ShallowSlot * const stab2 = g_shallow;
        size_t j = (size_t)key & SHALLOW_MASK;
        while (stab2[j].key != 0) j = (j + 1) & SHALLOW_MASK;
        stab2[j].key = key;
        stab2[j].depth = depth;
        g_shallow_count++;
    }
    /* --- Insert into recent if missing --- */
    if (!seen_recent) {
        if (g_recent_count * 5 >= (long long)RECENT_CAP * 4) recent_evict();
        RecentSlot * const rtab2 = g_recent;
        size_t j = (size_t)key & RECENT_MASK;
        while (rtab2[j].key != 0) j = (j + 1) & RECENT_MASK;
        rtab2[j].key = key;
        rtab2[j].depth = depth;
        rtab2[j].clock = now;
        g_recent_count++;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Frontier (LIFO stack of BState).
 *
 * We use depth-first instead of breadth-first because BFS-by-depth gets
 * stuck enumerating millions of shallow states once variant 4 (un-consume)
 * lifts the per-state branching factor.  DFS reaches deep states quickly,
 * updates g_best_depth, and lets the shortcut check do its job along the
 * way.  Trade-off: when the search hits the time cap, the result is a
 * lower bound on the true optimum (only the explored subtree is sound).
 * The "exhausted" report still reflects a complete search.
 * ------------------------------------------------------------------------- */

static BState   *g_queue   = NULL;
static size_t    g_q_cap   = 0;
static long long g_q_tail  = 0;        /* stack top                              */
static long long g_q_peak  = 0;        /* high-water mark for stats              */

/* Forward decls — q_push redirects to beam-mode push when --beam is set. */
extern int g_beam_width;
static void beam_push_to_next(const BState *s);
static int  count_potential_children(const BState *s);

static void q_push(const BState *s) {
    if (g_beam_width > 0) {
        beam_push_to_next(s);
        return;
    }
    if ((size_t)g_q_tail == g_q_cap) {
        g_q_cap = g_q_cap ? g_q_cap * 2 : 65536;
        g_queue = realloc(g_queue, g_q_cap * sizeof(BState));
        if (!g_queue) { perror("realloc queue"); exit(1); }
    }
    g_queue[g_q_tail++] = *s;
    if (g_q_tail > g_q_peak) g_q_peak = g_q_tail;
}

static int q_pop(BState *out) {
    if (g_q_tail == 0) return 0;
    *out = g_queue[--g_q_tail];
    return 1;
}

/* -------------------------------------------------------------------------
 * Beam search (alternative to DFS).  When --beam K is set, the search
 * proceeds depth-by-depth: from the current frontier, expand all states,
 * sort the resulting next-depth states by a heuristic score, and keep the
 * top K.  Trades exhaustiveness for tractability.
 *
 * In beam mode, q_push() redirects to g_beam_next instead of g_queue.
 * try_successor's dedup, shortcut check, and best-update logic are
 * unchanged.
 * ------------------------------------------------------------------------- */
int             g_beam_width  = 0;          /* 0 = DFS mode (default) */
static BState  *g_beam_curr   = NULL;
static int      g_beam_curr_n = 0;
static int      g_beam_curr_cap = 0;
static BState  *g_beam_next   = NULL;
static int      g_beam_next_n = 0;
static int      g_beam_next_cap = 0;
static long long g_beam_peak_frontier = 0;   /* before-trim high-water */

static double beam_score(const BState *s);                       /* forward decl */
static double beam_score_handtuned(const BState *s);             /* forward decl */
extern int    g_nn_loaded;                                       /* forward decl */
extern float  g_nn_blend;                                        /* forward decl */
extern float  g_nn_additive;                                     /* forward decl */
extern int    g_nn_use_additive;                                 /* forward decl */
#define NN_CHANNELS 9
static void   state_to_features(const BState *s, int exit_pos,   /* forward decl */
                                int rows, int cols, float *out);
static void   surrogate_pending_push(const BState *s, long long sid, long long parent_id);
static void   flush_surrogate_pending(void);

static void beam_push_to_next(const BState *s) {
    if (g_beam_next_n == g_beam_next_cap) {
        g_beam_next_cap = g_beam_next_cap ? g_beam_next_cap * 2 : 65536;
        g_beam_next = realloc(g_beam_next, (size_t)g_beam_next_cap * sizeof(BState));
        if (!g_beam_next) { perror("realloc beam_next"); exit(1); }
    }
    g_beam_next[g_beam_next_n] = *s;
    /* Cache 1-ply child-count for beam ranking. */
    g_beam_next[g_beam_next_n].kids_lookahead = (int8_t)count_potential_children(s);
    /* Cache "# holes adjacent to exit" — captures the strategic pattern
     * "block must be pushed through hole near exit", which empirically
     * yields the deepest puzzles. */
    int hae = 0;
    for (int d = 0; d < 4; d++) {
        int nbr = g_adj[g_exit_pos][d];
        if (nbr < 0) continue;
        for (int i = 0; i < s->nholes; i++) {
            if (s->hole_pos[i] == nbr) { hae++; break; }
        }
    }
    g_beam_next[g_beam_next_n].hole_adj_exit = (int8_t)hae;
    /* Cache sum of popcount(block_mask[i]).  Lower = more constrained
     * blocks = harder forward solve = deeper backward path. */
    int mps = 0;
    for (int i = 0; i < s->nblocks; i++)
        mps += __builtin_popcount(s->block_mask[i]);
    g_beam_next[g_beam_next_n].mask_pop_sum = (int8_t)mps;
    /* Cache hand-tuned score now (cheap).  In NN mode, compute_batch_scores
     * later replaces this with the blended NN+hand value before qsort.
     * In --beam-score-branching mode, ignore the hand-tuned features and
     * NN entirely: score is just -branch_factor (lower forward-search
     * explosion ranks higher). */
    if (g_beam_score_branching) {
        g_beam_next[g_beam_next_n].score_cached =
            -(float)g_beam_next[g_beam_next_n].branch_factor;
        g_beam_next[g_beam_next_n].score_valid = 1;
    } else {
        g_beam_next[g_beam_next_n].score_cached = (float)beam_score_handtuned(&g_beam_next[g_beam_next_n]);
        /* Only mark score as "needs NN batching" if the NN will actually contribute. */
        int nn_will_contribute = g_nn_loaded && (
            (g_nn_use_additive && g_nn_additive != 0.0f) ||
            (!g_nn_use_additive && g_nn_blend > 0.0f));
        g_beam_next[g_beam_next_n].score_valid = nn_will_contribute ? 0 : 1;
    }
    g_beam_next_n++;
    if (g_beam_next_n > g_beam_peak_frontier) g_beam_peak_frontier = g_beam_next_n;
}

/* int32 compare for qsort, used by rescore_branching_mid below. */
static int cmp_int32_asc(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

/* --beam-score-branching-mid: compute the median branch_factor across the
 * current beam_next level, then set each state's score to -|bf - median|.
 * Candidates closest to the median rank highest (those whose forward
 * shortcut_check did "neither too little nor too much" work).  Called
 * once per beam level just before qsort. */
static int32_t *g_branching_tmp     = NULL;
static int      g_branching_tmp_cap = 0;

static void rescore_branching_mid(void) {
    if (g_beam_next_n <= 0) return;
    if (g_branching_tmp_cap < g_beam_next_n) {
        g_branching_tmp_cap = g_beam_next_n;
        g_branching_tmp = realloc(g_branching_tmp, (size_t)g_branching_tmp_cap * sizeof(int32_t));
        if (!g_branching_tmp) { perror("realloc branching_tmp"); exit(1); }
    }
    for (int i = 0; i < g_beam_next_n; i++)
        g_branching_tmp[i] = g_beam_next[i].branch_factor;
    qsort(g_branching_tmp, (size_t)g_beam_next_n, sizeof(int32_t), cmp_int32_asc);
    /* Pick the configured percentile, clamped to [0, n-1]. */
    int idx = (int)(((long long)g_branching_target_pct * g_beam_next_n) / 100);
    if (idx < 0) idx = 0;
    if (idx >= g_beam_next_n) idx = g_beam_next_n - 1;
    int32_t target = g_branching_tmp[idx];
    for (int i = 0; i < g_beam_next_n; i++) {
        int32_t bf = g_beam_next[i].branch_factor;
        int32_t dist = bf > target ? (bf - target) : (target - bf);
        g_beam_next[i].score_cached = -(float)dist;
        g_beam_next[i].score_valid  = 1;
    }
}

/* Build NN features for all g_beam_next states into one contiguous buffer,
 * call nn_score_batch once, and populate each state's score_cached.
 * Reuses a heap buffer that grows on demand. */
static float *g_batch_features = NULL;
static int    g_batch_features_cap = 0;
static float *g_batch_scores = NULL;
static int    g_batch_scores_cap = 0;

static void compute_batch_scores(BState *arr, int n) {
    if (n <= 0 || !g_nn_loaded) return;
    int ncells = g_rows * g_cols;
    int stride = NN_CHANNELS * ncells;
    size_t needed_feats = (size_t)n * (size_t)stride;
    if ((size_t)g_batch_features_cap < needed_feats) {
        free(g_batch_features);
        g_batch_features = malloc(needed_feats * sizeof(float));
        if (!g_batch_features) { perror("malloc batch features"); exit(1); }
        g_batch_features_cap = (int)needed_feats;
    }
    if (g_batch_scores_cap < n) {
        free(g_batch_scores);
        g_batch_scores = malloc((size_t)n * sizeof(float));
        if (!g_batch_scores) { perror("malloc batch scores"); exit(1); }
        g_batch_scores_cap = n;
    }
    for (int i = 0; i < n; i++) {
        state_to_features(&arr[i], g_exit_pos, g_rows, g_cols,
                          g_batch_features + (size_t)i * stride);
    }
    nn_score_batch(g_batch_features, n, g_batch_scores);
    /* hand_tuned was cached eagerly by beam_push_to_next; we overwrite
     * with either the weighted blend or additive form. */
    if (g_nn_use_additive) {
        float lam = g_nn_additive;
        for (int i = 0; i < n; i++) {
            arr[i].score_cached = arr[i].score_cached + lam * g_batch_scores[i];
            arr[i].score_valid  = 1;
        }
    } else {
        float a = g_nn_blend;
        float one_minus_a = 1.0f - a;
        for (int i = 0; i < n; i++) {
            arr[i].score_cached = one_minus_a * arr[i].score_cached + a * g_batch_scores[i];
            arr[i].score_valid  = 1;
        }
    }
}

/* 1-ply lookahead: how many successors would expand(s) generate, without
 * dedup or shortcut filtering?  Mirror of the expand() loop with no side
 * effects.  Used as a beam-score feature — empirically, the optimal lineage
 * tends to have high child-count. */
static int count_potential_children(const BState *s) {
    int count = 0;
    int P = s->player_pos;

    uint64_t blk_occ = 0;
    int8_t blk_idx_at[MAX_NCELLS];
    for (int i = 0; i < g_ncells; i++) blk_idx_at[i] = -1;
    for (int i = 0; i < s->nblocks; i++) {
        blk_occ |= (1ULL <<s->block_pos[i]);
        blk_idx_at[s->block_pos[i]] = (int8_t)i;
    }
    uint64_t hole_occ = 0;
    for (int i = 0; i < s->nholes; i++) hole_occ |= (1ULL <<s->hole_pos[i]);

    for (int D = 0; D < 4; D++) {
        int C = g_adj[P][D ^ 2];
        if (C < 0) continue;
        if (C == g_exit_pos) continue;
        if (!(g_walkable_mask & (1ULL <<C))) continue;
        if (blk_occ & (1ULL <<C)) continue;
        if (hole_occ & (1ULL <<C)) continue;
        count++;  /* variant 1: walk-back */

        if (!g_allow_exit_transit && P == g_exit_pos) continue;
        int B = g_adj[P][D];
        if (B < 0) continue;
        if (!g_allow_exit_transit && B == g_exit_pos) continue;
        if (!(g_walkable_mask & (1ULL <<B))) continue;

        if (blk_idx_at[B] >= 0) {
            count++;  /* variant 2 */
        } else if (hole_occ & (1ULL <<B)) {
            /* skip */
        } else if (B == g_exit_pos
                   || (P == g_exit_pos && !g_allow_exit_transit)) {
            /* skip — see expand() comment for rationale. */
        } else {
            if (!(s->committed_empty & (1ULL <<B)) && s->nblocks < g_max_blocks) {
                count++;  /* variant 3 */
            }
            int v4 = !g_holeless && s->nblocks < g_max_blocks && s->nholes < g_max_holes;
            if (g_fixed_nholes > 0 && !(g_fixed_holes_mask & (1ULL <<B))) v4 = 0;
            if (v4) count++;  /* variant 4 */
        }
    }
    return count;
}

/* Beam-score weights.  Tunable via --score-weights "wRoom,wBlocks,wHoles,wKids".
 * Defaults derived from 4x4 trace analysis (4×4 trace shows the optimal
 * lineage favors high room, low nblocks/nholes, high child-count). */
static double g_w_room    =  1.0;
static double g_w_blocks  =  0.5;
static double g_w_holes   =  0.3;
static double g_w_kids    =  0.4;
/* New features (May 2026 R&D round): direct rewards for the structural
 * patterns we observed in deep puzzles. */
static double g_w_holeadj =  5.0;   /* per hole adjacent to exit */
static double g_w_maskpop =  0.2;   /* per direction-bit set across all blocks */

/* Stochastic beam: at clip time, take only (1 - F)*K deterministically
 * by score, then fill the remaining F*K slots with uniform-random states
 * from the tail.  F=0 (default) is pure top-K.  F=1.0 is uniform random.
 * Used to escape local optima when the deterministic beam stalls. */
static double g_beam_random_frac = 0.0;

/* --beam-save-tail T: at each beam clip, save T states from the *clipped
 * tail* (ranked just below the kept top-K) into the DFS stack.  After
 * beam terminates, continue with DFS from those saved states.  Lets the
 * search recover branches the beam dropped without re-exploring the
 * top-ranked sub-graph. */
static int g_beam_save_tail = 0;

/* --nn-value-model: when set, beam_score consults a libtorch model
 * instead of the hand-tuned features.  The model predicts a state's
 * eventual max_descendant_depth — higher = more promising. */
/* g_nn_loaded is forward-declared near the top of the file. */
int                g_nn_loaded        = 0;

/* --nn-surrogate-model: load a forward_solve predictor.  Before each
 * exact shortcut_check, ask the surrogate; if it predicts
 * forward_solve + margin ≤ depth - 2, skip the exact solver and prune.
 *
 * Gated by --nn-surrogate-min-depth N: only consult the surrogate for
 * states at depth ≥ N.  At shallow depths the exact solver is faster
 * than NN inference (5-10μs vs ~200μs); the surrogate only wins at
 * depth ≥ ~30 where the cutoff BFS explores enough states to be slow. */
static int         g_nn_surrogate_loaded     = 0;
static const char *g_nn_surrogate_model_path = NULL;
static float       g_nn_surrogate_target_scale = 1.0f;
static int         g_nn_surrogate_meta_rows  = 0;
static int         g_nn_surrogate_meta_cols  = 0;
static int         g_nn_surrogate_meta_channels = 9;
static int         g_nn_surrogate_is_classification = 0;  /* 1 if model outputs logit for P(prune) */
static float       g_nn_surrogate_margin     = 0.0f;       /* regression mode: safety margin */
static float       g_nn_surrogate_threshold  = 0.9f;       /* classification mode: P(prune) cutoff */
static int         g_nn_surrogate_min_depth  = 30;
static long long   g_nn_surrogate_calls      = 0;
static long long   g_nn_surrogate_skips      = 0;
static float       g_nn_surrogate_features[NN_CHANNELS * MAX_NCELLS];
/* Blend modes — set via --nn-blend or --nn-additive.
 *
 *   weighted (default): final = (1-α)*hand_tuned + α*nn.  α∈[0,1].
 *     α=1.0 → pure NN replacement.  α=0 → pure hand-tuned (no NN cost).
 *
 *   additive: final = hand_tuned + λ*nn.  λ unbounded (typically [0, 2]).
 *     NN can only ADD signal to hand-tuned; can never subtract it.
 *     Useful when NN is a tiebreaker / refinement of hand-tuned features.
 */
float              g_nn_blend         = 1.0f;
float              g_nn_additive      = 0.0f;   /* 0 → additive disabled */
int                g_nn_use_additive  = 0;
static const char *g_nn_model_path    = NULL;
static float       g_nn_target_scale  = 1.0f;
static int         g_nn_meta_rows     = 0;
static int         g_nn_meta_cols     = 0;
static int         g_nn_meta_channels = 9;

/* Channel layout MUST match corpus_features.state_to_tensor:
 *   0  player one-hot
 *   1  exit one-hot
 *   2-5  block-mask channels (U=bit1, R=bit2, D=bit4, L=bit8)
 *   6  hole one-hot
 *   7  committed_empty
 *   8  unknown (NOT committed AND NOT exit)
 *
 * NN_CHANNELS is defined near the top of the file with the forward declarations.
 */
static float g_nn_features[NN_CHANNELS * MAX_NCELLS];

static void state_to_features(const BState *s, int exit_pos,
                              int rows, int cols, float *out) {
    int ncells = rows * cols;
    int stride = ncells;
    memset(out, 0, (size_t)NN_CHANNELS * (size_t)stride * sizeof(float));

    if (s->player_pos >= 0 && s->player_pos < ncells)
        out[0 * stride + s->player_pos] = 1.0f;
    if (exit_pos >= 0 && exit_pos < ncells)
        out[1 * stride + exit_pos] = 1.0f;

    for (int i = 0; i < s->nblocks; i++) {
        int pos = s->block_pos[i];
        if (pos < 0 || pos >= ncells) continue;
        uint8_t m = s->block_mask[i];
        if (m & 1) out[2 * stride + pos] = 1.0f;
        if (m & 2) out[3 * stride + pos] = 1.0f;
        if (m & 4) out[4 * stride + pos] = 1.0f;
        if (m & 8) out[5 * stride + pos] = 1.0f;
    }
    for (int i = 0; i < s->nholes; i++) {
        int p = s->hole_pos[i];
        if (p >= 0 && p < ncells)
            out[6 * stride + p] = 1.0f;
    }

    uint64_t committed = s->committed_empty;
    for (int p = 0; p < ncells; p++)
        if ((committed >> p) & 1) out[7 * stride + p] = 1.0f;

    uint64_t exit_bit = (exit_pos >= 0 && exit_pos < 64) ? (1ULL << exit_pos) : 0;
    uint64_t mask_ncells = (ncells == 64) ? ~0ULL : ((1ULL << ncells) - 1);
    uint64_t not_committed = (~committed) & mask_ncells & ~exit_bit;
    for (int p = 0; p < ncells; p++)
        if ((not_committed >> p) & 1) out[8 * stride + p] = 1.0f;
}

/* Hand-tuned beam score from cached features.  Cheap.  Forward-declared
 * at top of file so beam_push_to_next can call it. */
double beam_score_handtuned(const BState *s) {
    int active_size = __builtin_popcountll(g_active_mask);
    int popcount    = __builtin_popcountll(s->committed_empty & g_active_mask);
    int room        = active_size - popcount;
    return  g_w_room    * (double)room
          - g_w_blocks  * (double)s->nblocks
          - g_w_holes   * (double)s->nholes
          + g_w_kids    * (double)s->kids_lookahead
          + g_w_holeadj * (double)s->hole_adj_exit
          - g_w_maskpop * (double)s->mask_pop_sum;
}

/* Heuristic score for beam ranking.  Higher = more promising.
 *
 * Three modes (controlled by g_nn_loaded and g_nn_blend):
 *   - No NN loaded: pure hand-tuned.
 *   - NN loaded, blend=1.0: pure NN replacement.
 *   - NN loaded, 0 < blend < 1: (1-blend)*hand_tuned + blend*nn.
 *
 * In NN mode this is the SINGLE-STATE fallback path (cmp_beam_score may
 * call it for an uncached state).  The hot path is compute_batch_scores
 * which batches everything into one libtorch call. */
static double beam_score(const BState *s) {
    double hand = beam_score_handtuned(s);
    if (!g_nn_loaded) return hand;
    state_to_features(s, g_exit_pos, g_rows, g_cols, g_nn_features);
    double nn = (double)nn_score(g_nn_features);
    if (g_nn_use_additive) return hand + (double)g_nn_additive * nn;
    return (1.0 - (double)g_nn_blend) * hand + (double)g_nn_blend * nn;
}

static int cmp_beam_score(const void *a, const void *b) {
    /* Reads the score cached during beam_push_to_next.  Falls back to
     * calling beam_score directly if uncached (shouldn't happen in
     * normal flow but is harmless if it does). */
    const BState *sa = (const BState *)a;
    const BState *sb = (const BState *)b;
    double va = sa->score_valid ? sa->score_cached : beam_score(sa);
    double vb = sb->score_valid ? sb->score_cached : beam_score(sb);
    if (va > vb) return -1;
    if (va < vb) return 1;
    /* Tie-breakers — important for --beam-score-branching where the primary
     * score has lots of ties.  Deterministic across runs and across beam
     * widths, which restores monotonicity in beam width. */
    /* (1) Deeper wins.  Within a beam level all states share the same
     *     depth, so this only fires when beam_curr mixes depths (rare;
     *     e.g. task seeds at different depths). */
    if (sa->depth > sb->depth) return -1;
    if (sa->depth < sb->depth) return 1;
    /* (2) More committed cells wins — captures "more constrained state". */
    int pa = __builtin_popcountll(sa->committed_empty);
    int pb = __builtin_popcountll(sb->committed_empty);
    if (pa > pb) return -1;
    if (pa < pb) return 1;
    /* (3) Final deterministic break by player position then nblocks so the
     *     order is reproducible regardless of input layout. */
    if (sa->player_pos != sb->player_pos) return sa->player_pos < sb->player_pos ? -1 : 1;
    if (sa->nblocks    != sb->nblocks)    return sa->nblocks    > sb->nblocks    ? -1 : 1;
    return 0;
}

/* -------------------------------------------------------------------------
 * Search bookkeeping
 * ------------------------------------------------------------------------- */

static int      g_best_depth      = 0;
static BState   g_best_state;
static long long g_states_checked = 0;
static long long g_pruned_short   = 0;
static long long g_pruned_dedup   = 0;
static long long g_pruned_cap     = 0;   /* states pruned by --shortcut-state-cap */
static long long g_pruned_axis    = 0;   /* states pruned by --single-axis-blocks */
static long long g_solver_calls   = 0;

/* Cross-exit overall best (for streaming new bests as they happen). */
static int             g_overall_best_depth = 0;
static BState          g_overall_best_state;
static int             g_overall_best_exit  = -1;
static struct timespec g_t_session_start;

/* New-best stream is debounced: each new-best event starts (or extends)
 * a STREAM_DEBOUNCE_S-second timer anchored to the *first* event since
 * the last print.  When the timer elapses, the latest pending state is
 * printed.  After printing, the timer resets and we wait for the next
 * new-best event before starting another timer. */
#define STREAM_DEBOUNCE_S  1.0
static double  g_first_new_best_time  = -1.0;  /* -1.0 = no timer running */
static int     g_pending_print_depth  = 0;     /* 0 = no pending */
static BState  g_pending_print_state;
static int     g_pending_print_exit   = -1;

static double session_elapsed(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec  - g_t_session_start.tv_sec)
         + (t.tv_nsec - g_t_session_start.tv_nsec) * 1e-9;
}

/* Forward decls — defined after print_puzzle_for_exit. */
static void stream_new_best(const BState *s);
static void flush_pending_new_best(double t);
static int g_rollout_quiet = 0;     /* 1 = collapse per-gen trace into one self-
                                     *    overwriting status line (\r + clear-to-EOL);
                                     *    puzzles still print normally.  Implies trace. */
static void print_puzzle_for_exit(const BState *s, int exit_pos);

/* -------------------------------------------------------------------------
 * Build the partial puzzle for the per-step shortcut check.
 * walls = active region minus committed_empty, plus inactive cells (always walls).
 * ------------------------------------------------------------------------- */
static void build_partial_puzzle(const BState *s, Puzzle *pz) {
    memset(pz, 0, sizeof *pz);
    pz->exit_pos     = g_exit_pos;
    pz->player_start = s->player_pos;
    pz->num_blocks   = s->nblocks;
    for (int i = 0; i < s->nblocks; i++) {
        pz->block_pos[i]      = s->block_pos[i];
        uint8_t m = s->block_mask[i];
        if (g_axis_both_ways) {
            /* Relax to both directions on each axis the block was pulled
             * along: vertical bit -> U|D (0x5), horizontal bit -> L|R (0xA). */
            uint8_t exp = 0;
            if (m & 0x5u) exp |= 0x5u;
            if (m & 0xAu) exp |= 0xAu;
            m = exp;
        }
        pz->block_pushable[i] = m;
    }
    /* All tracked holes are active; the forward solver assumes initial-active. */
    pz->num_holes = s->nholes;
    for (int i = 0; i < s->nholes; i++) {
        pz->hole_pos[i] = s->hole_pos[i];
    }
    /* walls: NCELLS cells that are NOT committed_empty.  Inactive cells
     * (outside g_active_mask), block cells, and hole cells are all not-wall
     * (block/hole cells are in committed_empty by construction). */
    pz->walls = g_active_mask & ~s->committed_empty;
}

/* Cutoff-aware shortcut check.  We only care whether a forward path of
 * length < depth exists (which would be a shortcut).  By a parity argument
 * — every move flips the player-cell color, so forward_solve has fixed
 * parity matching the cell-distance to exit — forward_solve at this state
 * has the *opposite* parity from the parent's depth.  Since the parent's
 * depth was d and forward_solve(parent) = d, this state's forward_solve
 * is in {1, 3, 5, ..., d+1}∩{same parity as d+1}.  Anything < d+1 is
 * therefore at most d-1 (one parity step below).  So we only need to
 * search for paths of length <= depth-2 (depth = d+1 here, so d-1 = depth-2).
 *
 * Returns:
 *    >= 0 : a real shortcut of that length exists — caller should prune.
 *    -1   : no shortcut within the cutoff — caller should accept the state.
 *    -2   : solver overflow (treat as prune for safety). */
/* Per-depth solver-call accounting (profiling).  Buckets correspond to
 * the state's backward depth at the moment of the shortcut call. */
#define SHORTCUT_PROFILE_BUCKETS 96
static long long g_solver_calls_by_depth[SHORTCUT_PROFILE_BUCKETS];
static double    g_solver_time_by_depth[SHORTCUT_PROFILE_BUCKETS];
static int       g_solver_profile = 0;   /* print per-depth solver-call profile; opt-in via --solver-profile */

/* shortcut_check return codes, extended:
 *   >= 0 : real shortcut of that length exists (caller prunes)
 *   -1   : no shortcut within the cutoff (caller accepts the state)
 *   -2   : heap overflow (caller prunes conservatively)
 *   -4   : --shortcut-state-cap exceeded; treated as a prune.  The forward
 *          solver explored more than g_shortcut_state_cap states without
 *          finding a shortcut, signalling a "branchy" subtree unlikely to
 *          deepen further. */
static int shortcut_check(const BState *s) {
    Puzzle pz;
    build_partial_puzzle(s, &pz);
    g_solver_calls++;
    int max_cost = s->depth - 2;
    BfsProfile prof = { .peak_heap_sz = 0, .states_popped = 0 };
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = sokoban_solve_cutoff(&pz, NULL, &prof, max_cost);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    /* Branching signal stamped into BState.branch_factor.  Default: states_popped
     * (total unique expanded states) — grows with how much work the forward
     * solver did.  With --beam-score-tailwidth: a deep-weighted sum of the
     * forward frontier's width in the cost-levels just below the cutoff
     * (max_cost = depth-2), capturing "is the tree narrow near the goal".
     * Deepest levels weigh most; smaller = a more forced corridor. */
    if (g_use_tailwidth) {
        long long acc = 0;
        int W = g_tailwidth_window;
        if (W > BFS_TAIL_W) W = BFS_TAIL_W;
        for (int j = 0; j < W; j++)
            acc += (long long)(W - j) * prof.tail_width[BFS_TAIL_W - 1 - j];
        if (acc > INT_MAX) acc = INT_MAX;
        g_last_peak_heap = (int)acc;
    } else {
        g_last_peak_heap = prof.states_popped;
    }
    if (g_bf_dump) fprintf(g_bf_dump, "%d,%d\n", s->depth, g_last_peak_heap);
    int b = s->depth;
    if (b < 0) b = 0;
    if (b >= SHORTCUT_PROFILE_BUCKETS) b = SHORTCUT_PROFILE_BUCKETS - 1;
    g_solver_calls_by_depth[b]++;
    g_solver_time_by_depth[b] += (t1.tv_sec - t0.tv_sec)
                              + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    /* Only enforce the cap when no real shortcut was found.  If rc >= 0
     * the state is pruned anyway; if rc == -2 the heap already overflowed
     * (separate signal).  Otherwise high states_popped means the forward
     * solver expanded a lot of states without finding a shortcut, which is
     * the "too branchy to deepen" signal we want to prune on. */
    if (g_shortcut_state_cap > 0 && rc == -1
        && prof.states_popped > g_shortcut_state_cap) {
        return -4;
    }
    return rc;
}

/* -------------------------------------------------------------------------
 * Surrogate pending buffer (Phase 8).
 *
 * try_successor pushes candidates here instead of calling the NN
 * surrogate single-state.  flush_surrogate_pending() runs one batched
 * libtorch call, then for each candidate:
 *   - if NN says confident-prune: harvest 'N', skip
 *   - else: exact shortcut_check, then accept-handling on success
 *
 * In beam mode the flush happens at the beam-clip boundary (large
 * batches).  In DFS mode the flush happens at the end of every expand()
 * (small batches but correct semantics).
 *
 * Pending entries carry their captured parent_id because flush happens
 * AFTER g_current_parent_id has been overwritten by subsequent
 * expansions; we restore it per-entry for correct harvest records.
 * ------------------------------------------------------------------------- */
typedef struct {
    BState     state;
    long long  sid;
    long long  parent_id;
} SurrogatePending;

static SurrogatePending *g_surrogate_pending = NULL;
static int               g_surrogate_pending_n   = 0;
static int               g_surrogate_pending_cap = 0;

static float *g_surrogate_batch_features      = NULL;
static int    g_surrogate_batch_features_cap  = 0;
static float *g_surrogate_batch_scores        = NULL;
static int    g_surrogate_batch_scores_cap    = 0;

static void surrogate_pending_push(const BState *s, long long sid, long long parent_id) {
    if (g_surrogate_pending_n == g_surrogate_pending_cap) {
        g_surrogate_pending_cap = g_surrogate_pending_cap ? g_surrogate_pending_cap * 2 : 4096;
        g_surrogate_pending = realloc(g_surrogate_pending,
                                      (size_t)g_surrogate_pending_cap * sizeof(SurrogatePending));
        if (!g_surrogate_pending) { perror("realloc surrogate_pending"); exit(1); }
    }
    g_surrogate_pending[g_surrogate_pending_n].state     = *s;
    g_surrogate_pending[g_surrogate_pending_n].sid       = sid;
    g_surrogate_pending[g_surrogate_pending_n].parent_id = parent_id;
    g_surrogate_pending_n++;
}

static void flush_surrogate_pending(void) {
    int n = g_surrogate_pending_n;
    if (n == 0 || !g_nn_surrogate_loaded) { g_surrogate_pending_n = 0; return; }

    int ncells = g_rows * g_cols;
    int stride = NN_CHANNELS * ncells;
    size_t needed_feats = (size_t)n * (size_t)stride;
    if ((size_t)g_surrogate_batch_features_cap < needed_feats) {
        free(g_surrogate_batch_features);
        g_surrogate_batch_features = malloc(needed_feats * sizeof(float));
        if (!g_surrogate_batch_features) { perror("malloc surrogate batch features"); exit(1); }
        g_surrogate_batch_features_cap = (int)needed_feats;
    }
    if (g_surrogate_batch_scores_cap < n) {
        free(g_surrogate_batch_scores);
        g_surrogate_batch_scores = malloc((size_t)n * sizeof(float));
        if (!g_surrogate_batch_scores) { perror("malloc surrogate batch scores"); exit(1); }
        g_surrogate_batch_scores_cap = n;
    }

    for (int i = 0; i < n; i++) {
        state_to_features(&g_surrogate_pending[i].state, g_exit_pos, g_rows, g_cols,
                          g_surrogate_batch_features + (size_t)i * stride);
    }
    nn_surrogate_score_batch(g_surrogate_batch_features, n, g_surrogate_batch_scores);
    g_nn_surrogate_calls += n;

    long long save_parent = g_current_parent_id;
    for (int i = 0; i < n; i++) {
        const BState *s = &g_surrogate_pending[i].state;
        long long sid   = g_surrogate_pending[i].sid;
        g_current_parent_id = g_surrogate_pending[i].parent_id;
        float raw = g_surrogate_batch_scores[i];

        int do_prune;
        if (g_nn_surrogate_is_classification) {
            float p = 1.0f / (1.0f + expf(-raw));
            do_prune = (p >= g_nn_surrogate_threshold);
        } else {
            do_prune = (raw + g_nn_surrogate_margin <= (float)(s->depth - 2));
        }

        if (do_prune) {
            g_nn_surrogate_skips++;
            g_pruned_short++;
            harvest_emit(s, sid, 'N', (int)raw);
            continue;
        }

        int x = shortcut_check(s);
        if (x >= 0 || x == -2 || x == -4) {
            g_pruned_short++;
            if (x == -4) g_pruned_cap++;
            harvest_emit(s, sid, x >= 0 ? 'S' : (x == -4 ? 'V' : 'E'), x);
            continue;
        }

        if (s->depth > g_best_depth) {
            int block_on_exit = 0;
            if (!g_allow_block_on_exit)
                for (int j = 0; j < s->nblocks; j++)
                    if (s->block_pos[j] == g_exit_pos) { block_on_exit = 1; break; }
            if (!block_on_exit) {
                g_best_depth = s->depth;
                g_best_state = *s;
                if (s->depth > g_overall_best_depth) {
                    g_overall_best_depth = s->depth;
                    g_overall_best_state = *s;
                    g_overall_best_exit  = g_exit_pos;
                    stream_new_best(s);
                }
            }
        }
        harvest_emit(s, sid, 'A', -1);

        BState ns = *s;
        ns.state_id = (int32_t)sid;
        ns.branch_factor = g_last_peak_heap;
        if (g_trace_csv) {
            int bx = 0;
            for (int j = 0; j < s->nblocks; j++)
                if (s->block_pos[j] == g_exit_pos) { bx = 1; break; }
            fprintf(g_trace_csv, "%lld,%lld,%d,%d,%d,%d,%d,%d,%d\n",
                    sid, g_current_parent_id, s->depth,
                    __builtin_popcountll(s->committed_empty & g_active_mask),
                    s->nblocks, s->nholes, s->player_pos, g_exit_pos, bx);
        }
        q_push(&ns);
    }
    g_current_parent_id = save_parent;
    g_surrogate_pending_n = 0;
}

/* -------------------------------------------------------------------------
 * Process one candidate successor: dedup, shortcut, enqueue.
 *
 * Dedup runs first (cheap hash probe).  If the state was already seen at
 * a depth ≤ the current depth, we skip it without paying for sokoban_solve.
 * Only states that survive dedup get the expensive shortcut check.  The
 * dedup table thus holds both shortcut-pass states and shortcut-fail
 * states from prior visits — that's correct: a shortcut-failing state
 * still fails at any *higher* depth, so future revisits should still be
 * pruned.  At a *lower* depth, dedup updates the recorded depth and
 * lets the visit through to re-check the shortcut (since the threshold
 * is lower, it may now pass).
 * ------------------------------------------------------------------------- */

/* Parse "U2,R3,D4" or "U2 R3 D4" into g_seed_path.  Returns 1 on success,
 * 0 on malformed input. */
static int parse_seed_path(const char *s) {
    g_seed_path_n = 0;
    g_seed_path_overflow = 0;
    while (*s) {
        while (*s == ' ' || *s == ',' || *s == '\t') s++;
        if (!*s) break;
        int dir;
        switch (*s++) {
            case 'U': case 'u': dir = 0; break;
            case 'R': case 'r': dir = 1; break;
            case 'D': case 'd': dir = 2; break;
            case 'L': case 'l': dir = 3; break;
            default: return 0;
        }
        if (!*s || *s < '1' || *s > '3') return 0;
        int user_action = *s++ - '0';
        /* Remap user-facing digit to internal variant number:
         *   1 -> 1 (walk-back), 2 -> 2 (push, auto 2/3), 3 -> 4 (un-consume) */
        int var = (user_action == 3) ? 4 : user_action;
        if (g_seed_path_n >= (int)(sizeof(g_seed_path)/sizeof(*g_seed_path))) { g_seed_path_overflow = 1; return 0; }
        g_seed_path[g_seed_path_n].direction = (int8_t)dir;
        g_seed_path[g_seed_path_n].variant   = (int8_t)var;
        g_seed_path_n++;
    }
    return 1;
}

/* Apply one backward step in direction D with variant V (1..4) to state
 * *s.  Mirrors the expand() logic exactly but for a single (D, V) choice
 * with no branching.  Returns 1 on success, 0 if the requested step is
 * invalid at this state.  Modifies *s in place on success. */
static int apply_seed_step(BState *s, int D, int variant) {
    int P = s->player_pos;
    int C = g_adj[P][D ^ 2];
    if (C < 0) return 0;
    if (C == g_exit_pos) return 0;
    if (!(g_walkable_mask & (1ULL << C))) return 0;

    uint64_t blk_occ = 0;
    int8_t blk_idx_at[MAX_NCELLS];
    for (int i = 0; i < g_ncells; i++) blk_idx_at[i] = -1;
    for (int i = 0; i < s->nblocks; i++) {
        blk_occ |= 1ULL << s->block_pos[i];
        blk_idx_at[s->block_pos[i]] = (int8_t)i;
    }
    uint64_t hole_occ = 0;
    for (int i = 0; i < s->nholes; i++) hole_occ |= 1ULL << s->hole_pos[i];

    if (blk_occ  & (1ULL << C)) return 0;
    if (hole_occ & (1ULL << C)) return 0;

    uint64_t new_E = s->committed_empty | (1ULL << C);

    if (variant == 1) {
        /* walk-back */
        s->player_pos      = (int8_t)C;
        s->committed_empty = new_E;
        s->depth          += 1;
        return 1;
    }

    /* Variants 2/3/4 require B = adj[P][D]. */
    if (!g_allow_exit_transit && P == g_exit_pos) return 0;
    int B = g_adj[P][D];
    if (B < 0) return 0;
    if (!g_allow_exit_transit && B == g_exit_pos) return 0;
    if (!(g_walkable_mask & (1ULL << B))) return 0;

    /* Digits 2 and 3 both mean "push-back".  Internally they're two
     * variants (existing block at B vs newly-discovered block at B),
     * which fire under mutually exclusive preconditions on the state.
     * We try the right one automatically. */
    if (variant == 2 || variant == 3) {
        if (blk_idx_at[B] >= 0) {
            /* Variant 2: block already tracked at B. */
            int idx = blk_idx_at[B];
            s->block_pos [idx]   = (int8_t)P;
            s->block_mask[idx]  |= (uint8_t)(1 << D);
            sort_blocks(s->block_pos, s->block_mask, s->nblocks);
            s->player_pos      = (int8_t)C;
            s->committed_empty = new_E;
            s->depth          += 1;
            return 1;
        }
        /* Variant 3: introduce a new block.  B must be uncommitted, no
         * hole there, not the exit, and we must have room. */
        if (hole_occ & (1ULL << B)) return 0;
        if (B == g_exit_pos) return 0;
        if (P == g_exit_pos && !g_allow_exit_transit) return 0;
        if (s->committed_empty & (1ULL << B)) return 0;
        if (s->nblocks >= g_max_blocks) return 0;
        s->block_pos [s->nblocks] = (int8_t)P;
        s->block_mask[s->nblocks] = (uint8_t)(1 << D);
        s->nblocks++;
        sort_blocks(s->block_pos, s->block_mask, s->nblocks);
        s->player_pos      = (int8_t)C;
        s->committed_empty = new_E | (1ULL << B);
        s->depth          += 1;
        return 1;
    }

    /* Variant 4: B must be empty (no block, no hole, not exit). */
    if (blk_idx_at[B] >= 0) return 0;
    if (hole_occ & (1ULL << B)) return 0;
    if (B == g_exit_pos) return 0;
    if (P == g_exit_pos && !g_allow_exit_transit) return 0;

    if (variant == 4) {
        if (g_holeless) return 0;
        if (s->nblocks >= g_max_blocks) return 0;
        if (s->nholes  >= g_max_holes ) return 0;
        if (g_fixed_nholes > 0 && !(g_fixed_holes_mask & (1ULL << B))) return 0;
        s->block_pos [s->nblocks] = (int8_t)P;
        s->block_mask[s->nblocks] = (uint8_t)(1 << D);
        s->nblocks++;
        sort_blocks(s->block_pos, s->block_mask, s->nblocks);
        s->hole_pos[s->nholes++] = (int8_t)B;
        sort_holes(s->hole_pos, s->nholes);
        s->player_pos      = (int8_t)C;
        s->committed_empty = new_E | (1ULL << B);
        s->depth          += 1;
        return 1;
    }
    return 0;
}

/* Build the seed-path seed (bare depth-0 root + replayed --seed-path steps)
 * into *out.  Returns 1 on success; on an invalid step prints a diagnostic and
 * returns 0.  Shared by run_exit_search and run_rollout so both honor
 * --seed-path identically. */
static int build_seed_path_seed(BState *out) {
    BState seed = {
        .player_pos      = (int8_t)g_exit_pos,
        .nblocks         = 0,
        .nholes          = 0,
        .committed_empty = 1ULL << g_exit_pos,
        .depth           = 0,
    };
    for (int i = 0; i < g_seed_path_n; i++) {
        int D = g_seed_path[i].direction;
        int V = g_seed_path[i].variant;
        int user_digit = (V == 4) ? 3 : V;   /* invert the parse-time remap for messages */
        if (!apply_seed_step(&seed, D, V)) {
            fprintf(stderr, "error: --seed-path step %d (%c%d) is invalid at depth %d (player=%d)\n",
                    i, "URDL"[D], user_digit, seed.depth, seed.player_pos);
            return 0;
        }
    }
    *out = seed;
    return 1;
}

static void try_successor(const BState *s) {
    g_states_checked++;
    /* Periodic debounce check: if a new-best timer is running and
     * STREAM_DEBOUNCE_S has elapsed since it started, fire the print.
     * Cheap unless a timer is active.  Multi-thread-safe via the mutex. */
    if ((g_states_checked & 8191) == 0 && g_first_new_best_time >= 0) {
        double t = session_elapsed();
        if ((t - g_first_new_best_time) >= STREAM_DEBOUNCE_S)
            flush_pending_new_best(t);
    }
    int trace_active = (g_trace_csv != NULL) || HARVEST_ACTIVE;
    long long sid = trace_active ? g_next_state_id++ : -1;

    /* --num-walls cap: refuse states whose committed_empty has overgrown
     * the active region beyond the limit set by --num-walls. */
    if (__builtin_popcountll(s->committed_empty & g_active_mask) > g_max_committed_in_active) {
        g_pruned_short++;
        harvest_emit(s, sid, 'W', -99);
        return;
    }
    /* --max-depth cap: stop exploring past the requested depth. */
    if (s->depth > g_max_depth) {
        g_pruned_short++;
        harvest_emit(s, sid, 'X', -99);
        return;
    }
    /* --single-axis-blocks: at most one block may be pushed along both
     * axes.  Directions 0=U,2=D form the vertical axis (mask 0x5); 1=R,3=L
     * the horizontal axis (mask 0xA).  A block spans both axes when its
     * accumulated push-mask has a bit in each.  Prune once a second such
     * block appears. */
    if (g_single_axis_blocks) {
        int both = 0;
        for (int i = 0; i < s->nblocks; i++) {
            uint8_t m = s->block_mask[i];
            if ((m & 0x5u) && (m & 0xAu)) both++;
        }
        if (both > 1) {
            g_pruned_axis++;
            harvest_emit(s, sid, 'M', -99);
            return;
        }
    }
    if (s->depth <= g_dupe_threshold) {
        uint64_t key = canonical_state_key(s);
        int dup = g_two_tables ? dedup_two_tables(key, s->depth)
                               : dedup_check_and_insert(key, s->depth);
        if (dup) {
            g_pruned_dedup++;
            harvest_emit(s, sid, 'D', -99);
            return;
        }
    } else {
        g_skipped_dedup++;
    }
    /* NN solver surrogate: defer to batched flush.  Per-state libtorch
     * inference is ~200μs (vs ~5-50μs for exact shortcut_check at low
     * depth), so we collect candidates into a pending buffer and
     * batch-evaluate at clip time (beam mode) or expand-end (DFS).
     * One batched libtorch call amortises to ~10μs per state. */
    if (g_nn_surrogate_loaded && s->depth >= g_nn_surrogate_min_depth) {
        surrogate_pending_push(s, sid, g_current_parent_id);
        return;
    }

    int x = shortcut_check(s);
    /* shortcut_check (cutoff variant) returns:
     *   >= 0 : a shortcut of that length exists (definitely prune).
     *   -1   : no shortcut within the cutoff (accept).
     *   -2   : heap overflow (conservatively prune).
     *   -4   : --shortcut-state-cap exceeded (prune as too-branchy). */
    if (x >= 0 || x == -2 || x == -4) {
        g_pruned_short++;
        if (x == -4) g_pruned_cap++;
        harvest_emit(s, sid, x >= 0 ? 'S' : (x == -4 ? 'V' : 'E'), x);
        return;
    }
    if (s->depth > g_best_depth) {
        /* Reject candidates with a block sitting on the exit cell, unless
         * --allow-block-on-exit permits it. */
        int block_on_exit = 0;
        if (!g_allow_block_on_exit)
            for (int i = 0; i < s->nblocks; i++)
                if (s->block_pos[i] == g_exit_pos) { block_on_exit = 1; break; }
        if (!block_on_exit) {
            g_best_depth = s->depth;
            g_best_state = *s;
            if (s->depth > g_overall_best_depth) {
                g_overall_best_depth = s->depth;
                g_overall_best_state = *s;
                g_overall_best_exit  = g_exit_pos;
                stream_new_best(s);
            }
        }
    }
    harvest_emit(s, sid, 'A', -1);
    BState ns = *s;
    ns.branch_factor = g_last_peak_heap;
    if (trace_active) {
        ns.state_id = (int32_t)sid;
        if (g_trace_csv) {
            int bx = 0;
            for (int i = 0; i < s->nblocks; i++)
                if (s->block_pos[i] == g_exit_pos) { bx = 1; break; }
            fprintf(g_trace_csv, "%lld,%lld,%d,%d,%d,%d,%d,%d,%d\n",
                    sid, g_current_parent_id, s->depth,
                    __builtin_popcountll(s->committed_empty & g_active_mask),
                    s->nblocks, s->nholes, s->player_pos, g_exit_pos, bx);
        }
    }
    q_push(&ns);
}

/* -------------------------------------------------------------------------
 * Successor enumeration from one state.
 * ------------------------------------------------------------------------- */
static void expand(const BState *s) {
    int P = s->player_pos;
    /* Fast occupancy lookup for the current state's blocks and active holes. */
    uint64_t blk_occ = 0;
    int8_t blk_idx_at[MAX_NCELLS];
    for (int i = 0; i < g_ncells; i++) blk_idx_at[i] = -1;
    for (int i = 0; i < s->nblocks; i++) {
        blk_occ |= (1ULL <<s->block_pos[i]);
        blk_idx_at[s->block_pos[i]] = (int8_t)i;
    }
    uint64_t hole_occ = 0;
    for (int i = 0; i < s->nholes; i++) hole_occ |= (1ULL <<s->hole_pos[i]);

    /* Canonical-root pruning: at the standard-root state (depth 0, player
     * at exit, no blocks, no holes) the four first-backstep directions sit
     * in symmetry orbits.  Restricting to one representative per orbit
     * prunes redundant subtrees without changing the set of reachable
     * (modulo symmetry) puzzle layouts.  At any state that doesn't match
     * the standard root, dir_mask stays 0xF (no pruning). */
    int dir_mask = 0xF;
    if (s->depth == 0 && s->nblocks == 0 && s->nholes == 0
        && P == g_exit_pos) {
        dir_mask = g_canonical_dir_mask;
    }

    for (int D = 0; D < 4; D++) {
        if (!(dir_mask & (1 << D))) continue;
        int C = g_adj[P][D ^ 2];                   /* player came from C */
        if (C < 0) continue;
        if (C == g_exit_pos) continue;             /* optimal play doesn't revisit exit */
        if (!(g_walkable_mask & (1ULL <<C))) continue;
        if (blk_occ  & (1ULL <<C)) continue;        /* can't walk into a block */
        if (hole_occ & (1ULL <<C)) continue;        /* can't walk into an active hole */

        uint64_t new_E = s->committed_empty | (1ULL << C);

        /* Buffer this direction's successors so --reverse can flip variant
         * priority within the direction without affecting direction order.
         * Max 3 per direction (walk + var3 + var4). */
        BState d_bufs[4];
        int d_n = 0;

        /* Successor 1: walk-back. */
        {
            BState ns = *s;
            ns.player_pos      = (int8_t)C;
            ns.committed_empty = new_E;
            ns.depth           = s->depth + 1;
            d_bufs[d_n++] = ns;
        }

        /* Strict-rule fast-out: when transit is disallowed, no push variant
         * may involve the exit cell at all. */
        if (g_allow_exit_transit || P != g_exit_pos) {
            int B = g_adj[P][D];
            if (B >= 0
                && (g_allow_exit_transit || B != g_exit_pos)
                && (g_walkable_mask & (1ULL <<B))) {

                if (blk_idx_at[B] >= 0) {
                    /* Successor 2: backward-push existing block from B to P.
                     * Mask of that block gets bit D OR'd in. */
                    int idx = blk_idx_at[B];
                    BState ns = *s;
                    ns.block_pos [idx]  = (int8_t)P;
                    ns.block_mask[idx] |= (uint8_t)(1 << D);
                    sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                    ns.player_pos      = (int8_t)C;
                    ns.committed_empty = new_E;
                    ns.depth           = s->depth + 1;
                    d_bufs[d_n++] = ns;
                } else if (hole_occ & (1ULL <<B)) {
                    /* B has an active hole — variant 3 (no consume) is impossible
                     * since a block can't sit on an active hole.  Variant 4
                     * (un-consume) is also impossible because B is already an
                     * active hole and the un-consume would re-introduce one. */
                } else if (B == g_exit_pos
                           || (P == g_exit_pos && !g_allow_exit_transit)) {
                    /* Variants 3 and 4 introduce a new block at P and (variant 4)
                     * a new hole at B.  Holes can never *originate* at the exit
                     * cell — that's a wholly different rule we don't support, so
                     * B == exit is always forbidden.  Blocks can transit through
                     * the exit when --allow-exit-transit is set. */
                } else {
                    /* B has no block and no active hole.  Two possible variants: */

                    /* Successor 3: introduce new block at B (continuously
                     * occupying B for the entire backward trace up to now),
                     * then push it back to P. */
                    if (!(s->committed_empty & (1ULL <<B)) && s->nblocks < g_max_blocks) {
                        BState ns = *s;
                        ns.block_pos [ns.nblocks] = (int8_t)P;
                        ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                        ns.nblocks++;
                        sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                        ns.player_pos      = (int8_t)C;
                        ns.committed_empty = new_E | (1ULL <<B);
                        ns.depth           = s->depth + 1;
                        d_bufs[d_n++] = ns;
                    }

                    /* Successor 4: backward un-consume.  Reverses a forward push
                     * that landed a block onto an active hole, consuming both.
                     * We introduce both: a new block at P (the cell from which
                     * it was pushed) with mask D, and a new active hole at B
                     * (the cell where it landed and was consumed).  In the
                     * puzzle, the hole has been at B since setup; we just
                     * hadn't tracked it because it was already inactive when we
                     * entered the trace.  Skipped entirely under --holeless. */
                    /* When --fixedholes is non-empty, restrict variant 4's hole
                     * placement: B must be one of the listed cells. */
                    int v4_blocked = (g_fixed_nholes > 0 && !(g_fixed_holes_mask & (1ULL <<B)));
                    if (!v4_blocked && !g_holeless
                        && s->nblocks < g_max_blocks && s->nholes < g_max_holes) {
                        BState ns = *s;
                        ns.block_pos [ns.nblocks] = (int8_t)P;
                        ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                        ns.nblocks++;
                        sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                        ns.hole_pos[ns.nholes++] = (int8_t)B;
                        sort_holes(ns.hole_pos, ns.nholes);
                        ns.player_pos      = (int8_t)C;
                        ns.committed_empty = new_E | (1ULL <<B);
                        ns.depth           = s->depth + 1;
                        d_bufs[d_n++] = ns;
                    }
                }
            }
        }

        /* Dispatch this direction's successors.  --reverse flips the
         * variant priority within the direction; direction priority
         * (determined by the outer D loop) is unchanged. */
        if (g_reverse_order) {
            for (int i = d_n - 1; i >= 0; i--) try_successor(&d_bufs[i]);
        } else {
            for (int i = 0; i < d_n; i++) try_successor(&d_bufs[i]);
        }
    }
}

/* -------------------------------------------------------------------------
 * Final puzzle output
 * ------------------------------------------------------------------------- */
static void print_puzzle_for_exit(const BState *s, int exit_pos) {
    char grid[MAX_ROWS][MAX_COLS + 1];
    for (int r = 0; r < g_rows; r++) {
        for (int c = 0; c < g_cols; c++) grid[r][c] = '.';
        grid[r][g_cols] = '\0';
    }
    /* Walls = active-region cells not in committed_empty.  Cells outside
     * the active region (relevant only for nxm grids embedded historically
     * in 5x5) don't apply now: the grid IS the active region. */
    for (int i = 0; i < g_ncells; i++)
        if (!(s->committed_empty & (1ULL << i))) grid[i / g_cols][i % g_cols] = '#';
    /* Holes (active in puzzle setup) — drawn before blocks so blocks shadow
     * any cell that was a block in our state but not a hole. */
    for (int i = 0; i < s->nholes; i++) {
        int p = s->hole_pos[i];
        grid[p / g_cols][p % g_cols] = 'O';
    }
    /* Exit drawn before blocks so a block resting on the exit (possible
     * under --allow-block-on-exit) shadows the '$' and stays visible as its
     * letter; the block's side annotation still identifies it. */
    grid[exit_pos / g_cols][exit_pos % g_cols] = '$';
    for (int i = 0; i < s->nblocks; i++) {
        int p = s->block_pos[i];
        grid[p / g_cols][p % g_cols] = (char)('A' + i);
    }
    /* Player drawn last so it overlays whatever cell it stands on. */
    grid[s->player_pos / g_cols][s->player_pos % g_cols] = '@';

    for (int r = 0; r < g_rows; r++) {
        printf("  ");
        for (int c = 0; c < g_cols; c++) putchar(grid[r][c]);
        if (r < s->nblocks) {
            uint8_t m = s->block_mask[r];
            printf("   %c=[%s%s%s%s]",
                   (char)('A' + r),
                   m & 1 ? "U" : "", m & 2 ? "R" : "",
                   m & 4 ? "D" : "", m & 8 ? "L" : "");
        }
        putchar('\n');
    }
    for (int i = g_grid_rows; i < s->nblocks; i++) {
        uint8_t m = s->block_mask[i];
        printf("            %c=[%s%s%s%s]\n",
               (char)('A' + i),
               m & 1 ? "U" : "", m & 2 ? "R" : "",
               m & 4 ? "D" : "", m & 8 ? "L" : "");
    }
}

/* Print whatever is pending right now and reset the debounce timer.
 * Caller has verified the timer has elapsed and there's something pending. */
static void flush_pending_new_best(double t) {
    if (g_pending_print_depth == 0) return;
    /* Human-readable elapsed: pick largest unit ≥ 1 (s / m / h),
     * 2 decimal places. */
    double val;
    const char *unit;
    if (t < 60.0)        { val = t;            unit = "s"; }
    else if (t < 3600.0) { val = t / 60.0;     unit = "m"; }
    else                 { val = t / 3600.0;   unit = "h"; }
    if (g_rollout_quiet) fputs("\r\033[K", stdout);  /* wipe live status line first */
    printf("%d (%.1f%s)\n", g_pending_print_depth, val, unit);
    (void)g_pending_print_exit;
    print_puzzle_for_exit(&g_pending_print_state, g_pending_print_exit);
    putchar('\n');
    fflush(stdout);
    g_pending_print_depth = 0;
    g_first_new_best_time = -1.0;     /* timer reset; next new best will start a fresh one */
}

/* Debounced "new best" event.  Always overwrite the pending slot with
 * the latest state.  If no timer is running, start one; otherwise leave
 * the existing timer alone — the print will fire STREAM_DEBOUNCE_S
 * seconds after the FIRST event in this batch (handled by the periodic
 * check in try_successor). */
static void stream_new_best(const BState *s) {
    g_pending_print_depth = s->depth;
    g_pending_print_state = *s;
    g_pending_print_exit  = g_exit_pos;
    if (g_first_new_best_time < 0)
        g_first_new_best_time = session_elapsed();
}

/* -------------------------------------------------------------------------
 * Task enumeration  (--list-tasks, --task-id N).
 *
 * Generates all valid (1st-backstep, 2nd-backstep) sequences for the
 * current exit and groups the resulting depth-2 states by the user's
 * partition: (transit, hole_loc).
 *
 *   transit  : 1 if a block was moved onto the exit on the 1st backstep
 *              (only possible when --allow-exit-transit is set), else 0.
 *   hole_loc : cell index where a new hole was placed on the 2nd backstep
 *              via variant 4, else -1.
 *
 * Each task ID maps to one (transit, hole_loc) bucket, which holds a
 * list of seed depth-2 states.  When --task-id N is set, run_exit_search
 * pushes only those seeds (instead of the standard depth-0 roots).
 * ------------------------------------------------------------------------- */

#define MAX_TASK_SEEDS    1024
#define MAX_TASKS_PER_EXIT 32

typedef struct {
    BState seeds[MAX_TASK_SEEDS];
    int    n_seeds;
    int    transit;     /* 0 or 1 */
    int    hole_loc;    /* -1 (none) or cell index */
} TaskGroup;

/* Compute all valid backward-step successors of state s into out_buf.
 * Same enumeration logic as expand(), but appends raw successors instead
 * of feeding them through try_successor (no dedup, no shortcut check). */
static int enumerate_successors(const BState *s, BState *out_buf, int max_out) {
    int n = 0;
    int P = s->player_pos;

    uint64_t blk_occ = 0;
    int8_t blk_idx_at[MAX_NCELLS];
    for (int i = 0; i < g_ncells; i++) blk_idx_at[i] = -1;
    for (int i = 0; i < s->nblocks; i++) {
        blk_occ |= (1ULL <<s->block_pos[i]);
        blk_idx_at[s->block_pos[i]] = (int8_t)i;
    }
    uint64_t hole_occ = 0;
    for (int i = 0; i < s->nholes; i++) hole_occ |= (1ULL <<s->hole_pos[i]);

    int dir_mask = 0xF;
    if (s->depth == 0 && s->nblocks == 0 && s->nholes == 0
        && P == g_exit_pos) {
        dir_mask = g_canonical_dir_mask;
    }

    for (int D = 0; D < 4; D++) {
        if (!(dir_mask & (1 << D))) continue;
        int C = g_adj[P][D ^ 2];
        if (C < 0) continue;
        if (C == g_exit_pos) continue;
        if (!(g_walkable_mask & (1ULL <<C))) continue;
        if (blk_occ  & (1ULL <<C)) continue;
        if (hole_occ & (1ULL <<C)) continue;

        uint64_t new_E = s->committed_empty | (1ULL << C);

        /* Variant 1 — walk-back. */
        if (n < max_out) {
            BState ns = *s;
            ns.player_pos      = (int8_t)C;
            ns.committed_empty = new_E;
            ns.depth           = s->depth + 1;
            out_buf[n++] = ns;
        }

        /* Push variants. */
        if (!g_allow_exit_transit && P == g_exit_pos) continue;
        int B = g_adj[P][D];
        if (B < 0) continue;
        if (!g_allow_exit_transit && B == g_exit_pos) continue;
        if (!(g_walkable_mask & (1ULL <<B))) continue;

        if (blk_idx_at[B] >= 0) {
            /* Variant 2 — push existing block. */
            if (n < max_out) {
                int idx = blk_idx_at[B];
                BState ns = *s;
                ns.block_pos [idx]  = (int8_t)P;
                ns.block_mask[idx] |= (uint8_t)(1 << D);
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E;
                ns.depth           = s->depth + 1;
                out_buf[n++] = ns;
            }
        } else if (hole_occ & (1ULL <<B)) {
            /* Skip — block can't sit on active hole. */
        } else if (B == g_exit_pos
                   || (P == g_exit_pos && !g_allow_exit_transit)) {
            /* Skip — B == exit always forbidden (holes never originate
             * at exit).  P == exit only forbidden under strict rule;
             * with --allow-exit-transit, block can transit the exit. */
        } else {
            /* Variant 3 — introduce new block at B. */
            if (n < max_out
                && !(s->committed_empty & (1ULL <<B))
                && s->nblocks < g_max_blocks) {
                BState ns = *s;
                ns.block_pos [ns.nblocks] = (int8_t)P;
                ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                ns.nblocks++;
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E | (1ULL <<B);
                ns.depth           = s->depth + 1;
                out_buf[n++] = ns;
            }
            /* Variant 4 — un-consume (introduce block + hole). */
            int v4_allowed = 1;
            if (g_fixed_nholes > 0 && !(g_fixed_holes_mask & (1ULL <<B))) v4_allowed = 0;
            if (g_holeless) v4_allowed = 0;
            if (s->nblocks >= g_max_blocks || s->nholes >= g_max_holes) v4_allowed = 0;
            if (v4_allowed && n < max_out) {
                BState ns = *s;
                ns.block_pos [ns.nblocks] = (int8_t)P;
                ns.block_mask[ns.nblocks] = (uint8_t)(1 << D);
                ns.nblocks++;
                sort_blocks(ns.block_pos, ns.block_mask, ns.nblocks);
                ns.hole_pos[ns.nholes++] = (int8_t)B;
                sort_holes(ns.hole_pos, ns.nholes);
                ns.player_pos      = (int8_t)C;
                ns.committed_empty = new_E | (1ULL <<B);
                ns.depth           = s->depth + 1;
                out_buf[n++] = ns;
            }
        }
    }

    return n;
}

/* Find or create the task-bucket matching (transit, hole_loc).  Returns
 * -1 if no slot available. */
static int task_bucket_for(TaskGroup *tasks, int *p_n_tasks, int max_tasks,
                           int transit, int hole_loc) {
    for (int t = 0; t < *p_n_tasks; t++) {
        if (tasks[t].transit == transit && tasks[t].hole_loc == hole_loc) return t;
    }
    if (*p_n_tasks >= max_tasks) return -1;
    int t = (*p_n_tasks)++;
    tasks[t].transit  = transit;
    tasks[t].hole_loc = hole_loc;
    tasks[t].n_seeds  = 0;
    return t;
}

/* Enumerate the partition tasks for one exit.  Reads g_exit_pos (must
 * be set by the caller).  Returns the number of non-empty task buckets. */
static int enumerate_tasks_for_exit(int exit_pos, TaskGroup *tasks, int max_tasks) {
    int n_tasks = 0;
    int saved_exit = g_exit_pos;
    g_exit_pos = exit_pos;
    refresh_canonical_for_exit(exit_pos);

    /* Build the depth-0 roots. */
    BState roots[8];
    int n_roots = 0;

    /* Standard root. */
    {
        BState r = {0};
        r.player_pos      = (int8_t)exit_pos;
        r.nblocks         = 0;
        r.nholes          = 0;
        r.committed_empty = 1ULL <<exit_pos;
        r.depth           = 0;
        roots[n_roots++]  = r;
    }
    /* Push-off-exit seeds (only when --allow-exit-transit).  These are
     * depth-1 states representing "right before the push-off win move":
     * player at Y=adj[exit][D'^2], block at exit, mask={D'}.  Seeding
     * here (rather than at depth 0 with player at exit) avoids spurious
     * variant-1 walk-back successors from the exit cell — see the
     * matching comment in run_exit_search.  Note: tasks built from these
     * seeds reach depth-3 in two enumerate_successors steps, while
     * standard-root tasks need three; this is the natural consequence
     * of the push-off win consuming one move. */
    if (g_allow_exit_transit && g_max_blocks >= 1) {
        for (int D = 0; D < 4; D++) {
            if (!(g_canonical_dir_mask & (1 << D))) continue;
            int X = g_adj[exit_pos][D];
            if (X < 0) continue;
            if (!(g_walkable_mask & (1ULL <<X))) continue;
            if (g_fixed_holes_mask & (1ULL <<X)) continue;
            int Y = g_adj[exit_pos][D ^ 2];
            if (Y < 0) continue;
            if (!(g_walkable_mask & (1ULL <<Y))) continue;
            BState r = {0};
            r.player_pos      = (int8_t)Y;
            r.nblocks         = 1;
            r.nholes          = 0;
            r.committed_empty = (1ULL <<exit_pos) | (1ULL <<X) | (1ULL <<Y);
            r.depth           = 1;
            r.block_pos [0]   = (int8_t)exit_pos;
            r.block_mask[0]   = (uint8_t)(1u << D);
            if (n_roots < (int)(sizeof roots / sizeof roots[0]))
                roots[n_roots++] = r;
        }
    }

    /* Bucket key (transit, hole_loc) classifies the LAST 3 forward moves:
     *   transit  : 1 if forward move 1 is a push-off-exit win (the root
     *              encodes a state with a block already at exit), else 0.
     *   hole_loc : cell pd1 (= the cell the player walks from on its way
     *              to the exit) IF that cell was a hole that got filled
     *              by a block push as forward move 3, AND the player then
     *              walked onto the just-filled tile as forward move 2.
     *              I.e., the LAST 3 forward moves are precisely
     *                  fill(pd1) → walk-onto(pd1) → walk/push to exit.
     *              hole_loc = -1 in any other case (no strategic hole-fill,
     *              or a hole filled at some non-path cell).
     *
     * Equivalently in backward-trace terms: hole_loc = pd1 iff the depth-3
     * state has a hole at pd1 that wasn't in the depth-2 state — i.e., the
     * hole was introduced exactly at backstep 3 via variant 4, with B=pd1.
     * Variant 4 at backstep 3 with B=pd1 forces the depth-2 player to be a
     * neighbor of pd1, which combined with pd1 being adjacent to exit means
     * the player's path traversed pd1.
     *
     * pd1 is the player position one backstep from the win, which differs
     * by root type:
     *   standard root (depth 0): pd1 = state_1.player_pos = d1.player_pos
     *   push-off root  (depth 1): pd1 = root.player_pos (= Y)
     * For push-off roots the "post-backstep-3" state is d2 (in code), one
     * level shallower than for standard roots; the seed itself remains at
     * d3 (one level beyond the partition key), reused as a search starting
     * point that already encodes the partition's classifying event. */
    uint64_t exit_adj_mask = 0;
    for (int D = 0; D < 4; D++) {
        int X = g_adj[exit_pos][D];
        if (X < 0) continue;
        exit_adj_mask |= (1ULL <<X);
    }
    for (int r = 0; r < n_roots; r++) {
        const BState *root = &roots[r];
        int transit = (root->depth == 1) ? 1 : 0;

        BState d1_buf[16];
        int n_d1 = enumerate_successors(root, d1_buf, 16);
        for (int i = 0; i < n_d1; i++) {
            BState *d1 = &d1_buf[i];

            /* pd1 = state_1 player position. */
            int pd1 = (root->depth == 0) ? d1->player_pos : root->player_pos;
            int pd1_is_exit_adj = (exit_adj_mask & (1ULL <<pd1)) != 0;

            BState d2_buf[16];
            int n_d2 = enumerate_successors(d1, d2_buf, 16);
            for (int j = 0; j < n_d2; j++) {
                BState *d2 = &d2_buf[j];

                /* For push-off roots, the partition key is determined here:
                 * state_2 = d1, state_3 = d2.  Compute hole_loc once per d2
                 * and reuse it for every d3 child below. */
                int hole_loc_pushoff = -1;
                if (root->depth == 1 && pd1_is_exit_adj) {
                    int d1_has = 0, d2_has = 0;
                    for (int h = 0; h < d1->nholes; h++)
                        if (d1->hole_pos[h] == pd1) { d1_has = 1; break; }
                    for (int h = 0; h < d2->nholes; h++)
                        if (d2->hole_pos[h] == pd1) { d2_has = 1; break; }
                    if (d2_has && !d1_has) hole_loc_pushoff = pd1;
                }

                BState d3_buf[16];
                int n_d3 = enumerate_successors(d2, d3_buf, 16);
                for (int k = 0; k < n_d3; k++) {
                    BState *d3 = &d3_buf[k];

                    int hole_loc;
                    if (root->depth == 1) {
                        hole_loc = hole_loc_pushoff;
                    } else {
                        /* Standard root: state_2 = d2, state_3 = d3. */
                        hole_loc = -1;
                        if (pd1_is_exit_adj) {
                            int d2_has = 0, d3_has = 0;
                            for (int h = 0; h < d2->nholes; h++)
                                if (d2->hole_pos[h] == pd1) { d2_has = 1; break; }
                            for (int h = 0; h < d3->nholes; h++)
                                if (d3->hole_pos[h] == pd1) { d3_has = 1; break; }
                            if (d3_has && !d2_has) hole_loc = pd1;
                        }
                    }

                    int t = task_bucket_for(tasks, &n_tasks, max_tasks, transit, hole_loc);
                    if (t >= 0 && tasks[t].n_seeds < MAX_TASK_SEEDS) {
                        tasks[t].seeds[tasks[t].n_seeds++] = *d3;
                    }
                }
            }
        }
    }

    g_exit_pos = saved_exit;
    return n_tasks;
}

/* -------------------------------------------------------------------------
 * Per-exit search.  Returns elapsed seconds; updates the per-exit globals
 * (best, counters) as side effects.  Stops early when the *remaining*
 * shared budget runs out.
 * ------------------------------------------------------------------------- */
static double elapsed_s(struct timespec t0, struct timespec t1) {
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* =========================================================================
 * ROLLOUT MODE  (--rollout K,M,C,P) — population / stochastic-beam search
 *
 * Instead of committing to a single trajectory, carry a *population* of up to
 * M endpoints forward across generations.  Each generation:
 *   1. every population member spawns C random rollouts of K backward steps
 *      (each step picks uniformly among the non-shortcut successors);
 *   2. all surviving endpoints are pooled, deduped by canonical state key,
 *      and ranked by branch_factor (states_popped, or --beam-score-tailwidth);
 *   3. the next population is an M-wide band of that ranking centered on the
 *      percentile-P index — so P selects which slice of the branchiness
 *      distribution survives, and M sets how much diversity we carry;
 *   4. if NO rollout survives, DFS-rescue any valid K-step continuation from
 *      some member and reseed the population with it; if even that fails, stop.
 * Single-trajectory rollout is the M=1 special case.  Starts from the standard
 * depth-0 root (player at exit, no blocks/holes).
 * ========================================================================= */

static int g_rollout_steps = 0;     /* K: backward steps per rollout (0 = mode off) */
static int g_rollout_pop   = 100;   /* M: population / band size */
static int g_rollout_child  = 10;   /* C: rollouts (children) spawned per member */
static int g_rollout_pct   = 50;    /* P: percentile center of the next band (0..100) */
/* --rollout-pct-window LO,HI,P: in percentile gens whose representative depth is
 * in [LO,HI], use percentile P instead of g_rollout_pct. Experiment knob to test
 * a non-monotone P schedule (record-depth levels ride high-percentile/branchy
 * states in the early band gens; flat P=10 prunes them). lo<0 = disabled. */
static int g_rollout_pctwin_lo  = -1;
static int g_rollout_pctwin_hi  = -1;
static int g_rollout_pctwin_pct = 50;

/* --rollout-bf-target FILE: in percentile (non-rand) gens, instead of a
 * percentile-centered band, select the M candidates whose branch_factor is
 * CLOSEST to a per-depth target value loaded from FILE ("depth,bf" CSV, e.g.
 * a record level's pathbf dump). This walks an exemplar level's actual
 * branchiness signature (absolute bf per depth) rather than a relative
 * percentile — auto-encoding its non-monotone profile (narrow early, branchy
 * deep). g_bf_target indexed by depth; NULL = disabled. */
static int32_t *g_bf_target      = NULL;   /* g_bf_target[d] = target bf at depth d */
static int      g_bf_target_max  = -1;     /* max depth index loaded (-1 = disabled) */
static const char *g_bf_target_path = NULL; /* deferred until after arg parse */
static int      g_bf_smooth_w    = 0;      /* moving-average half-width (0 = raw) */
static int g_rollout_gen1_pct = -1; /* >=0: force gen 1 to be a bf-band at this percentile
                                     *      (full M*C pool + rank) instead of random, while
                                     *      leaving gens 2+ untouched.  -1 = gen 1 obeys the
                                     *      normal rand-gens/percentile rules.  Experiment knob:
                                     *      does any gen-1 percentile beat pure-random seeding? */
static int g_rollout_trace = 0;     /* 1 = log per-generation selectivity to stderr */
static int g_rollout_pushoff = 0;   /* 1 = seed each trajectory from the V3 "block pulled
                                     *     off the exit" states instead of the bare root */
static int g_rollout_dfsfill = -1;  /* N: when the deduped pool has < N unique survivors,
                                     *    exhaustively DFS-enumerate distinct depth+K states
                                     *    to top the pool up to N.  Keeps the pool above M so
                                     *    the random pool reaches ~M.  -1 = auto: M in rand-gens
                                     *    (keep-all), OFF in percentile mode (fill there chases
                                     *    degenerate low-bf states into dead-ends); 0 = off. */
static long g_rollout_gen_budget = 0; /* >0: target this many total rollouts per percentile
                                       *    generation, scaling c_lim up when pop_n is small
                                       *    (early gens). 0 = legacy fixed C-per-root. */
static int g_rollout_stratify = 1;  /* DEFAULT ON.  Stratify the FIRST backstep: divide each
                                     *    member's c_lim children evenly across its valid first
                                     *    moves (round-robin -- 5 children over 2 moves -> 3 and 2),
                                     *    each then continuing with true random rollout for the
                                     *    remaining K-1 steps.  Evens first-move coverage every
                                     *    generation.  --no-rollout-stratify-first reverts to the
                                     *    legacy per-child random first move (for the H2 control). */
static int g_rollout_rand_fullgen = 1; /* DEFAULT ON (validated best, see g_rollout_rand_gens).
                                     *    1 = in rand-gens generations, lift the c_lim<=C cap so
                                     *    the ~2M-raw target (-> ~M unique, a population's worth)
                                     *    is honored even when pop_n is small (gen 1).  Off = legacy
                                     *    cap at C, which starves gen 1 to ~C unique.  This is the
                                     *    H1 knob: control (off) ~C unique vs treat (on) ~M unique,
                                     *    selection held random.  NOT an M*C over-generate. */
static int g_rollout_alloc = 1;     /* DEFAULT ON.  Budget-balanced recursive rollout
                                     *    allocation: deal each member's c_lim children across
                                     *    valid backsteps (random order, ceil-fair share), with a
                                     *    dead/saturated subtree's budget reflowing onto siblings;
                                     *    each K-deep endpoint absorbs one child.  Spreads the
                                     *    budget over non-stuck branches in a single pass, so it
                                     *    needs no DFS-fill (auto-disabled when on unless
                                     *    --rollout-dfs-fill N is set explicitly).  Subsumes
                                     *    --rollout-stratify-first (which only evens the FIRST
                                     *    step).  --no-rollout-alloc reverts to per-child walks. */
static int g_rollout_flow = 1;      /* DEFAULT ON.  Memoized single-pass "flow" allocation: the
                                     *    same budget-balanced descent as --rollout-alloc, plus
                                     *    (a) a per-generation canonical-key memo so each state's
                                     *    shortcut_check (forward solve) runs at most once even when
                                     *    the DAG re-converges or members overlap, (b) a global
                                     *    collected-set so a leaf already emitted (incl. D4-symmetric
                                     *    / transposition duplicates) reflows its budget to a sibling
                                     *    that yields something NEW, and (c) cross-member budget carry
                                     *    with a global stop once M distinct are collected.  Reaches
                                     *    min(M, reachable-distinct) in one pass, no re-walk.
                                     *    --no-rollout-flow falls back to plain --rollout-alloc. */
static int g_fill_probe = 0;        /* 1 = dump the bf distribution of the deduped pool and
                                     *    where the percentile band lands (per percentile gen).
                                     *    Diagnostic for the DFS-fill-vs-sample comparison. */
static int g_rollout_rand_gens = 2; /* DEFAULT 2 (validated best for fw67 6x6: random gens 1-2,
                                     *    bf-band from gen 3).  Swept k=1..5 (5 restarts each):
                                     *    k=2 dominated (median stall 144 vs 96-104 for k=3/4,
                                     *    40 for k=5).  Forcing gen 1 into a bf-band at any
                                     *    percentile (--rollout-gen1-pct sweep) did NOT beat
                                     *    random seeding (baseline median 112 >= band 96).
                                     *    Set 0 to disable (pure percentile-P from gen 1).
                                     *    N>0: for the first N generations, pick the M
                                     *    survivors as a pure random subset of the unique
                                     *    pool (no branch_factor ranking) to inject diversity
                                     *    at the top, then revert to percentile-P selection.
                                     *    Random gens cap production at ~2M candidates (the
                                     *    big M*C pool would just be randomly discarded), so
                                     *    they cost ~C x less than percentile gens. */
static int g_rollout_max_restarts = 0; /* N>0: stop after N dead-end restarts (episodes),
                                     *    ignoring --time.  Normalizes the episode budget so
                                     *    depth-per-restart can be compared across P/M (the
                                     *    fundamental knob) instead of depth-per-wall-clock
                                     *    (which just rewards whichever P spawns fastest). */
static int g_rollout_max_gen_depth = 0; /* N>0: force a restart once the population's depth
                                     *    reaches N.  Purely a sampling aid (cheap shallow-curve
                                     *    collection): lets --rollout-restarts gather many
                                     *    episodes without each one mining the expensive deep tail. */

static void rollout_record(const BState *s);

/* One random backward step that avoids shortcuts.  On success advances *cur
 * and writes the new state's branch_factor to *out_bf, returning 1.  Returns
 * 0 if no non-shortcut successor exists (this rollout is stuck).  Visiting a
 * random permutation and taking the first valid candidate samples uniformly
 * among the valid successors without shortcut-checking all of them. */
static int rollout_step(BState *cur, int32_t *out_bf) {
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    if (n == 0) return 0;
    int order[16];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (int k = 0; k < n; k++) {
        BState *cand = &buf[order[k]];
        if (shortcut_check(cand) == -1) {   /* -1 = no shortcut: valid */
            rollout_record(cand);           /* every traversed state is a real puzzle */
            *out_bf = g_last_peak_heap;
            *cur = *cand;
            return 1;
        }
    }
    return 0;
}

/* Record a committed checkpoint as a candidate best (mirrors try_successor's
 * best-tracking, including the block-on-exit rejection). */
static void rollout_record(const BState *s) {
    if (s->depth <= g_best_depth) return;
    for (int i = 0; i < s->nblocks; i++)
        if (s->block_pos[i] == g_exit_pos) return;   /* block on exit: reject */
    g_best_depth = s->depth;
    g_best_state = *s;
    if (s->depth > g_overall_best_depth) {
        g_overall_best_depth = s->depth;
        g_overall_best_state = *s;
        g_overall_best_exit  = g_exit_pos;
        stream_new_best(s);
    }
}

/* A valid backward successor plus its branch_factor (g_last_peak_heap captured
 * when shortcut_check accepted it). */
typedef struct { BState st; int32_t bf; } RollSucc;

/* Enumerate the valid (non-shortcut) backward successors of *cur into out[],
 * capturing each one's branch_factor and recording it as a traversed puzzle.
 * Returns the count (<= max).  Used to stratify the first rollout step: rather
 * than each child redrawing the first move at random (a noisy multinomial when
 * c_lim ~ branching), the caller divides the children evenly across these. */
static int rollout_valid_successors(const BState *cur, RollSucc *out, int max) {
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    int nv = 0;
    for (int i = 0; i < n && nv < max; i++) {
        if (shortcut_check(&buf[i]) == -1) {
            out[nv].st = buf[i];
            out[nv].bf = g_last_peak_heap;
            rollout_record(&buf[i]);
            nv++;
        }
    }
    return nv;
}

/* Probe-only: count valid (non-shortcut) backward successors of *cur WITHOUT
 * recording them (no rollout_record) so it can't perturb best-depth/dedup
 * state.  Used by the watch-key probe to read out-degree as a heuristic
 * signal.  Costs one forward shortcut_check per geometric successor. */
static int count_valid_successors(const BState *cur) {
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    int nv = 0;
    for (int i = 0; i < n; i++)
        if (shortcut_check(&buf[i]) == -1) nv++;
    return nv;
}

/* DFS for ANY valid K-step backward continuation from cur.  Returns 1 and
 * writes the endpoint to *out on success. */
static int rollout_dfs_extend(const BState *cur, int remaining, BState *out) {
    if (remaining == 0) { *out = *cur; return 1; }
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    for (int i = 0; i < n; i++) {
        if (shortcut_check(&buf[i]) == -1) {
            rollout_record(&buf[i]);
            if (rollout_dfs_extend(&buf[i], remaining - 1, out)) return 1;
        }
    }
    return 0;
}

/* A surviving rollout endpoint: branch_factor, canonical key (for dedup), and
 * the state itself. */
typedef struct { int32_t bf; uint64_t key; BState st; } RollCand;
static int cmp_rollcand_bf(const void *a, const void *b) {
    int32_t x = ((const RollCand *)a)->bf, y = ((const RollCand *)b)->bf;
    return (x > y) - (x < y);
}
static int cmp_rollcand_key(const void *a, const void *b) {
    uint64_t x = ((const RollCand *)a)->key, y = ((const RollCand *)b)->key;
    return (x > y) - (x < y);
}
/* Sort by |bf - g_rollcand_bf_target| ascending: candidates whose branch_factor
 * is closest to the per-depth target rank first. Used by --rollout-bf-target. */
static int32_t g_rollcand_bf_target = 0;
static int cmp_rollcand_bfdist(const void *a, const void *b) {
    long da = labs((long)((const RollCand *)a)->bf - (long)g_rollcand_bf_target);
    long db = labs((long)((const RollCand *)b)->bf - (long)g_rollcand_bf_target);
    return (da > db) - (da < db);
}

/* Load a "depth,bf" CSV into g_bf_target (indexed by depth), optionally applying
 * a ±smooth_w moving-average so the steppy/cliffy raw bf curve becomes a gentle
 * per-depth target ramp. Returns 1 on success. */
static int load_bf_target(const char *path, int smooth_w) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("fopen --rollout-bf-target"); return 0; }
    int cap = 256, maxd = -1;
    int32_t *raw = calloc(cap, sizeof(int32_t));
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int d; long bf;
        if (sscanf(line, "%d,%ld", &d, &bf) != 2) continue;  /* header/blank */
        if (d < 0) continue;
        if (d >= cap) { int nc = cap; while (d >= nc) nc *= 2;
                        raw = realloc(raw, (size_t)nc * sizeof(int32_t));
                        for (int k = cap; k < nc; k++) raw[k] = 0; cap = nc; }
        raw[d] = (int32_t)bf;
        if (d > maxd) maxd = d;
    }
    fclose(f);
    if (maxd < 0) { fprintf(stderr, "error: --rollout-bf-target %s: no depth,bf rows\n", path);
                    free(raw); return 0; }
    if (smooth_w > 0) {
        int32_t *sm = malloc((size_t)(maxd + 1) * sizeof(int32_t));
        for (int d = 0; d <= maxd; d++) {
            long sum = 0; int n = 0;
            for (int k = d - smooth_w; k <= d + smooth_w; k++)
                if (k >= 0 && k <= maxd) { sum += raw[k]; n++; }
            sm[d] = (int32_t)(sum / n);
        }
        free(raw); raw = sm;
    }
    g_bf_target = raw;
    g_bf_target_max = maxd;
    fprintf(stderr, "[bf-target] loaded %s: depths 0..%d, smooth=%d "
            "(target@20=%d @40=%d @41=%d @50=%d)\n",
            path, maxd, smooth_w,
            maxd >= 20 ? g_bf_target[20] : -1,
            maxd >= 40 ? g_bf_target[40] : -1,
            maxd >= 41 ? g_bf_target[41] : -1,
            maxd >= 50 ? g_bf_target[50] : -1);
    return 1;
}

/* Exhaustively enumerate every distinct valid state `remaining` backward steps
 * deeper than *cur, appending each endpoint (with its branch_factor) to
 * cand[*pnc..].  cur_bf is *cur's branch_factor (the value the band ranks on);
 * it is carried to the endpoint only at remaining==0.  Stops once *pnc reaches
 * cap.  Dedup is left to the caller's key-sort pass. */
static void rollout_dfs_collect(const BState *cur, int remaining, int32_t cur_bf,
                                RollCand *cand, size_t *pnc, size_t cap) {
    if (*pnc >= cap) return;
    if (remaining == 0) {
        cand[*pnc].bf  = cur_bf;
        cand[*pnc].key = canonical_state_key(cur);
        cand[*pnc].st  = *cur;
        (*pnc)++;
        return;
    }
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    /* Visit successors in a random order so a cap-truncated collection is an
     * unbiased sample of the reachable set rather than whatever enumerate_
     * successors happens to emit first.  In rand-gens mode (fill target = M)
     * this is the sole source of the "reasonably random" population. */
    int order[16];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (int oi = 0; oi < n && *pnc < cap; oi++) {
        BState *nx = &buf[order[oi]];
        if (shortcut_check(nx) == -1) {
            int32_t bf = g_last_peak_heap;
            rollout_record(nx);
            rollout_dfs_collect(nx, remaining - 1, bf, cand, pnc, cap);
        }
    }
}

/* Budget-balanced recursive rollout allocation (replaces independent per-child
 * random walks + DFS-fill).  Distributes `budget` K-step backward rollouts from
 * *cur across its valid (non-shortcut) successors: in a random order each
 * successor is dealt a ceil-fair share of the budget still left, and whatever a
 * dead or saturated subtree cannot place reflows onto the siblings dealt after
 * it.  A K-deep endpoint absorbs exactly ONE budget unit, so a narrow corridor
 * takes one child and the remainder spreads to wider branches -- no budget
 * wasted on stuck branches, and no exhaustive fill pass needed.  Appends each
 * endpoint (with its branch_factor) to cand[*pnc .. cap); returns the budget the
 * whole subtree below *cur could not place (so the caller can reflow it).
 *
 * Single pass by design: re-walking a subtree to drain leftover budget would
 * re-emit the same endpoints as duplicates.  The cost is that a dead branch's
 * budget only reaches siblings visited AFTER it, so on an unlucky random order a
 * few units stay unplaced rather than back-filling an earlier live sibling --
 * minor and random, and far better than a per-child walk where any stuck step
 * kills the whole child with no reallocation. */
static int rollout_alloc(const BState *cur, int remaining, int32_t cur_bf,
                         int budget, RollCand *cand, size_t *pnc, size_t cap)
{
    if (budget <= 0)  return 0;
    if (*pnc >= cap)  return budget;            /* no room: report all unplaced */
    if (remaining == 0) {                       /* K-deep endpoint: place one */
        cand[*pnc].bf  = cur_bf;
        cand[*pnc].key = canonical_state_key(cur);
        cand[*pnc].st  = *cur;
        (*pnc)++;
        return budget - 1;                      /* rest reflow to siblings */
    }
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    RollSucc sv[16];
    int nb = 0;
    for (int i = 0; i < n; i++) {
        if (shortcut_check(&buf[i]) == -1) {    /* -1 = no shortcut: valid */
            sv[nb].st = buf[i];
            sv[nb].bf = g_last_peak_heap;
            rollout_record(&buf[i]);            /* every traversed state is a puzzle */
            nb++;
        }
    }
    if (nb == 0) return budget;                 /* dead end: nothing placed */

    int order[16];
    for (int i = 0; i < nb; i++) order[i] = i;
    for (int i = nb - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    int left = budget;
    for (int oi = 0; oi < nb && left > 0 && *pnc < cap; oi++) {
        int kids_left = nb - oi;
        int share = (left + kids_left - 1) / kids_left;   /* ceil split of remainder */
        int un = rollout_alloc(&sv[order[oi]].st, remaining - 1,
                               sv[order[oi]].bf, share, cand, pnc, cap);
        left -= (share - un);                             /* placed = share - unplaced */
    }
    return left;
}

/* ---- Per-generation canonical-key memo (open addressing, lazy stamp clear) ----
 * One entry per distinct canonical state touched this generation.  `sc` caches
 * the shortcut_check verdict (the expensive forward solve) and its bf so each
 * state is solved at most once per generation; `collected` marks leaves already
 * emitted so the flow descent reflows their budget instead of re-emitting.
 * Slots whose stamp != g_hstamp are treated as empty, so bumping g_hstamp clears
 * the whole table in O(1). */
typedef struct {
    uint64_t key;
    uint32_t stamp;
    int32_t  bf;
    uint8_t  sc;          /* 0 = unknown, 1 = valid, 2 = invalid (shortcut) */
    uint8_t  collected;   /* leaf already emitted this generation */
} HEnt;
static HEnt    *g_htab   = NULL;
static size_t   g_hmask  = 0;
static uint32_t g_hstamp = 0;
static long     g_flow_collected = 0;   /* distinct leaves emitted this generation */
static long     g_flow_target    = 0;   /* stop once g_flow_collected reaches this */

static HEnt *hfind(uint64_t key) {
    size_t i = (size_t)(key * 0x9E3779B97F4A7C15ULL) & g_hmask;
    for (;;) {
        HEnt *e = &g_htab[i];
        if (e->stamp != g_hstamp) {     /* stale/empty: claim for this generation */
            e->stamp = g_hstamp;
            e->key = key;
            e->bf = 0;
            e->sc = 0;
            e->collected = 0;
            return e;
        }
        if (e->key == key) return e;
        i = (i + 1) & g_hmask;
    }
}

/* Memoized + collected-aware version of rollout_alloc (the "gravity water" flow:
 * deal `budget` units down the live backstep DAG, reflowing dead/saturated/dup
 * branches onto siblings, stopping globally at g_flow_target distinct leaves).
 * Returns the unplaced budget so the caller can carry it to the next member. */
static int rollout_alloc_flow(const BState *cur, int remaining, int32_t cur_bf,
                              int budget, RollCand *cand, size_t *pnc, size_t cap)
{
    if (budget <= 0)  return 0;
    if (*pnc >= cap)  return budget;
    if (g_flow_collected >= g_flow_target) return budget;   /* have M: stop */
    if (remaining == 0) {                                   /* K-deep endpoint */
        uint64_t k = canonical_state_key(cur);
        HEnt *e = hfind(k);
        if (e->collected) return budget;                    /* dup leaf: reflow */
        e->collected = 1;
        cand[*pnc].bf  = cur_bf;
        cand[*pnc].key = k;
        cand[*pnc].st  = *cur;
        (*pnc)++;
        g_flow_collected++;
        return budget - 1;
    }
    BState buf[16];
    int n = enumerate_successors(cur, buf, 16);
    if (n == 0) return budget;                              /* dead end */

    /* LAZY validity probing: visit successors in random order and forward-solve
     * each only when reached, collecting valid ones until we have enough to
     * spread the whole budget (one per unit) or run out.  When budget < branching
     * we stop after `budget` valid successors, so we never pay the forward solve
     * for successors we'd never descend -- a budget-1 chain costs ~1 solve per
     * level instead of the full fan-out.  When budget >= branching we collect
     * them all (== solve-all).  Splitting `budget` among the collected valid set
     * (not the raw successor count) keeps the fair share exact -- no budget leaks
     * back to the parent from trailing invalid successors. */
    int order[16];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    RollSucc sv[16];
    int nb = 0;
    for (int oi = 0; oi < n && nb < budget; oi++) {
        BState *nx = &buf[order[oi]];
        HEnt *e = hfind(canonical_state_key(nx));
        int valid;
        if (e->sc == 0) {                                   /* unknown: solve once */
            valid = (shortcut_check(nx) == -1);
            e->bf = g_last_peak_heap;
            e->sc = valid ? 1 : 2;
            if (valid) rollout_record(nx);
            g_states_checked++;
        } else {
            valid = (e->sc == 1);
        }
        if (valid) { sv[nb].st = *nx; sv[nb].bf = e->bf; nb++; }
    }
    if (nb == 0) return budget;                             /* no valid backstep */

    int left = budget;
    for (int oi = 0; oi < nb && left > 0 && *pnc < cap; oi++) {
        if (g_flow_collected >= g_flow_target) break;
        int kids_left = nb - oi;
        int share = (left + kids_left - 1) / kids_left;     /* ceil-fair share */
        int un = rollout_alloc_flow(&sv[oi].st, remaining - 1,
                                    sv[oi].bf, share, cand, pnc, cap);
        left -= (share - un);
    }
    return left;
}

/* Fill pop[] with the initial trajectory seeds and return the count.  Default:
 * the bare depth-0 root.  With --rollout-pushoff: the V3 "block pulled off the
 * exit" states (block sitting on the exit cell, player adjacent, no new hole),
 * one per canonical first direction.  Falls back to the bare root if none exist
 * (e.g. without --allow-exit-transit, a block may not sit on the exit). */
static int seed_rollout_pop(const BState *root, BState *pop) {
    if (!g_rollout_pushoff) { pop[0] = *root; return 1; }
    BState buf[16];
    int n = enumerate_successors(root, buf, 16);
    int m = 0;
    for (int i = 0; i < n; i++) {
        int on_exit = 0;
        for (int b = 0; b < buf[i].nblocks; b++)
            if (buf[i].block_pos[b] == g_exit_pos) { on_exit = 1; break; }
        if (on_exit && buf[i].nholes == root->nholes
            && shortcut_check(&buf[i]) == -1) {
            pop[m++] = buf[i];
        }
    }
    if (m == 0) { pop[0] = *root; return 1; }
    return m;
}

static void run_rollout(double remaining_s, int *out_exhausted) {
    struct timespec t0, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int unlimited = (remaining_s <= 0);

    BState root = {
        .player_pos      = (int8_t)g_exit_pos,
        .nblocks         = 0,
        .nholes          = 0,
        .committed_empty = 1ULL << g_exit_pos,
        .depth           = 0,
    };
    /* Honor --seed-path: start (and restart) the population from the replayed
     * seed-path state rather than the bare root.  Validity was already checked
     * in run_exit_search before dispatch, so this build cannot fail here. */
    if (g_have_seed_path) build_seed_path_seed(&root);

    int M = g_rollout_pop, C = g_rollout_child;
    BState   *pop  = malloc((size_t)M * sizeof(BState));
    RollCand *cand = malloc((size_t)M * (size_t)C * sizeof(RollCand));
    if (!pop || !cand) { perror("malloc rollout population"); exit(1); }
    int pop_n = seed_rollout_pop(&root, pop);

    /* Memo table for the flow allocator: sized to comfortably exceed the distinct
     * states one generation can touch (~M*C*(K+1)), kept at load < 0.5.  Allocated
     * once for the process; OS reclaims on exit. */
    if (g_rollout_flow && !g_htab) {
        size_t want = (size_t)M * (size_t)C * (size_t)(g_rollout_steps + 1) * 2;
        size_t sz = 1u << 16;
        while (sz < want && sz < (1u << 24)) sz <<= 1;
        g_htab = calloc(sz, sizeof(HEnt));
        if (!g_htab) { perror("malloc rollout memo"); exit(1); }
        g_hmask = sz - 1;
        g_hstamp = 0;   /* calloc zeroed stamps; first bump -> 1 != 0 entries */
    }

    int no_continuation = 0;
    int gen = 0;
    int restarts = 0;
    while (1) {
        gen++;
        /* --rollout-max-gen-depth: sampling aid.  Once the carried population has
         * reached the cap depth, force a restart instead of mining the (expensive)
         * deeper tail -- lets --rollout-restarts collect many shallow episodes fast. */
        if (g_rollout_max_gen_depth > 0 && pop_n > 0
            && pop[0].depth >= g_rollout_max_gen_depth) {
            restarts++;
            if (g_rollout_max_restarts > 0 && restarts >= g_rollout_max_restarts) {
                if (g_rollout_trace)
                    fprintf(stderr, "%s[rollout] restart cap %d reached -> stop\n",
                            g_rollout_quiet ? "\r\033[K" : "", g_rollout_max_restarts);
                break;
            }
            pop_n = seed_rollout_pop(&root, pop);
            gen = 0;
            continue;
        }
        int pop_n_prev = pop_n;   /* members that spawn this generation */
        /* 1. every member spawns up to c_lim K-step rollouts; pool surviving endpoints.
         *    A RANDOM-selected generation discards all but M survivors at random, so the
         *    full M*C pool is wasted work — cap total production at ~2M (-> ~M after dedup),
         *    which is ~C x cheaper.  Percentile generations keep the full pool to rank. */
        /* gen1_band: force gen 1 into the percentile-band path (full M*C pool,
         * bf-rank at g_rollout_gen1_pct) instead of random, leaving gens 2+ as-is. */
        int gen1_band = (gen == 1 && g_rollout_gen1_pct >= 0);
        int is_rand_gen = (g_rollout_rand_gens > 0 && gen <= g_rollout_rand_gens)
                          && !gen1_band;
        int c_lim = C;
        if (is_rand_gen && pop_n > 0) {
            /* Random-selection generations generate exactly a POPULATION'S WORTH
             * of candidates -- target M raw rollouts, dedup, and KEEP THEM ALL.
             * No over-generation + downselect: a random subset of a random sample
             * is just a smaller random sample, so generating more then randomly
             * discarding is pointless (only a non-random selector like the bf-band
             * benefits from a larger pool).  Targeting M (not 2M) also keeps the
             * deduped pool below M, so the keep-all rule fires and there is NO
             * selection phase at all -- the shuffle just reorders, discards nothing.
             *
             * Legacy capped c_lim at C, which STARVES small populations: on gen 1
             * (pop_n=1) the cap pins raw production at C(=40) instead of M(=400),
             * yielding ~C unique << M.  --rollout-rand-fullgen lifts the cap so the
             * M target is honored regardless of pop_n (gen 1 spawns M, not C). */
            long target = (long)M;                           /* M raw -> dedup -> keep all */
            c_lim = (int)((target + pop_n - 1) / pop_n);     /* ceil(M / pop_n) */
            if (c_lim < 1) c_lim = 1;
            int cap = g_rollout_rand_fullgen ? (int)(((long)M * (long)C) / pop_n) : C;
            if (cap < 1) cap = 1;
            if (c_lim > cap) c_lim = cap;
        } else if (pop_n > 0) {
            /* Small-population boost (DEFAULT).  C is children-per-root assuming
             * a full M-root population, so the intended generation budget is
             * M*C raw candidates.  But gen 1 starts from a SINGLE seed (pop_n=1)
             * and post-stall restarts are tiny, so pop_n*C produces M-fold fewer
             * candidates than intended -- starving exactly the generation whose
             * cuts matter most.  Scale c_lim up to target a full population's
             * worth (M*C by default; --rollout-gen-budget overrides the target),
             * capped at cand[]'s M*C capacity.  When pop_n >= M this is a no-op
             * (ceil(M*C/pop_n) <= C), so full populations keep c_lim = C. */
            long cap_total = (long)M * (long)C;
            long target = (g_rollout_gen_budget > 0) ? g_rollout_gen_budget : cap_total;
            if (target > cap_total) target = cap_total;
            int scaled = (int)((target + pop_n - 1) / pop_n);  /* ceil(target/pop_n) */
            if (scaled > c_lim) c_lim = scaled;              /* only ever raise */
            if ((long)pop_n * c_lim > cap_total) c_lim = (int)(cap_total / pop_n);
            if (c_lim < 1) c_lim = 1;
        }
        size_t nc = 0;
        size_t alloc_cap = (size_t)M * (size_t)C;
        /* Flow allocator: one memoized single-pass descent per member, carrying
         * unplaced budget forward and stopping globally once we hold a full
         * deduped population.  Target M distinct in keep-all/random gens; the big
         * M*C pool in percentile gens (which need volume to rank a band). */
        int flow_leftover = 0;
        long long solves0 = g_states_checked;   /* real forward-solve count this gen (flow) */
        /* Flow is ONLY for rand-gens (produce exactly M distinct cheaply).  Band
         * (percentile) gens must NOT use it: flow's online global dedup + budget
         * cones discover far fewer distinct candidates than the independent-walk
         * pool, starving the bf-band and roughly halving achievable depth. */
        int use_flow = g_rollout_flow && is_rand_gen;
        if (use_flow) {
            g_hstamp++;
            g_flow_collected = 0;
            g_flow_target = (long)M;
        }
        for (int m = 0; m < pop_n; m++) {
            if (use_flow) {
                long need = g_flow_target - g_flow_collected;
                if (need <= 0) break;                 /* population full */
                int budget = c_lim + flow_leftover;
                if ((long)budget > need) budget = (int)need;
                flow_leftover = rollout_alloc_flow(&pop[m], g_rollout_steps, 0,
                                                    budget, cand, &nc, alloc_cap);
                continue;
            }
            if (g_rollout_alloc) {
                /* Budget-balanced allocation subsumes both the independent
                 * per-child walk and the stratify-first even-coverage trick:
                 * deal c_lim children across the live backstep tree, reflowing
                 * dead-branch budget onto siblings. */
                rollout_alloc(&pop[m], g_rollout_steps, 0, c_lim,
                              cand, &nc, alloc_cap);
                continue;
            }
            RollSucc fs[16];
            int nb = 0;
            if (g_rollout_stratify) {
                nb = rollout_valid_successors(&pop[m], fs, 16);
                if (nb == 0) continue;   /* member has no valid backstep */
            }
            for (int c = 0; c < c_lim; c++) {
                BState  cur;
                int32_t bf      = 0;
                int32_t last_bf = 0;
                int ok = 1;
                int step0 = 0;
                if (g_rollout_stratify) {
                    /* Even-coverage first backstep: assign this child to first
                     * move (c % nb), so each valid move gets ~c_lim/nb children.
                     * fs[] already recorded the move and captured its bf. */
                    int pick = c % nb;
                    cur     = fs[pick].st;
                    bf      = fs[pick].bf;
                    last_bf = bf;
                    step0   = 1;
                } else {
                    cur = pop[m];
                }
                for (int step = step0; step < g_rollout_steps; step++) {
                    if (!rollout_step(&cur, &bf)) { ok = 0; break; }
                    last_bf = bf;
                }
                if (ok) {
                    cand[nc].bf  = last_bf;
                    cand[nc].key = canonical_state_key(&cur);
                    cand[nc].st  = cur;
                    nc++;
                }
            }
        }
        if (!use_flow)
            g_states_checked += (long long)pop_n * c_lim; /* flow counts real solves itself */

        if (nc > 0) {
            /* multi-key watch, stage 1: raw multiplicity before dedup.  Under
             * flow this is ~0/1 (online dedup); under independent walks it is
             * the true duplicate-generation count (path mass). */
            int wraw[16] = {0}, wded[16] = {0}, wpool[16] = {0};
            int wod[16];                      /* valid out-degree of each watched key (-1 absent) */
            for (int w = 0; w < 16; w++) wod[w] = -1;
            int pod_med = -1;                 /* sampled pool median valid out-degree */
            size_t nc_raw = nc;
            if (g_watch_nkeys > 0)
                for (size_t i = 0; i < nc; i++)
                    for (int w = 0; w < g_watch_nkeys; w++)
                        if (cand[i].key == g_watch_keys[w]) wraw[w]++;
            /* 2. dedup by canonical key (sort by key, drop adjacent dups). */
            qsort(cand, nc, sizeof(RollCand), cmp_rollcand_key);
            size_t uniq = 0;
            for (size_t i = 0; i < nc; i++)
                if (i == 0 || cand[i].key != cand[uniq - 1].key)
                    cand[uniq++] = cand[i];

            /* 2b. DFS-fill: random backstepping dries up with depth.  Top the
             *     pool up with an exhaustive (randomly-ordered) enumeration of
             *     distinct depth+K states — but ONLY in rand-gens mode, where we
             *     keep the whole pool and just want ~M randomly-produced
             *     candidates (randomness from the shuffled DFS order).
             *
             *     In percentile mode fill is OFF by default: exhaustive
             *     enumeration surfaces degenerate extreme-low-branch_factor
             *     states (near-shallow-solution, few backward continuations)
             *     that a low-P band then chases straight into early dead-ends.
             *     Measured: filling percentile mode to 5M collapsed rg0 from
             *     best 134 / dead-end med 96 down to 66 / 40.  The random-
             *     rollout pool's implicit bias protects bf-ranking; keep it.
             *     g_rollout_dfsfill overrides: >0 = explicit N, 0 = disabled. */
            long fill_target;
            if (g_rollout_dfsfill == 0)      fill_target = 0;                 /* disabled */
            else if (g_rollout_dfsfill > 0)  fill_target = g_rollout_dfsfill;  /* explicit */
            else if (g_rollout_flow)         fill_target = 0;                 /* flow reflows budget */
            else if (g_rollout_alloc)        fill_target = 0;                 /* alloc reflows budget */
            else if (is_rand_gen)            fill_target = M;                  /* keep-all */
            else                             fill_target = 0;                 /* percentile: off */
            size_t uniq_pre = uniq;
            if (fill_target > 0 && uniq < (size_t)fill_target) {
                size_t cap = (size_t)fill_target;
                if (cap > (size_t)M * (size_t)C) cap = (size_t)M * (size_t)C;
                size_t nc2 = uniq;
                for (int m = 0; m < pop_n_prev && nc2 < cap; m++)
                    rollout_dfs_collect(&pop[m], g_rollout_steps, 0, cand, &nc2, cap);
                qsort(cand, nc2, sizeof(RollCand), cmp_rollcand_key);
                uniq = 0;
                for (size_t i = 0; i < nc2; i++)
                    if (i == 0 || cand[i].key != cand[uniq - 1].key)
                        cand[uniq++] = cand[i];
            }

            /* 2c. median-bf probe: gen-agnostic (fires for rand AND band gens),
             *     so the full depth-vs-pool-bf curve can be reconstructed.  Sorts
             *     a scratch copy of the deduped pool's bf values; does NOT touch
             *     cand[] ordering that selection below relies on. */
            if (g_fill_probe && uniq > 0) {
                static int32_t *bfscratch = NULL;
                static size_t   bfscratch_cap = 0;
                if (uniq > bfscratch_cap) {
                    bfscratch = realloc(bfscratch, uniq * sizeof(int32_t));
                    bfscratch_cap = uniq;
                }
                for (size_t i = 0; i < uniq; i++) bfscratch[i] = cand[i].bf;
                qsort(bfscratch, uniq, sizeof(int32_t), cmp_int32_asc);
                fprintf(stderr, "BFMED gen %d depth %d med %d n %zu\n",
                        gen, cand[0].st.depth, bfscratch[uniq / 2], uniq);
            }

            /* 2d. watch-key probe: report whether a specific canonical key is
             *     present in this gen's deduped pool.  cand[] is sorted by key
             *     here, so binary-search it.  Used for Monte-Carlo estimating
             *     P(gem prefix in random pool). */
            if (g_watch_key && uniq > 0
                && (g_watch_key_depth < 0 || cand[0].st.depth == g_watch_key_depth)) {
                size_t lo2 = 0, hi2 = uniq, found = 0;
                while (lo2 < hi2) {
                    size_t mid = lo2 + (hi2 - lo2) / 2;
                    if (cand[mid].key == g_watch_key) { found = 1; break; }
                    if (cand[mid].key < g_watch_key) lo2 = mid + 1; else hi2 = mid;
                }
                fprintf(stderr, "WATCHKEY gen %d depth %d hit %d n %zu\n",
                        gen, cand[0].st.depth, (int)found, uniq);
            }

            /* multi-key watch, stage 2: presence in the deduped (post-fill)
             * candidate pool.  cand[] sorted by key here. */
            if (g_watch_nkeys > 0 && uniq > 0) {
                for (int w = 0; w < g_watch_nkeys; w++) {
                    size_t lo2 = 0, hi2 = uniq;
                    while (lo2 < hi2) {
                        size_t mid = lo2 + (hi2 - lo2) / 2;
                        if (cand[mid].key == g_watch_keys[w]) {
                            wded[w] = 1;
                            wod[w] = count_valid_successors(&cand[mid].st);
                            break;
                        }
                        if (cand[mid].key < g_watch_keys[w]) lo2 = mid + 1; else hi2 = mid;
                    }
                }
                /* pool out-degree median over a capped random sample */
                int sample[64], sn = 0;
                int want = (uniq < 64) ? (int)uniq : 64;
                for (int s = 0; s < want; s++) {
                    size_t idx = (size_t)(rand() % (int)uniq);
                    sample[sn++] = count_valid_successors(&cand[idx].st);
                }
                for (int a = 1; a < sn; a++) {       /* insertion sort (sn<=64) */
                    int v = sample[a], b = a - 1;
                    while (b >= 0 && sample[b] > v) { sample[b+1] = sample[b]; b--; }
                    sample[b+1] = v;
                }
                if (sn > 0) pod_med = sample[sn / 2];
            }

            /* 3. select M survivors.  For the first g_rollout_rand_gens
             *    generations, take a PURE RANDOM subset of the unique pool
             *    (inject diversity where the most consequential cuts happen);
             *    afterwards rank by branch_factor and take an M-wide band
             *    centered on the percentile-P index. */
            int take = (uniq < (size_t)M) ? (int)uniq : M;
            int lo = 0;
            const char *mode;
            if (is_rand_gen) {
                /* partial Fisher-Yates: shuffle the first `take` into place */
                for (int i = 0; i < take; i++) {
                    int j = i + (rand() % ((int)uniq - i));
                    RollCand tmp = cand[i]; cand[i] = cand[j]; cand[j] = tmp;
                }
                mode = "RANDOM";
            } else if (g_bf_target_max >= 0) {
                /* bf-target: take the M candidates whose branch_factor is closest
                 * to the exemplar level's value at this generation's depth.
                 * All candidates in a generation share one depth (clustered),
                 * so cand[0].st.depth is the representative. */
                int d = cand[0].st.depth;
                if (d < 0) d = 0;
                g_rollcand_bf_target = (d <= g_bf_target_max)
                                       ? g_bf_target[d]
                                       : g_bf_target[g_bf_target_max];  /* clamp */
                qsort(cand, uniq, sizeof(RollCand), cmp_rollcand_bfdist);
                lo = 0;  /* M closest-to-target are now first */
                mode = "bftarget";
            } else {
                qsort(cand, uniq, sizeof(RollCand), cmp_rollcand_bf);
                int pct = gen1_band ? g_rollout_gen1_pct : g_rollout_pct;
                if (!gen1_band && g_rollout_pctwin_lo >= 0) {
                    int gd = cand[0].st.depth;
                    if (gd >= g_rollout_pctwin_lo && gd <= g_rollout_pctwin_hi)
                        pct = g_rollout_pctwin_pct;
                }
                int center = (int)(((long long)pct * ((long long)uniq - 1)) / 100);
                lo = center - M / 2;
                if (lo + M > (int)uniq) lo = (int)uniq - M;
                if (lo < 0) lo = 0;
                mode = (uniq > (size_t)M) ? "yes" : "NO(keep-all)";
            }
            pop_n = take;
            for (int i = 0; i < take; i++) {
                pop[i] = cand[lo + i].st;
                rollout_record(&pop[i]);
            }
            /* multi-key watch, stage 3: presence in the selected pool, then
             * one parseable line per gen. */
            if (g_watch_nkeys > 0) {
                for (int i = 0; i < take; i++)
                    for (int w = 0; w < g_watch_nkeys; w++)
                        if (cand[lo + i].key == g_watch_keys[w]) wpool[w] = 1;
                fprintf(stderr, "WKEYS gen %d depth %d nc %zu uniq %zu podmed %d",
                        gen, pop[0].depth, nc_raw, uniq, pod_med);
                for (int w = 0; w < g_watch_nkeys; w++)
                    fprintf(stderr, " k%d raw %d ded %d pool %d od %d",
                            w, wraw[w], wded[w], wpool[w], wod[w]);
                fprintf(stderr, "\n");
            }
            /* PROBE: in percentile gens cand[] is bf-sorted ascending here, so we
             * can read the pool's bf quantiles directly and report where the band
             * (cand[lo .. lo+take-1]) sits.  uniq_pre vs uniq shows how many states
             * DFS-fill injected; comparing the two runs reveals which way fill
             * shifts the selected window. */
            if (g_fill_probe && !is_rand_gen && uniq > 0) {
                long long u1 = (long long)uniq - 1;
                int32_t p10 = cand[(10 * u1) / 100].bf;
                int32_t p25 = cand[(25 * u1) / 100].bf;
                int32_t p50 = cand[(50 * u1) / 100].bf;
                int32_t p75 = cand[(75 * u1) / 100].bf;
                int32_t p90 = cand[(90 * u1) / 100].bf;
                fprintf(stderr,
                    "PROBE gen %d depth %d uniq_pre %zu uniq %zu fill_added %zu "
                    "bf[min %d p10 %d p25 %d p50 %d p75 %d p90 %d max %d] "
                    "band[lo %d bf_lo %d bf_med %d bf_hi %d]\n",
                    gen, pop[0].depth, uniq_pre, uniq, uniq - uniq_pre,
                    cand[0].bf, p10, p25, p50, p75, p90, cand[uniq - 1].bf,
                    lo, cand[lo].bf, cand[lo + take / 2].bf, cand[lo + take - 1].bf);
            }
            if (g_rollout_trace) {
                /* Under flow, "spawned = pop_n*c_lim" is fictional (flow stops at
                 * the global target and skips remaining members), so report the
                 * real cost instead: distinct states forward-solved this gen. */
                const char *work_lbl = use_flow ? "solved" : "spawned";
                size_t work = use_flow ? (size_t)(g_states_checked - solves0)
                                       : (size_t)pop_n_prev * c_lim;
                fprintf(stderr, "%sgen %3d  depth %3d  %s %zu  survived %zu  "
                        "unique %zu (rnd %zu) kept %d  selective %s%s",
                        g_rollout_quiet ? "\r" : "",
                        gen, pop[0].depth, work_lbl, work, nc, uniq, uniq_pre, take,
                        mode, g_rollout_quiet ? "\033[K" : "\n");
                if (g_rollout_quiet) fflush(stderr);
            }
        } else {
            /* 4. nothing survived — DFS-rescue from some member.  If that also
             *    fails the trajectory is dead: reseed from the root and keep
             *    going (global best persists across restarts).  Restarting is
             *    bounded by --time and/or --rollout-restarts; with neither it
             *    loops until interrupted. */
            BState rescued;
            int rescued_any = 0;
            for (int m = 0; m < pop_n; m++) {
                if (rollout_dfs_extend(&pop[m], g_rollout_steps, &rescued)) {
                    pop[0] = rescued; pop_n = 1;
                    rollout_record(&rescued);
                    rescued_any = 1;
                    break;
                }
            }
            if (rescued_any) {
                if (g_rollout_trace) {
                    fprintf(stderr, "%sgen %3d  stall (0 survivors of %zu spawned)  "
                            "DFS-rescue ok%s", g_rollout_quiet ? "\r" : "",
                            gen, (size_t)pop_n_prev * c_lim, g_rollout_quiet ? "\033[K" : "\n");
                    if (g_rollout_quiet) fflush(stderr);
                }
            } else {
                restarts++;
                if (g_rollout_trace) {
                    fprintf(stderr, "%sgen %3d  DEAD-END at depth %d  -> RESTART #%d%s",
                            g_rollout_quiet ? "\r" : "",
                            gen, (gen - 1) * g_rollout_steps, restarts,
                            g_rollout_quiet ? "\033[K" : "\n");
                    if (g_rollout_quiet) fflush(stderr);
                }
                if (g_rollout_max_restarts > 0 && restarts >= g_rollout_max_restarts) {
                    if (g_rollout_trace)
                        fprintf(stderr, "%s[rollout] restart cap %d reached -> stop\n",
                                g_rollout_quiet ? "\r\033[K" : "", g_rollout_max_restarts);
                    break;
                }
                pop_n = seed_rollout_pop(&root, pop);
                gen = 0;   /* ++ at loop top makes the next generation #1 */
            }
        }

        /* try_successor's periodic debounce flush never runs here, so flush
         * any pending streamed best directly. */
        if (g_first_new_best_time >= 0) {
            double t = session_elapsed();
            if ((t - g_first_new_best_time) >= STREAM_DEBOUNCE_S)
                flush_pending_new_best(t);
        }

        if (!unlimited) {
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            if (elapsed_s(t0, t_now) >= remaining_s) break;
        }
    }

    if (g_rollout_trace)
        fprintf(stderr, "%s[rollout done] restarts=%d best_depth=%d\n",
                g_rollout_quiet ? "\r\033[K" : "", restarts, g_best_depth);
    free(cand);
    free(pop);
    *out_exhausted = no_continuation;   /* "exhausted" iff no continuation exists */
}

/* When non-NULL, run_exit_search uses this list of seeds (depth-2 states
 * from a task) instead of generating depth-0 roots. */
static const BState *g_task_seeds = NULL;
static int           g_task_seed_count = 0;

static double run_exit_search(double remaining_s, int *out_exhausted, int *out_dedup_full) {
    refresh_canonical_for_exit(g_exit_pos);
    /* Reset per-exit state. */
    if (g_two_tables) {
        memset(g_shallow, 0, SHALLOW_CAP * sizeof *g_shallow);
        memset(g_recent,  0, RECENT_CAP  * sizeof *g_recent);
        g_shallow_count = 0;
        g_recent_count  = 0;
        g_recent_clock  = 1;
        g_evictions_shallow = 0;
        g_evictions_recent  = 0;
    } else {
        memset(g_visited, 0, HASH_CAP * sizeof *g_visited);
    }
    g_visited_count  = 0;
    g_q_tail         = 0;
    g_q_peak         = 0;
    g_best_depth     = 0;
    g_states_checked = 0;
    g_pruned_short   = 0;
    g_pruned_dedup   = 0;
    g_pruned_cap     = 0;
    g_pruned_axis    = 0;
    g_solver_calls   = 0;
    g_dedup_full     = 0;
    g_skipped_dedup  = 0;

    if (g_have_seed_path) {
        /* Seed-path mode: start from the standard root and replay the
         * user-supplied sequence of (direction, variant) steps.  The
         * resulting state is the SOLE seed.  DFS then explores its
         * subtree exhaustively (subject to --time / --max-depth). */
        BState seed;
        if (!build_seed_path_seed(&seed)) {
            if (out_exhausted)  *out_exhausted = 0;
            if (out_dedup_full) *out_dedup_full = 0;
            return 0.0;
        }
        fprintf(stderr, "[seed-path] seed built: depth=%d player=%d nblocks=%d nholes=%d committed_pop=%d\n",
                seed.depth, seed.player_pos, seed.nblocks, seed.nholes,
                __builtin_popcountll(seed.committed_empty & g_active_mask));
        g_best_state = seed;
        try_successor(&seed);
    } else if (g_task_seeds && g_task_seed_count > 0) {
        /* Task mode: skip default depth-0 roots; feed each seed through
         * try_successor.  This is what registers the seed itself as a
         * candidate "best" — bypassing it (a previous bug) caused tight
         * --num-walls runs to miss valid depth-3 puzzles whose only
         * successors were over the popcount cap.  try_successor also
         * handles dedup-insert, the shortcut check, and q_push. */
        g_best_state = g_task_seeds[0];
        for (int i = 0; i < g_task_seed_count; i++) {
            BState s = g_task_seeds[i];
            try_successor(&s);
        }
    } else {
        /* Standard root: winning move was a free walk onto an empty exit. */
        BState init = {
            .player_pos      = (int8_t)g_exit_pos,
            .nblocks         = 0,
            .nholes          = 0,
            .committed_empty = 1ULL <<g_exit_pos,
            .depth           = 0,
        };
        g_best_state = init;
        {
            uint64_t k = canonical_state_key(&init);
            if (g_two_tables) dedup_two_tables(k, 0);
            else              dedup_check_and_insert(k, 0);
        }
        if (g_trace_csv || HARVEST_ACTIVE) {
            long long sid = g_next_state_id++;
            init.state_id = (int32_t)sid;
            if (g_trace_csv) {
                fprintf(g_trace_csv, "%lld,-1,%d,%d,%d,%d,%d,%d,0\n",
                        sid, init.depth,
                        __builtin_popcountll(init.committed_empty & g_active_mask),
                        init.nblocks, init.nholes, init.player_pos, g_exit_pos);
            }
            /* Roots use parent_id=-1 — temporarily reset g_current_parent_id
             * around the emit so harvest_emit picks that up. */
            long long save_parent = g_current_parent_id;
            g_current_parent_id = -1;
            harvest_emit(&init, sid, 'A', -1);
            g_current_parent_id = save_parent;
        }
        q_push(&init);

        /* Push-off-exit seeds (only when --allow-exit-transit).  The win
         * move is "player at Y=adj[exit][D'^2] walked to exit pushing
         * block from exit to X=adj[exit][D']"; we seed the depth-1 state
         * that immediately *precedes* this win move (player at Y, block
         * at exit).  Seeding the post-win state at depth 0 instead would
         * also let expand() generate variant-1 walk-backs from the exit,
         * which are inconsistent with the win move and produce phantom
         * puzzles whose transit block's mask points into a wall in the
         * final layout.  Each seed carries one block, so it's only valid
         * when --num-blocks allows at least one. */
        if (g_allow_exit_transit && g_max_blocks >= 1) {
            for (int D = 0; D < 4; D++) {
                if (!(g_canonical_dir_mask & (1 << D))) continue;
                int X = g_adj[g_exit_pos][D];
                if (X < 0) continue;
                if (!(g_walkable_mask & (1ULL <<X))) continue;
                if (g_fixed_holes_mask & (1ULL <<X)) continue;
                int Y = g_adj[g_exit_pos][D ^ 2];
                if (Y < 0) continue;
                if (!(g_walkable_mask & (1ULL <<Y))) continue;
                BState seed = {
                    .player_pos      = (int8_t)Y,
                    .nblocks         = 1,
                    .nholes          = 0,
                    .committed_empty = (1ULL <<g_exit_pos) | (1ULL <<X) | (1ULL <<Y),
                    .depth           = 1,
                };
                seed.block_pos [0] = (int8_t)g_exit_pos;
                seed.block_mask[0] = (uint8_t)(1u << D);
                {
                    uint64_t k = canonical_state_key(&seed);
                    if (g_two_tables) dedup_two_tables(k, seed.depth);
                    else              dedup_check_and_insert(k, seed.depth);
                }
                if (g_trace_csv || HARVEST_ACTIVE) {
                    long long sid = g_next_state_id++;
                    seed.state_id = (int32_t)sid;
                    if (g_trace_csv) {
                        /* push-off-exit seed: the block IS at exit_pos. */
                        fprintf(g_trace_csv, "%lld,-1,%d,%d,%d,%d,%d,%d,1\n",
                                sid, seed.depth,
                                __builtin_popcountll(seed.committed_empty & g_active_mask),
                                seed.nblocks, seed.nholes, seed.player_pos, g_exit_pos);
                    }
                    long long save_parent = g_current_parent_id;
                    g_current_parent_id = -1;
                    harvest_emit(&seed, sid, 'A', -1);
                    g_current_parent_id = save_parent;
                }
                q_push(&seed);
            }
        }
    }

    struct timespec t0, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int exhausted = 1;
    long long iter = 0;
    int unlimited = (remaining_s <= 0);

    if (g_rollout_steps > 0) {
        /* DFS-fill target (g_rollout_dfsfill < 0 = auto) is mode-dependent and
         * resolved per generation inside run_rollout: M in rand-gens, the
         * percentile-band fit otherwise. */
        run_rollout(remaining_s, &exhausted);
    } else if (g_beam_width > 0) {
        /* Beam mode: process frontier depth-by-depth, sort + truncate to
         * top-K each iteration.  Initial roots were q_push'd above, which
         * redirected into g_beam_next.  Swap into g_beam_curr to start. */
        BState *tmp_buf = g_beam_curr;
        int     tmp_n   = g_beam_curr_n;
        int     tmp_cap = g_beam_curr_cap;
        g_beam_curr     = g_beam_next;
        g_beam_curr_n   = g_beam_next_n;
        g_beam_curr_cap = g_beam_next_cap;
        g_beam_next     = tmp_buf;
        g_beam_next_n   = tmp_n;
        g_beam_next_cap = tmp_cap;
        g_beam_next_n   = 0;

        while (g_beam_curr_n > 0) {
            if (g_beam_level_report)
                fprintf(stderr, "BEAMLVL depth %d distinct %d%s\n",
                        g_beam_curr[0].depth, g_beam_curr_n,
                        (g_beam_curr_n >= g_beam_width) ? " (CLIPPED)" : "");
            g_beam_next_n = 0;
            for (int i = 0; i < g_beam_curr_n; i++) {
                if (g_trace_csv || HARVEST_ACTIVE) g_current_parent_id = g_beam_curr[i].state_id;
                expand(&g_beam_curr[i]);
                if (g_dedup_full) { exhausted = 0; goto beam_done; }
            }
            /* Flush deferred surrogate decisions for the whole beam level
             * before sort/clip — gives us the largest possible batch and
             * thus the lowest per-state libtorch overhead. */
            if (g_nn_surrogate_loaded) flush_surrogate_pending();
            /* Sort and truncate.  Optionally fill F*K slots from the
             * sorted tail uniformly at random (stochastic beam). */
            if (g_beam_next_n > g_beam_width) {
                /* NN mode: batch-score all candidates in one libtorch call.
                 * Drops 10k×200μs of single-state inference to one 30ms call. */
                int nn_will_contribute = g_nn_loaded && (
                    (g_nn_use_additive && g_nn_additive != 0.0f) ||
                    (!g_nn_use_additive && g_nn_blend > 0.0f));
                if (nn_will_contribute)
                    compute_batch_scores(g_beam_next, g_beam_next_n);
                /* --beam-score-branching-mid: rewrite scores to -|bf - median|
                 * for the whole level before sort.  Overrides whatever
                 * beam_push_to_next cached for each state. */
                if (g_beam_score_branching_mid)
                    rescore_branching_mid();
                qsort(g_beam_next, (size_t)g_beam_next_n,
                      sizeof(BState), cmp_beam_score);
                if (g_beam_random_frac > 0.0) {
                    int n_det  = (int)(g_beam_width * (1.0 - g_beam_random_frac));
                    int n_rand = g_beam_width - n_det;
                    int tail_lo = n_det;
                    int tail_hi = g_beam_next_n;
                    /* Sample n_rand items from [tail_lo, tail_hi) into
                     * positions [n_det, n_det + n_rand) without replacement. */
                    for (int i = 0; i < n_rand; i++) {
                        int span = tail_hi - tail_lo - i;
                        if (span <= 0) break;
                        int j = tail_lo + i + (rand() % span);
                        BState tmp = g_beam_next[tail_lo + i];
                        g_beam_next[tail_lo + i] = g_beam_next[j];
                        g_beam_next[j] = tmp;
                    }
                }
                /* Save up to g_beam_save_tail states from the clipped tail
                 * into the DFS stack for later exploration. */
                if (g_beam_save_tail > 0) {
                    int tail_avail = g_beam_next_n - g_beam_width;
                    int n_save = tail_avail < g_beam_save_tail ? tail_avail : g_beam_save_tail;
                    for (int i = 0; i < n_save; i++) {
                        if ((size_t)g_q_tail == g_q_cap) {
                            g_q_cap = g_q_cap ? g_q_cap * 2 : 65536;
                            g_queue = realloc(g_queue, g_q_cap * sizeof(BState));
                            if (!g_queue) { perror("realloc q for tail save"); exit(1); }
                        }
                        g_queue[g_q_tail++] = g_beam_next[g_beam_width + i];
                    }
                }
                g_beam_next_n = g_beam_width;
            }
            /* Swap. */
            tmp_buf = g_beam_curr; tmp_n = g_beam_curr_n; tmp_cap = g_beam_curr_cap;
            g_beam_curr = g_beam_next; g_beam_curr_n = g_beam_next_n; g_beam_curr_cap = g_beam_next_cap;
            g_beam_next = tmp_buf; g_beam_next_cap = tmp_cap; g_beam_next_n = 0;
            (void)tmp_n;
            if (!unlimited) {
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                if (elapsed_s(t0, t_now) >= remaining_s) { exhausted = 0; break; }
            }
        }
beam_done:
        /* If --beam-save-tail accumulated states in g_queue, switch to
         * DFS mode and drain.  This recovers branches dropped by beam
         * clipping. */
        if (g_q_tail > 0 && exhausted) {
            g_beam_width = 0;
            while (1) {
                BState s;
                if (!q_pop(&s)) break;
                if (g_trace_csv || HARVEST_ACTIVE) g_current_parent_id = s.state_id;
                expand(&s);
                if (g_dedup_full) { exhausted = 0; break; }
                if (!unlimited && (++iter & 1023) == 0) {
                    clock_gettime(CLOCK_MONOTONIC, &t_now);
                    if (elapsed_s(t0, t_now) >= remaining_s) { exhausted = 0; break; }
                }
            }
        }
    } else {
        while (1) {
            BState s;
            if (!q_pop(&s)) break;
            if (g_trace_csv || HARVEST_ACTIVE) g_current_parent_id = s.state_id;
            expand(&s);
            /* DFS mode: flush after every expand() — no level boundary to
             * batch larger.  Smaller batches but correct ordering. */
            if (g_nn_surrogate_loaded) flush_surrogate_pending();
            if (g_dedup_full) { exhausted = 0; break; }
            if (!unlimited && (++iter & 1023) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t_now);
                if (elapsed_s(t0, t_now) >= remaining_s) { exhausted = 0; break; }
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_now);
    *out_exhausted   = exhausted;
    *out_dedup_full  = g_dedup_full;
    return elapsed_s(t0, t_now);
}

/* -------------------------------------------------------------------------
 * CLI
 * ------------------------------------------------------------------------- */
static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [--grid RxC] [--time SEC] [--exit N] [--allow-exit-transit]\n"
        "                       [--fixedwalls c,c,...] [--fixedholes c,c,...]\n"
        "  --grid RxC            grid dimensions; R*C must be <= %d (= MAX_NCELLS).  Default: 5x5\n"
        "  --time SEC            wall-clock cap in seconds, shared across exits.  Default: %.0f\n"
        "                          (0 = no time limit, run until queue is empty for every exit)\n"
        "  --exit N              restrict to one exit cell.  Default: iterate canonical {0,1,2,6,7,12}\n"
        "  --exitloc c,c,...     iterate the listed exit cells (comma-separated).  Overrides the\n"
        "                          canonical default; --exit takes priority if both are given.\n"
        "  --allow-exit-transit  allow blocks to be pushed onto the exit and back off during play\n"
        "  --allow-block-on-exit permit the puzzle to start with a block on the exit (implies transit)\n"
        "                          (block still cannot start at exit at puzzle setup).  Default: off.\n"
        "  --reverse             invert variant priority within each direction.  In DFS (LIFO),\n"
        "                          default order explores un-consume (4) then new-block / push-\n"
        "                          existing (3 / 2) then walk-back (1); with --reverse, walk-back\n"
        "                          (1) goes first and un-consume (4) last.  Direction priority\n"
        "                          (L, D, R, U) is unchanged.  Default: off.\n"
        "  --shortcut-state-cap N treat shortcut_check as a prune when the forward solver expanded\n"
        "                          more than N unique states (states_popped) without finding a\n"
        "                          shortcut.  Heuristic: states whose forward search explodes are\n"
        "                          unlikely to deepen further.  Per-exit stats line gains a 'cap K'\n"
        "                          field showing how many states were pruned this way.\n"
        "                          Default: 0 (disabled).\n"
        "  --beam-score-branching  beam mode: replace the hand-tuned score (room/blocks/holes/kids/\n"
        "                          hole-adj-exit/mask-pop) with -branch_factor, where branch_factor\n"
        "                          is the peak heap size from this state's shortcut_check call.\n"
        "                          Lower forward-search explosion ranks higher.  Overrides NN/blend\n"
        "                          terms — this is a pure branching-only ranker.  Requires --beam.\n"
        "                          Default: off.\n"
        "  --beam-score-branching-mid  like --beam-score-branching, but each beam level computes\n"
        "                          the target branch_factor across all candidates at that level\n"
        "                          (median by default, tunable via --branching-target-pct) and\n"
        "                          ranks by -|bf - target|.  Rationale: very low bf = trivial\n"
        "                          state with no real puzzle; very high bf = shortcut nearby.\n"
        "                          The middle of the per-depth range is empirically where the\n"
        "                          deepest puzzles live.  Implies --beam-score-branching.\n"
        "                          Default: off.\n"
        "  --branching-target-pct P  percentile target used by --beam-score-branching-mid (0..100).\n"
        "                          50 = median.  Higher values (60-80) bias toward slightly more\n"
        "                          branchy states.  Default: 50.\n"
        "  --beam-score-tailwidth  redefine branch_factor as a deep-weighted sum of the forward\n"
        "                          frontier's WIDTH in the cost-levels just below the cutoff\n"
        "                          (max_cost = depth-2), instead of total states_popped.  Captures\n"
        "                          'is the search tree narrow near the goal' — a forced corridor,\n"
        "                          the hallmark of a deep level.  Implies --beam-score-branching\n"
        "                          (ranks by -branch_factor, smaller = narrower = better).  Combine\n"
        "                          with --beam-score-branching-mid to rank by percentile instead.\n"
        "                          Default: off.\n"
        "  --tailwidth-window K    number of deepest cost-levels summed by --beam-score-tailwidth,\n"
        "                          weighted linearly toward the deepest (the level at max_cost).\n"
        "                          Default: 24.\n"
        "  --rollout K[,M[,C[,P]]] population / stochastic-beam search (own mode; ignores --beam/DFS).\n"
        "                          Carry up to M endpoints across generations.  Each generation every\n"
        "                          member spawns C random rollouts of K backward steps (each step\n"
        "                          uniform among non-shortcut moves); surviving endpoints are pooled,\n"
        "                          deduped by canonical key, ranked by branch_factor, and the next\n"
        "                          population is an M-wide band centered on the P-th percentile.  If\n"
        "                          no rollout survives, DFS-rescue any valid K-step continuation;\n"
        "                          if none exists, stop.  branch_factor honours --beam-score-tailwidth\n"
        "                          (else states_popped); smaller ranks first, so P picks how deep into\n"
        "                          the branchiness distribution the band sits.  M=1 is single-trajectory.\n"
        "                          Defaults: K=10, M=100, C=10, P=50 (e.g. --rollout 10,100,10,50).\n"
        "  --solver-profile      print the per-depth solver-call profile table at the end of the\n"
        "                          run.  Default: off (the table can be ~90 rows).\n"
        "  --rollout-trace       log per-generation selectivity to stderr (depth, spawned, survived,\n"
        "                          unique-after-dedup, kept, and whether the band actually filters or\n"
        "                          keeps-all).  Diagnostic for tuning M/C/P.  Default: off.\n"
        "  --quiet               collapse the per-generation trace into a single self-overwriting\n"
        "                          status line (only one visible at a time); new-best puzzles still\n"
        "                          print normally.  Implies --rollout-trace.  Default: off.\n"
        "  --rollout-pushoff     seed every rollout trajectory from the V3 'block pulled off the exit'\n"
        "                          opening (block on the exit cell, player adjacent) instead of the bare\n"
        "                          root.  Needs --allow-exit-transit to apply; else falls back to root.\n"
        "  --rollout-dfs-fill N  when the deduped survivor pool has < N unique states, exhaustively DFS-\n"
        "                          enumerate distinct depth+K states (in random successor order) from the\n"
        "                          spawning members to top it up to N, since random rollouts run dry with\n"
        "                          depth.  Default: auto = M in rand-gens (keep the whole randomly-produced\n"
        "                          pool, no selection); OFF in percentile mode (exhaustive fill surfaces\n"
        "                          degenerate low-branch_factor states that a low-P band chases into early\n"
        "                          dead-ends).  Pass an explicit N to force it on anywhere; 0 disables.\n"
        "  --rollout-restarts N  stop after N dead-end restarts (episodes) instead of on --time.  Lets you\n"
        "                          compare deepest-depth-per-restart across P/M without the throughput bias\n"
        "                          of a wall-clock cap.  Default: 0 (off; use --time).\n"
        "  --rollout-rand-gens N for the first N generations pick the M survivors as a pure random subset\n"
        "                          (no branch_factor ranking) to inject diversity at the top, then revert\n"
        "                          to percentile-P selection.  Pair with a low P (e.g. 0).  Random gens\n"
        "                          produce only ~M candidates (not the full M*C pool, which would be\n"
        "                          randomly discarded anyway), so they run ~C x faster.  Default: 0.\n"
        "  --holeless            forbid all holes (disables variant-4 un-consume).  Incompatible\n"
        "                          with --fixedholes.  Default: off (holes allowed).\n"
        "  --two-tables          use a two-table dedup that auto-evicts: a SHALLOW table keeps\n"
        "                          the lowest-depth states (evicts deep ones when full) and a\n"
        "                          RECENT table keeps recently-hit states (evicts old ones when\n"
        "                          full).  No more dedup overflow at the cost of some redundant\n"
        "                          subtree exploration.  Default: off (single table).\n"
        "  --dupe-threshold N    only dedup states at depth <= N; deeper states free-fly without\n"
        "                          dedup tracking.  Trades wall-clock for memory headroom on long\n"
        "                          runs that would otherwise hit the dedup table cap.  Default: no\n"
        "                          threshold (all states deduped).\n"
        "  --fixedwalls c,c,...  cells that must be walls in the reported puzzle setup.\n"
        "                          The search prevents these cells from being walked on or used\n"
        "                          for blocks/holes.  Default: none.\n"
        "  --num-walls N         require at least N cells of the active region to remain walls\n"
        "  --single-axis-blocks  allow at most one block pushed along both axes; others single-axis\n"
        "  --single-axis-strict  implies --single-axis-blocks; shortcut check treats each block as\n"
        "                        pushable both ways on any axis it was pulled along\n"
        "                          in the reported puzzle (any cells, the search picks).  Counted\n"
        "                          as TOTAL walls including any --fixedwalls.  Default: 0.\n"
        "  --num-blocks N        cap the number of blocks introduced by the search to at most N.\n"
        "                          Default: %d (no extra constraint beyond MAX_BLOCKS).\n"
        "  --num-holes N         cap the number of active holes to at most N (counts --fixedholes).\n"
        "                          Default: %d (no extra constraint beyond MAX_HOLES).\n"
        "  --fixedholes c,c,...  if non-empty, holes are *only* allowed at these cells.  The\n"
        "                          search may use 0..N holes drawn from this list.  Pairs cleanly\n"
        "                          with --num-holes (cap) and --holeless (forbid all).  Default:\n"
        "                          none (holes may appear anywhere in the active region).\n"
        "  --harvest FILE        emit a packed binary record per *visited* state (accepted +\n"
        "                          shortcut-pruned + dedup-pruned + cap-pruned) with full state\n"
        "                          blob, canonical key, and forward-solve value.  If FILE ends in\n"
        "                          .gz it is gzip-compressed on the fly (~6x smaller).  Use\n"
        "                          harvest_load.py to post-process into a SQLite DB with derived\n"
        "                          max_descendant_depth.  See harvest_format.h for record layout.\n"
        "                          Default: off.\n"
        "  --nn-value-model PATH replace beam_score() with a libtorch TorchScript model that\n"
        "                          predicts max_descendant_depth.  Expects a companion\n"
        "                          PATH.meta.json carrying target_scale and the canonical grid.\n"
        "                          Use export_torchscript.py to generate.  Default: off (hand-\n"
        "                          tuned beam features).\n"
        "  --nn-blend ALPHA      weighted-average blend: score = (1-α)*hand + α*nn.  α=1.0 →\n"
        "                          pure NN.  α=0 short-circuits NN inference for zero overhead.\n"
        "  --nn-additive LAMBDA  additive blend: score = hand_tuned + λ*nn.  Use when the NN\n"
        "                          should refine hand-tuned signal instead of replacing it.\n"
        "  --nn-surrogate-model PATH  load a fast shortcut-prune predictor.  Reads PATH.meta.json\n"
        "                          for task=regression|classification.  Regression: predicts\n"
        "                          forward_solve, prune if pred+margin ≤ depth-2.  Classification:\n"
        "                          predicts P(prune), prune if sigmoid(logit) ≥ threshold.\n"
        "  --nn-surrogate-margin K   regression-mode safety margin (default 0).\n"
        "  --nn-surrogate-threshold P  classification-mode prune threshold ∈ [0,1] (default 0.9).\n"
        "  --nn-surrogate-min-depth N  only consult surrogate at depth ≥ N (default 30) — at\n"
        "                          shallow depths the exact solver is faster than NN inference.\n"
        "\nVisited table is %lu MB (2^%d slots × 16 B).  On overflow the search\n"
        "exits the current root gracefully.  To enlarge, recompile with -DHASH_LG2=N.\n"
        "\nNew bests are streamed to stdout as they are found, prefixed with elapsed time.\n",
        prog, MAX_NCELLS, DEFAULT_CAP_S, MAX_BLOCKS, MAX_HOLES,
        (unsigned long)((size_t)HASH_CAP * sizeof(HashSlot) / (1024*1024)), HASH_LG2);
}

static int parse_grid(const char *s) {
    int r, c;
    if (sscanf(s, "%dx%d", &r, &c) != 2) return 0;
    if (r < 1 || c < 1 || r > MAX_ROWS || c > MAX_COLS || r * c > MAX_NCELLS) return 0;
    g_grid_rows = r;
    g_grid_cols = c;
    return 1;
}

int main(int argc, char **argv) {
    /* Seed rand() so --stochastic produces a different sequence per run.
     * Mixes wall-clock and PID so multi-worker ensembles diverge. */
    srand((unsigned)(time(NULL) ^ (long)getpid()));

    /* --argfile FILE: splice the file's tokens (one per line, blank lines and
     * '#' comments skipped) into the argument stream in place.  Args authored
     * with a text editor / Write tool stay one-per-line and are NEVER subject
     * to shell word-splitting, so packed-string invocations can't malform args.
     * Multiple --argfile are allowed; nesting is not expanded recursively. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--argfile") != 0) continue;
        int cap = argc + 64, n = 0;
        char **av = malloc((size_t)cap * sizeof *av);
        av[n++] = argv[0];
        for (int j = 1; j < argc; j++) {
            if (strcmp(argv[j], "--argfile") == 0 && j + 1 < argc) {
                const char *path = argv[++j];
                FILE *f = fopen(path, "r");
                if (!f) { fprintf(stderr, "error: cannot open --argfile %s\n", path); return 1; }
                char line[1024];
                while (fgets(line, sizeof line, f)) {
                    char *s = line;
                    while (*s == ' ' || *s == '\t') s++;
                    char *e = s + strlen(s);
                    while (e > s && (e[-1]=='\n'||e[-1]=='\r'||e[-1]==' '||e[-1]=='\t')) *--e = '\0';
                    if (*s == '\0' || *s == '#') continue;
                    if (n + 1 >= cap) { cap *= 2; av = realloc(av, (size_t)cap * sizeof *av); }
                    av[n++] = strdup(s);
                }
                fclose(f);
            } else {
                if (n + 1 >= cap) { cap *= 2; av = realloc(av, (size_t)cap * sizeof *av); }
                av[n++] = argv[j];
            }
        }
        argv = av; argc = n;
        break;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --seed requires N\n"); return 1; }
            srand((unsigned)strtoul(argv[i], NULL, 10));   /* reproducible RNG for A/B runs */
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]); return 0;
        } else if (strcmp(argv[i], "--grid") == 0) {
            if (++i >= argc || !parse_grid(argv[i])) {
                fprintf(stderr, "error: --grid requires RxC with R>=1, C>=1, R*C<=%d\n", MAX_NCELLS); return 1;
            }
        } else if (strcmp(argv[i], "--time") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --time requires SEC\n"); return 1; }
            g_time_cap_s = atof(argv[i]);
            if (g_time_cap_s < 0) { fprintf(stderr, "error: --time must be >= 0 (0 = no limit)\n"); return 1; }
        } else if (strcmp(argv[i], "--exit") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --exit requires N\n"); return 1; }
            g_only_exit = atoi(argv[i]);
            /* Bounds-checked later, once sokoban_set_grid has populated g_ncells. */
        } else if (strcmp(argv[i], "--exitloc") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --exitloc requires c,c,...\n"); return 1; }
            const char *p = argv[i];
            uint32_t seen = 0;
            g_n_only_exits = 0;
            while (*p) {
                char *end;
                long v = strtol(p, &end, 10);
                /* Use the maximum allowable cell here; actual grid bounds
                 * are re-checked after sokoban_set_grid populates g_ncells. */
                if (end == p || v < 0 || v >= MAX_NCELLS) {
                    fprintf(stderr, "error: invalid cell '%s' in --exitloc\n", p); return 1;
                }
                if (seen & (1ULL <<v)) {
                    fprintf(stderr, "error: duplicate cell %ld in --exitloc\n", v); return 1;
                }
                seen |= (1ULL <<v);
                g_only_exit_list[g_n_only_exits++] = (int)v;
                p = end;
                if (*p == ',') p++;
                else if (*p) { fprintf(stderr, "error: expected ',' in --exitloc\n"); return 1; }
            }
            if (g_n_only_exits == 0) {
                fprintf(stderr, "error: --exitloc requires at least one cell\n"); return 1;
            }
        } else if (strcmp(argv[i], "--allow-exit-transit") == 0) {
            g_allow_exit_transit = 1;
        } else if (strcmp(argv[i], "--allow-block-on-exit") == 0) {
            g_allow_block_on_exit = 1;
            g_allow_exit_transit  = 1;   /* required to generate block-on-exit states */
        } else if (strcmp(argv[i], "--reverse") == 0) {
            g_reverse_order = 1;
        } else if (strcmp(argv[i], "--shortcut-state-cap") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --shortcut-state-cap requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) { fprintf(stderr, "error: --shortcut-state-cap must be >= 0\n"); return 1; }
            g_shortcut_state_cap = n;
        } else if (strcmp(argv[i], "--beam-score-branching") == 0) {
            g_beam_score_branching = 1;
        } else if (strcmp(argv[i], "--beam-score-branching-mid") == 0) {
            g_beam_score_branching_mid = 1;
            g_beam_score_branching     = 1;   /* needed so beam_push_to_next caches branch_factor */
        } else if (strcmp(argv[i], "--beam-score-tailwidth") == 0) {
            g_use_tailwidth        = 1;
            g_beam_score_branching = 1;   /* rank by -branch_factor (smaller tail = forced corridor) */
        } else if (strcmp(argv[i], "--tailwidth-window") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --tailwidth-window requires K\n"); return 1; }
            int k = atoi(argv[i]);
            if (k < 1 || k > BFS_TAIL_W) { fprintf(stderr, "error: --tailwidth-window must be in [1,%d]\n", BFS_TAIL_W); return 1; }
            g_tailwidth_window = k;
        } else if (strcmp(argv[i], "--rollout") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout requires K[,M[,C[,P]]]\n"); return 1; }
            int k = 10, mm = 100, cc = 10, pp = 50;
            if (sscanf(argv[i], "%d,%d,%d,%d", &k, &mm, &cc, &pp) < 1) {
                fprintf(stderr, "error: --rollout expects K[,M[,C[,P]]] (e.g. 10,100,10,50)\n"); return 1;
            }
            if (k < 1)               { fprintf(stderr, "error: --rollout K (steps) must be >= 1\n"); return 1; }
            if (mm < 1)              { fprintf(stderr, "error: --rollout M (population) must be >= 1\n"); return 1; }
            if (cc < 1)              { fprintf(stderr, "error: --rollout C (children) must be >= 1\n"); return 1; }
            if (pp < 0 || pp > 100)  { fprintf(stderr, "error: --rollout P (percentile) must be in [0,100]\n"); return 1; }
            g_rollout_steps = k;
            g_rollout_pop   = mm;
            g_rollout_child = cc;
            g_rollout_pct   = pp;
        } else if (strcmp(argv[i], "--solver-profile") == 0) {
            g_solver_profile = 1;
        } else if (strcmp(argv[i], "--rollout-trace") == 0) {
            g_rollout_trace = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            g_rollout_quiet = 1;
            g_rollout_trace = 1;   /* quiet collapses the trace; it implies trace */
        } else if (strcmp(argv[i], "--rollout-pushoff") == 0) {
            g_rollout_pushoff = 1;
        } else if (strcmp(argv[i], "--rollout-dfs-fill") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-dfs-fill requires N\n"); return 1; }
            g_rollout_dfsfill = atoi(argv[i]);
            if (g_rollout_dfsfill < 0) { fprintf(stderr, "error: --rollout-dfs-fill N must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--fill-probe") == 0) {
            g_fill_probe = 1;
        } else if (strcmp(argv[i], "--rollout-stratify-first") == 0) {
            g_rollout_stratify = 1;
        } else if (strcmp(argv[i], "--no-rollout-stratify-first") == 0) {
            g_rollout_stratify = 0;
        } else if (strcmp(argv[i], "--rollout-rand-fullgen") == 0) {
            g_rollout_rand_fullgen = 1;
        } else if (strcmp(argv[i], "--no-rollout-rand-fullgen") == 0) {
            g_rollout_rand_fullgen = 0;
        } else if (strcmp(argv[i], "--rollout-alloc") == 0) {
            g_rollout_alloc = 1;
        } else if (strcmp(argv[i], "--no-rollout-alloc") == 0) {
            g_rollout_alloc = 0;
        } else if (strcmp(argv[i], "--rollout-flow") == 0) {
            g_rollout_flow = 1;
        } else if (strcmp(argv[i], "--no-rollout-flow") == 0) {
            g_rollout_flow = 0;
        } else if (strcmp(argv[i], "--rollout-gen-budget") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-gen-budget requires N\n"); return 1; }
            g_rollout_gen_budget = strtol(argv[i], NULL, 10);
            if (g_rollout_gen_budget < 0) { fprintf(stderr, "error: --rollout-gen-budget N must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--rollout-restarts") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-restarts requires N\n"); return 1; }
            g_rollout_max_restarts = atoi(argv[i]);
            if (g_rollout_max_restarts < 0) { fprintf(stderr, "error: --rollout-restarts N must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--rollout-max-gen-depth") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-max-gen-depth requires N\n"); return 1; }
            g_rollout_max_gen_depth = atoi(argv[i]);
            if (g_rollout_max_gen_depth < 0) { fprintf(stderr, "error: --rollout-max-gen-depth N must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--rollout-rand-gens") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-rand-gens requires N\n"); return 1; }
            g_rollout_rand_gens = atoi(argv[i]);
            if (g_rollout_rand_gens < 0) { fprintf(stderr, "error: --rollout-rand-gens N must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--rollout-gen1-pct") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-gen1-pct requires P in [0,100]\n"); return 1; }
            int p = atoi(argv[i]);
            if (p < 0 || p > 100) { fprintf(stderr, "error: --rollout-gen1-pct must be in [0,100]\n"); return 1; }
            g_rollout_gen1_pct = p;
        } else if (strcmp(argv[i], "--branching-target-pct") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --branching-target-pct requires P in [0,100]\n"); return 1; }
            int p = atoi(argv[i]);
            if (p < 0 || p > 100) { fprintf(stderr, "error: --branching-target-pct must be in [0,100]\n"); return 1; }
            g_branching_target_pct = p;
        } else if (strcmp(argv[i], "--no-state-canon") == 0) {
            g_state_canon = 0;
        } else if (strcmp(argv[i], "--beam") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --beam requires K\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 1) { fprintf(stderr, "error: --beam K must be >= 1\n"); return 1; }
            g_beam_width = n;
        } else if (strcmp(argv[i], "--beam-save-tail") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --beam-save-tail requires T\n"); return 1; }
            g_beam_save_tail = atoi(argv[i]);
            if (g_beam_save_tail < 0) { fprintf(stderr, "error: --beam-save-tail must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--stochastic") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --stochastic requires F\n"); return 1; }
            g_beam_random_frac = atof(argv[i]);
            if (g_beam_random_frac < 0.0 || g_beam_random_frac > 1.0) {
                fprintf(stderr, "error: --stochastic F must be in [0, 1]\n"); return 1;
            }
        } else if (strcmp(argv[i], "--score-weights") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --score-weights requires wRoom,wBlocks,wHoles,wKids[,wHoleAdj,wMaskPop]\n"); return 1; }
            /* Accept 4 (legacy) or 6 (with new features) comma-separated weights. */
            int n6 = sscanf(argv[i], "%lf,%lf,%lf,%lf,%lf,%lf",
                            &g_w_room, &g_w_blocks, &g_w_holes, &g_w_kids,
                            &g_w_holeadj, &g_w_maskpop);
            if (n6 != 4 && n6 != 6) {
                fprintf(stderr, "error: --score-weights expects 4 or 6 comma-separated floats\n"); return 1;
            }
        } else if (strcmp(argv[i], "--w-hole-adj") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --w-hole-adj requires F\n"); return 1; }
            g_w_holeadj = atof(argv[i]);
        } else if (strcmp(argv[i], "--w-mask-pop") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --w-mask-pop requires F\n"); return 1; }
            g_w_maskpop = atof(argv[i]);
        } else if (strcmp(argv[i], "--seed-path") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --seed-path requires \"D1V1,D2V2,...\"\n"); return 1; }
            if (!parse_seed_path(argv[i])) {
                if (g_seed_path_overflow)
                    fprintf(stderr, "error: --seed-path too long (max %d tokens)\n",
                            (int)(sizeof(g_seed_path)/sizeof(*g_seed_path)));
                else
                    fprintf(stderr, "error: --seed-path must be tokens like U2,R3,L1 (dir letter U/R/D/L + action digit 1=walk, 2=push, 3=consume)\n");
                return 1;
            }
            g_have_seed_path = 1;
        } else if (strcmp(argv[i], "--print-seed-key") == 0) {
            g_print_seed_key = 1;
        } else if (strcmp(argv[i], "--beam-level-report") == 0) {
            g_beam_level_report = 1;
        } else if (strcmp(argv[i], "--rollout-watch-key") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-watch-key requires KEY[,DEPTH]\n"); return 1; }
            g_watch_key = strtoull(argv[i], NULL, 10);
            const char *comma = strchr(argv[i], ',');
            g_watch_key_depth = comma ? atoi(comma + 1) : -1;
        } else if (strcmp(argv[i], "--rollout-watch-keys") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-watch-keys requires K1,K2,...\n"); return 1; }
            const char *p = argv[i];
            while (*p && g_watch_nkeys < 16) {
                g_watch_keys[g_watch_nkeys++] = strtoull(p, NULL, 10);
                const char *c = strchr(p, ',');
                if (!c) break;
                p = c + 1;
            }
        } else if (strcmp(argv[i], "--trace-csv") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --trace-csv requires FILE\n"); return 1; }
            g_trace_csv = fopen(argv[i], "w");
            if (!g_trace_csv) { perror("fopen --trace-csv"); return 1; }
            fprintf(g_trace_csv, "id,parent_id,depth,popcount,nblocks,nholes,player_pos,exit_pos,block_on_exit\n");
        } else if (strcmp(argv[i], "--bf-dump") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --bf-dump requires FILE\n"); return 1; }
            g_bf_dump = fopen(argv[i], "w");
            if (!g_bf_dump) { perror("fopen --bf-dump"); return 1; }
            fprintf(g_bf_dump, "depth,branch_factor\n");
        } else if (strcmp(argv[i], "--rollout-pct-window") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-pct-window requires LO,HI,P\n"); return 1; }
            if (sscanf(argv[i], "%d,%d,%d", &g_rollout_pctwin_lo,
                       &g_rollout_pctwin_hi, &g_rollout_pctwin_pct) != 3) {
                fprintf(stderr, "error: --rollout-pct-window expects LO,HI,P\n"); return 1;
            }
        } else if (strcmp(argv[i], "--rollout-bf-target") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-bf-target requires FILE\n"); return 1; }
            g_bf_target_path = argv[i];   /* loaded after arg parse (smooth may follow) */
        } else if (strcmp(argv[i], "--rollout-bf-smooth") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --rollout-bf-smooth requires W\n"); return 1; }
            g_bf_smooth_w = atoi(argv[i]);
            if (g_bf_smooth_w < 0) { fprintf(stderr, "error: --rollout-bf-smooth W must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--harvest") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --harvest requires FILE\n"); return 1; }
            if (!harvest_open(argv[i], argc, argv)) return 1;
            /* Header write is deferred until sokoban_set_grid() has populated
             * g_rows/g_cols.  Flags are gathered as the rest of argv parses. */
        } else if (strcmp(argv[i], "--nn-value-model") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-value-model requires PATH\n"); return 1; }
            /* Parse companion .meta.json to recover target_scale + canonical grid.
             * Done as a tiny manual scan to avoid a JSON dep.  Falls back to
             * sensible defaults if the meta file is missing. */
            char meta_path[1024];
            snprintf(meta_path, sizeof meta_path, "%s.meta.json", argv[i]);
            float target_scale = 1.0f;
            int meta_rows = 0, meta_cols = 0, meta_channels = NN_CHANNELS;
            FILE *mf = fopen(meta_path, "r");
            if (mf) {
                char buf[2048] = {0};
                size_t n = fread(buf, 1, sizeof buf - 1, mf);
                buf[n] = 0;
                fclose(mf);
                const char *p;
                if ((p = strstr(buf, "\"target_scale\""))) sscanf(p, "%*[^:]:%f", &target_scale);
                if ((p = strstr(buf, "\"rows\"")))         sscanf(p, "%*[^:]:%d", &meta_rows);
                if ((p = strstr(buf, "\"cols\"")))         sscanf(p, "%*[^:]:%d", &meta_cols);
                if ((p = strstr(buf, "\"channels\"")))     sscanf(p, "%*[^:]:%d", &meta_channels);
            } else {
                fprintf(stderr, "warning: %s not found; assuming target_scale=1.0 "
                                "and grid from --grid\n", meta_path);
            }
            /* Note: actual nn_load happens after sokoban_set_grid runs (below)
             * so we know g_rows/g_cols match what the model expects. */
            g_nn_model_path   = argv[i];
            g_nn_target_scale = target_scale;
            g_nn_meta_rows    = meta_rows;
            g_nn_meta_cols    = meta_cols;
            g_nn_meta_channels = meta_channels;
        } else if (strcmp(argv[i], "--nn-blend") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-blend requires FLOAT in [0,1]\n"); return 1; }
            g_nn_blend = atof(argv[i]);
            g_nn_use_additive = 0;
            if (g_nn_blend < 0.0f || g_nn_blend > 1.0f) {
                fprintf(stderr, "error: --nn-blend must be in [0,1] (got %f)\n", g_nn_blend);
                return 1;
            }
        } else if (strcmp(argv[i], "--nn-additive") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-additive requires FLOAT\n"); return 1; }
            g_nn_additive = atof(argv[i]);
            g_nn_use_additive = 1;
        } else if (strcmp(argv[i], "--nn-surrogate-model") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-surrogate-model requires PATH\n"); return 1; }
            char meta_path[1024];
            snprintf(meta_path, sizeof meta_path, "%s.meta.json", argv[i]);
            float ts = 1.0f;
            int mr = 0, mc = 0, mch = NN_CHANNELS;
            int is_cls = 0;
            FILE *mf = fopen(meta_path, "r");
            if (mf) {
                char buf[2048] = {0};
                size_t n = fread(buf, 1, sizeof buf - 1, mf);
                buf[n] = 0;
                fclose(mf);
                const char *p;
                if ((p = strstr(buf, "\"target_scale\""))) sscanf(p, "%*[^:]:%f", &ts);
                if ((p = strstr(buf, "\"rows\"")))         sscanf(p, "%*[^:]:%d", &mr);
                if ((p = strstr(buf, "\"cols\"")))         sscanf(p, "%*[^:]:%d", &mc);
                if ((p = strstr(buf, "\"channels\"")))     sscanf(p, "%*[^:]:%d", &mch);
                if ((p = strstr(buf, "\"task\""))) {
                    /* Match the value (string in quotes). */
                    const char *q = strchr(p + 6, '"');  /* skip past "task": */
                    if (q && q[1] && q[2] && q[3]) {
                        /* look for "classification" or "regression" */
                        if (strncmp(q + 1, "classification", 14) == 0) is_cls = 1;
                    }
                }
            }
            g_nn_surrogate_model_path    = argv[i];
            g_nn_surrogate_target_scale  = ts;
            g_nn_surrogate_meta_rows     = mr;
            g_nn_surrogate_meta_cols     = mc;
            g_nn_surrogate_meta_channels = mch;
            g_nn_surrogate_is_classification = is_cls;
        } else if (strcmp(argv[i], "--nn-surrogate-margin") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-surrogate-margin requires FLOAT\n"); return 1; }
            g_nn_surrogate_margin = atof(argv[i]);
        } else if (strcmp(argv[i], "--nn-surrogate-threshold") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-surrogate-threshold requires FLOAT in [0,1]\n"); return 1; }
            g_nn_surrogate_threshold = atof(argv[i]);
        } else if (strcmp(argv[i], "--nn-surrogate-min-depth") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --nn-surrogate-min-depth requires INT\n"); return 1; }
            g_nn_surrogate_min_depth = atoi(argv[i]);
        } else if (strcmp(argv[i], "--holeless") == 0) {
            g_holeless = 1;
        } else if (strcmp(argv[i], "--two-tables") == 0) {
            g_two_tables = 1;
        } else if (strcmp(argv[i], "--list-tasks") == 0) {
            g_list_tasks = 1;
        } else if (strcmp(argv[i], "--task-id") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --task-id requires N\n"); return 1; }
            g_only_task = atoi(argv[i]);
            if (g_only_task < 0) { fprintf(stderr, "error: --task-id must be >= 0\n"); return 1; }
        } else if (strcmp(argv[i], "--num-walls") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --num-walls requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) { fprintf(stderr, "error: --num-walls must be >= 0\n"); return 1; }
            g_min_walls = n;
        } else if (strcmp(argv[i], "--single-axis-blocks") == 0) {
            g_single_axis_blocks = 1;
        } else if (strcmp(argv[i], "--single-axis-strict") == 0) {
            g_single_axis_blocks = 1;
            g_axis_both_ways     = 1;
        } else if (strcmp(argv[i], "--max-depth") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --max-depth requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) { fprintf(stderr, "error: --max-depth must be >= 0\n"); return 1; }
            g_max_depth = n;
        } else if (strcmp(argv[i], "--num-blocks") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --num-blocks requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0 || n > MAX_BLOCKS) {
                fprintf(stderr, "error: --num-blocks must be in [0..%d]\n", MAX_BLOCKS); return 1;
            }
            g_max_blocks = n;
        } else if (strcmp(argv[i], "--num-holes") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --num-holes requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0 || n > MAX_HOLES) {
                fprintf(stderr, "error: --num-holes must be in [0..%d]\n", MAX_HOLES); return 1;
            }
            g_max_holes = n;
        } else if (strcmp(argv[i], "--dupe-threshold") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --dupe-threshold requires N\n"); return 1; }
            int n = atoi(argv[i]);
            if (n < 0) {
                fprintf(stderr, "error: --dupe-threshold must be >= 0\n"); return 1;
            }
            g_dupe_threshold = n;
        } else if (strcmp(argv[i], "--fixedwalls") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --fixedwalls requires c,c,...\n"); return 1; }
            const char *p = argv[i];
            g_fixed_walls_mask = 0;
            while (*p) {
                char *end;
                long v = strtol(p, &end, 10);
                /* Use the maximum allowable cell here; actual grid bounds
                 * are re-checked after sokoban_set_grid populates g_ncells. */
                if (end == p || v < 0 || v >= MAX_NCELLS) {
                    fprintf(stderr, "error: invalid cell '%s' in --fixedwalls\n", p); return 1;
                }
                if (g_fixed_walls_mask & (1ULL <<v)) {
                    fprintf(stderr, "error: duplicate cell %ld in --fixedwalls\n", v); return 1;
                }
                g_fixed_walls_mask |= (1ULL <<v);
                p = end;
                if (*p == ',') p++;
                else if (*p) { fprintf(stderr, "error: expected ',' in --fixedwalls\n"); return 1; }
            }
        } else if (strcmp(argv[i], "--fixedholes") == 0) {
            if (++i >= argc) { fprintf(stderr, "error: --fixedholes requires c,c,...\n"); return 1; }
            const char *p = argv[i];
            g_fixed_nholes = 0;
            g_fixed_holes_mask = 0;
            while (*p) {
                char *end;
                long v = strtol(p, &end, 10);
                /* Use the maximum allowable cell here; actual grid bounds
                 * are re-checked after sokoban_set_grid populates g_ncells. */
                if (end == p || v < 0 || v >= MAX_NCELLS) {
                    fprintf(stderr, "error: invalid cell '%s' in --fixedholes\n", p); return 1;
                }
                if (g_fixed_holes_mask & (1ULL <<v)) {
                    fprintf(stderr, "error: duplicate cell %ld in --fixedholes\n", v); return 1;
                }
                if (g_fixed_nholes >= MAX_HOLES) {
                    fprintf(stderr, "error: too many fixed holes (max %d)\n", MAX_HOLES); return 1;
                }
                g_fixed_hole_pos[g_fixed_nholes++] = (int)v;
                g_fixed_holes_mask |= (1ULL <<v);
                p = end;
                if (*p == ',') p++;
                else if (*p) { fprintf(stderr, "error: expected ',' in --fixedholes\n"); return 1; }
            }
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    /* Load the bf-target curve now that --rollout-bf-smooth has also been seen. */
    if (g_bf_target_path && !load_bf_target(g_bf_target_path, g_bf_smooth_w))
        return 1;

    /* Configure the solver for the requested grid.  This populates the
     * solver's g_rows / g_cols / g_ncells / g_adj / g_all_cells globals,
     * which we then read directly from the header. */
    sokoban_set_grid(g_grid_rows, g_grid_cols);

    /* Snapshot the harvest flags now that argv has been fully parsed. */
    g_harvest_flags = 0;
    if (g_allow_exit_transit) g_harvest_flags |= HARVEST_FLAG_ALLOW_EXIT_TRANSIT;
    if (g_two_tables)         g_harvest_flags |= HARVEST_FLAG_TWO_TABLES;
    if (g_holeless)           g_harvest_flags |= HARVEST_FLAG_HOLELESS;

    /* Load the value-head NN now that g_rows/g_cols are known.  If the
     * checkpoint's canonical grid doesn't match, abort — the model
     * weights are grid-specific (channels are shared but the conv field
     * sizes differ). */
    if (g_nn_model_path) {
        int use_rows = g_nn_meta_rows ? g_nn_meta_rows : g_rows;
        int use_cols = g_nn_meta_cols ? g_nn_meta_cols : g_cols;
        if (use_rows != g_rows || use_cols != g_cols) {
            fprintf(stderr,
                "warning: --nn-value-model trained for %dx%d but --grid is %dx%d; "
                "scoring will still run (fully-conv model handles any size) but "
                "may produce out-of-distribution predictions\n",
                use_rows, use_cols, g_rows, g_cols);
        }
        if (!nn_load(g_nn_model_path, g_nn_target_scale,
                     g_rows, g_cols, g_nn_meta_channels)) {
            fprintf(stderr, "fatal: failed to load NN model %s\n", g_nn_model_path);
            return 1;
        }
        g_nn_loaded = 1;
        fprintf(stderr, "[nn] loaded %s (grid %dx%d, target_scale=%.3f)\n",
                g_nn_model_path, g_rows, g_cols, g_nn_target_scale);
    }

    if (g_nn_surrogate_model_path) {
        if (!nn_surrogate_load(g_nn_surrogate_model_path, g_nn_surrogate_target_scale,
                               g_rows, g_cols, g_nn_surrogate_meta_channels)) {
            fprintf(stderr, "fatal: failed to load NN surrogate %s\n", g_nn_surrogate_model_path);
            return 1;
        }
        g_nn_surrogate_loaded = 1;
        if (g_nn_surrogate_is_classification) {
            fprintf(stderr, "[nn-surrogate] loaded %s (grid %dx%d, task=classification, "
                            "threshold=%.2f, min_depth=%d)\n",
                    g_nn_surrogate_model_path, g_rows, g_cols,
                    g_nn_surrogate_threshold, g_nn_surrogate_min_depth);
        } else {
            fprintf(stderr, "[nn-surrogate] loaded %s (grid %dx%d, task=regression, "
                            "target_scale=%.3f, margin=%.1f, min_depth=%d)\n",
                    g_nn_surrogate_model_path, g_rows, g_cols, g_nn_surrogate_target_scale,
                    g_nn_surrogate_margin, g_nn_surrogate_min_depth);
        }
    }

    /* Now that g_ncells is known, finish bounds-checking flags that were
     * parsed before sokoban_set_grid (--exit, --exitloc, --fixedwalls,
     * --fixedholes accept up to MAX_NCELLS at parse time). */
    if (g_only_exit >= 0 && g_only_exit >= g_ncells) {
        fprintf(stderr, "error: --exit must be 0..%d (got %d)\n", g_ncells - 1, g_only_exit);
        return 1;
    }
    for (int i = 0; i < g_n_only_exits; i++) {
        if (g_only_exit_list[i] >= g_ncells) {
            fprintf(stderr, "error: --exitloc cell %d is outside the %dx%d grid\n",
                    g_only_exit_list[i], g_grid_rows, g_grid_cols);
            return 1;
        }
    }
    if (g_fixed_walls_mask & ~((g_ncells == 64) ? ~0ULL : ((1ULL << g_ncells) - 1))) {
        fprintf(stderr, "error: --fixedwalls includes a cell outside the %dx%d grid\n",
                g_grid_rows, g_grid_cols);
        return 1;
    }
    if (g_fixed_holes_mask & ~((g_ncells == 64) ? ~0ULL : ((1ULL << g_ncells) - 1))) {
        fprintf(stderr, "error: --fixedholes includes a cell outside the %dx%d grid\n",
                g_grid_rows, g_grid_cols);
        return 1;
    }

    /* Build active mask from grid dimensions.  In the new generalized
     * model the entire grid is the active region, so this equals the
     * solver's g_all_cells — but we keep the loop for clarity and in case
     * future use re-introduces an active sub-region. */
    g_active_mask = 0;
    for (int r = 0; r < g_grid_rows; r++)
        for (int c = 0; c < g_grid_cols; c++)
            g_active_mask |= 1ULL << (r * g_cols + c);

    sokoban_init();

    /* Validate fixed holes against the active region. */
    for (int i = 0; i < g_fixed_nholes; i++) {
        int h = g_fixed_hole_pos[i];
        if (!(g_active_mask & (1ULL <<h))) {
            fprintf(stderr, "error: --fixedholes cell %d is outside the %dx%d active region\n",
                    h, g_grid_rows, g_grid_cols);
            return 1;
        }
    }

    /* Validate fixed walls: must be in active region, must not coincide
     * with a fixed hole. */
    if (g_fixed_walls_mask & ~g_active_mask) {
        fprintf(stderr, "error: --fixedwalls includes a cell outside the %dx%d active region\n",
                g_grid_rows, g_grid_cols);
        return 1;
    }
    if (g_fixed_walls_mask & g_fixed_holes_mask) {
        fprintf(stderr, "error: --fixedwalls and --fixedholes overlap\n");
        return 1;
    }
    if (g_holeless && g_max_holes > 0) {
        /* --holeless implies cap=0; just enforce. */
        g_max_holes = 0;
    }
    /* Note: --holeless + --fixedholes is now consistent — both restrict
     * holes; combined they mean "no holes at all" which is fine.
     * --num-holes can be any value <= MAX_HOLES; the combination "at most
     * N holes, only at these cells" simply means up to min(N, len) holes. */

    /* Effective walkable region: active region minus fixed walls. */
    g_walkable_mask = g_active_mask & ~g_fixed_walls_mask;

    /* Build the D4 transforms for this grid (depends only on g_grid_rows/cols). */
    build_d4();

    /* Enumerate canonical exit reps under the D4 group.  For 5x5 this
     * reproduces the historical {0,1,2,6,7,12}. */
    build_canonical_exits();

    /* Translate --num-walls into a popcount cap on committed_empty
     * within the active region. */
    {
        int active_size = __builtin_popcountll(g_active_mask);
        if (g_min_walls >= active_size) {
            fprintf(stderr, "error: --num-walls (%d) >= active region size (%d) leaves no room for the puzzle\n",
                    g_min_walls, active_size);
            return 1;
        }
        g_max_committed_in_active = active_size - g_min_walls;
    }

    if (g_two_tables) {
        g_shallow = calloc(SHALLOW_CAP, sizeof *g_shallow);
        g_recent  = calloc(RECENT_CAP,  sizeof *g_recent);
        if (!g_shallow || !g_recent) { perror("calloc two-tables"); return 1; }
    } else {
        g_visited = calloc(HASH_CAP, sizeof *g_visited);
        if (!g_visited) { perror("calloc visited"); return 1; }
    }

    /* Build the list of exits to search.  Sources, in priority order:
     *   1. --exit N             single cell (overrides everything).
     *   2. --exitloc c,c,...    explicit list.
     *   3. default              canonical {0,1,2,6,7,12} in active region.
     * In all cases, cells that conflict with --fixedholes or --fixedwalls
     * are invalid (single-cell forms error; lists silently filter). */
    int exits[MAX_NCELLS];
    int n_exits = 0;
    if (g_only_exit >= 0) {
        if (!(g_active_mask & (1ULL <<g_only_exit))) {
            fprintf(stderr, "error: --exit %d is outside the %dx%d active region\n",
                    g_only_exit, g_grid_rows, g_grid_cols);
            return 1;
        }
        if (g_fixed_holes_mask & (1ULL <<g_only_exit)) {
            fprintf(stderr, "error: --exit %d coincides with a fixed hole\n", g_only_exit);
            return 1;
        }
        if (g_fixed_walls_mask & (1ULL <<g_only_exit)) {
            fprintf(stderr, "error: --exit %d coincides with a fixed wall\n", g_only_exit);
            return 1;
        }
        exits[n_exits++] = g_only_exit;
    } else if (g_n_only_exits > 0) {
        for (int i = 0; i < g_n_only_exits; i++) {
            int e = g_only_exit_list[i];
            if (!(g_active_mask & (1ULL <<e))) {
                fprintf(stderr, "error: --exitloc cell %d is outside the %dx%d active region\n",
                        e, g_grid_rows, g_grid_cols);
                return 1;
            }
            if (g_fixed_holes_mask & (1ULL <<e)) {
                fprintf(stderr, "error: --exitloc cell %d coincides with a fixed hole\n", e);
                return 1;
            }
            if (g_fixed_walls_mask & (1ULL <<e)) {
                fprintf(stderr, "error: --exitloc cell %d coincides with a fixed wall\n", e);
                return 1;
            }
            exits[n_exits++] = e;
        }
    } else {
        for (int i = 0; i < g_n_canonical_exits; i++) {
            int e = g_canonical_exits[i];
            if (!(g_active_mask & (1ULL << e))) continue;
            if (g_fixed_holes_mask & (1ULL << e)) continue;  /* exit can't be a hole */
            if (g_fixed_walls_mask & (1ULL << e)) continue;  /* exit can't be a wall */
            exits[n_exits++] = e;
        }
        if (n_exits == 0) {
            fprintf(stderr, "error: no canonical exit fits in the active region (and is neither a fixed hole nor wall)\n");
            return 1;
        }
    }

    /* Initialize cross-exit overall best (streamed during search). */
    g_overall_best_depth = 0;
    g_overall_best_exit  = -1;
    g_first_new_best_time = -1.0;
    g_pending_print_depth = 0;

    long long total_states      = 0;
    long long total_pruned_short= 0;
    long long total_pruned_dedup= 0;
    long long total_solver_calls= 0;
    double   total_elapsed      = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &g_t_session_start);
    struct timespec t_global0 = g_t_session_start, t_global_now;

    /* Handle --list-tasks: enumerate tasks per exit, print total, exit.
     * When --seed-path is in effect, the seed-path overrides task seeds
     * inside run_exit_search anyway — so partitioning into N tasks would
     * just have N workers explore the same subtree redundantly.  Emit
     * a single synthetic task so the wrapper falls through to the
     * single-worker path. */
    if (g_print_seed_key) {
        if (!g_have_seed_path) {
            fprintf(stderr, "error: --print-seed-key requires --seed-path\n");
            return 1;
        }
        BState seed;
        if (!build_seed_path_seed(&seed)) return 1;
        printf("depth %d canonical_key %llu state_key %llu\n",
               seed.depth,
               (unsigned long long)canonical_state_key(&seed),
               (unsigned long long)state_key(&seed));
        free(g_visited); free(g_queue);
        free(g_shallow); free(g_recent);
        return 0;
    }
    if (g_list_tasks && g_have_seed_path) {
        printf("seed-path: %d step%s\ntotal: 1\n",
               g_seed_path_n, g_seed_path_n == 1 ? "" : "s");
        free(g_visited); free(g_queue);
        free(g_shallow); free(g_recent);
        return 0;
    }
    if (g_list_tasks) {
        int total = 0;
        /* Heap-allocate: a TaskGroup is large (~120 KB with MAX_BLOCKS/HOLES
         * bumped for 8x8), and the array of 32 of them would overflow the
         * main-thread stack. */
        TaskGroup *tasks = calloc(MAX_TASKS_PER_EXIT, sizeof(TaskGroup));
        if (!tasks) { perror("calloc tasks"); return 1; }
        for (int ei = 0; ei < n_exits; ei++) {
            int n = enumerate_tasks_for_exit(exits[ei], tasks, MAX_TASKS_PER_EXIT);
            printf("exit %d: %d task%s\n", exits[ei], n, n == 1 ? "" : "s");
            for (int i = 0; i < n; i++) {
                printf("  task %d (global %d): transit=%d hole_loc=%d seeds=%d\n",
                       i, total + i, tasks[i].transit, tasks[i].hole_loc, tasks[i].n_seeds);
            }
            total += n;
        }
        printf("\ntotal: %d\n", total);
        free(tasks);
        free(g_visited);
        free(g_queue);
        free(g_shallow);
        free(g_recent);
        return 0;
    }

    /* If --task-id is set, find the (exit, local_task) for that global ID. */
    int task_target_exit = -1;
    TaskGroup task_target = {0};
    if (g_only_task >= 0) {
        int idx = 0;
        TaskGroup *tasks = calloc(MAX_TASKS_PER_EXIT, sizeof(TaskGroup));
        if (!tasks) { perror("calloc tasks"); return 1; }
        for (int ei = 0; ei < n_exits; ei++) {
            int n = enumerate_tasks_for_exit(exits[ei], tasks, MAX_TASKS_PER_EXIT);
            if (g_only_task < idx + n) {
                task_target_exit = exits[ei];
                task_target = tasks[g_only_task - idx];
                break;
            }
            idx += n;
        }
        free(tasks);
        if (task_target_exit < 0) {
            fprintf(stderr, "error: --task-id %d out of range (total tasks = %d)\n",
                    g_only_task, idx);
            return 1;
        }
        printf("[task %d: exit=%d transit=%d hole_loc=%d seeds=%d]\n",
               g_only_task, task_target_exit, task_target.transit,
               task_target.hole_loc, task_target.n_seeds);
    }

    int unlimited = (g_time_cap_s == 0);
    for (int ei = 0; ei < n_exits; ei++) {
        /* In --task-id mode, skip exits that don't host the target task. */
        if (g_only_task >= 0 && exits[ei] != task_target_exit) continue;

        double remain = 0.0;  /* sentinel for unlimited */
        if (!unlimited) {
            clock_gettime(CLOCK_MONOTONIC, &t_global_now);
            double used = elapsed_s(t_global0, t_global_now);
            remain = g_time_cap_s - used;
            if (remain <= 0) {
                printf("[exit %d skipped — time budget exhausted]\n", exits[ei]);
                continue;
            }
        }

        g_exit_pos = exits[ei];

        if (g_only_task >= 0) {
            g_task_seeds = task_target.seeds;
            g_task_seed_count = task_target.n_seeds;
        } else {
            g_task_seeds = NULL;
            g_task_seed_count = 0;
        }

        int exhausted, dedup_full;
        double exit_elapsed = run_exit_search(remain, &exhausted, &dedup_full);

        /* Force-flush any new-best still buffered when the search ended
         * (the periodic timer in try_successor only fires every 8192
         * states, so a best discovered just before the time cap or queue
         * drain could otherwise be missed by the stream). */
        flush_pending_new_best(session_elapsed());

        /* Verify and report this exit. */
        int verify = -1;
        if (g_best_depth > 0) {
            Puzzle pz;
            build_partial_puzzle(&g_best_state, &pz);
            verify = sokoban_solve(&pz, NULL, NULL);
        }

        printf("--- Exit %d (%s) ---\n", g_exit_pos,
               dedup_full ? "dedup table full"
                          : (exhausted ? "exhausted" : "time cap"));
        printf("  elapsed:        %.3f s\n", exit_elapsed);
        printf("  states checked: %lld (shortcut %lld, dedup %lld",
               g_states_checked, g_pruned_short, g_pruned_dedup);
        if (g_dupe_threshold < INT_MAX)
            printf(", free-fly %lld", g_skipped_dedup);
        if (g_shortcut_state_cap > 0)
            printf(", cap %lld", g_pruned_cap);
        if (g_single_axis_blocks)
            printf(", axis %lld", g_pruned_axis);
        printf(")\n");
        printf("  solver calls:   %lld\n", g_solver_calls);
        if (g_two_tables) {
            printf("  shallow / recent: %lld / %lld  (evicts %lld / %lld)\n",
                   g_shallow_count, g_recent_count,
                   g_evictions_shallow, g_evictions_recent);
        } else {
            printf("  visited:        %lld\n", g_visited_count);
        }
        printf("  stack peak:     %lld\n", g_q_peak);
        printf("  best depth:     %d", g_best_depth);
        if (g_best_depth > 0)
            printf("  (verify %d %s)", verify, verify == g_best_depth ? "OK" : "MISMATCH");
        printf("\n");

        total_states       += g_states_checked;
        total_pruned_short += g_pruned_short;
        total_pruned_dedup += g_pruned_dedup;
        total_solver_calls += g_solver_calls;
        total_elapsed      += exit_elapsed;

        /* Cross-exit overall best is updated incrementally inside try_successor
         * (so streaming sees it).  Nothing to do here. */
    }

    printf("\n=== Backward search complete ===\n");
    if (g_time_cap_s == 0)
        printf("Total elapsed:     %.2f s (no time limit)\n", total_elapsed);
    else
        printf("Total elapsed:     %.2f s (cap %.2f s)\n", total_elapsed, g_time_cap_s);
    printf("Exits searched:    %d / %d\n", n_exits, n_exits);
    printf("States checked:    %lld\n", total_states);
    printf("  pruned shortcut: %lld\n", total_pruned_short);
    printf("  pruned dedup:    %lld\n", total_pruned_dedup);
    printf("Solver calls:      %lld\n", total_solver_calls);
    if (g_nn_surrogate_loaded) {
        printf("NN surrogate:      %lld calls, %lld skips (%.1f%% skip rate)\n",
               g_nn_surrogate_calls, g_nn_surrogate_skips,
               g_nn_surrogate_calls > 0 ?
                  100.0 * g_nn_surrogate_skips / g_nn_surrogate_calls : 0.0);
    }
    printf("Best depth:        %d  (exit %d)\n", g_overall_best_depth, g_overall_best_exit);

    if (g_overall_best_depth > 0) {
        printf("\n");
        print_puzzle_for_exit(&g_overall_best_state, g_overall_best_exit);
    } else {
        printf("(no non-trivial puzzle found)\n");
    }

    /* Solver-call profile by state depth — useful for evaluating cheap
     * lower-bound bypass strategies. */
    {
        long long total_calls = 0;
        double    total_time  = 0.0;
        int       max_seen    = -1;
        for (int b = 0; b < SHORTCUT_PROFILE_BUCKETS; b++) {
            total_calls += g_solver_calls_by_depth[b];
            total_time  += g_solver_time_by_depth[b];
            if (g_solver_calls_by_depth[b] > 0) max_seen = b;
        }
        if (total_calls > 0 && g_solver_profile) {
            printf("\nSolver-call profile (by state depth):\n");
            printf("  depth   calls       time(s)   %%calls  %%time   cum%%time\n");
            double cum_time = 0.0;
            for (int b = 0; b <= max_seen; b++) {
                long long c = g_solver_calls_by_depth[b];
                double    t = g_solver_time_by_depth[b];
                if (c == 0 && t == 0.0) continue;
                cum_time += t;
                printf("  %3d   %10lld   %8.3f   %5.1f%%  %5.1f%%   %5.1f%%\n",
                       b, c, t,
                       100.0 * (double)c / (double)total_calls,
                       100.0 * t / total_time,
                       100.0 * cum_time / total_time);
            }
            printf("  total %10lld   %8.3f s\n", total_calls, total_time);
        }
    }

    free(g_visited);
    free(g_shallow);
    free(g_recent);
    free(g_queue);
    if (g_trace_csv) fclose(g_trace_csv);
    if (g_harvest_csv) { fclose(g_harvest_csv); free(g_harvest_buf); g_harvest_buf = NULL; }
    if (g_harvest_gz)  { gzclose(g_harvest_gz); g_harvest_gz = NULL; }
    if (g_nn_loaded) nn_close();
    if (g_nn_surrogate_loaded) nn_surrogate_close();
    return 0;
}
