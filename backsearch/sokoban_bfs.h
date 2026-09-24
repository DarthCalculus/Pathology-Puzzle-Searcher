#pragma once
#include <stdint.h>

/* Maximum grid bounds.  Bitmask cell occupancy uses a single uint64_t,
 * so the addressable region is capped at MAX_NCELLS=64 cells total.
 * Any RxC with R*C <= 64 is supported — 8x8, 7x9, 3x21, 1x64, etc. */
#define MAX_ROWS    64
#define MAX_COLS    64
#define MAX_NCELLS  64
#define MAX_BLOCKS  32     /* enough for 8x8 puzzles in practice; --num-blocks rejects more */
#define MAX_HOLES   64
/* CONSUMED sentinel: encodes "block fell into a hole".  Set by
 * sokoban_set_grid() to g_ncells, which is the smallest value larger
 * than any legal cell index AND still fits in g_bits_per_cell bits. */
extern int g_consumed;

/* Runtime grid dimensions — set by sokoban_set_grid() and used by both
 * the solver and the back-search generator. */
extern int g_rows;
extern int g_cols;
extern int g_ncells;

/* Adjacency table built from grid dimensions in sokoban_set_grid().
 * g_adj[cell][dir] = neighbor cell or -1.  Directions 0=U 1=R 2=D 3=L. */
extern int8_t g_adj[MAX_NCELLS][4];

/* Pushable-direction bits stored in block_pushable[]:  U=1 R=2 D=4 L=8 */

typedef struct {
    uint64_t walls;               /* bitmask: bit i set → cell i is a wall    */
    int      exit_pos;            /* cell index the block must reach           */
    int      player_start;        /* cell index where the player begins        */
    int      num_blocks;
    int      block_pos[MAX_BLOCKS];      /* current cell index, or CONSUMED    */
    uint8_t  block_pushable[MAX_BLOCKS]; /* direction bitmask per block        */
    int      num_holes;
    int      hole_pos[MAX_HOLES];        /* cell indices of holes              */
} Puzzle;

/* Returns the cell index for row r, column c */
static inline int pos(int r, int c) { return r * g_cols + c; }
static inline int row_(int p)       { return p / g_cols; }
static inline int col_(int p)       { return p % g_cols; }

/*
 * sokoban_set_grid(rows, cols)
 *
 * Must be called once before sokoban_init() and before any solve.  Sets
 * the runtime grid dimensions, rebuilds the adjacency table, and
 * recomputes the bitmask edge constants.  Safe to call repeatedly to
 * change grid size (e.g., between tests), but the call is not
 * thread-safe — only the main thread should change the grid.
 */
void sokoban_set_grid(int rows, int cols);

/*
 * Solver return codes.  x >= 0 is a solution length; -1 means "no solution
 * (within the cutoff)", proven by an exhaustive search.  Every other negative
 * value means the search did NOT finish, so neither answer is proven:
 *   SOK_PENDING_CAP  -2  the pending queue reached HP64_SIZE / HP128_SIZE
 *   SOK_PROBE_LIMIT  -3  a hash probe chain exceeded HT_PROBE_LIMIT (or the
 *                        optional heap cap, sokoban_set_heap_cap, was hit)
 *   SOK_TABLE_FULL   -5  the big hash table reached its 85% insert limit
 *                        (the solve is not retried: the rerun is identical)
 *   SOK_NO_FIT       -6  the state does not fit the widest packing
 *                        (g_bits_per_cell*(1+nb)+nh > SOK_STATE_BITS_MAX, or
 *                        nh > 30); nothing was searched
 * -2/-3/-5 are UNKNOWN: a caller must never read them as "shortcut" (prune)
 * nor as "no solution" (accept); backsearch explores such a state without
 * counting it (PROTOCOL3 §1).  -6 is an internal error for the caller.
 * (-4 is backsearch's own --shortcut-state-cap code, never returned here.)
 */
#define SOK_PENDING_CAP   (-2)
#define SOK_PROBE_LIMIT   (-3)
#define SOK_TABLE_FULL    (-5)
#define SOK_NO_FIT        (-6)
#ifndef SOK_STATE_BITS_MAX
#define SOK_STATE_BITS_MAX 128   /* widest state packing the solvers support (a smaller -D value is a test knob) */
#endif
int sokoban_state_bits_max(void);   /* == SOK_STATE_BITS_MAX */

/*
 * sokoban_solve(pz, used_dirs, prof)
 *
 * Runs optimised BFS on the puzzle.  Returns the minimum number of moves
 * to reach exit_pos, -1 if unsolvable, or one of the undecided codes above
 * (-2, -3, -5) or SOK_NO_FIT (-6).  Callers must treat the undecided codes
 * as "state space not fully explored", never as unsolvable.
 *
 * used_dirs: no longer tracked (no caller used it, and the old 32-bit
 *   tracking overflowed for 8+ blocks).  Pass NULL.  A non-NULL buffer is
 *   filled with 0xF per block on success (the conservative "any direction").
 *
 * prof: if non-NULL, filled with profiling data for this call.
 *   Pass NULL for normal operation.
 */
/* Width of the near-goal frontier window recorded by the cutoff solvers.
 * tail_width[BFS_TAIL_W-1] is the number of states that settle at exactly
 * cost == max_cost; tail_width[BFS_TAIL_W-1-j] is the count at max_cost-j. */
#define BFS_TAIL_W 64

typedef struct {
    int peak_heap_sz;  /* max heap entries live at any point during the solve */
    int states_popped; /* unique states actually expanded (post-staleness check) */
    int max_cost_seen; /* max_cost passed to the cutoff solve (-1 / unset otherwise) */
    /* Frontier-width profile for the deepest BFS_TAIL_W cost levels of the
     * forward solve: tail_width[i] = #states settling at cost
     * (max_cost_seen - (BFS_TAIL_W-1-i)).  Only the *cutoff* solvers populate
     * this; the uncapped solvers leave it untouched. */
    int32_t tail_width[BFS_TAIL_W];
} BfsProfile;

int  sokoban_solve(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof);

/*
 * sokoban_solve_cutoff(pz, used_dirs, prof, max_cost)
 *
 * Like sokoban_solve, but bounded by max_cost.  Returns:
 *   x in [0, max_cost] : a forward solve of length x (the shortest one unless
 *                        decision mode is on, see sokoban_set_decision_only).
 *   -1                 : no path of length <= max_cost (or unsolvable).
 *   -2 / -3 / -5       : undecided (pending cap / probe limit / table full).
 *   -6                 : the state does not fit the packing (SOK_NO_FIT).
 *
 * Useful when the caller knows answers above max_cost are uninteresting.
 * E.g., a backward shortcut check at depth d+1 only cares whether some
 * path of length <= d-1 exists (parity rules out length d), so it can
 * pass max_cost = d-1 and skip the entire Dijkstra shell from d-1 to d+1.
 */
int  sokoban_solve_cutoff(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof, int max_cost);

void sokoban_init(void);   /* call once before spawning threads */

/* Allocation failures call this handler with a message; it must not return.
 * Default: print "fatal: <msg>" to stderr and exit(5). */
void sokoban_set_fatal_handler(void (*fn)(const char *msg));

/* The build's effective table-size knobs as a JSON object body (no braces). */
const char *sokoban_knobs_json(void);

/*
 * sokoban_set_heap_cap(n)
 *
 * Soft upper bound on the BFS priority-queue size for all subsequent
 * solves.  When heap_sz exceeds n, the solver aborts and returns -3.
 * Useful for capping per-eval cost: solvable puzzles with very wide
 * state spaces (and unsolvable puzzles that aren't trivially walled
 * off) can otherwise burn seconds before BFS exhausts the heap.
 *
 * Set n = 0 (default) to disable.  Setting n >= HP64_SIZE is effectively
 * disabled — the heap can't grow past HP64_SIZE anyway (returns -2).
 * Like -2, a -3 from the cap is undecided, never "no solution".
 *
 * Not thread-safe; call from a single thread before parallel solves.
 */
void sokoban_set_heap_cap(int n);

/*
 * sokoban_set_hole_prune(on)
 *
 * Enable (on != 0) or disable (the default) the mandatory-hole prune.  A hole
 * is "mandatory" if the player cannot reach the exit while it stays open, so
 * any solution must fill it with a block.  During the solve, a successor is
 * dropped when fewer blocks can still ever reach an unfilled mandatory hole
 * than there are such holes — a sound relaxation (it can only under-report
 * dead states), so the optimal solve length is unchanged; it is purely a
 * speedup.  Exposed as a toggle so the effect can be A/B benchmarked without
 * rebuilding.  Not thread-safe; set before parallel solves.
 */
void sokoban_set_hole_prune(int on);

/* Decision mode for sokoban_solve_cutoff(): return at the FIRST solution found
 * within the cutoff (a valid length <= max_cost, not necessarily the shortest)
 * and order the search by g + walk-distance-to-exit.  Off by default so that
 * callers needing exact lengths (harvest, verification) are unaffected. */
void sokoban_set_decision_only(int on);

/* Multi-start decision solve (see sokoban_bfs.c).  All starts[] lie in one
 * player component of pz's board; for each i sets out[i] = 1 iff a solution
 * of length <= cut[i] exists from starts[i] (exactly what sokoban_solve_cutoff
 * from that start would decide).  pred[i] (or NULL): bitmask of starts that
 * are shortest-path predecessors of start i; a shortcut there implies one at i.
 * n <= 24.  Returns 0, or -2 when the solve
 * did not fit (table overflow, heap cap, 128-bit puzzle): then out[] is
 * meaningless and the caller must fall back to per-start solves. */
/* Parent-table reuse (see REFERENCE TABLE in sokoban_bfs.c).  After a cutoff
 * solve returned -1 exhaustively on the small table, sokoban_export_settled()
 * copies its settled (key, cost) pairs (returns n, or -1 if unavailable / too
 * many).  sokoban_set_reference() installs such a table for the next cutoff
 * solves of children whose forward puzzle is identical and whose cutoff is the
 * parent's + k; sokoban_clear_reference() removes it.  A solve made with a
 * reference installed is not exportable (its table is incomplete). */
int  sokoban_export_settled(uint64_t *keys, int32_t *costs, int max);
void sokoban_set_reference(const uint64_t *keys, const int32_t *costs, int n, int k);
void sokoban_clear_reference(void);
void sokoban_set_reference_delta(uint64_t delta, uint64_t walls, int exit_pos);   /* cells floor now, wall for the table owner */
typedef struct { uint64_t *hk; int32_t *hc; uint32_t mask, cap; int n; } SokRefTable;   /* open-addressing (key -> cost), 0 = empty slot */
int  sokoban_ref_build(const uint64_t *keys, const int32_t *costs, int n, SokRefTable *t);   /* build once (duplicates keep the minimum) ... */
void sokoban_ref_attach(const SokRefTable *t);                              /* ... attach by pointer for an expansion (k = 0, no delta/xor) */
void sokoban_ref_load(const uint64_t *keys, const int32_t *costs, int n);   /* copying variant of build + attach */
void sokoban_ref_set_k(int k);                                             /* ... then per child: chain offset k (clears delta and xor) */
void sokoban_ref_set_xor(uint64_t x);                                      /* child key ^ x = owner key for the states they share */
uint64_t sokoban_zobrist_block(int mask, int cell);
void sokoban_ref_suspend(int suspend);                                     /* REF_CHECK: disable/enable the loaded table */
int  sokoban_ref_active(void);

int sokoban_solve_multi(const Puzzle *pz, const int8_t *starts, const int16_t *cut, const uint32_t *pred,
                        const uint8_t *refk, int n, uint8_t *out, BfsProfile *prof);   /* refk: per-start chain offset for an installed reference, or NULL.
                                                                                        * The true offsets must be passed: a capped k would prune more than
                                                                                        * proven, so a caller whose offset exceeds 250 passes NULL instead. */

/*
 * sokoban_set_forced_mandatory(cell_mask)
 *
 * Bitmask of cell indices whose holes are treated as mandatory unconditionally,
 * skipping the per-solve reachability check for them (union with the check —
 * other holes are still classified normally).  Only takes effect when the
 * mandatory-hole prune is enabled.  WARNING: forcing a hole that is not truly
 * mandatory is unsound — it can drop states that belong to a valid solution and
 * thus change which puzzles/solve-lengths are found.  Pass 0 to clear.  Not
 * thread-safe; set before parallel solves.
 */
void sokoban_set_forced_mandatory(uint64_t cell_mask);
