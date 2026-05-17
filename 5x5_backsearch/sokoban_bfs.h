#pragma once
#include <stdint.h>

/* Maximum grid bounds.  Bitmask cell occupancy uses a single uint64_t,
 * so the addressable region is capped at MAX_NCELLS=64 cells total.
 * Any RxC with R*C <= 64 is supported — 8x8, 7x9, 3x21, 1x64, etc. */
#define MAX_ROWS    64
#define MAX_COLS    64
#define MAX_NCELLS  64
#define MAX_BLOCKS  32     /* enough for 8x8 puzzles in practice; --num-blocks rejects more */
#define MAX_HOLES   32
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
 * sokoban_solve(pz, used_dirs, prof)
 *
 * Runs optimised BFS on the puzzle.  Returns the minimum number of moves
 * to reach exit_pos, -1 if unsolvable, -2 if the BFS queue overflowed
 * (out of memory), or -3 if the hash table probe limit was exceeded.
 * Callers must treat -2 and -3 as fatal errors (state space not fully
 * explored), not as unsolvable.
 *
 * used_dirs: if non-NULL and the puzzle is solvable, filled with the
 *   per-block bitmask of push directions that were actually used on the
 *   optimal path (U=1 R=2 D=4 L=8).  A block that was never pushed gets 0.
 *   Pass NULL to skip path tracking.
 *
 * prof: if non-NULL, filled with profiling data for this call.
 *   Pass NULL for normal operation.
 */
typedef struct {
    int peak_heap_sz;  /* max heap entries live at any point during the solve */
} BfsProfile;

int  sokoban_solve(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof);

/*
 * sokoban_solve_cutoff(pz, used_dirs, prof, max_cost)
 *
 * Like sokoban_solve, but bounded by max_cost.  Returns:
 *   x in [0, max_cost] : shortest forward solve length x.
 *   -1                 : no path of length <= max_cost (or unsolvable).
 *   -2                 : heap overflow.
 *
 * Useful when the caller knows answers above max_cost are uninteresting.
 * E.g., a backward shortcut check at depth d+1 only cares whether some
 * path of length <= d-1 exists (parity rules out length d), so it can
 * pass max_cost = d-1 and skip the entire Dijkstra shell from d-1 to d+1.
 */
int  sokoban_solve_cutoff(const Puzzle *pz, uint8_t *used_dirs, BfsProfile *prof, int max_cost);

void sokoban_init(void);   /* call once before spawning threads */
