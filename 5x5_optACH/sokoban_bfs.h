#pragma once
#include <stdint.h>

/* Grid dimensions */
#define ROWS       5
#define COLS       5
#define NCELLS     25   /* ROWS * COLS */
#define MAX_BLOCKS 8
#define MAX_HOLES  13
#define CONSUMED   25   /* sentinel: block fell into a hole */

/* Pushable-direction bits stored in block_pushable[]:  U=1 R=2 D=4 L=8 */

typedef struct {
    uint32_t walls;               /* bitmask: bit i set → cell i is a wall    */
    int      exit_pos;            /* cell index the block must reach           */
    int      player_start;        /* cell index where the player begins        */
    int      num_blocks;
    int      block_pos[MAX_BLOCKS];      /* current cell index, or CONSUMED    */
    uint8_t  block_pushable[MAX_BLOCKS]; /* direction bitmask per block        */
    int      num_holes;
    int      hole_pos[MAX_HOLES];        /* cell indices of holes              */
} Puzzle;

/* Returns the cell index for row r, column c */
static inline int pos(int r, int c) { return r * COLS + c; }
static inline int row_(int p)       { return p / COLS; }
static inline int col_(int p)       { return p % COLS; }

/*
 * sokoban_solve(pz, used_dirs)
 *
 * Runs optimised BFS on the puzzle.  Returns the minimum number of moves
 * to reach exit_pos, -1 if unsolvable, or -2 if the BFS queue overflowed
 * (out of memory — caller must handle this as a fatal error, not as
 * unsolvable, since the state space was not fully explored).
 *
 * used_dirs: if non-NULL and the puzzle is solvable, filled with the
 *   per-block bitmask of push directions that were actually used on the
 *   optimal path (U=1 R=2 D=4 L=8).  A block that was never pushed gets 0.
 *   Pass NULL to skip path tracking.
 */
int sokoban_solve(const Puzzle *pz, uint8_t *used_dirs);
