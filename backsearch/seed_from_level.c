/* seed_from_level — given a puzzle in the printed-text format and an
 * integer N, output a --seed-path string that builds the depth-N
 * back-search state on the unique optimal-solution trace.
 *
 * Build:
 *   cc -O3 -pthread -I . -o seed_from_level seed_from_level.c sokoban_bfs.c
 *
 * Usage:
 *   ./seed_from_level <N> < puzzle.txt
 *   ./seed_from_level <N> < <(pbpaste)
 *
 * Input format:  the puzzle in the form printed by backsearch_worker, e.g.
 *
 *     99 (3.7s)                  <-- optional header line
 *       .O$.#@..   A=[U]
 *       ABCD.EF.   B=[RD]
 *       ........   C=[UR]
 *                  D=[URL]
 *                  E=[DL]
 *                  F=[RDL]
 *
 * Algorithm:
 *   1. Parse the puzzle from stdin.
 *   2. Forward-solve with sokoban_solve to get the optimal depth D.
 *   3. Reconstruct the optimal move sequence greedily: at each forward
 *      state, try every legal move; accept the one whose post-state
 *      solves to (remaining - 1).  This is O(D) solver calls.
 *   4. Classify each forward move as walk (1) / push (2) / consume (3).
 *   5. The backward seed-path for depth N is the last N forward moves
 *      reversed.  Print as comma-separated tokens.
 *
 * Caveats:
 *   - Greedy reconstruction relies on sokoban_solve returning the optimal
 *     depth for every intermediate state.  For very large puzzles this
 *     can be slow.
 *   - The optimal path may not be unique; we pick the first move that
 *     decreases solve depth by 1 — that may differ from another reverse
 *     trace that also reaches the puzzle.
 */
#include "sokoban_bfs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 512
#define MAX_LINES 200

static char input_lines[MAX_LINES][MAX_LINE];
static int  n_input_lines = 0;

static void read_input(FILE *f) {
    while (n_input_lines < MAX_LINES &&
           fgets(input_lines[n_input_lines], MAX_LINE, f)) {
        n_input_lines++;
    }
}

/* Split a single C-string on newlines into input_lines[]. */
static void load_from_string(const char *s) {
    while (*s && n_input_lines < MAX_LINES) {
        char *buf = input_lines[n_input_lines];
        int n = 0;
        while (*s && *s != '\n' && n < MAX_LINE - 2) {
            buf[n++] = *s++;
        }
        buf[n++] = '\n';
        buf[n]   = '\0';
        n_input_lines++;
        if (*s == '\n') s++;
    }
}

static void rstrip(char *s) {
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r' || s[n-1] == ' '))
        s[--n] = 0;
}

static int is_grid_char(char c) {
    return c == '.' || c == '#' || c == 'O' || c == '$' || c == '@'
           || (c >= 'A' && c <= 'Z');
}

/* Find "X=[...]" anywhere in line; on success fill out_letter and out_mask. */
static int find_mask(const char *line, char *out_letter, uint8_t *out_mask) {
    for (const char *s = line; *s; s++) {
        if (*s != '=' || s[1] != '[') continue;
        if (s == line) continue;
        char letter = s[-1];
        if (letter < 'A' || letter > 'Z') continue;
        /* Cell before letter must be a space or start-of-line. */
        if (s - line >= 2 && s[-2] != ' ') continue;
        uint8_t m = 0;
        const char *p = s + 2;
        while (*p && *p != ']') {
            switch (*p) {
                case 'U': m |= 1; break;
                case 'R': m |= 2; break;
                case 'D': m |= 4; break;
                case 'L': m |= 8; break;
                default: return 0;
            }
            p++;
        }
        if (*p != ']') return 0;
        *out_letter = letter;
        *out_mask   = m;
        return 1;
    }
    return 0;
}

typedef struct {
    int      rows, cols;
    int      exit_pos, player_start;
    uint64_t walls;
    int      nblocks;
    int      block_pos[MAX_BLOCKS];
    uint8_t  block_mask[MAX_BLOCKS];
    int      nholes;
    int      hole_pos[MAX_HOLES];
} Parsed;

static int parse_puzzle(Parsed *pz) {
    /* Find first line whose chars at col 2+ are grid chars. */
    int first = -1;
    for (int i = 0; i < n_input_lines; i++) {
        const char *s = input_lines[i];
        if (s[0] == ' ' && s[1] == ' ' && is_grid_char(s[2])) { first = i; break; }
    }
    if (first < 0) { fprintf(stderr, "error: no grid lines found\n"); return 0; }

    /* Detect width: consecutive grid chars starting at col 2. */
    int W = 0;
    for (int i = 2; input_lines[first][i] && is_grid_char(input_lines[first][i]); i++)
        W++;
    if (W < 1 || W > MAX_COLS) {
        fprintf(stderr, "error: bad detected width %d\n", W); return 0;
    }

    pz->cols   = W;
    pz->rows   = 0;
    pz->walls  = 0;
    pz->nblocks = 0;
    pz->nholes  = 0;
    pz->exit_pos = -1;
    pz->player_start = -1;

    int letter_pos[26];   for (int i = 0; i < 26; i++) letter_pos[i] = -1;
    uint8_t letter_mask[26] = {0};
    int letter_mask_seen[26] = {0};

    int i;
    for (i = first; i < n_input_lines; i++) {
        const char *s = input_lines[i];
        /* A grid row has at least 2 leading spaces and grid chars at col 2..2+W-1. */
        if (s[0] != ' ' || s[1] != ' ') break;
        int all_grid = 1;
        for (int j = 0; j < W; j++) {
            if (!is_grid_char(s[2+j])) { all_grid = 0; break; }
        }
        if (!all_grid) break;
        if (pz->rows >= MAX_ROWS) break;

        int r = pz->rows;
        for (int c = 0; c < W; c++) {
            int cell = r * W + c;
            char ch = s[2+c];
            switch (ch) {
                case '.': break;
                case '#': pz->walls |= 1ULL << cell; break;
                case 'O':
                    if (pz->nholes >= MAX_HOLES) { fprintf(stderr, "too many holes\n"); return 0; }
                    pz->hole_pos[pz->nholes++] = cell;
                    break;
                case '$': pz->exit_pos = cell; break;
                case '@': pz->player_start = cell; break;
                default:
                    if (ch >= 'A' && ch <= 'Z') letter_pos[ch - 'A'] = cell;
                    break;
            }
        }
        pz->rows++;

        char letter; uint8_t mask;
        if (find_mask(s, &letter, &mask)) {
            int idx = letter - 'A';
            letter_mask[idx] = mask;
            letter_mask_seen[idx] = 1;
        }
    }

    /* Continuation lines for blocks whose mask isn't on a grid row. */
    for (; i < n_input_lines; i++) {
        char letter; uint8_t mask;
        if (find_mask(input_lines[i], &letter, &mask)) {
            int idx = letter - 'A';
            letter_mask[idx] = mask;
            letter_mask_seen[idx] = 1;
        }
    }

    /* Build block array in alphabetical order. */
    for (int idx = 0; idx < 26; idx++) {
        if (letter_pos[idx] >= 0) {
            if (!letter_mask_seen[idx]) {
                fprintf(stderr, "warning: no mask for block %c, assuming URDL\n", 'A'+idx);
                letter_mask[idx] = 0xF;
            }
            if (pz->nblocks >= MAX_BLOCKS) { fprintf(stderr, "too many blocks\n"); return 0; }
            pz->block_pos [pz->nblocks] = letter_pos[idx];
            pz->block_mask[pz->nblocks] = letter_mask[idx];
            pz->nblocks++;
        }
    }

    if (pz->exit_pos < 0) {
        fprintf(stderr, "error: no exit '$' in grid\n"
                        "  (if you passed the puzzle in DOUBLE quotes, zsh/bash ate the '$'\n"
                        "   as a variable.  Use single quotes or a heredoc with 'EOF' marker.)\n");
        return 0;
    }
    if (pz->player_start < 0) {
        fprintf(stderr, "error: no player '@' in grid\n");
        return 0;
    }
    return 1;
}

/* ----- Forward simulation ----- */

typedef struct {
    int      player;
    int      nblocks;
    int      block_pos[MAX_BLOCKS];
    uint8_t  block_mask[MAX_BLOCKS];
    int      nholes;
    int      hole_pos[MAX_HOLES];
    int      hole_active[MAX_HOLES];
    uint64_t walls;
    int      exit_pos;
    int      cols;
    int      rows;
} State;

static int solve_state(const State *s) {
    Puzzle pz = {0};
    pz.walls = s->walls;
    pz.exit_pos = s->exit_pos;
    pz.player_start = s->player;
    pz.num_blocks = s->nblocks;
    for (int i = 0; i < s->nblocks; i++) {
        pz.block_pos[i]      = s->block_pos[i];
        pz.block_pushable[i] = s->block_mask[i];
    }
    int nh = 0;
    for (int i = 0; i < s->nholes; i++)
        if (s->hole_active[i]) pz.hole_pos[nh++] = s->hole_pos[i];
    pz.num_holes = nh;
    return sokoban_solve(&pz, NULL, NULL);
}

static const int  DR[] = {-1, 0, 1, 0};
static const int  DC[] = { 0, 1, 0,-1};
static const char DN[] = "URDL";

/* Apply forward move in direction dir.  Returns 1 on legal move with
 * out_action ∈ {1=walk, 2=push, 3=consume}.  Returns 0 if illegal. */
static int forward_move(State *s, int dir, int *out_action) {
    int pr = s->player / s->cols, pc = s->player % s->cols;
    int nr = pr + DR[dir], nc = pc + DC[dir];
    if (nr < 0 || nr >= s->rows || nc < 0 || nc >= s->cols) return 0;
    int np = nr * s->cols + nc;
    if (s->walls & (1ULL << np)) return 0;

    /* Active hole blocks walking. */
    for (int i = 0; i < s->nholes; i++)
        if (s->hole_active[i] && s->hole_pos[i] == np) return 0;

    int blk_at = -1;
    for (int i = 0; i < s->nblocks; i++)
        if (s->block_pos[i] == np) { blk_at = i; break; }

    if (blk_at < 0) {
        s->player = np;
        *out_action = 1;
        return 1;
    }

    /* Push */
    if (!(s->block_mask[blk_at] & (1 << dir))) return 0;
    int br = nr + DR[dir], bc = nc + DC[dir];
    if (br < 0 || br >= s->rows || bc < 0 || bc >= s->cols) return 0;
    int bnp = br * s->cols + bc;
    if (s->walls & (1ULL << bnp)) return 0;
    for (int i = 0; i < s->nblocks; i++)
        if (i != blk_at && s->block_pos[i] == bnp) return 0;

    int hole_at = -1;
    for (int i = 0; i < s->nholes; i++)
        if (s->hole_active[i] && s->hole_pos[i] == bnp) { hole_at = i; break; }

    s->player = np;
    if (hole_at >= 0) {
        s->block_pos[blk_at] = g_consumed;
        s->hole_active[hole_at] = 0;
        *out_action = 3;
    } else {
        s->block_pos[blk_at] = bnp;
        *out_action = 2;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <N> [\"<puzzle-text>\"] \n"
            "  Reads a puzzle in the printed-text format and outputs a\n"
            "  --seed-path token-string producing the first N backward\n"
            "  steps on the optimal solution's reverse trace.\n"
            "  N=0: depth-only mode — print 'optimal depth: D' and exit.\n"
            "\n"
            "  Puzzle input: provide either\n"
            "    - As argv[2]:  %s 10 \"  .$@.\\n  ABC.   A=[R]\\n  ...\"\n"
            "    - Or via stdin: cat puzzle.txt | %s 10\n"
            "    - Or via heredoc:\n"
            "        %s 10 <<'EOF'\n"
            "          ...puzzle text here...\n"
            "        EOF\n"
            "    - Or via macOS clipboard: pbpaste | %s 10\n",
            argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    if (N < 0) { fprintf(stderr, "error: N must be >= 0 (0 = depth-only)\n"); return 1; }

    if (argc >= 3) {
        /* Puzzle text passed as a single argv string. */
        load_from_string(argv[2]);
    } else {
        read_input(stdin);
    }

    Parsed pp;
    if (!parse_puzzle(&pp)) return 1;

    sokoban_set_grid(pp.rows, pp.cols);
    sokoban_init();

    State st = {0};
    st.cols = pp.cols; st.rows = pp.rows;
    st.walls = pp.walls; st.exit_pos = pp.exit_pos;
    st.player = pp.player_start;
    st.nblocks = pp.nblocks;
    for (int i = 0; i < pp.nblocks; i++) {
        st.block_pos[i]  = pp.block_pos[i];
        st.block_mask[i] = pp.block_mask[i];
    }
    st.nholes = pp.nholes;
    for (int i = 0; i < pp.nholes; i++) {
        st.hole_pos[i]    = pp.hole_pos[i];
        st.hole_active[i] = 1;
    }

    int D = solve_state(&st);
    if (D <= 0) { fprintf(stderr, "error: puzzle unsolvable (rc=%d)\n", D); return 1; }
    if (N == 0) {
        printf("optimal depth: %d\n", D);
        return 0;
    }
    if (N > D) {
        fprintf(stderr, "warning: requested N=%d > optimal depth %d; using N=%d\n", N, D, D);
        N = D;
    }

    /* Greedy reconstruct optimal move sequence. */
    int directions[1024];
    int actions[1024];
    int n_moves = 0;
    State cur = st;
    int remaining = D;
    while (remaining > 0) {
        int found = 0;
        for (int dir = 0; dir < 4; dir++) {
            State next = cur;
            int act;
            if (!forward_move(&next, dir, &act)) continue;
            int new_d;
            if (next.player == cur.exit_pos) new_d = 0;
            else                              new_d = solve_state(&next);
            if (new_d == remaining - 1) {
                directions[n_moves] = dir;
                actions[n_moves]    = act;
                n_moves++;
                cur = next;
                remaining = new_d;
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "error: stuck at step %d, remaining=%d\n", n_moves, remaining);
            return 1;
        }
    }

    /* The first N backward steps = reverse of the last N forward moves.
     * Backward step direction = forward direction (unchanged).
     * Backward action digit (new 1/2/3 scheme) = forward action digit. */
    for (int i = 0; i < N; i++) {
        int fwd_idx = n_moves - 1 - i;
        if (i > 0) printf(",");
        printf("%c%d", DN[directions[fwd_idx]], actions[fwd_idx]);
    }
    printf("\n");
    return 0;
}
