/* pathbf — given a puzzle in the printed-text format, forward-solve it and
 * walk the optimal solution path, printing each path state's
 *   depth,states_popped
 * (depth = moves-to-go; states_popped = forward-solver branch_factor, the
 * exact signal the rollout percentile band ranks on).  One line per state,
 * from the start state (depth D) down to depth 1.
 *
 * Build:  cc -O3 -pthread -I . -o pathbf pathbf.c sokoban_bfs.c
 * Usage:  ./pathbf < puzzle.txt        (or  pbpaste | ./pathbf)
 *
 * Parsing + greedy optimal-path reconstruction lifted verbatim from
 * seed_from_level.c; only solve_state and main's tail differ.
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

static int is_grid_char(char c) {
    return c == '.' || c == '#' || c == 'O' || c == '$' || c == '@'
           || (c >= 'A' && c <= 'Z');
}

static int find_mask(const char *line, char *out_letter, uint8_t *out_mask) {
    for (const char *s = line; *s; s++) {
        if (*s != '=' || s[1] != '[') continue;
        if (s == line) continue;
        char letter = s[-1];
        if (letter < 'A' || letter > 'Z') continue;
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
    int first = -1;
    for (int i = 0; i < n_input_lines; i++) {
        const char *s = input_lines[i];
        if (s[0] == ' ' && s[1] == ' ' && is_grid_char(s[2])) { first = i; break; }
    }
    if (first < 0) { fprintf(stderr, "error: no grid lines found\n"); return 0; }

    int W = 0;
    for (int i = 2; input_lines[first][i] && is_grid_char(input_lines[first][i]); i++)
        W++;
    if (W < 1 || W > MAX_COLS) {
        fprintf(stderr, "error: bad detected width %d\n", W); return 0;
    }

    pz->cols = W; pz->rows = 0; pz->walls = 0;
    pz->nblocks = 0; pz->nholes = 0;
    pz->exit_pos = -1; pz->player_start = -1;

    int letter_pos[26];   for (int i = 0; i < 26; i++) letter_pos[i] = -1;
    uint8_t letter_mask[26] = {0};
    int letter_mask_seen[26] = {0};

    int i;
    for (i = first; i < n_input_lines; i++) {
        const char *s = input_lines[i];
        if (s[0] != ' ' || s[1] != ' ') break;
        int all_grid = 1;
        for (int j = 0; j < W; j++)
            if (!is_grid_char(s[2+j])) { all_grid = 0; break; }
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
            letter_mask[idx] = mask; letter_mask_seen[idx] = 1;
        }
    }
    for (; i < n_input_lines; i++) {
        char letter; uint8_t mask;
        if (find_mask(input_lines[i], &letter, &mask)) {
            int idx = letter - 'A';
            letter_mask[idx] = mask; letter_mask_seen[idx] = 1;
        }
    }

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

    if (pz->exit_pos < 0) { fprintf(stderr, "error: no exit '$' in grid\n"); return 0; }
    if (pz->player_start < 0) { fprintf(stderr, "error: no player '@' in grid\n"); return 0; }
    return 1;
}

typedef struct {
    int player, nblocks;
    int block_pos[MAX_BLOCKS];
    uint8_t block_mask[MAX_BLOCKS];
    int nholes;
    int hole_pos[MAX_HOLES];
    int hole_active[MAX_HOLES];
    uint64_t walls;
    int exit_pos, cols, rows;
} State;

static void build_pz(const State *s, Puzzle *pz) {
    memset(pz, 0, sizeof(*pz));
    pz->walls = s->walls;
    pz->exit_pos = s->exit_pos;
    pz->player_start = s->player;
    pz->num_blocks = s->nblocks;
    for (int i = 0; i < s->nblocks; i++) {
        pz->block_pos[i]      = s->block_pos[i];
        pz->block_pushable[i] = s->block_mask[i];
    }
    int nh = 0;
    for (int i = 0; i < s->nblocks && i < MAX_BLOCKS; i++) {} /* no-op, keep struct shape */
    nh = 0;
    for (int i = 0; i < s->nholes; i++)
        if (s->hole_active[i]) pz->hole_pos[nh++] = s->hole_pos[i];
    pz->num_holes = nh;
}

/* Full optimal-depth solve (used to drive path reconstruction). */
static int solve_state(const State *s, int *out_pop) {
    Puzzle pz; build_pz(s, &pz);
    BfsProfile prof; memset(&prof, 0, sizeof(prof));
    int d = sokoban_solve(&pz, NULL, &prof);
    if (out_pop) *out_pop = prof.states_popped;
    return d;
}

/* branch_factor EXACTLY as the rollout band computes it: states_popped of a
 * CUTOFF solve at max_cost = depth-2 (mirrors shortcut_check in backsearch.c). */
static int bf_cutoff(const State *s, int max_cost) {
    Puzzle pz; build_pz(s, &pz);
    BfsProfile prof; memset(&prof, 0, sizeof(prof));
    sokoban_solve_cutoff(&pz, NULL, &prof, max_cost);
    return prof.states_popped;
}

static const int  DR[] = {-1, 0, 1, 0};
static const int  DC[] = { 0, 1, 0,-1};

static int forward_move(State *s, int dir, int *out_action) {
    int pr = s->player / s->cols, pc = s->player % s->cols;
    int nr = pr + DR[dir], nc = pc + DC[dir];
    if (nr < 0 || nr >= s->rows || nc < 0 || nc >= s->cols) return 0;
    int np = nr * s->cols + nc;
    if (s->walls & (1ULL << np)) return 0;
    for (int i = 0; i < s->nholes; i++)
        if (s->hole_active[i] && s->hole_pos[i] == np) return 0;
    int blk_at = -1;
    for (int i = 0; i < s->nblocks; i++)
        if (s->block_pos[i] == np) { blk_at = i; break; }
    if (blk_at < 0) { s->player = np; *out_action = 1; return 1; }
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
    (void)argc; (void)argv;
    read_input(stdin);

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
        st.hole_pos[i] = pp.hole_pos[i];
        st.hole_active[i] = 1;
    }

    int D = solve_state(&st, NULL);
    if (D <= 0) { fprintf(stderr, "error: puzzle unsolvable (rc=%d)\n", D); return 1; }
    fprintf(stderr, "optimal depth D=%d, nblocks=%d, nholes=%d\n", D, pp.nblocks, pp.nholes);

    printf("depth,states_popped\n");
    State cur = st;
    int remaining = D;
    while (remaining > 0) {
        int bf = bf_cutoff(&cur, remaining - 2);
        printf("%d,%d\n", remaining, bf);
        fflush(stdout);

        int found = 0;
        for (int dir = 0; dir < 4; dir++) {
            State next = cur;
            int act;
            if (!forward_move(&next, dir, &act)) continue;
            int new_d = (next.player == cur.exit_pos) ? 0 : solve_state(&next, NULL);
            if (new_d == remaining - 1) { cur = next; remaining = new_d; found = 1; break; }
        }
        if (!found) {
            fprintf(stderr, "error: stuck at remaining=%d\n", remaining);
            return 1;
        }
    }
    return 0;
}
