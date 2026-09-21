/* level_to_pathology — convert a puzzle in backsearch_worker's printed-text
 * format into a Pathology level "code" (newline-joined tile grid).
 *
 * Build (self-contained; no solver/POSIX dependency):
 *   cc -O3 -o level_to_pathology level_to_pathology.c
 *
 * Usage:
 *   ./level_to_pathology < puzzle.txt
 *   pbpaste | ./level_to_pathology
 *   ./level_to_pathology "  .$@.\n  ABC.   A=[R]\n  ..."
 *
 * Input format (same as seed_from_level / backsearch_worker output):
 *
 *     54 (2.3s)                  <-- optional header line, ignored
 *       $..AO   A=[R]
 *       OB..O   B=[U]
 *       @C.#O   C=[URD]
 *       .#DE.   D=[UR]
 *       .F..O   E=[URL]
 *                 F=[R]
 *
 *   Grid tiles:  '.' floor  '#' wall  'O' hole  '$' exit  '@' player
 *                'A'-'Z' block identifiers, each with a push-mask given by an
 *                'X=[URDL]' annotation (U=up R=right D=down L=left).
 *
 * Output: the Pathology tile grid, one row per line, no indentation. Tiles:
 *   '0' floor  '1' wall  '3' exit  '4' player  '5' hole
 *   '2' block (all 4 directions), and directional blocks '6'-'9','A'-'J'
 *   per the server's CHAR_TO_MASK table (see server/lib/level.js).
 *
 * Paste the output straight into the Pathology submission box; the server
 * canonicalizes (reflect/transpose) on its own.
 *
 * Limitation: a block resting on the exit (from --allow-block-on-exit) cannot
 * be represented — the printed format draws the exit under the block, so the
 * exit position is already lost, and the Pathology format rejects on-exit
 * tiles (K-Z) anyway. Such inputs error out at the "no exit" check.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_LINE   512
#define MAX_LINES  200
#define MAX_COLS   64
#define MAX_ROWS   64
#define MAX_BLOCKS 32
#define MAX_HOLES  64

/* Pathology tile char for a push-mask (U=1 R=2 D=4 L=8), index 1..15.
 * Mirrors the reverse of CHAR_TO_MASK in server/lib/level.js. Index 0 is
 * invalid (a block with no push directions has no representation). */
static const char MASK_TO_CHAR[16] = {
    0,   /* 0  invalid */
    '7', /* 1  U      */
    '8', /* 2  R      */
    'B', /* 3  UR     */
    '9', /* 4  D      */
    'J', /* 5  UD     */
    'C', /* 6  RD     */
    'E', /* 7  URD    */
    '6', /* 8  L      */
    'A', /* 9  UL     */
    'I', /* 10 RL     */
    'H', /* 11 URL    */
    'D', /* 12 DL     */
    'G', /* 13 UDL    */
    'F', /* 14 RDL    */
    '2', /* 15 URDL   */
};

static char input_lines[MAX_LINES][MAX_LINE];
static int  n_input_lines = 0;

static void read_input(FILE *f) {
    while (n_input_lines < MAX_LINES &&
           fgets(input_lines[n_input_lines], MAX_LINE, f)) {
        n_input_lines++;
    }
}

/* Split a single C-string on newlines into input_lines[]. Supports a literal
 * two-character "\n" escape so the puzzle can be passed as one argv token. */
static void load_from_string(const char *s) {
    while (*s && n_input_lines < MAX_LINES) {
        char *buf = input_lines[n_input_lines];
        int n = 0;
        while (*s && n < MAX_LINE - 2) {
            if (*s == '\\' && s[1] == 'n') { s += 2; break; }
            if (*s == '\n') { s++; break; }
            buf[n++] = *s++;
        }
        buf[n++] = '\n';
        buf[n]   = '\0';
        n_input_lines++;
    }
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

    pz->cols = W;
    pz->rows = 0;
    pz->walls = 0;
    pz->nblocks = 0;
    pz->nholes  = 0;
    pz->exit_pos = -1;
    pz->player_start = -1;

    int     letter_pos[26];       for (int i = 0; i < 26; i++) letter_pos[i] = -1;
    uint8_t letter_mask[26]       = {0};
    int     letter_mask_seen[26]  = {0};

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
            letter_mask[idx] = mask;
            letter_mask_seen[idx] = 1;
        }
    }

    for (; i < n_input_lines; i++) {
        char letter; uint8_t mask;
        if (find_mask(input_lines[i], &letter, &mask)) {
            int idx = letter - 'A';
            letter_mask[idx] = mask;
            letter_mask_seen[idx] = 1;
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

    if (pz->exit_pos < 0) {
        fprintf(stderr, "error: no exit '$' in grid\n"
                        "  (a block resting on the exit cannot be converted: the printed\n"
                        "   format hides the exit under the block, and Pathology rejects\n"
                        "   on-exit tiles.  Also: if you passed the puzzle in DOUBLE quotes,\n"
                        "   the shell may have eaten the '$'.  Use single quotes.)\n");
        return 0;
    }
    if (pz->player_start < 0) {
        fprintf(stderr, "error: no player '@' in grid\n");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc >= 2 &&
        (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        fprintf(stderr,
            "usage: %s [\"<puzzle-text>\"]\n"
            "  Convert a puzzle in backsearch_worker's printed-text format into a\n"
            "  Pathology level code (newline-joined tile grid) on stdout.\n"
            "  Reads the puzzle from argv[1] if given, otherwise from stdin.\n",
            argv[0]);
        return 1;
    }

    if (argc >= 2) load_from_string(argv[1]);
    else           read_input(stdin);

    Parsed pp;
    if (!parse_puzzle(&pp)) return 1;

    if (pp.rows * pp.cols > 64)
        fprintf(stderr, "warning: %dx%d = %d cells exceeds Pathology's 64-cell max\n",
                pp.rows, pp.cols, pp.rows * pp.cols);

    /* Build the Pathology tile grid, all floor to start. */
    char grid[MAX_ROWS][MAX_COLS + 1];
    for (int r = 0; r < pp.rows; r++) {
        for (int c = 0; c < pp.cols; c++) grid[r][c] = '0';
        grid[r][pp.cols] = '\0';
    }
    for (int cell = 0; cell < pp.rows * pp.cols; cell++)
        if (pp.walls & (1ULL << cell)) grid[cell / pp.cols][cell % pp.cols] = '1';
    for (int i = 0; i < pp.nholes; i++) {
        int p = pp.hole_pos[i];
        grid[p / pp.cols][p % pp.cols] = '5';
    }
    for (int i = 0; i < pp.nblocks; i++) {
        uint8_t m = pp.block_mask[i];
        if (m == 0 || m > 15 || MASK_TO_CHAR[m] == 0) {
            fprintf(stderr, "error: block at cell %d has un-representable mask %u\n",
                    pp.block_pos[i], m);
            return 1;
        }
        int p = pp.block_pos[i];
        grid[p / pp.cols][p % pp.cols] = MASK_TO_CHAR[m];
    }
    grid[pp.exit_pos / pp.cols][pp.exit_pos % pp.cols]           = '3';
    grid[pp.player_start / pp.cols][pp.player_start % pp.cols]   = '4';

    for (int r = 0; r < pp.rows; r++)
        printf("%s\n", grid[r]);
    return 0;
}
