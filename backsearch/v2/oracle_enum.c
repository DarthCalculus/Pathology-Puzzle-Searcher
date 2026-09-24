/* oracle_enum.c -- independent oracle for the backward generator (review M89; v2/test_oracle.py).
 *
 * The worker (backsearch.c) finds levels by walking BACKWARD from the solved state and pruning with
 * a forward solver, a dedup table whose key ignores walk-history flags, state canonicalisation, bulk
 * walk-back, adjflags / exit-block prunes, parent tables and a root symmetry reduction.  This
 * program shares none of that code.  It computes, with its own move rules and its own breadth-first
 * judge, the set the worker's BS_TRACE_VALID trace must equal:
 *
 *   T = { level L of the class : some optimal solution S of L touches every non-wall cell, pushes
 *         every block in every direction of its mask and fills every hole }   (with its length |S|)
 *
 * Why T: for any level L and optimal solution S, the restriction tight(L,S) (untouched cells ->
 * walls, masks -> directions S uses, unpushed blocks -> walls, unfilled holes -> walls) still admits
 * S and is a restriction of L, so its optimum is |S|.  Every class maximum is therefore attained in
 * T, and T is exactly what the backward search can reach: the configuration after t moves of an
 * optimal S, restricted to what the rest of S uses (call it I_t), has optimum |S| - t.
 *
 * Rules (the site's, PathologyRecords/solver/solver6.c): one move = one step of the player (a walk,
 * or the push of one block).  The player cannot enter walls or unfilled holes.  A block moves one
 * cell in the push direction if the direction is in its mask and the landing cell is on the board,
 * not a wall and not a block; landing on an unfilled hole fills it (block and hole disappear).  A
 * block may land on the exit and be pushed off again (exit transit, the campaign rule).  A move that
 * ends with the player on the exit wins (walking onto the empty exit, or pushing a block off it).
 * The class forbids a block on the exit at the start.
 *
 * Three independent enumerations (test_oracle.py checks that they agree):
 *   --mode bf    brute force: every assignment of wall / floor / hole / block(mask) to the non-exit
 *                cells and every start cell; the BFS judge gives the optimum and a covering DP over
 *                the DAG of all optimal solutions decides tightness.  Exact filters only (non-wall
 *                cells connected to the start; every mask bit usable somewhere in the block's
 *                wall-only push closure; every hole in some closure; direction bits no cell of the
 *                board can ever use are not enumerated).  Needs small --max-blocks.
 *   --mode lazy  forward generation: from each start cell, depth-first over move sequences; a cell's
 *                content is decided when first touched (floor, a block pushed now, a hole a block
 *                falls into), mask bits are added when used, untouched cells end as walls.  Each
 *                sequence reaching the exit yields tight(S), kept iff the judge's optimum equals |S|.
 *                Prunes (sound; neither can cut an optimal S): |S| + manhattan(player, exit) > dmax,
 *                and "the configuration is reachable in fewer moves in the partial level with the
 *                unknown cells as walls" (the final level relaxes it).  Memo on exact (partial level,
 *                configuration, t) keys.
 *   --mode back  a deliberately naive backward DFS: states are (touched cells, player, blocks with
 *                used masks, open holes); moves undo a walk, a push, a push of a block that then rests
 *                forever (new block), a fall into a hole; a state is kept iff the judge's optimum of
 *                its level equals its depth (sound by the I_t argument; the judge also checks it is
 *                never above the depth, which verifies the move rules), dedup on the exact state and
 *                depth.  No flags, no canonicalisation, no bulk walk, no symmetry reduction.  It ends
 *                when no state extends, so its T is complete for every depth (no dmax needed).
 *   --extended   (bf, lazy) also admit a block on the exit at the start: lines 'E'.  They are not
 *                class levels but they prove the maximum: if a class level of length n exists, every
 *                I_t is a T or E configuration of length n - t, so a depth m > max(T) with no T and
 *                no E configuration bounds the class ("proof").  back always reports E (its search
 *                passes through them) and proves by exhaustion.
 *   --judge      read level codes (rows joined by '/' or '|') on stdin, print the BFS optimum or -1.
 *
 * Output: "T\t<depth>\t<code>" per distinct tight class level, "E\t<depth>\t<code>\t<exit>" per
 * distinct block-on-exit configuration (the block drawn over the exit), then "STATS\t{json}".  Codes:
 * rows joined by '/', 0 floor, 1 wall, 3 exit, 4 player start, 5 hole, a block by its mask (U=1 R=2
 * D=4 L=8: 7 8 B 9 J C E 6 A I H D G F 2).  Exit status 0, or 2 on a capacity or usage error (never a
 * silent drop).  Grids up to 32 cells, at most 12 blocks.
 *
 *   cc -O2 -o oracle_enum oracle_enum.c
 *   ./oracle_enum --grid 4x4 --exit 5 --mode back
 *   ./oracle_enum --grid 3x4 --exit 1 --max-holes 2 --dmax 30 --mode lazy --extended
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXN   32
#define MAXBL  12
static int R, C, N, EXITC = -1, MAXH = 0, MAXB = MAXBL, DMAX = 10000, EXTENDED = 0;
static uint32_t ALL;
static int ADJ[MAXN][4];  /* 0=U 1=R 2=D 3=L, -1 off the board */
static const char MCH[16] = {'?','7','8','B','9','J','C','E','6','A','I','H','D','G','F','2'};

static void die(const char *m) { fprintf(stderr, "oracle_enum: %s\n", m); exit(2); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int manh(int a, int b) { int d = a / C - b / C, e = a % C - b % C; return (d < 0 ? -d : d) + (e < 0 ? -e : e); }
static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) die("out of memory"); return p; }
static void *xcalloc(size_t n, size_t s) { void *p = calloc(n, s); if (!p) die("out of memory"); return p; }
static uint64_t fnv(const void *p, size_t n) { const uint8_t *b = p; uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; } return h ^ (h >> 31); }

/* ------------------------------------------------------------------ level, configuration, moves */
typedef struct {
    uint32_t wall, hole;          /* cell masks at the start */
    int start, exitc, nb;
    int8_t bpos[MAXBL];           /* start cells */
    uint8_t bmask[MAXBL];
} Level;
typedef struct { int8_t pl; uint32_t hole; int8_t bp[MAXBL]; } St;   /* hole = unfilled holes; bp -1 = consumed */

static int block_at(const St *s, int nb, int c) { for (int i = 0; i < nb; i++) if (s->bp[i] == c) return i; return -1; }

enum { NOMOVE = 0, WALK = 1, PUSH = 2 };
/* One move of the player in direction d.  The move wins iff o->pl == exit afterwards. */
static int try_move(const Level *L, const St *s, int d, St *o, int *pb, int *pland) {
    int t = ADJ[s->pl][d];
    if (t < 0 || (L->wall >> t & 1) || (s->hole >> t & 1)) return NOMOVE;
    int b = block_at(s, L->nb, t);
    if (b < 0) { *o = *s; o->pl = (int8_t)t; return WALK; }
    if (!(L->bmask[b] >> d & 1)) return NOMOVE;
    int l = ADJ[t][d];
    if (l < 0 || (L->wall >> l & 1) || block_at(s, L->nb, l) >= 0) return NOMOVE;
    *o = *s; o->pl = (int8_t)t;
    if (s->hole >> l & 1) { o->hole = s->hole & ~(1u << l); o->bp[b] = -1; } else o->bp[b] = (int8_t)l;
    *pb = b; *pland = l;
    return PUSH;
}

/* Configuration keys.  merged: blocks of equal mask are interchangeable (player, unfilled holes, the
 * sorted multiset of (mask, cell) of live blocks).  labeled: the raw block array (for the DP). */
typedef struct { uint8_t b[5 + 2 * MAXBL + 3]; } Key;
static void mkkey(const Level *L, const St *s, int labeled, Key *k) {
    memset(k, 0xFF, sizeof *k);
    k->b[0] = (uint8_t)s->pl; memcpy(k->b + 1, &s->hole, 4);
    if (labeled) { for (int i = 0; i < L->nb; i++) k->b[5 + i] = (uint8_t)s->bp[i]; return; }
    uint16_t v[MAXBL]; int n = 0;
    for (int i = 0; i < L->nb; i++) if (s->bp[i] >= 0) v[n++] = (uint16_t)(L->bmask[i] << 8 | (uint8_t)s->bp[i]);
    for (int i = 1; i < n; i++) { uint16_t x = v[i]; int j = i; while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; } v[j] = x; }
    for (int i = 0; i < n; i++) { k->b[5 + 2 * i] = (uint8_t)(v[i] >> 8); k->b[6 + 2 * i] = (uint8_t)v[i]; }
}

/* BFS arena reused by every solve; the table uses stamps so a solve only pays for what it touches. */
#define TLG 22
#define TSZ (1u << TLG)
#define SMAX (1u << 21)
static St *ST; static Key *SK; static int16_t *SD; static uint32_t *TIDX, *TSTAMP, STAMP, NS;
static void arena_init(void) {
    ST = xmalloc(sizeof(St) * SMAX); SK = xmalloc(sizeof(Key) * SMAX); SD = xmalloc(sizeof(int16_t) * SMAX);
    TIDX = xmalloc(sizeof(uint32_t) * TSZ); TSTAMP = xcalloc(TSZ, sizeof(uint32_t));
}
static void bfs_reset(void) { NS = 0; if (++STAMP == 0) { memset(TSTAMP, 0, sizeof(uint32_t) * TSZ); STAMP = 1; } }
static uint32_t bfs_find(const Key *k, const St *s, int d, int insert, int *isnew) {
    uint32_t h = (uint32_t)fnv(k->b, sizeof k->b) & (TSZ - 1);
    for (;;) {
        if (TSTAMP[h] != STAMP) {
            if (!insert) return UINT32_MAX;
            if (NS >= SMAX) die("BFS state capacity exceeded (level too large for the oracle judge)");
            TSTAMP[h] = STAMP; TIDX[h] = NS; ST[NS] = *s; SK[NS] = *k; SD[NS] = (int16_t)d; *isnew = 1; return NS++;
        }
        if (!memcmp(SK[TIDX[h]].b, k->b, sizeof k->b)) { *isnew = 0; return TIDX[h]; }
        h = (h + 1) & (TSZ - 1);
    }
}
static St start_state(const Level *L) {
    St s; memset(&s, 0, sizeof s); s.pl = (int8_t)L->start; s.hole = L->hole;
    for (int i = 0; i < MAXBL; i++) s.bp[i] = i < L->nb ? L->bpos[i] : -1;
    return s;
}
static long long g_solves = 0, g_solve_states = 0;
/* Optimum of L if it is <= maxd, else -1 (also for unsolvable).  target != NULL: instead the BFS
 * distance of that configuration if it is < maxd, else -1 (the lazy generator's shortcut test). */
static int bfs(const Level *L, int maxd, const Key *target) {
    bfs_reset(); g_solves++;
    St s0 = start_state(L); Key k; mkkey(L, &s0, 0, &k); int nw;
    if (s0.pl == L->exitc) return 0;
    bfs_find(&k, &s0, 0, 1, &nw);
    if (target && !memcmp(k.b, target->b, sizeof k.b)) return 0;
    for (uint32_t qi = 0; qi < NS; qi++) {
        St u = ST[qi]; int du = SD[qi];
        if (target ? du + 1 >= maxd : du + 1 > maxd) break;
        for (int d = 0; d < 4; d++) {
            St v; int b, l, r = try_move(L, &u, d, &v, &b, &l);
            if (r == NOMOVE) continue;
            if (v.pl == L->exitc) { if (!target) { g_solve_states += NS; return du + 1; } continue; }
            mkkey(L, &v, 0, &k); bfs_find(&k, &v, du + 1, 1, &nw);
            if (nw && target && !memcmp(k.b, target->b, sizeof k.b)) { g_solve_states += NS; return du + 1; }
        }
    }
    g_solve_states += NS;
    return -1;
}

static void level_code(const Level *L, char *out) {     /* a block on the exit is drawn over the '3' */
    int o = 0;
    for (int r = 0; r < R; r++) {
        if (r) out[o++] = '/';
        for (int c = 0; c < C; c++) {
            int i = r * C + c; char ch = (L->wall >> i & 1) ? '1' : '0';
            if (L->hole >> i & 1) ch = '5';
            if (i == L->exitc) ch = '3';
            for (int b = 0; b < L->nb; b++) if (L->bpos[b] == i) ch = MCH[L->bmask[b]];
            if (i == L->start) ch = '4';
            out[o++] = ch;
        }
    }
    out[o] = 0;
}

/* ------------------------------------------------------------------ results (growable string set) */
typedef struct { char code[MAXN + 8]; int16_t depth; int8_t ext; } Found;
static Found *FOUND; static uint32_t NFOUND, FCAP, *FH, FHSZ;
static long long g_emitted = 0, g_judged = 0, g_rejected = 0;
static void found_init(void) { FHSZ = 1u << 16; FH = xmalloc(sizeof(uint32_t) * FHSZ); memset(FH, 0xFF, sizeof(uint32_t) * FHSZ); FCAP = 1u << 14; FOUND = xmalloc(sizeof(Found) * FCAP); }
static void found_rehash(void) {
    FHSZ *= 2; free(FH); FH = xmalloc(sizeof(uint32_t) * FHSZ); memset(FH, 0xFF, sizeof(uint32_t) * FHSZ);
    for (uint32_t i = 0; i < NFOUND; i++) { uint32_t h = (uint32_t)fnv(FOUND[i].code, strlen(FOUND[i].code)) & (FHSZ - 1); while (FH[h] != UINT32_MAX) h = (h + 1) & (FHSZ - 1); FH[h] = i; }
}
static int found_add(const char *code, int depth, int ext) {   /* 0 if already known */
    if ((NFOUND + 1) * 2 >= FHSZ) found_rehash();
    uint32_t h = (uint32_t)fnv(code, strlen(code)) & (FHSZ - 1);
    for (;;) {
        uint32_t i = FH[h];
        if (i == UINT32_MAX) break;
        if (!strcmp(FOUND[i].code, code)) {
            if (FOUND[i].depth != depth) { fprintf(stderr, "level %s at depths %d and %d\n", code, FOUND[i].depth, depth); die("inconsistent depth"); }
            return 0;
        }
        h = (h + 1) & (FHSZ - 1);
    }
    if (NFOUND == FCAP) { FCAP *= 2; FOUND = realloc(FOUND, sizeof(Found) * FCAP); if (!FOUND) die("out of memory"); }
    strcpy(FOUND[NFOUND].code, code); FOUND[NFOUND].depth = (int16_t)depth; FOUND[NFOUND].ext = (int8_t)ext;
    FH[h] = NFOUND++;
    return 1;
}

/* ------------------------------------------------------------------ exact visited set (lazy memo, back dedup) */
#define VKEY 64
static uint8_t *VK; static uint32_t VSZ, VN; static uint8_t *VUSED; static int VLG = 21; static long long g_visit_full = 0;
static void visit_init(void) { VSZ = 1u << VLG; VK = xmalloc((size_t)VKEY * VSZ); VUSED = xcalloc(VSZ, 1); }
/* 1 if the key was already present; inserts it otherwise.  full: returns -1 (caller decides). */
static int visit(const uint8_t *k) {
    uint32_t x = (uint32_t)fnv(k, VKEY) & (VSZ - 1);
    for (;;) {
        if (!VUSED[x]) {
            if ((uint64_t)VN * 4 >= (uint64_t)VSZ * 3) { g_visit_full++; return -1; }
            VUSED[x] = 1; memcpy(VK + (size_t)x * VKEY, k, VKEY); VN++; return 0;
        }
        if (!memcmp(VK + (size_t)x * VKEY, k, VKEY)) return 1;
        x = (x + 1) & (VSZ - 1);
    }
}

/* ------------------------------------------------------------------ mode lazy: forward generation */
enum { UNK = 0, FLR = 1, HOL = 2, BLK = 3, EXT = 4, EXB = 5 };   /* initial content of a touched cell */
typedef struct {
    uint8_t kind[MAXN];
    int nb, nh;
    int8_t bstart[MAXBL]; uint8_t bmask[MAXBL];
    St cur;
    int t;
} Gen;
static int g_start;
static long long g_nodes = 0, g_prune_depth = 0, g_prune_short = 0, g_prune_memo = 0;

static void gen_level(const Gen *g, Level *L, int *ext) {       /* unknown cells are walls */
    memset(L, 0, sizeof *L); *ext = 0;
    for (int i = 0; i < N; i++) {
        if (g->kind[i] == UNK) L->wall |= 1u << i;
        if (g->kind[i] == HOL) L->hole |= 1u << i;
        if (g->kind[i] == EXB) *ext = 1;
    }
    L->start = g_start; L->exitc = EXITC; L->nb = g->nb;
    for (int b = 0; b < g->nb; b++) { L->bpos[b] = g->bstart[b]; L->bmask[b] = g->bmask[b]; }
}
static int gen_memo_seen(const Gen *g) {
    uint8_t k[VKEY]; memset(k, 0xFF, sizeof k); int o = 0;
    for (int i = 0; i < N; i += 2) k[o++] = (uint8_t)(g->kind[i] | (i + 1 < N ? g->kind[i + 1] << 4 : 0));
    k[16] = (uint8_t)g_start; k[17] = (uint8_t)g->cur.pl; memcpy(k + 18, &g->cur.hole, 4); k[22] = (uint8_t)g->nb;
    k[23] = (uint8_t)(g->t & 0xFF); k[24] = (uint8_t)(g->t >> 8);
    for (int b = 0; b < g->nb; b++) { k[25 + 3 * b] = (uint8_t)g->bstart[b]; k[26 + 3 * b] = g->bmask[b]; k[27 + 3 * b] = (uint8_t)g->cur.bp[b]; }
    return visit(k) == 1;
}
static void emit_level(const Gen *g, int n) {
    Level L; int ext; gen_level(g, &L, &ext);
    g_emitted++;
    char code[MAXN + 8]; level_code(&L, code);
    g_judged++;
    if (bfs(&L, n, NULL) != n) { g_rejected++; return; }   /* S was not optimal in tight(S) */
    found_add(code, n, ext);
}
static void gen(Gen *g);
/* push the block b (now at t) in direction d; an unknown landing cell branches (floor, or a hole) */
static void do_push(Gen *g, int b, int t, int d) {
    int l = ADJ[t][d];
    if (l < 0 || block_at(&g->cur, g->nb, l) >= 0) return;
    uint8_t k = g->kind[l];
    Gen h;
    if (k == UNK) {
        h = *g; h.kind[l] = FLR; h.bmask[b] |= (uint8_t)(1 << d); h.cur.bp[b] = (int8_t)l; h.cur.pl = (int8_t)t; h.t++; gen(&h);
        if (g->nh < MAXH) { h = *g; h.kind[l] = HOL; h.nh++; h.bmask[b] |= (uint8_t)(1 << d); h.cur.bp[b] = -1; h.cur.pl = (int8_t)t; h.t++; gen(&h); }
        return;
    }
    h = *g; h.bmask[b] |= (uint8_t)(1 << d); h.cur.pl = (int8_t)t; h.t++;
    if (k == HOL && (g->cur.hole >> l & 1)) { h.cur.hole &= ~(1u << l); h.cur.bp[b] = -1; }
    else h.cur.bp[b] = (int8_t)l;
    gen(&h);
}
static void gen(Gen *g) {
    g_nodes++;
    int p = g->cur.pl;
    if (g->t + manh(p, EXITC) > DMAX) { g_prune_depth++; return; }
    if (p == EXITC) { emit_level(g, g->t); return; }     /* the move just made ended on the exit: a win */
    if (g->t >= 2) {
        Level P; int ext; gen_level(g, &P, &ext); Key tk; mkkey(&P, &g->cur, 0, &tk);
        if (bfs(&P, g->t, &tk) >= 0) { g_prune_short++; return; }
        if (gen_memo_seen(g)) { g_prune_memo++; return; }
    }
    for (int d = 0; d < 4; d++) {
        int t = ADJ[p][d];
        if (t < 0) continue;
        uint8_t k = g->kind[t];
        int b = block_at(&g->cur, g->nb, t);
        if (b >= 0) { do_push(g, b, t, d); continue; }
        if (k == UNK) {
            Gen h = *g; h.kind[t] = FLR; h.cur.pl = (int8_t)t; h.t++; gen(&h);          /* floor */
            if (g->nb < MAXB) {                                                          /* a block, pushed now */
                h = *g; h.kind[t] = BLK; int nb = h.nb++; h.bstart[nb] = (int8_t)t; h.bmask[nb] = 0; h.cur.bp[nb] = (int8_t)t;
                do_push(&h, nb, t, d);
            }
            continue;
        }
        if (k == HOL && (g->cur.hole >> t & 1)) continue;                                /* unfilled hole */
        Gen h = *g; h.cur.pl = (int8_t)t; h.t++; gen(&h);                               /* known floor or exit */
    }
}
static void run_lazy(void) {
    for (int s = 0; s < N; s++) {
        if (s == EXITC) continue;
        g_start = s;
        Gen g; memset(&g, 0, sizeof g);
        for (int i = 0; i < MAXBL; i++) { g.cur.bp[i] = -1; g.bstart[i] = -1; }
        g.kind[s] = FLR; g.kind[EXITC] = EXT; g.cur.pl = (int8_t)s;
        gen(&g);
        if (EXTENDED && MAXB > 0) {   /* second root: a block on the exit at the start, mask set when pushed */
            Gen e = g; e.kind[EXITC] = EXB; int nb = e.nb++; e.bstart[nb] = (int8_t)EXITC; e.bmask[nb] = 0; e.cur.bp[nb] = (int8_t)EXITC;
            gen(&e);
        }
    }
}

/* ------------------------------------------------------------------ mode back: naive backward DFS */
typedef struct {
    uint32_t com, hole;            /* cells touched by the suffix; holes open now (filled later in the suffix) */
    int8_t pl, nb, nh;
    int8_t bp[MAXBL]; uint8_t bm[MAXBL];   /* blocks alive now (all pushed later), masks used later */
    int16_t depth;
} BS;
static long long g_back_states = 0, g_back_pruned = 0, g_back_dups = 0, g_truncated = 0;
static void bs_level(const BS *s, Level *L) {
    memset(L, 0, sizeof *L);
    L->wall = ~s->com & ALL; L->hole = s->hole; L->start = s->pl; L->exitc = EXITC; L->nb = s->nb;
    for (int b = 0; b < s->nb; b++) { L->bpos[b] = s->bp[b]; L->bmask[b] = s->bm[b]; }
}
static int bs_seen(const BS *s) {   /* exact key: touched cells, holes, player, sorted (mask, cell) pairs, depth */
    uint8_t k[VKEY]; memset(k, 0xFF, sizeof k);
    memcpy(k, &s->com, 4); memcpy(k + 4, &s->hole, 4); k[8] = (uint8_t)s->pl; memcpy(k + 9, &s->depth, 2);
    uint16_t v[MAXBL]; int n = s->nb;
    for (int i = 0; i < n; i++) v[i] = (uint16_t)(s->bm[i] << 8 | (uint8_t)s->bp[i]);
    for (int i = 1; i < n; i++) { uint16_t x = v[i]; int j = i; while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; j--; } v[j] = x; }
    for (int i = 0; i < n; i++) { k[11 + 2 * i] = (uint8_t)(v[i] >> 8); k[12 + 2 * i] = (uint8_t)v[i]; }
    int r = visit(k);
    if (r < 0) die("back mode: visited table full (raise --table-lg2)");
    return r;
}
static BS *BSTK; static size_t BTOP, BCAP;
static void bs_push(const BS *s) { if (BTOP == BCAP) { BCAP = BCAP ? 2 * BCAP : 4096; BSTK = realloc(BSTK, sizeof(BS) * BCAP); if (!BSTK) die("out of memory"); } BSTK[BTOP++] = *s; }
/* judge a new state: kept iff its level's optimum equals its depth (never above: the suffix solves it) */
static void bs_consider(BS *y) {
    if (y->depth > DMAX) { g_truncated++; return; }
    if (bs_seen(y)) { g_back_dups++; return; }
    g_back_states++;
    Level L; bs_level(y, &L);
    int opt = bfs(&L, y->depth, NULL);
    if (opt < 0 || opt > y->depth) {
        char code[MAXN + 8]; level_code(&L, code);
        fprintf(stderr, "state %s at depth %d has optimum %d\n", code, y->depth, opt);
        die("back mode: a backward move produced a state its own suffix does not solve (move-rule bug)");
    }
    if (opt < y->depth) { g_back_pruned++; return; }
    char code[MAXN + 8]; level_code(&L, code);
    int ext = 0; for (int b = 0; b < y->nb; b++) if (y->bp[b] == EXITC) ext = 1;
    found_add(code, y->depth, ext);
    bs_push(y);
}
static void run_back(void) {
    BS root; memset(&root, 0, sizeof root);
    root.com = 1u << EXITC; root.pl = (int8_t)EXITC;
    bs_seen(&root); bs_push(&root);
    while (BTOP) {
        BS x = BSTK[--BTOP];
        int p = x.pl;
        for (int d = 0; d < 4; d++) {              /* the forward move went in direction d, from q to p */
            int q = ADJ[p][d ^ 2];
            if (q < 0 || q == EXITC || (x.hole >> q & 1)) continue;
            int bq = 0; for (int b = 0; b < x.nb; b++) if (x.bp[b] == q) bq = 1;
            if (bq) continue;
            BS y;
            /* undo a walk q -> p */
            y = x; y.pl = (int8_t)q; y.com |= 1u << q; y.depth++; bs_consider(&y);
            int l = ADJ[p][d];
            if (l < 0) continue;
            int bl = -1; for (int b = 0; b < x.nb; b++) if (x.bp[b] == l) bl = b;
            if (bl >= 0) {                          /* undo the push of the block now at l (it came from p) */
                y = x; y.bp[bl] = (int8_t)p; y.bm[bl] |= (uint8_t)(1 << d); y.pl = (int8_t)q; y.com |= 1u << q; y.depth++;
                bs_consider(&y);
                continue;
            }
            if (x.nb >= MAXB) continue;
            if (!(x.com >> l & 1)) {                /* a new block, pushed p -> l, resting at l ever after */
                y = x; int nb = y.nb++; y.bp[nb] = (int8_t)p; y.bm[nb] = (uint8_t)(1 << d); y.pl = (int8_t)q;
                y.com |= (1u << q) | (1u << l); y.depth++; bs_consider(&y);
            }
            if (l != EXITC && !(x.hole >> l & 1) && x.nh < MAXH) {   /* a block pushed p -> l fell into a hole at l */
                y = x; int nb = y.nb++; y.bp[nb] = (int8_t)p; y.bm[nb] = (uint8_t)(1 << d); y.pl = (int8_t)q;
                y.hole |= 1u << l; y.nh++; y.com |= (1u << q) | (1u << l); y.depth++; bs_consider(&y);
            }
        }
    }
}

/* ------------------------------------------------------------------ mode bf: brute force + covering DP */
static int g_usable_dirs = 0;
static uint32_t closure_of(const Level *L, int b) {   /* wall-only push closure of block b */
    uint32_t seen = 1u << L->bpos[b], fr = seen;
    while (fr) {
        uint32_t nf = 0;
        for (int c = 0; c < N; c++) if (fr >> c & 1)
            for (int d = 0; d < 4; d++) if (L->bmask[b] >> d & 1) {
                int pc = ADJ[c][d ^ 2], l = ADJ[c][d];
                if (pc < 0 || l < 0 || (L->wall >> pc & 1) || (L->wall >> l & 1)) continue;
                if (!(seen >> l & 1)) { seen |= 1u << l; nf |= 1u << l; }
            }
        fr = nf;
    }
    return seen;
}
static int filters_pass(const Level *L) {
    uint32_t open = ~L->wall & ALL, seen = 1u << L->start, fr = seen;
    while (fr) { uint32_t nf = 0; for (int c = 0; c < N; c++) if (fr >> c & 1) for (int d = 0; d < 4; d++) { int x = ADJ[c][d]; if (x >= 0 && (open >> x & 1) && !(seen >> x & 1)) { seen |= 1u << x; nf |= 1u << x; } } fr = nf; }
    if (seen != open) return 0;
    uint32_t reach = 0;
    for (int b = 0; b < L->nb; b++) {
        uint32_t cl = closure_of(L, b); reach |= cl;
        for (int d = 0; d < 4; d++) if (L->bmask[b] >> d & 1) {
            int ok = 0;
            for (int c = 0; c < N && !ok; c++) if (cl >> c & 1) { int pc = ADJ[c][d ^ 2], l = ADJ[c][d]; if (pc >= 0 && l >= 0 && !(L->wall >> pc & 1) && !(L->wall >> l & 1)) ok = 1; }
            if (!ok) return 0;
        }
    }
    return !(L->hole & ~reach);
}
typedef struct { uint64_t *m; int n, cap; } MSet;
static void mset_add(MSet *s, uint64_t x) {        /* keep only the maximal usage sets */
    for (int i = 0; i < s->n; i++) if ((s->m[i] | x) == s->m[i]) return;
    int j = 0; for (int i = 0; i < s->n; i++) if ((s->m[i] | x) != x) s->m[j++] = s->m[i];
    s->n = j;
    if (s->n == s->cap) { s->cap = s->cap ? 2 * s->cap : 4; s->m = realloc(s->m, sizeof(uint64_t) * s->cap); if (!s->m) die("out of memory"); }
    s->m[s->n++] = x;
}
/* does some optimal solution (length n) use every non-wall cell and every mask bit? */
static int tight_dp(const Level *L, int n) {
    if (N + 4 * L->nb > 64) die("covering DP: too many cells + block bits for a 64-bit mask");
    bfs_reset();
    St s0 = start_state(L); Key k; int nw;
    mkkey(L, &s0, 1, &k); bfs_find(&k, &s0, 0, 1, &nw);
    for (uint32_t qi = 0; qi < NS; qi++) {
        St u = ST[qi]; int du = SD[qi];
        if (du >= n - 1) continue;
        for (int d = 0; d < 4; d++) { St v; int b, l, r = try_move(L, &u, d, &v, &b, &l); if (r != NOMOVE && v.pl != L->exitc) { mkkey(L, &v, 1, &k); bfs_find(&k, &v, du + 1, 1, &nw); } }
    }
    uint32_t ns = NS;
    uint8_t *good = xcalloc(ns, 1);
    for (int layer = n - 1; layer >= 0; layer--)
        for (uint32_t i = 0; i < ns; i++) {
            if (SD[i] != layer) continue;
            for (int d = 0; d < 4 && !good[i]; d++) {
                St v; int b, l, r = try_move(L, &ST[i], d, &v, &b, &l);
                if (r == NOMOVE) continue;
                if (v.pl == L->exitc) { if (layer == n - 1) good[i] = 1; }
                else if (layer < n - 1) { mkkey(L, &v, 1, &k); uint32_t j = bfs_find(&k, &v, 0, 0, &nw); if (j != UINT32_MAX && SD[j] == layer + 1 && good[j]) good[i] = 1; }
            }
        }
    uint64_t full = ~L->wall & ALL;
    for (int b = 0; b < L->nb; b++) full |= (uint64_t)L->bmask[b] << (N + 4 * b);
    uint64_t init = 1ull << L->start;
    for (int b = 0; b < L->nb; b++) init |= 1ull << L->bpos[b];
    MSet *sets = xcalloc(ns, sizeof(MSet));
    int tight = 0;
    if (good[0]) mset_add(&sets[0], init);
    for (int layer = 0; layer <= n - 1 && !tight; layer++)
        for (uint32_t i = 0; i < ns && !tight; i++) {
            if (SD[i] != layer || !good[i] || !sets[i].n) continue;
            for (int d = 0; d < 4 && !tight; d++) {
                St v; int b, l, r = try_move(L, &ST[i], d, &v, &b, &l);
                if (r == NOMOVE) continue;
                uint64_t add = 1ull << v.pl;
                if (r == PUSH) add |= (1ull << l) | (1ull << (N + 4 * b + d));
                if (v.pl == L->exitc) {
                    if (layer != n - 1) continue;
                    for (int q = 0; q < sets[i].n; q++) if ((sets[i].m[q] | add) == full) tight = 1;
                    continue;
                }
                if (layer == n - 1) continue;
                mkkey(L, &v, 1, &k); uint32_t j = bfs_find(&k, &v, 0, 0, &nw);
                if (j == UINT32_MAX || SD[j] != layer + 1 || !good[j]) continue;
                for (int q = 0; q < sets[i].n; q++) mset_add(&sets[j], sets[i].m[q] | add);
            }
        }
    for (uint32_t i = 0; i < ns; i++) free(sets[i].m);
    free(sets); free(good);
    return tight;
}
static long long g_bf_levels = 0, g_bf_filtered = 0;
static int OPT_N = 0, OPTS[20], CELLS[MAXN], NCELLS_X;   /* cell options: 0 wall, 1 floor, 2 hole, 2+m block of mask m */
static void bf_check(const Level *L0) {
    g_bf_levels++;
    if (!filters_pass(L0)) return;
    g_bf_filtered++;
    int opt = bfs(L0, DMAX, NULL);
    if (opt <= 0 || !tight_dp(L0, opt)) return;
    char code[MAXN + 8]; level_code(L0, code);
    int ext = 0; for (int b = 0; b < L0->nb; b++) if (L0->bpos[b] == EXITC) ext = 1;
    found_add(code, opt, ext);
}
static void bf_rec(int idx, Level *L, int nh) {
    if (idx == NCELLS_X) {
        for (int s = 0; s < N; s++) {
            if (s == EXITC || (L->wall >> s & 1) || (L->hole >> s & 1)) continue;
            int isb = 0; for (int b = 0; b < L->nb; b++) if (L->bpos[b] == s) isb = 1;
            if (isb) continue;
            L->start = s; bf_check(L);
        }
        return;
    }
    int c = CELLS[idx];
    for (int o = 0; o < OPT_N; o++) {
        int v = OPTS[o];
        if (v == 0) { L->wall |= 1u << c; bf_rec(idx + 1, L, nh); L->wall &= ~(1u << c); }
        else if (v == 1) bf_rec(idx + 1, L, nh);
        else if (v == 2) { if (nh >= MAXH) continue; L->hole |= 1u << c; bf_rec(idx + 1, L, nh + 1); L->hole &= ~(1u << c); }
        else { if (L->nb >= MAXB) continue; L->bpos[L->nb] = (int8_t)c; L->bmask[L->nb] = (uint8_t)(v - 2); L->nb++; bf_rec(idx + 1, L, nh); L->nb--; }
    }
}
static void run_bf(void) {
    g_usable_dirs = 0;
    for (int c = 0; c < N; c++) for (int d = 0; d < 4; d++) if (ADJ[c][d] >= 0 && ADJ[c][d ^ 2] >= 0) g_usable_dirs |= 1 << d;
    OPT_N = 0; OPTS[OPT_N++] = 0; OPTS[OPT_N++] = 1; if (MAXH > 0) OPTS[OPT_N++] = 2;
    for (int m = 1; m < 16; m++) if (!(m & ~g_usable_dirs)) OPTS[OPT_N++] = m + 2;
    NCELLS_X = 0; for (int c = 0; c < N; c++) if (c != EXITC) CELLS[NCELLS_X++] = c;
    Level L; memset(&L, 0, sizeof L); L.exitc = EXITC;
    bf_rec(0, &L, 0);
    if (EXTENDED && MAXB > 0)
        for (int m = 1; m < 16; m++) {
            if (m & ~g_usable_dirs) continue;
            memset(&L, 0, sizeof L); L.exitc = EXITC; L.bpos[0] = (int8_t)EXITC; L.bmask[0] = (uint8_t)m; L.nb = 1;
            bf_rec(0, &L, 0);
        }
}

/* ------------------------------------------------------------------ judge mode */
static int parse_code(const char *s, Level *L) {
    memset(L, 0, sizeof *L); L->start = L->exitc = -1;
    int rows = 0, cols = -1, c = 0;
    for (const char *p = s;; p++) {
        if (*p == '/' || *p == '|' || *p == 0) { if (cols < 0) cols = c; else if (c != cols) return 0; rows++; c = 0; if (!*p) break; continue; }
        c++;
    }
    if (rows != R || cols != C) return 0;
    int r = 0; c = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '/' || *p == '|') { r++; c = 0; continue; }
        int i = r * C + c++; char ch = *p;
        if (ch == '0') continue;
        if (ch == '1') L->wall |= 1u << i;
        else if (ch == '3') { if (L->exitc >= 0) return 0; L->exitc = i; }
        else if (ch == '4') { if (L->start >= 0) return 0; L->start = i; }
        else if (ch == '5') L->hole |= 1u << i;
        else { int m = -1; for (int k = 1; k < 16; k++) if (MCH[k] == ch) m = k; if (m < 0 || L->nb >= MAXBL) return 0; L->bpos[L->nb] = (int8_t)i; L->bmask[L->nb++] = (uint8_t)m; }
    }
    return L->start >= 0 && L->exitc >= 0;
}

int main(int argc, char **argv) {
    const char *mode = "lazy"; int judge = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--grid") && v) { if (sscanf(v, "%dx%d", &R, &C) != 2) die("bad --grid"); i++; }
        else if (!strcmp(a, "--exit") && v) { EXITC = atoi(v); i++; }
        else if (!strcmp(a, "--max-holes") && v) { MAXH = atoi(v); i++; }
        else if (!strcmp(a, "--max-blocks") && v) { MAXB = atoi(v); i++; }
        else if (!strcmp(a, "--dmax") && v) { DMAX = atoi(v); i++; }
        else if (!strcmp(a, "--mode") && v) { mode = v; i++; }
        else if (!strcmp(a, "--table-lg2") && v) { VLG = atoi(v); i++; }
        else if (!strcmp(a, "--extended")) EXTENDED = 1;
        else if (!strcmp(a, "--judge")) judge = 1;
        else { fprintf(stderr, "unknown argument %s\n", a); die("usage: oracle_enum --grid RxC --exit E [--max-holes H] [--max-blocks B] [--dmax D] [--mode back|lazy|bf] [--extended] [--table-lg2 K] | --grid RxC --judge"); }
    }
    N = R * C;
    if (R < 1 || C < 1 || N > MAXN) die("grid must have at most 32 cells");
    if (VLG < 10 || VLG > 28) die("--table-lg2 in [10, 28]");
    ALL = N == 32 ? ~0u : (1u << N) - 1;
    if (MAXB > MAXBL) MAXB = MAXBL;
    if (MAXH > N) MAXH = N;
    for (int c = 0; c < N; c++) {
        int r = c / C, q = c % C;
        ADJ[c][0] = r > 0 ? c - C : -1; ADJ[c][1] = q < C - 1 ? c + 1 : -1;
        ADJ[c][2] = r < R - 1 ? c + C : -1; ADJ[c][3] = q > 0 ? c - 1 : -1;
    }
    arena_init();
    if (judge) {
        char line[512];
        while (fgets(line, sizeof line, stdin)) {
            line[strcspn(line, "\r\n")] = 0; if (!line[0]) continue;
            Level L; if (!parse_code(line, &L)) { printf("error\n"); continue; }
            printf("%d\n", bfs(&L, 30000, NULL)); fflush(stdout);
        }
        return 0;
    }
    if (EXITC < 0 || EXITC >= N) die("--exit required");
    int back = !strcmp(mode, "back");
    if (!back && DMAX > 200) DMAX = 200;
    found_init();
    double t0 = now();
    if (back) { visit_init(); run_back(); }
    else if (!strcmp(mode, "lazy")) { visit_init(); run_lazy(); }
    else if (!strcmp(mode, "bf")) run_bf();
    else die("--mode back|lazy|bf");
    int maxT = 0, maxE = 0; long long perT[512] = {0}, perE[512] = {0};
    for (uint32_t i = 0; i < NFOUND; i++) {
        Found *f = &FOUND[i];
        if (f->depth >= 512) die("depth >= 512");
        if (f->ext) { perE[f->depth]++; if (f->depth > maxE) maxE = f->depth; printf("E\t%d\t%s\t%d\n", f->depth, f->code, EXITC); }
        else { perT[f->depth]++; if (f->depth > maxT) maxT = f->depth; printf("T\t%d\t%s\n", f->depth, f->code); }
    }
    /* proof of the maximum: back = exhausted without truncation; bf/lazy --extended = the first depth
     * above max(T) with no T and no E configuration (within dmax) */
    int empty = -1, proof = 0;
    if (back) { proof = !g_truncated; empty = (maxT > maxE ? maxT : maxE) + 1; }
    else if (EXTENDED) for (int m = maxT + 1; m <= DMAX && m < 512; m++) if (!perT[m] && !perE[m]) { empty = m; proof = 1; break; }
    printf("STATS\t{\"mode\":\"%s\",\"grid\":\"%dx%d\",\"exit\":%d,\"max_holes\":%d,\"max_blocks\":%d,\"dmax\":%d,\"extended\":%d,"
           "\"tight\":%u,\"max\":%d,\"max_ext\":%d,\"empty_depth\":%d,\"proof\":%s,\"nodes\":%lld,\"prune_depth\":%lld,\"prune_short\":%lld,"
           "\"prune_memo\":%lld,\"memo_full\":%lld,\"emitted\":%lld,\"judged\":%lld,\"rejected\":%lld,\"bf_levels\":%lld,\"bf_filtered\":%lld,"
           "\"back_states\":%lld,\"back_pruned\":%lld,\"back_dups\":%lld,\"truncated\":%lld,\"solves\":%lld,\"solve_states\":%lld,\"elapsed\":%.3f,\"per_depth\":{",
           mode, R, C, EXITC, MAXH, MAXB, DMAX, back ? 1 : EXTENDED, NFOUND, maxT, maxE, empty, proof ? "true" : "false",
           g_nodes, g_prune_depth, g_prune_short, g_prune_memo, g_visit_full, g_emitted, g_judged, g_rejected, g_bf_levels, g_bf_filtered,
           g_back_states, g_back_pruned, g_back_dups, g_truncated, g_solves, g_solve_states, now() - t0);
    int first = 1;
    for (int m = 0; m < 512; m++) if (perT[m] || perE[m]) { printf("%s\"%d\":[%lld,%lld]", first ? "" : ",", m, perT[m], perE[m]); first = 0; }
    printf("}}\n");
    return 0;
}
