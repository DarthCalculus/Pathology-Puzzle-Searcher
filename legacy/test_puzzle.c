#include "sokoban_bfs.h"
#include <stdio.h>
#include <string.h>

static int solve3(int player,
                  int b0, int m0, int b1, int m1, int b2, int m2) {
    Puzzle pz;
    memset(&pz, 0, sizeof pz);
    pz.exit_pos      = 0;
    pz.player_start  = player;
    pz.num_holes     = 3;
    pz.hole_pos[0]   = 1;
    pz.hole_pos[1]   = 6;
    pz.hole_pos[2]   = 8;
    pz.num_blocks    = 3;
    pz.block_pos[0]  = b0; pz.block_pushable[0] = m0;
    pz.block_pos[1]  = b1; pz.block_pushable[1] = m1;
    pz.block_pos[2]  = b2; pz.block_pushable[2] = m2;
    return sokoban_solve(&pz, NULL);
}

int main(void) {
    /* Original found puzzle:
     *   EO@..
     *   AOBO.   A@5 push=0, B@7 push=6[RD], C@11 push=b[URL]
     *   .C...
     *   .....   player=2 */
    printf("Original (A@5[--],B@7[RD],C@11[URL], player=2): %d\n",
           solve3(2,  5,0x0,  7,0x6,  11,0xb));

    printf("\n--- Proposed modification: B moves to 12, player to 7, B gains U ---\n\n");

    /* Proposed: B@12, player=7, B push=7[URD] */
    printf("Proposed  (A@5[--],C@11[URL],B@12[URD], player=7): %d\n",
           solve3(7,  5,0x0,  11,0xb,  12,0x7));

    /* Same position but B push=U only — the subset the antichain would have tried first */
    printf("B push=U  (A@5[--],C@11[URL],B@12[U],   player=7): %d\n",
           solve3(7,  5,0x0,  11,0xb,  12,0x1));

    /* Verify monotonicity for a few more subsets */
    printf("B push=R  (A@5[--],C@11[URL],B@12[R],   player=7): %d\n",
           solve3(7,  5,0x0,  11,0xb,  12,0x2));
    printf("B push=D  (A@5[--],C@11[URL],B@12[D],   player=7): %d\n",
           solve3(7,  5,0x0,  11,0xb,  12,0x4));
    printf("B push=UD (A@5[--],C@11[URL],B@12[UD],  player=7): %d\n",
           solve3(7,  5,0x0,  11,0xb,  12,0x5));
    printf("B push=0  (A@5[--],C@11[URL],B@12[--],  player=7): %d\n",
           solve3(7,  5,0x0,  11,0xb,  12,0x0));
    return 0;
}
