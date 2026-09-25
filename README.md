# Pathology Puzzle Searcher

Exhaustive search for the longest [Pathology](https://pathology.thinky.gg) levels
on small grids, and machine-checked proofs that particular levels are the longest
possible in their class.  Everything current lives in
[`backsearch/`](backsearch/); results are published with an
"exhaustively proven champion" badge on
[pathology.georgespahn.com](https://pathology.georgespahn.com).

## The problem

A Pathology level is a grid of floor, walls, an exit, blocks (each pushable only
in some directions) and holes (a block pushed in fills the hole and vanishes).
The player wins by stepping onto the exit; the level's length is the number of
moves of its shortest solution.  The question is: for a given grid size and a cap
on blocks and holes, what is the longest level, and can we prove it?

Rules as searched: blocks may be pushed across the exit cell (the record levels
rely on it), and no block starts on the exit.  Levels with more holes than
blocks are never longer than one with the extra holes turned into walls, so hole
caps above the block cap are redundant.

## Method: backward search with a forward oracle

`backsearch.c` grows levels backwards from the solved position.  Each backward
step un-walks the player, un-pushes an existing block, introduces a block, or
un-consumes a block from a hole; every step commits cells as floor, and cells
never touched are walls.  A state at depth *d* is a level whose intended solution
has *d* moves; it is kept only if the forward solver (`sokoban_bfs.c`, a push-macro
Dijkstra with a move cutoff) certifies that no solution of length *d-2* or less
exists (parity rules out *d-1*).  Exhausting the backward tree under a cap
therefore enumerates every level of the class up to symmetry, and the deepest
accepted state is the proven maximum.

Most of the work is the forward check, so most of the engineering is there:
decision-mode Dijkstra with a bucket queue, a shortest-walk-segment prune, one
multi-start solve for a whole component of walk-back children, and reuse of a
parent's exhausted solve as a table of bounds for its children.  Every pruning
rule is exact and has a check build that re-runs the pruned solves without the
rule.  See [`backsearch/README.md`](backsearch/README.md) for the full
description and [`backsearch/CHANGES_2026-09-18.md`](backsearch/CHANGES_2026-09-18.md)
for the solver work of September 2026.

## Proven results (5x5, exit anywhere)

Longest optimal solution for a 5x5 level with at most the given blocks and holes.
Cells are blank where holes exceed blocks (equivalent to the row's "any" cell) and
`?` where the class is not exhausted yet.

| blocks \ holes | 0 | 1 | 2 | 3 | 4 | 5 | any |
|---|---|---|---|---|---|---|---|
| 0 | 16 | | | | | | 16 |
| 1 | 17 | 40 | | | | | 40 |
| 2 | 28 | 75 | 75 | | | | 75 |
| 3 | 53 | 96 | 96 | 110 | | | 110 |
| 4 | 56 | 106 | 119 | 127 | 127 | | 127 |
| 5 | 67 | 113 | 121 | 127 | ? | ? | ? |
| 6 | 69 | 122 | 124 | ? | ? | ? | ? |
| 7 | 69 | 122 | 149 | 149 | ? | ? | ? |
| 8 | 69 | 122 | 149 | 149 | ? | ? | ? |
| any | 69 | 122 | 149 | 149 | ? | ? | ? |

The 149 is davidspencer6174's *Capital C* (7 blocks, 2 holes); it is proven
longest among all 5x5 levels with at most 2 holes, and, by the first collective
campaign (September 2026: 597 CPU-hours from volunteers, published under
"Collective"), among all 5x5 levels with at most 3 holes (per exit 127, 149, 112,
138, 115 and 80; the 3-hole column's 149s for 7 and 8 blocks follow from it).
The second campaign, 6x6 with no holes (best known 165 moves), is running: see
[`backsearch/VOLUNTEERS.md`](backsearch/VOLUNTEERS.md) to help, and
[`backsearch/v2/DESIGN.md`](backsearch/v2/DESIGN.md) for how it works.  4x5 is fully enumerated
(70).  Per-exit proofs, states searched and CPU time for each entry are listed on
the site's Proofs page; the raw campaign records are in `backsearch/results/`.

## Running it

```bash
cd backsearch
./build_pgo.sh -o backsearch_worker_nt --no-torch          # profile-guided build

# one exit, one class, run until the tree is exhausted
./backsearch_worker_nt --grid 5x5 --two-tables --allow-exit-transit \
    --exit 7 --num-blocks 3 --time 0

# size a tree before committing to it (lower bound; real runs are 2-7x)
./backsearch_worker_nt --grid 5x5 --two-tables --allow-exit-transit \
    --exit 1 --num-holes 3 --estimate 1500000 --estimate-depth 10

# split an exit into resumable jobs and run them in parallel (the depth-K root
# listing is exact at every K with a worker from commit 27443a8 on; older
# workers lost subtrees at K >= 5 -- see backsearch/README.md)
python3 campaign.py --out results/camp_b6h2 --exits 0,1,2,6,7,12 \
    --extra "--num-blocks 6 --num-holes 2" --layer 8 --workers 2 \
    --worker ./backsearch_worker_nt --shuffle
python3 campaign.py --out results/camp_b6h2 --status

# publish a finished campaign as proofs + champion levels (site owner only)
python3 results/publish_campaign.py results/camp_b6h2 --max-blocks 6 --max-holes 2
```

`results/prove_loop.sh` automates the last three steps: it sizes every
(class, exit) in `results/prove_queue.txt`, runs them one at a time from the
cheapest, skips any that a finished superclass proof already settles (only when
that proof's champion lies inside the subclass), and publishes each as it
finishes.

## Layout

| path | what |
|---|---|
| `backsearch/backsearch.c` | backward generator: DFS, pruning, estimator, seed paths, campaign worker |
| `backsearch/sokoban_bfs.c`, `.h` | forward solver: cutoff Dijkstra, multi-start solve, reference tables |
| `backsearch/campaign.py` | resumable parallel driver over depth-K seed paths |
| `backsearch/results/` | campaign proofs, champions and logs; `prove_loop.sh` and the publishers |
| `backsearch/solvebench.c`, `multitest.c` | solver regression harnesses (replay a harvest; multi vs single) |
| `backsearch/level_to_pathology.c`, `seed_from_level.c` | format converters |
| `backsearch/nn_*.*`, `train_*.py`, `selfplay.py`, `harvest_*.py` | experimental learned surrogates (off by default; the proofs never use them) |
| `backsearch/old_experiments/` | superseded rollout / annealing record hunts |
| `legacy/` | the earlier forward-enumeration searchers for 4x5, 5x5 and 5x6 |

Verification of a new level's length independently of this code is done by the
site's solver on submission.
