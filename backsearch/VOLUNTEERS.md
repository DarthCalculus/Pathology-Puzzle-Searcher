# Helping with the collective proof

The tracker at https://pathology.georgespahn.com (Proofs tab) splits one
exhaustive search into chunks. You sign up for chunks by name, run them on
your own machine with one command, and paste the report the command prints.
When every chunk is done the site records the proof, attributed to everyone
who ran a chunk.

## 1. One-time setup (10 minutes)

You need a C compiler, Python 3, git and zlib headers.

* **Linux (Debian/Ubuntu)**: `sudo apt install build-essential git python3 zlib1g-dev`
* **macOS**: install the Xcode command-line tools (`xcode-select --install`).
* **Windows**: use WSL (Ubuntu) and follow the Linux line inside it.

Then:

```bash
git clone https://github.com/DarthCalculus/Pathology-Puzzle-Searcher.git
cd Pathology-Puzzle-Searcher/backsearch
./build_pgo.sh -o backsearch_worker_nt --no-torch
```

The build takes a minute (it runs a short profiling search and rebuilds).
If it fails, the plain build works too:

```bash
cc -O3 -o backsearch_worker_nt backsearch.c sokoban_bfs.c nn_stub.c -lz -lm
```

Check it: `./backsearch_worker_nt --grid 4x5 --exit 0 --max-depth 16 --time 20`
should end with `accepted: 1289049` and `best depth: 16`.

## 2. Sign up

On the tracker, enter your name and how many processes you want to run
(one per CPU core is fine; leave a core free if you use the machine), then
either click **Sign up** on specific chunks or type a number into
"Sign me up for N chunks" and press Go. The page shows a command like

```
python3 run_chunks.py --chunks 12,13 --workers 4 --name "Your Name"
```

Run it from the `backsearch` directory. "Show my command" reprints it any
time. Chunks are estimated at a few CPU-hours each, but the estimates are
rough: some will finish in minutes, a few may run for days.

## 3. While it runs

* Progress lines show jobs done, CPU time and the best length so far.
* **Ctrl-C is safe at any time.** Each worker prints a checkpoint and the
  run stops; the same command continues exactly where it left off. Laptop
  sleep, reboots and crashes are also fine: re-run the command.
* Keep the checkout unmodified. The runner refuses to report from a modified
  `backsearch.c`, `sokoban_bfs.c` or `campaign.py`, and `git pull` before a
  new sign-up is a good habit (rebuild after pulling).

## 4. Report

When the command finishes it prints a block between
`-----BEGIN CHUNK REPORT-----` and `-----END CHUNK REPORT-----` (also saved
under `results/chunks/`). Paste the whole block into "Report finished chunks"
on the tracker. That marks your chunks done with the longest level found,
the states searched and the time it took.

If you want to stop for good before a chunk finishes, run the command once
more and Ctrl-C immediately, then paste the report it prints anyway: it is
marked partial, with the unfinished ranges, so someone else can take over,
and you can release the chunk with the Release button.

## What a chunk is

The search is a depth-first walk of a tree of puzzle states in a fixed order.
A chunk is a contiguous range of that order, given by two paths; its interior
cut points are the sub-ranges your processes run in parallel. Every range is
searched exactly once across all chunks, so the union of all reports is the
full exhaustive search.
