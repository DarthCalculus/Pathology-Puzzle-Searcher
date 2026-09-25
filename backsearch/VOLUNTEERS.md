# Helping with the collective search

The site https://pathology.georgespahn.com runs exhaustive searches for the
longest Pathology levels, one *campaign* at a time (the current one is shown on
https://pathology.georgespahn.com/collective.html). You run one program, the
volunteer client. It fetches small pieces of the search from the server, runs
them on as many cores as you allow, and reports back by itself. Everything is
controlled from a page it opens in your browser. Stopping is always safe:
unfinished work is handed back within seconds and nothing is lost. Proofs that
come out of a campaign are credited to "Collective", and every volunteer is
listed with their CPU-hours.

## 1. One-time setup (10 minutes)

You need a C compiler, Python 3.9 or newer, git and the zlib headers.

* **Linux (Debian/Ubuntu)**: `sudo apt install build-essential git python3 zlib1g-dev`
* **macOS**: install the Xcode command-line tools (`xcode-select --install`).
  If you use Python from python.org and the client reports certificate errors,
  run its `Install Certificates.command` once.
* **Windows**: use WSL (Ubuntu) and follow the Linux line inside it.

Then:

```bash
git clone https://github.com/DarthCalculus/Pathology-Puzzle-Searcher.git
cd Pathology-Puzzle-Searcher/backsearch
./build_pgo.sh -o backsearch_worker_nt --no-torch
```

The build takes a few minutes (it runs a short profiling search and rebuilds).
If it fails, a plain build works too (slower, but it carries the source hash
the client checks; without `-DSRC_HASH_STR` the client refuses the binary):

```bash
cc -O3 -DSRC_HASH_STR="\"$(./build_pgo.sh --print-hash)\"" -o backsearch_worker_nt backsearch.c sokoban_bfs.c nn_stub.c -lz -lm
```

Check it: `./backsearch_worker_nt --version` prints an `SRC_HASH` line and
`PROTOCOL 3`.

## 2. Start the client

From the same `backsearch` directory:

```bash
python3 v2/volunteer.py --name "Your name" --workers 4
```

* `--name` is only a label (there are no accounts); it is fixed after the first
  registration.
* `--workers` is the number of search processes, one core each. The default is
  half your logical cores. Each worker uses about 0.5 GB of memory: keep
  workers x 0.5 GB well below your free memory. One per core is fine on a machine
  you are not using; leave a core or two free if you are.

Your browser opens http://127.0.0.1:8765/, the control panel. It shows what each
worker is doing, the longest levels found on your machine (last second, last
hour, this session) and the longest known for the campaign (with a copy button
for its level code), your statistics, and the campaign table: roots covered per
exit, whether an exit is already *exact*, and the unresolved candidates still to
check when the server reports them. There is no percentage of the whole search,
because the total amount of work is unknown until it is done.

**Closing the page stops the client** like Stop. Start it with
`--no-stop-on-close` to keep it running without the page, or with `--no-ui` on a
machine without a browser (`--no-browser` serves the page without opening it).

## 3. While it runs

* **Workers**: type a number in the panel and press Enter (or Apply). Extra
  workers finish their current job first; new ones start with the next lease.
  The panel says when the campaign limits the number.
* **Exit**: choose which exit position to work on, or leave "Any exit". It
  applies to the next lease; if that exit has no open jobs, the client takes
  jobs of other exits and says so.
* **Pause** freezes the workers instantly (their memory stays allocated) and,
  while the client stays online, keeps their jobs reserved (up to 12 hours after
  each lease; the panel shows the campaign's figure). **Resume** continues where
  they were.
* **Stop**, Ctrl-C once in the terminal, or closing the terminal: every worker
  prints the exact unexplored part of its job, the client reports it and exits.
  A second Ctrl-C kills the workers at once; finished reports stay on disk and are
  sent at the next start, and the unfinished jobs return to the pool after their
  lease expires. Ctrl-Z freezes the workers together with the client; `fg`
  continues both.
* Laptop sleep is fine: on wake the client checks which jobs it still holds.
* If the server is unreachable, finished reports wait in `v2/volunteer_outbox/`
  and are delivered when it answers again.
* The client keeps at most two jobs per worker (the one running and the next), so
  no job waits on your machine while someone else could run it.

## 4. When the campaign is complete

The client notices by itself: it stops asking for work, delivers its last
reports, prints a closing summary (your jobs and CPU-hours, and the result for
each exit) in the terminal and in the panel, and **exits with code 0**. The panel
then says you can close the window. Start it with `--keep-going` to wait for the
next campaign and join it automatically instead. If no campaign is running when
you start it, the client says so and exits 0.

## 5. Keeping up to date

The server accepts only whitelisted worker builds and recent clients. When an
update is needed, the client tells you and exits with code 2. Then, in the
`backsearch` directory:

```bash
git pull
./build_pgo.sh -o backsearch_worker_nt --no-torch
```

and start the client again. Stop the client before you rebuild. Keep the checkout
unmodified: the client refuses to run from a modified `backsearch.c`,
`sokoban_bfs.c`, `sokoban_bfs.h` or `v2/volunteer.py`.

## 6. More than one client on a machine

Each client needs its own outbox and panel port, for example
`python3 v2/volunteer.py --name "Your name" --outbox v2/volunteer_outbox2 --port 8766`.
Each outbox gets its own registration, so the two clients never share a token. A
second client on the same outbox refuses to start and tells you this.

More detail: [v2/README.md](v2/README.md) and the design notes in
[v2/DESIGN.md](v2/DESIGN.md).

## History

Until September 2026 the collective proof used a chunk tracker (`run_chunks.py`,
sign-ups by chunk, pasted reports). That tracker is retired; `run_chunks.py` no
longer works with the server.
