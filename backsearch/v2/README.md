# Collective search v2

The distributed layer around the `backsearch` worker. Design and contract:
[DESIGN.md](DESIGN.md). Status and live boards: the site's `hunt.html` page.

## Volunteers

The full guide is [../VOLUNTEERS.md](../VOLUNTEERS.md). In short, one-time setup
(Linux, macOS, or WSL on Windows):

```
git clone https://github.com/DarthCalculus/Pathology-Puzzle-Searcher.git
cd Pathology-Puzzle-Searcher/backsearch
./build_pgo.sh -o backsearch_worker_nt --no-torch      # a few minutes
```

Run it from that `backsearch` directory (leave it running; it fetches its own work):

```
python3 v2/volunteer.py --name "Your name" --workers 4
```

* `--workers` = how many search processes to run, one core each. The default is
  half your logical cores. Each worker uses about 0.5 GB of memory, so keep
  workers x 0.5 GB well below your free memory; one per core is fine on a machine
  you are not using, fewer if you are. You can change it in the page while it runs
  (type the number, press Enter).
* A page opens at http://127.0.0.1:8765/ (the control panel) showing, for each
  worker, the node it is expanding and its job, the longest levels found on this
  machine (last second, last hour, this session) and the longest known for the
  campaign, your statistics and the campaign table (roots covered, exact or not,
  unresolved candidates when the server reports them). Closing the page stops the
  client like **Stop** (add `--no-stop-on-close` to keep it running without the page).
* **Pause** freezes the workers instantly (memory stays allocated). While the
  client stays online the jobs stay reserved (up to 12 hours after each lease; the
  page shows the campaign's figure). **Resume** continues where they were.
* **Stop** (or Ctrl-C once, or closing the terminal) lets each worker print the
  exact remaining work of its job, reports it to the server, and exits. Nothing
  is lost; the remaining pieces become new jobs for anyone. A second Ctrl-C kills
  the workers and exits at once: finished reports stay in the outbox for the next
  start, and the running jobs are re-issued after their lease expires. Ctrl-Z
  freezes the workers together with the client; `fg` continues both.
* **When the campaign is complete** the client stops by itself: it delivers its
  last reports, prints a closing summary (your jobs and CPU-hours, and the result
  per exit; the page shows the same and says you can close it) and exits 0. Add
  `--keep-going` to wait for the next campaign and join it instead. With no
  campaign running, the client says so and exits 0.
* The client holds at most two jobs per worker (the one running and the next), so
  no job waits on your machine that someone else could run. Each window takes the
  length the server sets when it starts (shorter near the end of a campaign).
* Laptop sleep is fine: on wake the client checks which jobs it still holds.
* If the server is unreachable, finished reports queue in `v2/volunteer_outbox/`
  and are delivered later; work continues on the jobs already leased.
* When the worker code changes you will be told to run
  `git pull && ./build_pgo.sh -o backsearch_worker_nt --no-torch` in the
  backsearch directory (the server accepts only whitelisted worker versions);
  stop the client before you rebuild. When the client itself is too old, it
  prints the server's update instructions and exits 2.
* Headless machines: add `--no-ui`. Several clients on one machine: give each
  its own `--outbox DIR` and `--port`. Each outbox has its own registration (a
  second client on the same outbox refuses to start and says so), so two clients
  never share a token.
* The worker gets a clean environment: `BS_*` debug variables are not passed on
  (set `VOLUNTEER_ALLOW_DIRTY=1` only for development).

## Owners

The `node tools/...` commands run on the server, in `PathologyRecords/server` (the full
recipe, backups and deploys are in `server/ops/README.md` there); the root listing runs with the
worker, in `backsearch/`. Proofs from a campaign are attributed "Collective"; the owner's own
proofs "Panacea".

List the roots with the release worker (built by `./build_pgo.sh -o backsearch_worker_nt
--no-torch`, whose hash the campaign whitelists), for exactly the campaign's flags, and keep
the WHOLE stdout. For campaign #2 (6x6, no holes, layer 4):

```
./backsearch_worker_nt --grid 6x6 --two-tables --allow-exit-transit --num-holes 0 --time 0 \
    --estimate 30000 --estimate-depth 4 --estimate-dump dump.tsv > roots.tsv
grep -c '^LAYER[[:space:]]' roots.tsv        # 1,368 roots: exits 0/1/2/7/8/14 = 17/116/158/188/513/376
```

(`--list-layer 4` gives the same roots without the estimate; the estimate dump gives every root
its `est_s`, which the server leases largest first.) Then:

```
node tools/v2_seed.js --campaign-file plan.json --layer-file roots.tsv --estimate-file dump.tsv --dry
node tools/v2_seed.js --campaign-file plan.json --layer-file roots.tsv --estimate-file dump.tsv [--close-others]
node tools/v2_hashes.js list | add <src-hash> | remove <hash> [--reissue]   # --reissue only after a correctness fix
node tools/v2_jobs.js quarantined | mismatches | candidates | release ...    # what blocks an exit
node tools/v2_resolve.js        # solve the open unresolved candidates deeper than each exit's best
node tools/v2_verify.js         # re-solve champions the server has not confirmed yet
node tools/v2_finish.js [--solve]   # dry run; then --publish: solve, check, store the champion, add the proof
```

- `roots.tsv` has one LAYERINFO header per exit, then its LAYER lines (DESIGN §2.8); v2_seed
  refuses a file without the headers, with `seed` other than `""`, or with a root count or depth
  that does not match. A root is a job-tree node of depth ≥ K whose parent is shallower, so a
  listing is exact at every K with the current worker; campaigns still seed at layer 3 or 4
  (DESIGN §7.1). Workers before project45 27443a8 (the live 406e5500 included) listed with bulk
  walk-back off: never seed from a layer deeper than 4 listed by one of them.
- For the 5x5 ≤3-hole campaign (#1) the six exits' depth-4 layers are in `roots_h3/`
  (2,761 roots, `roots_h3/layer4_all.tsv`; listed before the header existed, so LAYER lines only).
- `plan.json` fields and their defaults are listed at the top of `v2_seed.js`; campaign #2's
  values are in DESIGN §6.1. The extra flags are an allowlist: `--allow-exit-transit` (always),
  `--num-holes N`, `--num-blocks N`, `--min-walls N`, nothing else (never `--allow-block-on-exit`).
- Audit anytime: `GET /api/v2/audit?exit=E` (add `campaign=ID` for a finished one) lists
  uncovered roots, jobs done under a hash that is no longer whitelisted, fingerprint mismatches,
  quarantined jobs, pending duplicates and unresolved candidates, and says whether the exit is
  clean and exact. A campaign turns complete by itself when every job is finished and every exit
  is clean (clients then exit by themselves); each exit's proof
  is published once that exit is **exact** (clean, and every unresolved candidate deeper than its
  best resolved by `v2_resolve.js`).

## Tests

Run from `backsearch/`, one test at a time:

```
python3 v2/test_split_exact.py ./backsearch_worker_nt --quick        # or --suite campaign [--only NAME]
python3 v2/test_split_exact.py ./backsearch_worker_nt --split 0.15 -- --grid 5x5 --exit 12 --num-blocks 3 --allow-exit-transit
python3 v2/test_roots_exact.py ./backsearch_worker_nt --quick        # or --suite campaign | deep [--jobs N]
python3 v2/test_oracle.py ./backsearch_worker_nt --quick             # independent enumerator (builds oracle_enum.c)
python3 v2/test_dkey.py ./backsearch_worker_nt                       # 96-bit dedup key sensitivity (~50 s)
bash v2/test_chaos.sh ./backsearch_worker_nt   # needs node and the server checkout; --config 6x6h0b1, --e2e
bash v2/test_client.sh [a e h s t ui ...]      # fake server + fake worker (test 'ui' needs node)
bash v2/test_load.sh 10 16 60 10               # server load: CLIENTS WORKERS SECONDS JOB_S (also 30 16 60 2)
(cd <PathologyRecords checkout> && node --test server/test/v2_gates.test.js)   # the server's gate tests
```

Every test but the chaos and load tests runs one worker at a time and finishes in about a
minute (the campaign and deep suites take longer; run them with `--only`). The chaos test runs
a server, two clients and their workers at once (two CPUs) for one to two minutes. What each
test asserts is in DESIGN §5; the worker equivalence configs are in `../README.md`.
