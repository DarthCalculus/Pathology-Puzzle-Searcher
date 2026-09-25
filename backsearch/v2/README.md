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
* If the server is unreachable, finished reports queue in `volunteer_outbox/`
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

Seed a campaign (on the server, in `PathologyRecords/server`):

```
node tools/v2_seed.js --campaign-file plan.json --layer-file roots.tsv [--dry]
node tools/v2_hashes.js add <src-hash>        # from `backsearch_worker_nt --version`
node tools/v2_hashes.js remove <hash> --reissue   # after a correctness fix
node tools/v2_finish.js [--publish]           # refuses any exit whose audit is not clean
```

`roots.tsv` is the worker's `--list-layer K` output; for the 5x5 ≤3-hole
campaign the six exits' depth-4 layers are in `roots_h3/` (2,761 roots, see DESIGN §7.1;
`roots_h3/layer4_all.tsv`). `plan.json` fields are listed at the top of
`v2_seed.js`; the campaign parameters chosen in DESIGN.md §6 are the defaults.

Audit anytime: `GET /api/v2/audit?exit=E` lists uncovered roots, jobs done
under a hash that is no longer whitelisted, and fingerprint mismatches. A proof
is published only when every exit's audit is clean.

## Tests

```
python3 v2/test_split_exact.py ./backsearch_worker_nt --split 0.15 -- --grid 5x5 --exit 12 --num-blocks 3 --allow-exit-transit
bash v2/test_chaos.sh ./backsearch_worker_nt          # needs node and the server checkout
bash v2/test_client.sh                                 # fake server + fake worker (test 'ui' needs node)
node --test ../PathologyRecords/server/test/v2_gates.test.js
```

Every test runs one worker at a time and finishes in about a minute.
