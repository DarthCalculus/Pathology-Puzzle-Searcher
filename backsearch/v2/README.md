# Collective search v2

The distributed layer around the `backsearch` worker. Design and contract:
[DESIGN.md](DESIGN.md). Status and live boards: the site's `hunt.html` page.

## Volunteers

One-time setup (Linux, macOS, or WSL on Windows):

```
git clone https://github.com/DarthCalculus/Pathology-Puzzle-Searcher.git
cd Pathology-Puzzle-Searcher/backsearch
./build_pgo.sh -o backsearch_worker_nt --no-torch      # a few minutes
```

Run (leave it running; it fetches its own work):

```
python3 v2/volunteer.py --name "Your name" --workers 4
```

* `--workers` = how many search processes to run. Default is your core count
  minus one; each uses about 200 MB of RAM. You can change it while running.
* A page opens at http://127.0.0.1:8765/ showing, for each worker, the node it
  is expanding, the best level of its current job and the best level found in
  the last second, plus totals. Closing the page does not stop the search
  unless you press **Stop**.
* **Pause** freezes the workers instantly (memory stays allocated) and keeps
  your jobs reserved for up to 12 hours. **Resume** continues where they were.
* **Stop** (or Ctrl-C once) lets each worker print the exact remaining work of
  its job, reports it to the server, and exits. Nothing is lost; the remaining
  pieces become new jobs for anyone. A second Ctrl-C kills the workers and
  leaves the jobs to expire (they are re-issued after an hour).
* Laptop sleep is fine: on wake the client checks which jobs it still holds.
* If the server is unreachable, finished reports queue in `volunteer_outbox/`
  and are delivered later; work continues on the jobs already leased.
* When the worker code changes you will be told to `git pull && ./build_pgo.sh`
  (the server accepts only whitelisted worker versions).
* Headless machines: add `--no-ui`. Several clients on one machine: give each
  its own `--outbox DIR` and `--port`.

## Owners

Seed a campaign (on the server, in `PathologyRecords/server`):

```
node tools/v2_seed.js --campaign-file plan.json --layer-file roots.tsv [--dry]
node tools/v2_hashes.js add <src-hash>        # from `backsearch_worker_nt --version`
node tools/v2_hashes.js remove <hash> --reissue   # after a correctness fix
node tools/v2_finish.js [--publish]           # refuses any exit whose audit is not clean
```

`roots.tsv` is the worker's `--list-layer K` output; for the 5x5 ≤3-hole
campaign the six exits' depth-8 layers are in `roots_h3/` (297,943 roots;
`cat roots_h3/*.tsv > roots.tsv`). `plan.json` fields are listed at the top of
`v2_seed.js`; the campaign parameters chosen in DESIGN.md §6 are the defaults.

Audit anytime: `GET /api/v2/audit?exit=E` lists uncovered roots, jobs done
under a hash that is no longer whitelisted, and fingerprint mismatches. A proof
is published only when every exit's audit is clean.

## Tests

```
python3 v2/test_split_exact.py ./backsearch_worker_nt --split 0.15 -- --grid 5x5 --exit 12 --num-blocks 3 --allow-exit-transit
bash v2/test_chaos.sh ./backsearch_worker_nt          # needs node and the server checkout
bash v2/test_client.sh                                 # fake server + fake worker
node --test ../PathologyRecords/server/test/v2_gates.test.js
```

Every test runs one worker at a time and finishes in about a minute.
