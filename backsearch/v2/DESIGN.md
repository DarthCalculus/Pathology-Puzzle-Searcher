# Collective search v2: server-scheduled subtree jobs

Status: design frozen 2026-09-23 for the restart. This file is the contract
between the three components. Change it here first, then the code.

## 1. Principles

1. A **job is one node's whole subtree**, named by its seed path (the
   `--seed-path` text form: tokens `[URDL][123]`, comma separated). A subtree
   exists in every run, so there is no ordering, no cut points, no fallbacks.
2. **Splitting is exact by construction.** When a worker stops early it prints
   the node it was expanding plus its *entire* pending stack as seed paths.
   Those subtrees are disjoint and their union is exactly the unexplored part.
3. **The server owns the queue.** Jobs form a tree (parent = the job that was
   split). An exit is proven when every root job is *covered*: exhausted, or
   split with all children covered.
4. **Every report carries the worker's compiled-in source hash.** The server
   whitelists hashes. A worker fix means re-issuing only jobs done under the
   old hash.
5. **Jobs are kept small (target 20 min, split at 30 min)** so the dedup
   tables never evict and a job's `(states, valid, best)` is a deterministic
   fingerprint of `(seed, hash)`. About 2 % of finished jobs are re-issued to
   a different client and the fingerprints compared.
6. **Trusted friends, untrusted input.** No accounts, but every field from a
   client is validated, size-capped and escaped before it reaches the DB or a
   page.

## 2. Worker (`backsearch`, C)

New flags and outputs; everything else unchanged. `--from/--until` stay for
local experiments but are NOT used by the client.

- `--split-after S` : after S seconds of search (checked at pop boundaries,
  like `--time`), stop and print the remaining work. SIGINT/SIGTERM do the
  same immediately. `--time 0` remains "no cap".
- `--status-every MS` : print a `STATUS` line at most every MS milliseconds.
- `--version` : print `SRC_HASH` and exit 0.
- Seed replay sets `adjflags` at every step (same prune strength as the DFS).

Output lines (tab separated, one per line, everything else is free text):

```
SRC_HASH\t<hex>                      first line of every run (also printed
                                     inside SUMMARY as src_hash)
REMAINING\t<seedpath>                one per pending subtree (cursor first,
                                     then the stack top-down); only on split
LEVEL\t<depth>\t<code rows joined by '/'>  the best VALID level (no block on
                                     exit) of this run, Pathology code chars
                                     (0 floor,1 wall,3 exit,4 player,5 hole,
                                     block letters by push mask); once, before
                                     SUMMARY; absent if nothing accepted
STATUS\t<json>                       {"depth":d,"cur":"rows/…","best":b,
                                     "win_best":w,"win":"rows/…"} where cur is
                                     the node being expanded rendered like a
                                     level with '?' for uncommitted cells, and
                                     win_* is the deepest accepted state since
                                     the previous STATUS line
SUMMARY\t<json>                      {"status":"exhausted"|"split",
                                     "seed":"…","exit":e,"states":n,
                                     "accepted":n,"valid":n,"best":d,
                                     "evict_shallow":n,"evict_recent":n,
                                     "elapsed":s,"solver_calls":n}
```

Rules: `SUMMARY` is the last *protocol* line (the worker's human-readable
final block may follow it; clients ignore non-protocol lines). `SRC_HASH` is
the first line. `status` is `exhausted` only if the
queue drained. A run killed before `SUMMARY` has no result; the client treats
the job as untouched. `REMAINING` lines are complete, never sampled; a stack of
N entries prints N+1 lines. The exit's root is the empty seed "": a client runs
it by omitting `--seed-path`. `SRC_HASH` = sha256 over `backsearch.c`,
`sokoban_bfs.c`, `sokoban_bfs.h` (`build_pgo.sh` computes it and passes
`-DSRC_HASH_STR`); the hash is of the source, so PGO and compiler differences
do not matter.

## 3. Server (Node, `server/lib/jobs.js`, DB file `data/jobs.db`)

Separate SQLite file (WAL, `busy_timeout 5000`, every write in a transaction).

Tables

```
campaigns(id, title, grid, extra_json, exits_json, layer, hashes_json,
          max_clients, workers_max, job_target_s, split_after_s,
          lease_s, paused_max_s, dup_fraction, status open|closed, created_at)
clients(token PK, campaign_id, name, workers, created_at, last_seen,
        paused INT, boards_json, revoked INT)
jobs(id PK, campaign_id, exit, seed TEXT, parent_id NULL, depth,
     status open|leased|done|split, client_token NULL, lease_until NULL,
     leased_at, done_at, src_hash, states, accepted, valid, best,
     level_code NULL, elapsed_s, dup_of NULL, created_at)
reports(id, job_id, client_token, src_hash, summary_json, received_at)
```

Indexes: `jobs(campaign_id, status, exit)`, `jobs(client_token, status)`,
`jobs(parent_id)`.

Routes (all JSON; POST bodies ≤ 4 MB; per-token rate limit 30/min)

- `POST /api/v2/register {name, workers}` → `{token, campaign, queue_position?}`.
  Fails with 503 + `queue_position` when active clients ≥ `max_clients`
  (active = `last_seen` within 3 min and not revoked). Name 1–40 chars.
- `POST /api/v2/heartbeat {token, workers, paused, boards:[{depth,cur,best}]}`
  → `{ok, leases:[job ids still valid], revoked?}`. Updates `last_seen`,
  clamps `workers` to `workers_max`, stores at most `workers` boards of at
  most 200 chars each. Paused leases are extended up to `paused_max_s`.
- `POST /api/v2/lease {token, n}` → `{jobs:[{id, exit, seed}]}`. Grants at most
  `3 × workers − currently leased` jobs, oldest open first, preferring the
  exit with the most open work. Sets `lease_until = now + lease_s`.
- `POST /api/v2/report {token, job_id, src_hash, summary, level?, remaining?}`
  - `src_hash` must be whitelisted, else 409 `unknown_hash` (report stored,
    job returned to open).
  - `summary.status == "exhausted"`: job → done with fingerprint fields.
  - `summary.status == "split"`: `remaining` must be a non-empty list of seed
    paths, each of which extends the job's seed (token-prefix check) and
    parses; else 400 and the job returns to open. Otherwise job → split and
    one open child per entry (depth from token count).
  - A report for a job not leased by this token is accepted if the job is
    still open/leased (first report wins) and ignored with 200 `{dup:true}` if
    the job is already done; a `dup_of` job's fingerprint is compared and any
    mismatch is stored on the job (`mismatch=1`) and logged.
  - `level`, when present, is validated: parses (`parseLevel`), grid equals the
    campaign grid, holes ≤ cap, blocks ≤ cap, single exit at the job's exit
    cell (after canonical exit mapping), player present. Then re-solved in the
    background with a 30 s budget; `verified_moves` stored.
- `GET /api/v2/status` → dashboard JSON: per exit open/leased/done/split
  counts, CPU hours, best per exit with level, active clients with boards,
  hashes.
- `GET /api/v2/audit?exit=E` → recomputes coverage from the job tree and lists
  uncovered roots, unknown-hash jobs, fingerprint mismatches.
- Lease expiry: a sweeper every 60 s returns `leased` jobs past `lease_until`
  to `open` (unless the client is paused and within `paused_max_s`).

Tools: `tools/v2_seed.js --campaign-file plan.json --layer-file layer.tsv`
creates the campaign and root jobs from a worker `--list-layer K` listing;
`tools/v2_hashes.js add|remove <hex>`; `tools/v2_finish.js` publishes a proof
per exit only when `audit` is clean for that exit.

Dashboard page `public/hunt.html` (+ `hunt.js`, reuses the tile renderer and
`.lvlgrid/.tile` CSS): totals, per-exit table, best levels, the live wall of
client boards (≤ max_clients), and the one-line install/run instructions.

## 4. Client (`volunteer.py`, Python 3.9+, stdlib only)

```
python3 volunteer.py --name "Your name" [--workers N] [--server URL] [--port 8765]
```

- Registers (or reuses the token in `~/.pathology_volunteer.json`), then runs
  a lease loop per worker: lease up to 3 jobs ahead, run
  `backsearch_worker_nt --grid G --two-tables --exit E --seed-path S --time 0
  --split-after <split_after_s> --status-every 250 <extra…>`, parse the lines
  above, POST the report. Reports that fail to send are queued in
  `volunteer_outbox/` and retried with backoff; the loop continues.
- Heartbeat every 30 s with one board per worker. On wake from sleep (gap
  > 2 × lease_s) it re-validates leases before letting workers continue and
  kills workers whose lease is gone.
- Controls: **Pause** = SIGSTOP all workers, heartbeat `paused:true`;
  **Resume** = SIGCONT; **Stop** = SIGINT all workers, wait for `SUMMARY`,
  report the splits, exit. Ctrl-C and window close = Stop. Changing
  `--workers` at runtime lets surplus workers finish their current job.
- Local GUI: the client serves `http://localhost:PORT/` (stdlib
  `http.server`, no dependencies) with `volunteer_ui.html`: fields name and
  workers, Pause/Resume and Stop buttons, per worker three boards (current
  node with '?' cells grey, best this job, best in the last second) with
  depths, totals (jobs done, CPU hours, best per exit, server connection).
  The page polls `/state` every 250 ms. The UI never talks to workers; it
  calls `/pause`, `/resume`, `/stop`, `/workers` on the client only. The
  client runs fine with no browser open (`--no-ui`).
- Refuses to run if the worker's `--version` hash is not in the campaign's
  whitelist (fetched at register) and prints the update command.
- Startup check: `git status --porcelain` clean for `backsearch.c`,
  `sokoban_bfs.c`, `sokoban_bfs.h`, `volunteer.py`; override with
  `VOLUNTEER_ALLOW_DIRTY=1` (only for development).

## 5. Tests (before any volunteer runs it)

1. **Split exactness** (worker, `v2/test_split_exact.py`): on 4x4 exit 0 and
   5x5 exit 12 ≤3 blocks, `--split-after S` (or SIGINT) repeatedly on the
   resulting REMAINING seeds until all exhaust. Invariant: the set of
   canonicalised valid levels (`BS_TRACE_VALID`, depth + code, min over the 8
   symmetries) found by the union of jobs equals the full run's, and the best
   depth agrees. Visit sets are NOT the invariant: the dedup key ignores
   walk-history flags, so a subtree job expands some twins the full run skipped
   and skips some it expanded; both dominate the same levels. Passed
   2026-09-23 (5 runs, 0 levels lost, 106,436 distinct levels on the 5x5 case).
2. **Chaos** (client+server, local, `v2/test_chaos.sh WORKER`): real node
   server on a temp `jobs.db`, real worker, two real clients; one client is
   SIGKILLed (workers orphaned, leases expire in 5 s) and one is stopped with
   SIGINT (hands back REMAINING) and both restart; runs until the audit is
   clean. Asserts: at least one job split, audit clean, union of the workers'
   canonical valid-level traces == monolithic run (levels at or below the root
   layer), best equal. Passed 2026-09-23 on 5x5 exit 7 ≤3 blocks (46 layer-2
   roots → 72 jobs, 473,034 levels, best 58, 62 s wall).
3. **Gates** (server, `server/test/v2_gates.test.js`, `node --test`): 12
   tests covering register/queue, lease caps, unknown hash, bad remaining,
   foreign level, duplicates and fingerprint mismatch, sweeper, audit, HTTP
   413/400/429/503. Passing 2026-09-23.
4. **Fingerprint duplicates**: re-issued jobs match (covered in the gate tests;
   the chaos run re-issued 5 of 71 done jobs with `dup_fraction` 0.05, no
   mismatch).
5. **Client unit tests** (`v2/test_client.sh`, fake server + fake worker):
   clean stop, kill -9 and restart, pause/resume/stop via the local HTTP
   controls, worker dying without SUMMARY (loud log, backoff, nothing
   reported).

Known follow-ups: a `POST /api/v2/release` route so a stopping client can
hand back leased-ahead jobs immediately (today they idle until `lease_s`);
the server should also re-check leases on every heartbeat, not only on wake.

## 6. Parameters for the 5x5 ≤3 holes campaign

grid 5x5; extra `--allow-exit-transit --num-holes 3`; exits 0,1,2,6,7,12;
layer 8; max_clients 40; workers_max 32; job_target 20 min; split_after 1800 s;
lease 3600 s; paused_max 12 h; dup_fraction 0.02.
