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
   Fingerprints are compared within one hash only. Across worker versions
   they agree only for jobs with no inconclusive (UNKNOWN) checks: solver
   speed-ups that change which checks run or how big a table may grow (the
   W4 growable tables, bigger parent tables, dead-end children decided
   without a check) can decide a state that an older worker left UNKNOWN,
   which changes that job's valid and unknown counts (never loses a level:
   UNKNOWN states are explored and reported as candidates).
   The dedup key is 96 bits (a 64-bit hash plus an independent 32-bit check
   value, worker W4 on): a false dedup match would silently skip a subtree;
   the chance of one anywhere in a campaign is about 1e-12 (with 64 bits
   alone, about 1 %). `v2/test_dkey.py` checks that the check value is
   really compared.
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
- `POST /api/v2/heartbeat {token, workers, paused, boards:[{depth,cur,best}],
  holding?:[job ids running or queued]}`
  → `{ok, leases:[ids held after this call], drop:[ids to abandon],
  reclaimed:[ids handed back], campaign, me, exits (object keyed by exit)}`.
  Updates `last_seen`, clamps `workers`, stores ≤ `workers` boards.
  **Lease renewal (2026-09-24):** every heartbeat pushes the deadline of every
  held lease to now + `lease_s` (paused: capped at leased_at + lease_s +
  paused_max_s), so a lease expires only `lease_s` after a client's LAST
  heartbeat, never while it is alive. The original design started the clock
  at grant time; with up to 200 jobs leased ahead, heavy members expired while
  still queued and were re-run by others (3 % of CPU on day one). `holding`
  ids that expired and are still open are re-leased to the caller
  (`reclaimed`); ids held by another client or finished come back in `drop`
  and the client kills that worker without reporting.
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
  a lease loop per worker: at most 2 jobs per worker held (running + queued, §7.6), run
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
6. **Dedup-key sensitivity** (worker, `v2/test_dkey.py WORKER`, ~50 s): two
   test-only builds keep 20 bits of the dedup hash so distinct states collide
   thousands of times per run. With the 32-bit check value compared, levels
   and counts must equal the tested worker's; with it disabled, the same runs
   must lose states and levels (two-table and single-table dedup). Passed
   2026-09-25.

Known follow-ups: a `POST /api/v2/release` route so a stopping client can
hand back leased-ahead jobs immediately (today they idle until `lease_s`);
the server should also re-check leases on every heartbeat, not only on wake.

## 6. Parameters for the 5x5 ≤3 holes campaign

grid 5x5; extra `--allow-exit-transit --num-holes 3`; exits 0,1,2,6,7,12;
layer 4 (as seeded: 2,761 roots in `roots_h3/`; §7 rule 1: seed shallow); max_clients 40;
workers_max 32; job_target 20 min; split_after 1800 s; lease 3600 s; paused_max 12 h;
dup_fraction 0.02.

(This section first said layer 8. Until 2026-09-24 `--list-layer` cut the tree with bulk
walk-back off, and at K >= 5 its roots lost subtrees; campaign #1 used layer 4, which that
defect did not affect. The listing now cuts the job tree and is exact at every K: a root is a
job-tree node of depth >= K whose parent is shallower (a bulk walk-back child can be a root
deeper than K), and the header's shallow_best covers the levels above the layer; PROTOCOL3 §2.4.
For campaign #2 (6x6, 0 holes, exits 0,1,2,7,8,14) the layer-4 listing has 1,368 roots.)

## 7. Load rules (added 2026-09-23 after the load analysis; these override §3/§4 where they differ)

Server load must depend on wall-clock only, never on how many tree nodes a
job turns out to have. Job durations range from 1 ms to hours; a worker on
trivial jobs completes ~20/s (process spawn), so "one job = one server event"
is not acceptable.

1. **Shallow roots.** Campaigns are seeded at layer 3 or 4 (hundreds to a few
   thousand roots per exit), never at a layer where most roots are trivial.
2. **One report per worker per window.** A worker leases a job J and works a
   LOCAL stack for a window of `split_after_s` seconds: run J; if it splits,
   push its REMAINING seeds locally (LIFO) and keep popping, each run capped
   at the time left in the window. Trivial children finish locally and never
   reach the server. At the window's end the worker reports J once with the
   whole local tree (§7.5). A window therefore contains at most ~2 splits and
   a report at most a few hundred nodes.
3. **Absorb trivial children before hand-back.** Before reporting, spend up to
   `absorb_total_s` (120) probing the still-open local seeds for
   `absorb_probe_s` (10) seconds each, **deepest first** (REMAINING lists the
   cursor and the top of the stack first: the cheap entries; review M18). A
   probe that exhausts is reported done. A probe that splits is kept as a split
   node with its summary and its REMAINING children (none of its time is lost),
   and the pass then continues with that probe's own children only: every seed
   still to come is no deeper than the one that split, i.e. larger. A probe that
   cannot expand its root in the probe time ends the pass. When the budget (or
   a stop) ends the pass inside the children of the last kept split, that split
   stays a split only if the work kept under it (its probe plus its children's
   probes) is at least 1 s per extra open job it creates (children never probed,
   minus one); otherwise it goes back to one `open` node like a fold-back (its
   children dropped, its deepest level kept). This matters for the budget-cut
   last probe, which gets only what is left, nearly always splits, and would
   otherwise hand its whole REMAINING (mostly trivial) to the pool; the work
   dropped instead is less than 1 s per job avoided. The open pool still gets the
   unprobed siblings at each level of the chain (no deeper than a child that
   did not finish in a probe, so mostly large, a few trivial) and any child the
   budget did not reach under a split that saved enough.
   (Was 2 s / 60 s; raised after the first load test measured ~1 ms of server
   time per row: the worst-case job rate is now 1,280 / 10 s = 128 jobs/s.)
4. **The server sets the window.** The lease response carries `split_after_s`:
   `ramp_split_after_s` (120) while open jobs < 3 × (sum of workers of active
   clients), else the campaign's `split_after_s` (1800). The client takes a
   window's length (and its absorption budget) when the window **starts**, from
   the latest lease response (or a heartbeat `window`, when the server sends
   one), not from the lease of that job (review M108); a fingerprint re-run keeps
   the window its grant carries. A campaign may set `min_window_s` (default 1 s,
   lower for tests) and `split_after_nodes` (adds `--split-after-nodes N`, tests).
5. **Tree reports.** `POST /api/v2/report` body:
   `{token, job_id, src_hash, nodes:[{seed, parent, status, summary?, level?}]}`
   where every `seed` extends the job's seed (token prefix, may equal it only
   for the job itself), `parent` is the job's seed or the seed of an earlier
   node in the list, `status` ∈ done|split|open; `done` requires a `summary`
   with `status:"exhausted"` and the fingerprint fields; `split` requires at
   least one node in the list whose parent it is; `open` has no summary but
   MAY carry a `level` (with its `depth`): a probe that ran out of time or a
   run that split still found real levels, and the deepest one belongs to the
   explored part, which no child will revisit — dropping it would lose a
   champion (found by the chaos test: status best 53 vs true 58). The
   job itself is the first node. All inserted in one transaction; the job's
   status becomes done or split accordingly. ≤ 5,000 nodes per report. The
   old `summary`+`remaining` shape stays accepted as the one-node case.
6. **Client-level batching.** A client sends at most one lease request and one
   `POST /api/v2/reports {token, reports:[…]}` (array of §7.5 bodies) per
   `batch_interval_s` (10), plus one heartbeat per 30 s. Lease requests are
   sized by expected work (review M108): at most 2 × workers jobs are held
   locally (running + queued; the job of a worker that retires after the
   worker count was lowered does not count), and a worker gets its next job when its current
   window has less than min(`lease_ahead_s`, window) left; a job's expected
   work is the mean window time at its depth (a depth nothing is known about
   counts as a whole window), and an idle worker always gets one. An idle worker
   with nothing queued may lease sooner than `batch_interval_s` (at least 2 s
   apart). The local queue runs the server's lease order: largest `est_s`
   first when the grant carries it (then shallowest, then grant order), and
   grant order for grants without it (the server sent them in its order). The
   server caps at `lease_cap` (200) per request and 3 × workers + 200 held.
   Failed batches go to the outbox as one file.
7. **No throttling between friends.** No per-token rate limit. The per-IP
   limit is a sanity cap honest clients cannot reach (600/min).
8. **Nothing scales with rows on the request path.** `status` cached 2 s;
   `audit` cached per exit for 60 s and recomputed at most once a minute;
   level re-verification only for a new per-exit best.

Worst-case rates under these rules (40 clients, 1,280 workers): 9.3 req/s;
rows/s ≤ 128 one-node reports + 0.7 splits × 300 children ≈ 340 rows/s;
≤ 100 KB per report; ≤ 2M rows for a 5,000 CPU-hour campaign.
Realistic deployment (owner, 2026-09-23): 10 clients × 16 workers = 160
workers; the 40 × 32 cap is a configuration ceiling, not a load target.
Acceptance by `v2/test_load.sh` before launch: at 10 × 16 with 10-s jobs,
report p95 latency < 1 s and server CPU < 30 %; at 30 × 16 with 2-s jobs
(3× stress) zero errors or timeouts; 40 × 32 is measured for information.
First measurement (2026-09-23, before optimisation, 2-s jobs): ~1,000 rows/s
saturated one core, report p95 25 s, socket timeouts. Not acceptable; see
`synchronous=NORMAL` + one transaction per batch below.
9. **Server write path.** `PRAGMA synchronous = NORMAL` (WAL; a power loss can
   drop the last committed reports, which only returns those jobs to the pool
   when their leases expire, so it is safe), one transaction per report batch
   (validate every element first, apply all valid ones together), prepared
   statements only, no per-node JSON re-parsing on the hot path.

## 8. Control-panel GUI (added 2026-09-23 from the owner's brief)

The local page is a control panel, not a status readout. It must let a
volunteer understand the system and steer it without reading any docs.

Server support:
- `lease {token, n, exit?}`: `exit` is a preference. Grant from that exit's
  open jobs first; if it has none, grant from any exit and set
  `exit_fallback: true` in the response so the panel can say so.
- `heartbeat` response gains `me`: `{jobs_done, splits, nodes_done, cpu_s,
  states, solver_calls, best: {moves, code, exit, at} | null, rank,
  contributors, first_seen}` computed for this token's *name* (a person may
  re-register), and `exits`: per exit `{roots, roots_covered, open, leased,
  done, split, cpu_s, best: {moves, code, by, at} | null}` where
  `roots_covered` is the audit's covered-root count (from the 60 s cache).
  "Done" for an exit is reported as covered roots / roots plus CPU hours;
  total work is unknown, so no percentage of work is ever shown.

Panel sections (top to bottom):
1. **Controls**: name (fixed), workers (number input, apply), exit selector
   (Any / 0 / 1 / 2 / 6 / 7 / 12 with each exit's covered-roots and open jobs
   in the label), Pause/Resume, Stop. Every control that is not instantaneous
   shows a "working…" state until `/state` reflects it: workers change ("2
   workers finishing their jobs…"), exit change ("current jobs finish first;
   new leases use exit 7"), pause ("freezing…" until all workers show T),
   stop ("waiting for N workers to print their remaining work…").
2. **What is happening**: per worker: job seed, exit, window time left, local
   stack size, nodes done this window, phase (search / absorbing / reporting),
   current board (grey = undecided cells).
3. **Levels**: four boards with Pathology codes and copy buttons: best on
   this machine in the last second, in the last hour and this session (fed by
   STATUS samples and LEVEL lines, so short runs count too), and best known
   globally for the selected exit, or the longest over all exits with "Any exit"
   (from `exits`; the server joins code rows with newlines, the panel shows and
   copies them joined with `/`). Each with depth, and for the global one who
   found it and when. A board is redrawn only when it changes, never under the
   pointer or a text selection.
4. **Your stats** (from `me`): jobs done, splits handed back, trivial
   children absorbed locally, CPU hours, states searched, solver calls, your
   deepest level, rank among contributors, first seen, session uptime, jobs
   this session, mean job length, longest job this session.
5. **Campaign**: per-exit table (roots covered / roots, open, leased, CPU
   hours, best), total CPU hours, active volunteers, worker version/hash.
6. **How it works**: short explanations of a job (a subtree named by a seed
   path), windows and splitting (a job that outlives its window prints the
   exact remaining subtrees, which become new jobs; nothing is lost), pausing
   (freezes processes, keeps memory, jobs reserved for up to 12 h), stopping
   (each worker hands back its remaining work, seconds), changing workers
   (extra workers finish their current job; new ones start at once), leases
   (a job left unfinished returns to the pool after an hour), what a "done"
   fraction means and why there is no percentage of total work.

Rules: the page never talks to workers or to the campaign server directly;
it only reads `/state` and posts to the client's control endpoints. Client
fetches `/api/v2/status` at most every 30 s for the campaign table; `me` and
`exits` arrive with each heartbeat. All text from the server is escaped.

## 9. Work stealing (added 2026-09-24)

Leases are advisory; coverage is decided by reports (first report wins). When
a lease request finds the pool empty, the server re-leases jobs held by other
clients: from the client holding the most, newest lease first, preferring jobs
the holder's last heartbeat did not list in `running` (zero progress lost); a
running job is taken only when no queued job exists anywhere, at most one per
request, and first-report-wins settles it. The previous holder sees the ids in
`drop` on its next heartbeat and removes them (killing the worker if it was
running one). Counters `empty_leases` and `stolen` are in `/api/v2/status`
totals. The client sends `running` (running job ids) alongside `holding`.
No held cap is needed: hoarding is harmless while the pool is deep and is
undone automatically the moment it would idle someone.

Release by omission (2026-09-24): when a heartbeat carries `holding`, every
other lease on that token older than 120 s is returned to the pool
(`released` in the response). A restarted client reuses its token but not its
old queue; without this its abandoned queue was renewed forever. Re-issued
duplicate jobs are stealable like any other job.
