# Protocol 3: the contract for campaign #2 (6x6, 0 holes)

Written 2026-09-24 from the full review (`v2/REVIEW-2026-09-24.md`, issue ids M*/N*/C*/H*).
Every component implements exactly this. Where it differs from DESIGN.md §1-§9, this file wins;
the docs pass folds it into DESIGN.md afterwards. The next campaign is **6x6 with 0 holes**:
`--grid 6x6 --allow-exit-transit --num-holes 0`, canonical exits 0,1,2,7,8,14. Levels there are
long (best known 165 moves, 10 blocks) and most solver time goes to states wider than 64 bits.

## 1. Exactness invariant (replaces DESIGN §1.1)

For each exit let T be the true set of canonical valid levels, V the levels the jobs verified,
and U the **unresolved candidates**: states whose shortcut check ended on a capacity limit.
The search guarantees V ⊆ T ⊆ V ∪ U. An exit is **exact** when every candidate in U deeper than
the exit's verified best has been resolved offline (valid with its true length, or refuted).
Capacity results never prune: a state whose check is inconclusive is explored like an accepted
state, is not counted as valid, does not update best, and never exports a parent table.

## 2. Worker (backsearch)

### 2.1 `--version`
Prints, one per line, tab-separated:
```
SRC_HASH	<64 hex>
GIT_SHA	<sha or empty>
PROTOCOL	3
LIMITS	{"path_tok_max":1024,"max_ncells":64,"max_blocks":32,"state_bits":<max supported>}
KNOBS	{"HASH_LG2":..,"SHALLOW_LG2":..,"RECENT_LG2":..,"HP64_SIZE":..,"HTP_SIZE":..,...}
```
SRC_HASH = sha256 over backsearch.c + sokoban_bfs.c + sokoban_bfs.h **plus** the sorted list of
`-D` knob overrides passed to the build (build_pgo.sh computes it; a knob build therefore gets a
different hash and can never be whitelisted by accident). Every table-size macro has `#ifndef`.

### 2.2 Protocol lines (stdout)
- `SRC_HASH\t<hex>` first, as today.
- `STATUS\t{json}` as today, plus `"unknown":N`.
- `LEVEL\t<depth>\t<code>`: the job's verified best, as today.
- `UNRESOLVED\t<depth>\t<code>\t<path>\t<cause>` (new): a candidate (no block on the exit) whose
  check was inconclusive. cause ∈ `pq` (pending-entry cap, old -2), `probe` (probe limit / heap
  cap, old -3), `big` (big table full, new -5). The worker keeps at most 1000 candidates, the
  deepest first, and prints at the end only those **deeper than its final verified best**
  (a candidate no deeper than a verified level can never raise a maximum).
- `REMAINING\t<path>` as today. Never printed after a path overflow.
- `SUMMARY\t{json}` last. New fields: `"protocol":3`, `"cpu_s"` (user+sys CPU of the process),
  `"unknown"` (all inconclusive checks), `"unknown_pq"`, `"unknown_probe"`, `"unknown_big"`,
  `"unresolved"` (candidates printed), `"flags"` (effective search definition:
  `{"grid":"6x6","exit":14,"transit":1,"block_on_exit":0,"max_holes":0,"max_blocks":32,
  "min_walls":0,"bulk_walk":1}`), and the existing fields unchanged.

### 2.3 Statuses and exit codes
| status | exit code | meaning |
|---|---|---|
| `exhausted` | 0 | subtree fully searched (U may be non-empty) |
| `split` | 0 | stopped by --split-after / --split-after-nodes / SIGINT; REMAINING printed |
| `bad_seed` | 3 | --seed-path does not replay (or is longer than path_tok_max) |
| `path_overflow` | 4 | a node deeper than path_tok_max tokens was reached; no REMAINING |
| `error` | 5 | allocation failure or internal error; message on stderr |

`time_cap` is gone from protocol mode (--time 0 is always passed). Any other exit is a crash.

### 2.4 Behaviour
- Solver capacity: one classify(x) at every consumer: x >= 0 shortcut (prune); -1 accept;
  -2/-3/-5 UNKNOWN (explore, count, candidate). Big-table overflow returns -5 instead of retrying
  forever; the end-of-run verify is bounded (cutoff solve at best-2, or skipped after a split).
- Paths: up to 1024 tokens (`plen` uint16). A path_push failure ends the run with `path_overflow`.
- Root listing (`--list-layer K`, `--estimate`, `--estimate-dump`) uses exactly the job tree's
  semantics (bulk walk-back on), so the union of K-layer roots covers the whole tree for every K.
  It prints a header `LAYERINFO\t{"k":K,"grid":..,"flags":{..},"src_hash":..,"roots":{"<exit>":n}}`.
  As implemented (worker 27443a8 on; the fallback "refuse K >= 5" is not used, the listing is exact
  at every K):
  - A root is a job-tree node of depth >= K whose tree parent is shallower. A bulk walk-back child
    is one tree edge, so a root can be deeper than K; its depth equals its token count and its
    tokens from K on are one walk run. Every node above the layer is expanded by the listing.
  - One header per exit, before that exit's LAYER lines. Its full form:
    `{"k":K,"grid":"RxC","flags":{as SUMMARY},"src_hash":"<hex>","protocol":3,"seed":"",
    "roots":{"<exit>":n},"shallow_best":{"<exit>":{"depth":d,"code":".."} | null},
    "above":{"<exit>":n},"root_depth":{"<exit>":[min,max]}}`. `k` is the absolute layer depth
    (under `--seed-path P`, `--list-layer K` lists K below P and k = depth(P) + K); `seed` is the `--seed-path` the listing started from (`""` = the exit root; a campaign accepts
    only `""`); `shallow_best` is the deepest valid level above the layer (depth < K, verified
    like a LEVEL line; null = none), which no job reports; `above` = nodes expanded above the layer.
  - Before printing, the listing replays every root's path as its job will and requires the very
    same node (path, cells, blocks, holes, bulk_walk, walk history and segment, adjflags). A
    mismatch, a full dedup table, or an inconclusive check above the layer ends the listing with
    status `error` (exit 5); a path over path_tok_max exits 4. An exit's header and LAYER lines are
    printed only after all of its checks pass (a failing listing must never be used: check the exit code).
  - `--estimate` prints the same header and LAYER lines on stdout (so its stdout is a layer file)
    and appends the header plus one row per root to the `--estimate-dump` file, in the same
    (DFS) order.
- `--split-after-nodes N`: split deterministically after N expansions (tests).
- In protocol mode (`--status-every` given) the worker ignores `BS_DUMP_SEED` and the `BS_TRACE_*`
  variables unless `BS_ALLOW_DEBUG=1`.
- Split, time and status checks run at least every 64 expansions and also inside long bulk solves.
- States wider than 64 bits use the same optimised solver (A*, reference tables, multi-start) as
  64-bit states; widths up to MAX_BLOCKS on every grid up to 64 cells are supported, and a state
  that does not fit is an `error`, never a silent prune.

## 3. Client (volunteer.py)

- Sends `"client_version":"3.0.0","protocol":3` in register, heartbeat, lease and reports.
- Worker command: `... --allow-exit-transit` etc. **only** from the allowlist below; any other
  flag in the campaign's `extra` makes the client refuse to run with a clear message.
  Allowlist: `--allow-exit-transit`, `--num-holes N`, `--num-blocks N`, `--min-walls N`
  (N a non-negative integer).
- HTTP 426 `client_too_old` → print the server's message (it names the release) and exit 2.
- HTTP 410 `campaign_closed` (or 401/403/404 on a token) → drop the token and register again.
- **Campaign completion (new feature):** every heartbeat and lease response carries
  `"campaign_state": "running" | "complete" | "closed"`. On `complete` the client stops asking
  for work, lets running workers finish (there should be none), delivers every pending report and
  outbox file, prints a closing summary (the volunteer's jobs and CPU-hours for this campaign and
  the campaign's result per exit), tells the control panel, and **exits 0**. `--keep-going` makes it
  wait for and join the next campaign instead. If register finds no campaign (404 `no_campaign`)
  the client exits 0 with a message, unless `--keep-going`.
- Identity: the token file is keyed by server URL **and outbox path**; the client holds an
  exclusive lock on `<outbox>/.lock`, so a second instance on the same outbox refuses to start and
  says to pass `--outbox DIR`. Two instances never share a token.
- Reports: never truncate a split's children. If a tree exceeds the node or byte cap, demote the
  deepest split nodes to `open` (drop their descendants, keep their levels) until it fits; if even
  that fails, return the job untouched (release) and log loudly. Batches are capped by bytes
  (≤ 1 MB) as well as count; a 413 halves the batch.
- A run whose output has any malformed, non-extending or overlong REMAINING line, or whose status
  is bad_seed / path_overflow / error / crash / no SUMMARY, is **void**: the node is reported as a
  failure (`{"job":id,"failed":"<reason>"}` in the batch) and nothing from that run is used.
- Unresolved candidates are forwarded per node: `"unresolved":[{"depth":d,"code":c,"path":p,"cause":k}]`.
- `holding` lists every job the client still owns, including finished ones whose reports are
  pending or in the outbox. The final heartbeat on Stop sends `"stopping":true,"holding":[],"running":[]`.
- Lease grants are deduped against jobs already queued or running locally.
- Environment: strips `BS_*` from the worker environment unless `VOLUNTEER_ALLOW_DIRTY=1`.
- Watchdog: a worker that prints no STATUS for 3 × status interval + 60 s after its split time is
  sent SIGINT, then SIGKILL after STOP_GRACE_S; the job is reported as failed (`hang`).

## 4. Server (jobs.js / v2routes.js / tools)

- Campaign definition allowlist (createCampaign, v2_seed, v2_finish): `extra` must contain
  `--allow-exit-transit`; may contain `--num-holes N`, `--num-blocks N`, `--min-walls N`; nothing
  else (in particular never `--allow-block-on-exit`). Grid rows <= cols. Exits = the grid's
  canonical exits.
- Per campaign: `min_client_version` (default "3.0.0") and `min_protocol` (3); older clients get
  426 `{error:"client_too_old", release:"...", message:"..."}`.
- `campaign_state` in heartbeat and lease responses: `complete` when no job is open or leased
  (dup re-issues included), no job is quarantined, and every exit's audit is clean; the campaign
  row's status becomes `complete` (still readable; its tokens get 200 with the state, not 410).
  `closed` = closed by the owner.
- Reports: whitelist the new SUMMARY fields; store `cpu_s`, `unknown*`, `flags`; reject a report
  whose `flags` disagree with the campaign definition; store unresolved candidates in a
  `candidates` table (campaign, exit, job, depth, code, path, cause, status open/valid/invalid,
  resolved_moves). An exit is `exact` only when it is clean and has no open candidate deeper than
  its best. v2_finish refuses to publish an exit that is not exact.
- Failures: `{"job":id,"failed":reason}` increments `fail_count` and reopens the job; after 3
  failures from at least 2 clients (or 5 in total) the job becomes `quarantined`; quarantined jobs
  block the audit and are listed in it.
- Seeding: v2_seed requires the LAYERINFO header, checks its flags/hash against the plan, requires
  `seed` "" and shallow_best < K, requires exactly the listed number of roots per exit, each with
  depth == tokens >= K (a root deeper than K must end in one walk run from token K on), validates
  before `--close-others`, and stores `est_s` per root when the input is an `--estimate-dump` file.
- Scheduling: lease largest `est_s` first across exits (split children inherit
  parent est_s × their share, or fall back to depth); work stealing never steals a job granted or
  stolen within the holder's last 60 s or listed as running in its last heartbeat; a job is never
  granted to the client already holding or running it.
- Status and audits are computed off the request path on a timer and served from cache.
- `jobs.db` is backed up daily (VACUUM INTO) and copied off the VPS.
