# Collective search v2: server-scheduled subtree jobs

**Status: the current specification, protocol 3 (2026-09-25).** This file is the contract between the
three components (worker, client, server). Change it here first, then the code.

- It describes the `campaign2` branches: worker sources with SRC_HASH `c3ee5d2b…` (project45 9430640, the
  release candidate), client 3.1.0 (`v2/volunteer.py`), server PathologyRecords `campaign2` (3ec0a4f).
- `PROTOCOL3.md` is the change record: the protocol-3 contract as written on 2026-09-24 from the review
  (`REVIEW-2026-09-24.md`, issue ids M*/N*/C*/H*, one detail file per issue in `review_issues/`). Everything
  in it is folded in here; §10 lists where the implementation differs from what it says.
- Campaign #2 (6x6, no holes) has run live since 2026-09-24 20:50 on the previous release (worker `406e5500…`
  = project45 6991199, client 3.0.0, server 914fa20 + ee3aebf). §6.3 says what that release lacks and how the
  release candidate is rolled out during the campaign.

Owner rules, enforced by every component:
- Real Pathology rules: every campaign passes `--allow-exit-transit` (blocks may be pushed over the exit).
  `--allow-block-on-exit` is never used: no level starts with a block on the exit. The server and the client
  both refuse a campaign definition that breaks either rule.
- Proofs from a campaign are attributed **"Collective"**; the owner's own proofs **"Panacea"**.
- Volunteers are trusted friends: the system must survive crashes, kills, sleep, slow machines and variance,
  not malice. Every input is still validated.
- Published proofs are never edited or revoked.

## 1. Principles

1. **A job is one node's whole subtree**, named by its seed path (the `--seed-path` text: tokens
   `[URDL][123]`, comma separated; the exit's root is the empty seed `""`). A subtree exists in every run, so
   there is no ordering, no cut points and no fallbacks.
2. **Splitting is exact by construction.** A worker that stops early prints the node it was expanding (the
   cursor) and its entire pending stack as seed paths. Together they cover exactly the unexplored part. They
   may overlap, which costs time but never loses anything: the cursor's subtree contains its children that are
   already on the stack (review M5; the old text said "disjoint"), and a chain head (§2.6) contains the stack
   entries below it.
3. **The server owns the queue.** Jobs form a tree (parent = the job that was split). A root is covered when
   it is done under a whitelisted hash, or split under a whitelisted hash with at least one live child and
   every child covered. The levels above the root layer come from the root listing (`shallow_best`, §2.8).
4. **Exactness invariant** (protocol 3; replaces "an exhausted job is proven"). For each exit let T be the
   true set of canonical valid levels, V the levels the jobs verified, and U the **unresolved candidates**:
   states whose shortcut check ended on a solver capacity limit. The search guarantees **V ⊆ T ⊆ V ∪ U**.
   Capacity results never prune: an inconclusive state is explored like an accepted one, is not counted as
   valid, does not update the best, and never exports a parent table. An exit is **clean** when its tree is
   covered and nothing is suspicious (§3.5), and **exact** when it is clean and every candidate deeper than
   its best has been resolved offline (valid with its true length, or refuted). Only exact exits are
   published.
5. **Every report carries the worker's compiled-in source hash**, and the server whitelists hashes.
   SRC_HASH = sha256 over `backsearch.c` + `sokoban_bfs.c` + `sokoban_bfs.h`, plus the sorted list of `-D`
   knob overrides when there are any (§2.1). A correctness fix in the worker means re-issuing only the jobs done
   under the old hash (§3.8); any other new worker is added beside the old hash, which stays whitelisted
   (§6.3).
6. **Fingerprints.** A job's `(states, valid, best)` is a deterministic function of `(seed, SRC_HASH)`.
   - The dedup tables DO evict (correcting the old premise "jobs are kept small so the dedup tables never
     evict", review M10). With SHALLOW_LG2 = RECENT_LG2 = 22 the two-table dedup starts evicting at 80 % fill,
     about 3.36M inserts. In campaign #1, 1,675 of 54,409 reports evicted, and they held 63 % of the CPU-hours
     and 77 % of the states.
   - Eviction is a deterministic function of the insert sequence and the compiled table sizes. The table
     sizes are part of SRC_HASH only for builds made by `build_pgo.sh`, which folds its `KNOBS="-D..."`
     overrides into the hash (§2.1). A hand `cc` build that embeds `$(./build_pgo.sh --print-hash)` and adds
     its own `-D` overrides carries the release hash with different tables: `--version` prints the effective
     sizes in KNOBS, but neither the client nor the server checks that line (§10). So a plain build must add no
     `-D` flag beyond `SRC_HASH_STR`. Within one hash and its tables the fingerprint holds: 1,009 of 1,009
     campaign-#1 duplicate pairs matched, evicting jobs included. An eviction costs only re-exploration,
     never a level.
   - About 2 % of done jobs are re-run as duplicates on another client and compared (§3.6). Fingerprints are
     compared within one hash only. Across worker versions they agree only for jobs with no UNKNOWN checks:
     solver changes that let a newer worker decide a state an older one left UNKNOWN change that job's valid
     and unknown counts (never a level: UNKNOWN states are explored and reported as candidates).
   - The dedup key is 96 bits (a 64-bit hash plus an independent 32-bit check value, worker W4 on). A false
     match would silently skip a subtree; the chance of one anywhere in a campaign is about 1e-12 (about 1 %
     with 64 bits alone). `v2/test_dkey.py` checks that the check value is really compared.
7. **Windows split for scheduling, not for dedup.** Forced splits after `split_after_s` (1,800 s) bound what a
   crash or kill loses and keep the pool full. They cost about 3 % extra states on the subtrees they split,
   because the children lose the parent's dedup table (35 natural monolithic-vs-split pairs in campaign #1:
   +3.3 %), plus the absorption probes (§7.3). This trade-off is kept for campaign #2 (§6.1).
8. **Trusted friends, untrusted input.** No accounts, but every field from a client is type-checked,
   range-checked, size-capped and escaped before it reaches the DB or a page. Every field from the server
   that reaches a volunteer's worker command line goes through the client's allowlist (§4).

## 2. Worker (`backsearch`, C)

### 2.1 Build, identity and `--version`

- **Release build**, in `backsearch/`: `./build_pgo.sh -o backsearch_worker_nt --no-torch`.
  - It computes SRC_HASH and embeds it (`-DSRC_HASH_STR`).
  - It trains PGO on two campaign-shaped protocol-mode jobs: a 5x5 ≤3-hole layer-4 root (500k expansions) and
    a 6x6 0-hole wide subtree run to exhaustion. The build fails if a training run fails, or if the profile is
    empty or misses the hot solver instances (a cold 16-byte instance once cost 27 % on heavy 6x6 jobs).
  - `--train "ARGS"` / `--train2 "ARGS"` replace the training runs' worker arguments (`--train2 ""` skips the
    second); `NATIVE=1` opts in to `-mcpu=native` / `-march=native`; `CC=gcc-13` picks the compiler. The
    default is portable.
  - `./build_pgo.sh --print-hash` prints the hash of the current sources and exits.
  - Knob builds: `KNOBS="-DX=Y ..." ./build_pgo.sh -o OUT` (every table-size macro has an `#ifndef`). The
    sorted overrides are folded into SRC_HASH, so a knob build can never be whitelisted by accident.
  - A plain `cc` build must pass `-DSRC_HASH_STR="\"$(./build_pgo.sh --print-hash)\""`; without it the worker
    prints an empty hash, which every client refuses. It must pass no other `-D` flag: that hash is the one of
    the default tables, and nothing checks KNOBS against it (§1.6, §10). Overrides go through `KNOBS=`.
- **`--version`** prints, tab separated, one per line:
  ```
  SRC_HASH	<64 hex>
  GIT_SHA	<sha or empty>
  PROTOCOL	3
  LIMITS	{"path_tok_max":1024,"max_ncells":64,"max_blocks":32,"state_bits":261}
  KNOBS	{"HASH_LG2":24,"SHALLOW_LG2":22,"RECENT_LG2":22,"PATH_TOK_MAX":1024,...,"solver_defs":"","defs":""}
  ```
  KNOBS lists every table-size and limit macro and any test `-D` definitions. The client reads PROTOCOL
  (it must be 3) and `path_tok_max`.

### 2.2 Command line

The client runs exactly:
```
backsearch_worker_nt --grid G --two-tables --exit E [--seed-path S] --time 0 --split-after W
    [--split-after-nodes N] --status-every 250 --allow-exit-transit [--num-holes N] [--num-blocks N] [--min-walls N]
```
- `--seed-path S`: the job's node (at most 1024 tokens; the exit root `""` runs without the flag). A path that
  is malformed, too long or does not replay ends with status `bad_seed`.
- `--split-after W`: split after W seconds (fractional values allowed; the client never passes 0).
  `--split-after-nodes N`: split deterministically after N expansions (test campaigns). SIGINT or SIGTERM:
  split at the next check. `--time 0`: no cap (`--time` stays for local runs; its status `time_cap` never
  occurs in protocol use).
- `--status-every MS`: a STATUS line at most every MS ms. It also turns on **protocol mode**: the `BS_*` debug
  variables are ignored unless `BS_ALLOW_DEBUG=1` (then `BS_DUMP_SEED` ends with status `debug`, never
  coverage), and `--shortcut-state-cap`, the NN surrogate, `--beam` and `--rollout` are refused.
- `--min-walls N` is the same as `--num-walls N`. `--num-holes 0` means no holes at all.
- Split, time and status checks run at least every 64 expansions and also from inside long solves.
- `--from/--until`, `--harvest`, `--beam` and the other experiment flags stay for local work; the client never
  passes them (it cannot: §4 allowlist).

### 2.3 Protocol lines (stdout, tab separated, one per line)

Everything else on stdout is free text; clients ignore it.

```
SRC_HASH\t<hex>                          first line of every run
STATUS\t<json>                           {"depth":d,"cur":"rows/…","best":b,"win_best":w,"win":"rows/…",
                                         "unknown":n}: cur is the node being expanded rendered like a level
                                         with '?' for uncommitted cells; win_* is the deepest accepted state
                                         since the previous STATUS line
REMAINING\t<seedpath>                    only on split: the cursor first, then the stack top-down
LEVEL\t<depth>\t<code rows joined by '/'>  the run's best VALID level (no block on the exit), verified;
                                         once, before SUMMARY; absent if nothing was accepted or the run is void
UNRESOLVED\t<depth>\t<code>\t<path>\t<cause>  an unresolved candidate (§2.5), after the run
SUMMARY\t<json>                          the last protocol line
```
- Level codes use the Pathology characters: 0 floor, 1 wall, 3 exit, 4 player, 5 hole, block letters by push
  mask.
- Every REMAINING line strictly extends the seed by whole tokens (never equal, never empty) and has at most
  `path_tok_max` tokens. A stack of N entries prints N+1 lines, fewer when chain heads (§2.6) replace entries.
  REMAINING is never printed after a path overflow or for a void status.
- An UNRESOLVED line's depth equals its path's token count; `cause` is `pq`, `probe` or `big`.
- SUMMARY fields:

| field | meaning |
|---|---|
| `status` | §2.4 |
| `seed`, `exit` | the job's seed and exit |
| `states`, `accepted`, `valid`, `best` | states checked; accepted (UNKNOWN states excluded); valid levels (accepted with no block on the exit); the deepest valid level |
| `verify` | end-of-run check of `best`: a cutoff solve at best-2 with no reference table. `best` when confirmed, -1 when skipped (no best, after a split, void runs), else the raw solver code |
| `evict_shallow`, `evict_recent` | dedup evictions |
| `elapsed`, `cpu_s` | wall seconds (pauses included) and user+sys CPU seconds |
| `solver_calls` | shortcut checks run |
| `src_hash`, `protocol` | the build's hash; 3 |
| `unknown`, `unknown_pq`, `unknown_probe`, `unknown_big` | inconclusive checks, by cause |
| `unresolved` | UNRESOLVED lines printed |
| `unresolved_dropped_max` | the depth of the deepest candidate that did not fit in the 1,000 kept (0 = none dropped) |
| `flags` | the effective search definition: `{"grid":"6x6","exit":14,"transit":1,"block_on_exit":0,"max_holes":0,"max_blocks":32,"min_walls":0,"bulk_walk":1}` |
| `max_pops`, `max_pending`, `max_rss_mb`, `table_grows`, `bulk_fallbacks`, `ptab_mb` | information only (capacity headroom and memory): the largest single check, peak RSS, solver table growths, multi-start fallbacks, parent-table pool peak |
| `error` / `detail` | the message of status `error` / of another void status |

### 2.4 Statuses and exit codes

| status | exit code | meaning | the client |
|---|---|---|---|
| `exhausted` | 0 | the subtree is fully searched (U may be non-empty) | done |
| `split` | 0 | stopped by --split-after, --split-after-nodes, SIGINT/SIGTERM or the chain guard; REMAINING printed | split |
| `bad_seed` | 3 | --seed-path malformed, longer than path_tok_max, or does not replay | void, failure |
| `path_overflow` | 4 | a node deeper than path_tok_max tokens was reached; no REMAINING, no LEVEL | void, failure |
| `error` | 5 | allocation failure, a state the solver cannot pack (more than 32 blocks or 30 holes), a failed listing check, internal error; message on stderr and in SUMMARY `error` | void, failure |
| `unknown_chain` | 6 | the UNKNOWN chain guard (§2.6): a capacity cascade that starts at the job's own root; SUMMARY `detail`; no REMAINING, LEVEL or UNRESOLVED | void, failure |
| `debug` | 0 | `BS_DUMP_SEED` with `BS_ALLOW_DEBUG=1` only | void, failure |
| `time_cap` | 0 | the --time cap (never in protocol use) | void, failure |
| `dedup_full` | 0 | the single dedup table filled (--no-two-tables only; the client always passes --two-tables) | void, failure |

Two exit codes come before any output line and print no SUMMARY: 1 for an unknown or malformed argument, and
2 when protocol mode (`--status-every`) refuses `--beam`, `--rollout`, `--shortcut-state-cap` or
`--nn-surrogate-model` (a v2 job must be exhaustive), or a listing option is malformed (`--list-layer`,
`--estimate-depth`, `--estimate-dump`, `--from` / `--until`, or `--estimate` with `--task-id`). The client never
passes these. Any other end (a signal, a missing SUMMARY, exit codes 1 and 2 included) is a crash. "Failure"
means a failure report when the run is the window's own job (§4); a void child run just stays open in the tree.

### 2.5 Solver capacity: UNKNOWN and candidates

- Every consumer of a shortcut check uses one `classify(x)`: x ≥ 0 or -4 is a shortcut (prune); -1 accepts;
  -2 (the pending-entry cap, cause `pq`), -3 (probe or heap limit, `probe`) and -5 (the last solver table tier
  full, `big`) are **UNKNOWN**; -6 (`SOK_NO_FIT`) is status `error`.
- An UNKNOWN state is explored like an accepted one. It is not counted as accepted or valid, never becomes
  `best` or the STATUS win, and never exports a parent table (it keeps an inherited ancestor's table).
- An UNKNOWN state with no block on the exit is a **candidate**. The worker keeps at most 1,000, deepest first,
  and at the end prints only those deeper than its final verified best (a candidate no deeper than a verified
  level can never raise a maximum). `unresolved_dropped_max` reports the deepest one it could not keep.
  `BS_TRACE_UNRESOLVED=<file>` traces every candidate (debug only).
- The solver never retries forever: a full table returns -5, and the end-of-run verify is bounded (§2.3).
- Headroom (release candidate): the push-solver table grows in place (64K → 256K → 1M → 4M → 16M slots; every
  tier but the last takes inserts up to half its size, the last up to 85 %), and the pending cap is 1,048,576
  entries. On the 12 heaviest campaign-#1 seeds (120 s each) and on 60-s runs from every 6x6 0-hole exit root
  there was no UNKNOWN result at all; the largest single checks had at most 158,749 pops and 52,499 pending
  entries.

### 2.6 The UNKNOWN chain guard (protocol mode)

A capacity limit tends to repeat down a subtree. Each node records how many inconclusive checks in a row end
at it (`unk_run`) and where that run started (the chain head). A run without two UNKNOWN states in a row
behaves exactly as without the guard.
- At a split, every stack entry two or more steps into such a run is listed as its chain head, printed once,
  so a cascade goes back to the server as one job.
- A run of UNKNOWN_CHAIN_MAX = 8 inconclusive checks in a row forces a split at once.
- In the head's own job the head is the root and cannot be handed back, so the split waits. The run ends with
  status `unknown_chain` (exit 6) after max(UNKNOWN_DEFER_S = 60 s, --split-after) seconds or --split-after-nodes
  expansions, or at once when the forced-split length is reached. The job then fails on every client and is
  quarantined (§3.6); resolve it offline with a build that has larger caps.

### 2.7 Splitting

- A split that comes due waits for the run's second expansion, so every REMAINING line strictly extends the
  seed. A due split with an empty stack ends the run as `exhausted`.
- SIGINT/SIGTERM before the root was expanded prints REMAINING equal to the seed. The client treats that as
  "not expanded" (no result, no failure) when it stopped the worker itself, and as failure `no_progress` when
  the worker split on its own timer.
- Chain heads (§2.6) may replace stack entries; the guard may force a split early or defer one.

### 2.8 Root listing (`--list-layer K`, `--estimate`, `--estimate-dump`)

The listing cuts exactly the tree the jobs search (bulk walk-back on), so the roots plus the levels above them
cover the whole exhaustive search **for every K** (worker 27443a8 on; reviews C2, N6, N20).
- **The rule.** A root is a job-tree node of depth ≥ K whose tree parent is shallower. Every node above the
  layer is expanded by the listing. A bulk walk-back child is one tree edge, so a root can be deeper than K;
  its depth equals its token count, and its tokens from position K on are one walk run.
- **The header**, before each exit's LAYER lines:
  `LAYERINFO\t{"k":K,"grid":"RxC","flags":{as SUMMARY, without exit},"src_hash":"<hex>","protocol":3,"seed":"",
  "roots":{"<exit>":n},"shallow_best":{"<exit>":{"depth":d,"code":"…"}|null},"above":{"<exit>":n},
  "root_depth":{"<exit>":[min,max]}}`.
  - `k` is the absolute layer depth (under `--seed-path P`, `--list-layer K` lists K below P and k =
    depth(P) + K; `seed` is P). A campaign accepts only `seed` `""`.
  - `shallow_best` is the deepest valid level above the layer (depth < K), verified like a LEVEL line, or
    null. No job reports it, so the server stores it as the exit's starting best.
  - `above` counts the nodes expanded above the layer.
- **Then** one line per root, in DFS order: `LAYER\t<exit>\t<seed path>\t<depth>\t<blocks>\t<holes>`.
- **Self-check.** Before printing, the listing replays every root's path as its job will and requires the very
  same node (path, cells, blocks, holes, bulk_walk, walk history and segment, adjflags). A mismatch, a full
  dedup table or an inconclusive check above the layer ends the listing with status `error` (exit 5); a path
  over path_tok_max exits 4. A listing that did not exit 0 must never be used.
- `--estimate N --estimate-depth K` prints the same header and LAYER lines on stdout (so its stdout is a layer
  file), then a Knuth estimate of the tree below the roots. `--estimate-dump F` appends the header and one row
  per root (exit, seed, depth, blocks, holes, estimated nodes, estimated seconds, SE) in the same order;
  v2_seed stores the estimated seconds as each root's `est_s`. The estimator runs low (measured 1-13x).
- **Campaign roots** are listed by the release binary for exactly the campaign's flags, without --seed-path,
  and the WHOLE stdout is the layer file (headers included). `grep -c '^LAYER[[:space:]]'` counts the roots.
- **Older workers** (before 27443a8, which includes the live 406e5500) listed with bulk walk-back off and print
  no LAYERINFO header. Their roots are exact only for K ≤ 4: at K ≥ 5 a root whose path walks onto a committed
  cell replays with bulk walk-back on and loses its parent's deeper bulk siblings (the lead measured 9,566
  lost levels at layer 8 on 5x5 exit 7 ≤3 blocks). No layer-4 root can contain such a walk (brute force on
  5x5, 4x6, 6x6, 3x3 and 2x5 finds the first one at step 5), so campaigns #1 and #2, both at layer 4, are not
  affected. Never seed from a deeper layer listed by such a worker.

### 2.9 Solver widths and memory

- **Widths (W3).** The push solvers (full, cutoff and multi-start) are written once and compiled for three
  packed-state widths: one 64-bit word (bits_per_cell × blocks + holes ≤ 64: on 6x6 with no holes up to 10
  blocks), 16 block bytes (≤ 16 blocks) and 32 block bytes (≤ 32 blocks). All widths share the Zobrist-keyed
  tables, the A* bound, decision mode, the reference (parent) tables and the multi-start solve. Only more than
  32 blocks or 30 holes does not fit (status `error`, never a prune). The multi-start label table checks a
  second, independent key on every key match (review M7); a collision falls back to exact per-start checks.
- **Performance (W4).** Solver tables grow in place instead of restarting; any checked state may export its
  table as a parent table within a 4M-slot (48 MB) live budget; a dead-end child (the new player cell's only
  way out is back) is decided without a check (proof above `deadend_child` in backsearch.c; off in
  beam/rollout/trace modes); fresh dedup tables are not cleared a second time (a trivial job costs 3 ms and
  4 MB). Wide checks are 1.5-1.8x faster than on the live worker, heavy 6x6 roots run about 2x faster again,
  and 60-s runs from the 6x6 exit roots do 1.05-3.2x the states per second of the pre-protocol-3 worker.
- **Memory.** Plan on **0.5 GB per worker**. Measured: a typical job 0.17-0.25 GB; the heaviest 6x6 0-hole job
  measured (exit 7 root U2,R2,U2,U1, 30k expansions) 417 MiB max RSS; a trivial job 4 MB. The dedup tables
  (about 160 MB once filled) dominate; the parent-table pool is bounded at 48 + 12 MB; solver tiers of 4M slots
  or more are freed as soon as a solve grows past them. SUMMARY reports `max_rss_mb`.

## 3. Server (Node, `server/lib/jobs.js`, `v2routes.js`; DB file `data/jobs.db`)

### 3.1 Storage

`jobs.db` is a separate SQLite file (WAL, busy_timeout 5000, every write in a `BEGIN IMMEDIATE` transaction,
every value bound). The server runs `synchronous=NORMAL` (a power loss can drop the last commits, which only
leaves jobs leased until their leases expire); the owner tools open it with `synchronous=FULL`, so an admin
action is on disk when the tool prints success. `JOBS_DB_MUST_EXIST=1` refuses to create a missing jobs.db.
The schema, with its migrations, is in `openJobsDB` in jobs.js. In short:

| table | holds |
|---|---|
| `campaigns` | the definition (grid, extra, exits, hashes, parameters, client params, version gate, release, LAYERINFO), status `open` / `complete` / `closed`, steal and empty-lease counters |
| `campaign_exits` | per exit: `expected_roots` and `shallow_best` from LAYERINFO |
| `clients` | token, name, workers, paused, boards, the last `running` and `holding` lists, client version |
| `jobs` | the job tree: seed, parent, depth, status, lease, hash, fingerprint and SUMMARY fields, level, `verified_moves`, `est_s`, `dup_of`, `mismatch`, failure count, lease and steal history |
| `reports` | every report (summary, outcome, client version) |
| `failures` | every failure report |
| `candidates` | unresolved candidates per exit (§3.5) |
| `events` | lease, steal, reclaim, release, stop_release, drop, expire, reset and supersede events (append-only tuning data; coverage never reads it) |

Job statuses: `open`, `leased`, `done` (exhausted), `split`, `quarantined` (never leased; blocks clean),
`superseded` (retired by a hash re-issue, §3.8; never counts again) and `cancelled` (a dropped duplicate).

### 3.2 Campaign definition and version gate

A campaign is created only by `tools/v2_seed.js` from a `plan.json` (§3.8). `validateCampaignPlan` checks:
- the **extra allowlist**: exactly one `--allow-exit-transit`; at most one each of `--num-holes N` (0..64),
  `--num-blocks N` (0..32) and `--min-walls N`; nothing else (`--allow-block-on-exit` gets its own refusal);
- the grid is written rows ≤ cols (6x6 or 4x6, never 6x4), and `exits` is the grid's full canonical exit set
  (6x6: 0, 1, 2, 7, 8, 14);
- at least one whitelisted hash;
- the parameters, with defaults: `max_clients` 40, `workers_max` 32, `job_target_s` 1200, `split_after_s` 1800,
  `ramp_split_after_s` 120, `lease_s` 3600, `paused_max_s` 43200, `dup_fraction` 0.02, `fail_regrant_s` 3600;
  client parameters `lease_cap` 200, `batch_interval_s` 10, `lease_ahead_s` 300, `absorb_total_s` 120,
  `absorb_probe_s` 10, `heartbeat_s` 30; test-only, with no default: `split_after_nodes` and `min_window_s`.

**Version gate**: `min_client_version` (default 3.0.0), `min_protocol` (3) and `release` (the tag the refusal
names). Every POST body carries `client_version` and `protocol`; a client below the minimum gets 426
`{error:"client_too_old", release, message, min_client_version, min_protocol}`. A pre-protocol-3 client
(which sends neither field) gets a code it already treats as fatal: 400 on register, 403 on heartbeat and
lease.

### 3.3 Routes (JSON; POST bodies ≤ 4 MB for report and reports, ≤ 64 KB for register, heartbeat and lease)

- `POST /api/v2/register {name, workers, client_version, protocol}` → `{token, campaign, campaign_state}`.
  503 with `queue_position` when `max_clients` are active; 404 `no_campaign` (with `last_campaign`) when none is
  open; 429 `too_many_registrations` beyond 20 new tokens per IP per hour (loopback exempt; behind the proxy it
  needs `TRUST_PROXY=1`). Names: NFC, control and format characters stripped, 1-40 code points.
- `POST /api/v2/heartbeat {token, workers, paused, boards, holding, running, stopping?}` →
  `{leases, drop, reclaimed, released, paused, stopping?, server_time, campaign, campaign_state, me, exits}`.
  - Every heartbeat pushes each held lease's deadline to now + `lease_s` (paused: capped at leased_at +
    `lease_s` + `paused_max_s`), so a lease expires only `lease_s` after a client's last heartbeat.
  - `holding` ids that expired while still open come back re-leased (`reclaimed`); ids held by another client
    or finished come back in `drop`, and the client kills that worker without reporting.
  - Release by omission: every other lease of the token older than 120 s is returned (`released`).
    `stopping:true` releases every lease of the token at once.
  - `me` is the volunteer's statistics (by name); `exits` per exit: roots, roots covered, counts, CPU, best,
    clean, exact and the forecast `progress` (these may be null for a few seconds after a campaign is created).
- `POST /api/v2/lease {token, n, exit?}` → `{jobs:[{id, exit, seed, depth, split_after_s?, dup?}], lease_s,
  split_after_s, absorb_total_s, window_mode, endgame?, held, cap, exit_fallback, stolen, reclaimed,
  campaign_state}` (§3.6).
- `POST /api/v2/report` (one tree report) and `POST /api/v2/reports {token, reports:[…]}` (a batch of up to
  1,000 tree reports and failure reports, applied in one transaction) → per element a result, plus
  `campaign_state` (§3.4).
- `GET /api/v2/status`: the dashboard (per-exit counts, CPU-hours, best levels, active clients and their
  boards, hashes, totals including re-checks, forecast). `GET /api/v2/campaign`: the open campaign.
  `GET /api/v2/campaigns`: completed campaigns (search tree only; re-checks listed apart).
  `GET /api/v2/audit[?campaign=ID][&exit=E]`: the audit (§3.5), also for closed and complete campaigns;
  503 `warming_up` for a few seconds after a campaign is created.
- The retired chunk tracker's POST routes answer 410.

### 3.4 Reports

- **Tree report**: `{job_id, src_hash, nodes:[{seed, parent, status, summary?, level?, unresolved?}]}` (§7.5).
  The job itself is the first node and carries its run's SUMMARY; every `done` node carries a SUMMARY with
  status `exhausted` and the fingerprint fields; every `split` node carries its run's SUMMARY and at least one
  child; `open` nodes carry no summary but may carry a level (with its depth) and candidates. At most 5,000
  nodes per report.
- **SUMMARY checks** (strict, protocol 3): `protocol` ≥ 3, and `flags` and `unresolved` present. The
  whitelisted fields are stored (`cpu_s`, `unknown*`, `unresolved`, `unresolved_dropped_max` as
  `cand_dropped_max`, `verify`, and `flags` in the report); the information-only fields are dropped. A report
  whose `flags` disagree with the campaign definition is refused (an unset cap counts as unconstrained:
  `max_holes` ≥ cells-2, `max_blocks` ≥ min(32, cells-2)). `best` ≤ 4,000. A `verify` in 0..best-1 is a verify
  mismatch and blocks clean.
- **Levels**: a level must parse, match the grid, the exit cell (after canonical mapping) and the caps
  (`--min-walls` included), and its depth must equal the node's `summary.best` (1..4,000); a node with best 0
  carries no level. A finished row with a best but no level is stored, logged, and listed as `levelless` (it
  blocks exact; the remedy is `v2_jobs rerun`). A new per-exit best is re-solved by the site solver in the
  background (30 s, 0.5 GiB) into `verified_moves`; a server sweep retries champions not confirmed yet.
- **Candidates**: `unresolved:[{depth, code, path, cause}]` per node, at most 1,000 per node and 20,000 per
  report; cause pq, probe or big; the path extends the node's seed and has exactly `depth` tokens; the count
  equals `summary.unresolved`. They are stored per exit, deduplicated by level code at the smallest depth.
- **Failure report** (a batch element): `{job, failed:"<reason>", detail?, src_hash?}`, the reason matching
  `^[a-z][a-z0-9_:.-]{0,39}$`, detail ≤ 300 chars. It is recorded, increments the job's `fail_count` and
  reopens the job if this client holds it (another client's lease is left alone; a failure on a finished job
  is ignored; a non-whitelisted `src_hash` gets 409). A deterministic 400 refusal (`bad_tree`, `bad_level`) of
  a report from a whitelisted hash counts as a failure too.
- **First report wins.** A report for a job that is open, leased (by anyone) or quarantined finishes it. A
  report for a finished job is a late duplicate (`{dup:true}`): its fingerprint is compared and a mismatch is
  flagged. A late report on a superseded or cancelled job returns `{dup:true, retired}`. A report under a
  hash that is not whitelisted is stored, the job goes back to open, and the answer is 409 `unknown_hash`.
- Every batch element re-reads its job row inside the transaction, so a batch naming a job twice is safe.

### 3.5 Audit, exactness and completion

Per exit, from the job tree (computed off the request path, §3.7):
- **clean** = at least one root; every root covered (§1.3); no job done under a hash that is no longer
  whitelisted; no fingerprint mismatch; nothing quarantined (duplicates included); no verify mismatch; and the
  root count equals LAYERINFO's `roots[exit]`.
- **best** = the maximum of the jobs' levels, the listing's `shallow_best` and the true lengths of resolved
  candidates (`best_source` says which).
- **exact** = clean, and no open candidate deeper than the best, no node whose candidate list was cut above
  the best (`cand_dropped_max` > best; for older workers, a full list of 1,000 whose shallowest kept candidate
  is deeper than the best), no contradicted candidate (a recorded true length that a later sighting at a
  smaller depth disproves), and no levelless row deeper than the best.
- The audit also lists uncovered roots, unknown-hash jobs, mismatches, quarantined jobs, duplicates by status
  (`pending_dups`), candidates (open, deeper than the best, contradicted) and a completion forecast from
  `est_s` (`progress`, shown once 5 % of the root estimate is covered; display and scheduling only).
- **campaign_state** `running | complete | closed` is in every register, heartbeat, lease and report response.
  - A campaign becomes `complete` by itself when no job is open, leased or quarantined (duplicates included)
    and every exit's fresh audit is clean (checked at most every 15 s, and 5 min after an unclean result).
  - Its tokens keep getting 200 with the state (lease returns no jobs; heartbeat still carries `me` and `exits`
    for the client's closing summary) until a newer campaign is open. Then they get 410 `campaign_closed` with
    `campaign_state:"complete"`, so a volunteer who restarts later joins the new campaign.
  - `closed` means closed by the owner (410).
  - Complete is not exact: the candidates are resolved offline afterwards (§3.8).

### 3.6 Scheduling

- **Lease order**: the preferred exit first (`exit` in the request); then the largest `est_s` first across
  exits, then depth ascending, then age. Split children inherit an equal share of their parent's `est_s`;
  without estimates the order is shallowest first. A request is capped at `lease_cap` (200); a client may hold
  at most 3 × workers + 200 (the client itself asks for far fewer, §7.6).
- **Never back to the holder**: a job the client listed as held or running at its last heartbeat is never
  granted to it again (if its lease lapsed it comes back as `reclaimed`), and a job is not granted to a client
  that reported a failure for it within `fail_regrant_s` (1 h).
- **Windows** (review M109). The lease response's `split_after_s` and `absorb_total_s` come from the pool: the
  open jobs plus the jobs their holders have queued but not started, each counted as `est_s` × the exit's
  calibration once 5 % of the exit's estimate is covered, else as `split_after_s` / 3.
  - `full` (`split_after_s`, 1,800 s) while pool work ≥ active workers × `split_after_s`, or open + queued jobs
    ≥ 3 × active workers;
  - otherwise, with idle workers, clamp(pool work / idle workers, 15 s, `ramp_split_after_s`): `endgame` when
    below the ramp window, else `ramp` (120 s);
  - the absorption budget is min(`absorb_total_s`, max(`absorb_probe_s`, 0.5 × window)): 120 s in a full
    window, 60 s in a ramp window, 0 in the endgame.
- **Work stealing** when the pool is empty: §9.
- **Duplicates** (fingerprints): when a job finishes `done`, a duplicate is created with probability
  `dup_fraction`. A duplicate is never stolen and never granted to a token with the original's name, except
  when that client has had nothing else to do for 600 s. Its window is max(window, min(7,200 s, 3 × the
  original's run)), so a long original is re-run whole. It is compared only when both are done under the same
  hash; a mismatch blocks clean until `v2_jobs resolve` decides. `v2_jobs drop-dup` cancels pending ones.
- **Failures and quarantine**: a job with 3 failures from at least 2 clients, or 5 in total, becomes
  `quarantined`: never leased, blocks clean, listed by the audit. A valid report still finishes it;
  `v2_jobs release [--all]` reopens.
- **Sweeper** (every 60 s): leases past `lease_until` go back to open (a paused client that still heartbeats
  keeps them within `paused_max_s`); registrations never used for a day are purged.

### 3.7 Off the request path: stats, backups, events

- A worker thread (`lib/v2stats.js`) with its own read-only connection computes status, every audit, the pools
  and the campaigns summary in one read transaction per tick (every max(2 s, 2 × the last tick's cost)) and
  decides completion; the routes only serve the cache. `V2_STATS_WORKER=0` computes on the request path instead
  (cached, stamped after the compute).
- Backups: the server runs `tools/v2_backup.js` daily as a child process (VACUUM INTO, quick_check, gzip into
  `data/backups/`, 7 days, the newest 3 always kept; `JOBS_BACKUP=0` disables it, `JOBS_BACKUP_DIR` moves it).
  On the VPS `server/ops/backup-jobs.sh` also runs from cron.daily (14 days), and `tools/pull_backups.sh` copies
  both off the box from the Mac (`server/ops/README.md`).
- On SIGTERM or SIGINT the server stops the stats worker and closes both databases (the WAL is checkpointed).

### 3.8 Owner tools (in `PathologyRecords/server/tools`, run on the server)

- `v2_seed.js --campaign-file plan.json --layer-file layer.tsv [--estimate-file dump.tsv] [--close-others] [--dry]`
  (and `--append ID` for exits without roots yet). The layer file is the whole stdout of the release worker's
  listing (§2.8). Every check must pass or nothing is written: the header's k, grid, flags and src_hash match
  the plan (the hash must be whitelisted), `seed` is `""`, `shallow_best` < K, exactly `roots[exit]` roots per
  exit, each with depth == tokens ≥ K (a deeper root ends in one walk run from token K on) and within the caps,
  no duplicates, no malformed or merged lines. `--dry` runs every check. A real run closes the open campaign
  (`--close-others`), creates the new one and inserts every root in one transaction.
- `v2_hashes.js list | add HASH | remove HASH [--reissue]`.
  - `remove` alone only takes the hash off the whitelist. New reports under it are refused (409
    `unknown_hash`), and every job finished under it stops counting as covered: the audit lists those jobs as
    unknown-hash, and the exit stays unclean (§3.5), so the campaign cannot complete and v2_finish refuses to
    publish, until the hash is added back or the jobs are re-issued. Never remove the hash of a worker whose
    finished work the campaign still relies on (§6.3).
  - `remove --reissue`, only after a correctness fix: per exit only the topmost finished rows under the removed
    hash are reset to open, everything below them becomes `superseded` (their levels dropped), and duplicates of
    retired rows are retired. That work is searched again. The tool prints the CPU-hours to redo and the open
    candidates recorded on retired rows.
- `v2_jobs.js quarantined | failures | release [--all] | mismatches | resolve | recheck | rerun JOB | drop-dup |
  candidates | candidate ID MOVES | solve-candidates`: what blocks an exit, and how to clear it.
- `v2_resolve.js`: solves the open candidates deeper than each exit's best (and the contradicted ones) with
  solver6, deepest first, and records `valid` (true length = depth) or `invalid` (shorter).
- `v2_verify.js`: re-solves each exit's champion whose `verified_moves` is empty.
- `v2_finish.js [--solve]` (a dry run), then `--publish`: for each exact exit, solve the champion with solver6
  at the offline budget, check that its length equals the maximum, store the champion, add the proof
  (attributed "Collective"), and re-check every overlapping proof. It is idempotent; it refuses non-exact
  exits, pending duplicates, contradicted candidates and `--min-walls` campaigns. `--accept-unverified` only
  overrides a solver that gives up; `--force-conflict` records a proof that disagrees with a nested-class proof.
  A 6x6 0-hole campaign publishes as "any number of blocks": all 7,350 6x6 levels with 33 or 34 blocks and no
  holes have a longest optimal solution of 2 moves.
- The finishing order: `v2_jobs candidates` → `v2_resolve` → `v2_verify` → `v2_finish --solve` →
  `v2_finish --publish` (all re-runnable; `server/ops/README.md`).

## 4. Client (`v2/volunteer.py` 3.1.0, Python 3.9+, stdlib only)

```
python3 v2/volunteer.py --name "Your name" [--workers N] [--exit E] [--server URL] [--port 8765]
        [--outbox DIR] [--keep-going] [--no-ui | --no-browser] [--no-stop-on-close] [--reregister]
```
It runs from `backsearch/`. The defaults: half the logical cores, the worker `./backsearch_worker_nt`, the
outbox `v2/volunteer_outbox`.

- **Identity.** Tokens are stored in `~/.pathology_volunteer.json`, keyed by server URL and the outbox's real
  path. The client holds an exclusive lock on `<outbox>/.lock`, so a second instance on the same outbox exits 2
  and says to pass `--outbox` and `--port`. Two instances never share a token.
- **Server answers.** 426 (`client_too_old`): print the server's message (it names the release) and exit 2.
  410 (`campaign_closed`): drop the token and register again; with `campaign_state` `complete` the client ends
  with the closing summary instead (below). 403 for an unknown token: register again; the batch stays in the
  outbox and is resent. 404 `no_campaign` at register: a message and exit 0 (with `--keep-going`: poll). 429 and
  5xx: retry with backoff.
- **The worker.** At startup the client checks the worker's `--version` (PROTOCOL 3, a whitelisted hash) and
  copies the binary to `<outbox>/.bin/` (a new inode); every run uses that pinned copy, and a run whose SRC_HASH
  differs from it stops the client. It checks that `backsearch.c`, `sokoban_bfs.c`, `sokoban_bfs.h` and
  `v2/volunteer.py` are unmodified (`VOLUNTEER_ALLOW_DIRTY=1` for development). The worker command (§2.2) takes
  the campaign's `extra` only through the allowlist `--allow-exit-transit` (required), `--num-holes N`,
  `--num-blocks N`, `--min-walls N` (N a non-negative integer); anything else makes the client refuse to run.
  The worker's `SUMMARY.flags` must match the campaign. `BS_*` variables are stripped from its environment.
- **Windows.** A worker leases a job and works a local stack for one window (§7.2-§7.3), then reports the job
  once with the whole local tree. The window's length and absorption budget come from the latest lease
  response when the window starts (§7.4).
- **Void runs.** A void run contributes nothing. When it is the window's own job, it is reported as a failure
  `{job, failed, detail, src_hash}` (a child's void run leaves that child open in the tree). Reasons: the
  statuses bad_seed, path_overflow, error, unknown_chain or any other non-exhausted/split status; a crash or a
  non-zero exit (`crash`); no SUMMARY (`no_summary`); a watchdog kill (`hang`); a SUMMARY without flags or
  unresolved (`bad_summary`); a bad candidate line or count (`bad_unresolved`); a malformed, non-extending or
  overlong REMAINING line (`bad_remaining`); REMAINING equal to the seed on the worker's own timer
  (`no_progress`); other bad lines (`bad_output`); and a tree too big to report even after fold-back
  (`too_big`). The failure is queued before the slot lets the job go.
- **Fold-back, never truncation.** A tree over 5,000 nodes, 3.5 MB or 20,000 candidates is made to fit by
  demoting the deepest split nodes (other than the job) to `open`, then done nodes, largest first (their
  descendants dropped, their deepest level kept). If even the job's own REMAINING does not fit, nothing is
  reported and the job is handed back with failure `too_big`.
- **Delivery.** Reports are batched by size (≤ 1 MB and ≤ 500 reports; a 413 halves the batch) and written to
  the outbox before the POST (write-ahead); a file is deleted only after the server answered. `holding` lists
  every job the client still owns: running, queued, and with reports pending or in outbox files younger than
  `lease_s`; `running` lists the running ones. Lease grants are deduped against everything held.
- **Campaign completion.** On `campaign_state` `complete` (in heartbeat, lease or report responses, or in a 410
  that says so) the client stops leasing, lets running workers finish, delivers every pending report and outbox
  file (for at most 120 s), sends the final heartbeat, and prints a closing summary: the volunteer's jobs,
  CPU-hours, rank and deepest level, this run's windows, and per exit the longest level, who found it, exact /
  not exact yet, and roots covered. The panel shows the same with "you can close this window", and the client
  **exits 0**. `--keep-going` waits for the next campaign (polling every 60 s) and joins it. A stored token of
  a completed campaign delivers its outbox, shows the summary once, then registers.
- **Stop and pause.** Stop, Ctrl-C once, SIGHUP or closing the panel (unless `--no-stop-on-close`) sends every
  worker SIGINT, reports what they print, sends a final heartbeat `stopping:true, holding:[], running:[]` (the
  server releases every lease at once) and exits. A second Ctrl-C writes pending reports to the outbox and
  exits at once. Pause freezes the workers (SIGSTOP) and heartbeats `paused:true`; Resume thaws them. Ctrl-Z
  freezes the workers with the client, and `fg` thaws them. Workers run in the client's session in their own
  process group; a pidfile lets the next start kill orphans still running this outbox's pinned binary.
- **Watchdog.** A worker silent for 3 × the status interval + 60 s after its split time, or still running
  max(600 s, split_after) past it, gets SIGINT, then SIGKILL after 90 s; the run is void (`hang`). A main-loop
  tick more than 5 s late (laptop sleep, a frozen client) first moves every slot's baselines by the lost time,
  so a sleep never looks like a hang; a wake after more than 2 × `lease_s` also re-validates the leases. A
  Stop-grace kill of one run leaves that node open and still reports the rest of the window.
- **Panel security.** The panel answers only `127.0.0.1:PORT` / `localhost:PORT` Host headers; POSTs also need
  the Origin and a per-session secret from the page; framing is denied.
- **Updates.** A refused hash prints `git pull && ./build_pgo.sh -o backsearch_worker_nt --no-torch` (built from
  `--worker`) and the campaign's release tag, and says that the build may be older or newer than what the
  campaign accepts.

## 5. Tests

Everything runs on a shared laptop under the owner's CPU guard; single runs take about a minute. Unless noted,
a test runs one worker at a time.

1. **Split exactness** (`v2/test_split_exact.py WORKER --quick | --suite campaign [--only NAME]`): splits a tree
   repeatedly (by time, SIGINT or node count) and requires the union of the jobs' canonical valid levels
   (`BS_TRACE_VALID`) to equal the monolithic run's exactly, depth by depth (lost and extra both fail), with an
   equal best. It asserts every protocol line (§2.3-§2.4) and cross-checks its worker command line against the
   client's. The campaign suite covers whole 4x4 and 4x5 exits, 3x4 and 3x5 node splits, 5x5 and 6x6 0-hole
   subtrees, real campaign-#1 subtrees, and small-table builds that must evict. Visit sets are not the
   invariant: the dedup key ignores walk-history flags, so a subtree job expands some twins the full run skipped.
2. **Chaos** (`v2/test_chaos.sh WORKER [--config 4x4h3|6x6h0b1] [--e2e]`, driver `v2/test_chaos.py`, two CPU
   slots): the real server on a temp data dir (`JOBS_DATA_DIR`, `RECORDS_DATA_DIR`, no backups), the real
   worker, and two real clients with one worker each. The campaign is an allowlisted protocol-3 plan over every
   canonical exit, seeded by v2_seed from the worker's own `--list-layer`; splits are deterministic
   (`split_after_nodes`), lease 10 s, heartbeat 5 s, 1-s windows, `dup_fraction` 0.2.
   - Scenario, each step waiting for its condition: client A is SIGKILLed while its worker runs and it holds
     leases, and restarted (before the restart the test plants a SIGSTOPped orphan, A's own pinned worker on a
     real search, in the pidfile A left: a real orphan dies of the broken pipe by itself, so this is the
     survivor the startup cleanup exists for; the restarted A must kill it and log it); client B gets SIGINT in
     the middle of a window (it must hand the window back, exit 0 and hold no lease) and is restarted; the
     server gets SIGTERM while work remains (exit 0), stays down 3 s (both clients must notice) and is
     restarted; both clients must then exit by themselves with code 0 when the campaign is complete.
   - Asserts: the campaign is complete; a FRESH audit of every exit is clean and exact; at least one split and
     one really compared duplicate (`done:match`); no mismatch, failure or rejected report; empty outboxes;
     every exit's best equals the monolithic run's; the union of the valid levels traced by the runs behind
     ACCEPTED nodes (tied through the client's development run log, `VOLUNTEER_RUN_LOG`) equals the
     monolithic canonical set at depth ≥ K, nothing lost or extra; no process of the test survives it; the
     owner's databases are untouched.
   - `--e2e` runs one client without disruptions to completion (exit 0).
3. **Gates** (`node --test server/test/v2_gates.test.js`, in PathologyRecords): 60 tests of the server:
   register and queue, the version gate, the allowlist, seeding checks, lease order and caps, windows,
   stealing, reclaim and release, tree and candidate validation, failures and quarantine, duplicates and
   mismatches, the hash re-issue, completion, the stats worker, the tools, HTTP 413/400/426/429/503.
4. **Fingerprint duplicates**: covered by the gates and the chaos test (§3.6).
5. **Client** (`v2/test_client.sh [tests]`, fake server + fake worker; tests a-x, y, z, pw and ui; ui needs
   node): stop, kill -9 and orphans, pause/resume and the panel's controls, dying workers, batching, 410 and
   426, completion and `--keep-going`, a second instance, the extra allowlist, fold-back, failure reports, the
   watchdog, 413 halving, unresolved forwarding, grant dedupe, the empty root, units, the exit check, the outbox
   after a server restart, sleep across the split time, grace kill and endgame, absorption order, lease sizing,
   pause with fewer workers, the panel script. Run in groups of about a minute, e.g. `a e h s t ui`,
   `b c g k q`, `d f j l m`, `n o p u v`, `w x i r`, `y z pw`.
6. **Dedup-key sensitivity** (`v2/test_dkey.py WORKER`, ~50 s): test builds keep 20 bits of the dedup hash, so
   distinct states collide thousands of times per run. With the 32-bit check value compared, levels and counts
   must equal the tested worker's; with it disabled, the same runs must lose states and levels.
7. **Root cover** (`v2/test_roots_exact.py WORKER --quick | --suite campaign | --suite deep [--jobs N]`): the
   union of the root jobs plus `shallow_best` must equal the monolithic run at every tested K (layers 2-8,
   including 5x5 exit 7 ≤3 blocks at K = 8 and 6x6 0-hole configs); a failed listing fails the test.
8. **Oracle** (`v2/test_oracle.py WORKER [--quick]`, `v2/oracle_enum.c`): an independent enumerator with its own
   move rules and BFS judge. On grids 2x4 to 4x5 every exit's traced valid levels must equal the oracle's set
   depth by depth, in every mode (bulk walk, parent tables, canonicalisation and two-table dedup each off) and
   in split trees; the judge agrees with the site's solver6; `wide6` re-solves every traced 6x6 0-hole level
   with 11 or more blocks with solver6. `--narrow-bits N` and `--build-knobs` force the 16- and 32-byte solver
   widths.
9. **Load** (`v2/test_load.sh`, §7): 10 clients × 16 workers with 10-s jobs, and 30 × 16 with 2-s jobs.
10. **Worker equivalence**: the configs in `backsearch/README.md` must reproduce states, accepted, valid, best
    and verify exactly after any worker change (solver_calls and eviction counts may change, and every change
    must be explained).

Last results on the release candidate (2026-09-25): split exactness quick OK; root cover quick OK (layer 8
loses 0); oracle quick OK (2,122 levels judged, 0 disagreements; wide6 754 levels with 11 blocks); chaos PASS on
4x4 ≤3 holes (70 s, 12 duplicates compared) and on 6x6 0 holes ≤1 block (89 s, 13 compared), union exact; gates
60/60; dkey passed; ASan+UBSan on 18 campaign roots with 0 reports; load at 10 × 16 p95 lease 77 ms, reports
90 ms, 0 errors (30 × 16: p95 540 ms, 0 errors); capacity knob builds end exhausted with
V_knob ⊆ V ⊆ V_knob ∪ U_knob; the equivalence configs match the baseline exactly.

## 6. Campaign parameters

### 6.1 Campaign #2: 6x6 with no holes (live since 2026-09-24 20:50)

- Title "6x6 with no holes: what is the longest level?"; grid 6x6; extra `--allow-exit-transit --num-holes 0`
  (any number of blocks: the worker's 32; the proof publishes as any block count, §3.8); exits 0, 1, 2, 7, 8,
  14. The best known level is 165 moves with 10 blocks, so levels are long and most states have 10+ blocks.
- **Root layer 4**: 1,368 roots (exits 0/1/2/7/8/14 = 17/116/158/188/513/376), all at depth 4. The release
  candidate's listing of the same roots (byte-identical LAYER lines) reports `shallow_best` 3 on every exit.
  The live campaign's header was written by hand, because 406e5500 predates LAYERINFO: it has `k`, `grid`,
  `flags`, `src_hash`, `roots` and `made_by`, and no `seed` or `shallow_best` key. That is harmless, since a
  level of at most 3 moves never decides a maximum. Layer 4 follows §7.1 (seed shallow): with the realistic
  10 × 16 = 160 workers, 1,368 roots are more than 3 × workers, so windows are full from the start; and no
  layer-4 root can be affected by the old listing defect (§2.8).
- **Root estimates**: the launch set holds, beside the plan and the listing, the 406e5500 estimate dump of the
  same 1,368 roots (`--estimate-depth 4`), and the launch record says the roots were seeded with it
  (`--estimate-file`). So they carry `est_s` and the lease order is largest `est_s` first (without estimates it
  would be shallowest first). To confirm on the server, as the service user (so any `-wal`/`-shm` file stays
  the service's; `-readonly` fails on a WAL database without them):
  `runuser -u pathology -- sqlite3 /opt/pathology/data/jobs.db "SELECT count(*), count(est_s) FROM jobs WHERE
  parent_id IS NULL AND dup_of IS NULL AND campaign_id = (SELECT max(id) FROM campaigns WHERE status = 'open')"`
  prints `1368|1368` when every root has one.
- **Size**: the Knuth estimate (the live worker, `--estimate-depth 4`, about 30k probes per exit) gives a
  dedup-free lower bound of about 5,100 CPU-hours, about 3,600 with dedup: exit 7 about 4,700 h (SE 70 %),
  exit 14 290 h, exit 8 110 h, exit 1 22 h, exit 2 5 h, exit 0 0.1 h. The estimator ran 1-13x low before, so
  expect several thousand CPU-hours at least.
- **Parameters** (as seeded): `split_after_s` 1800, `ramp_split_after_s` 120, `lease_s` 3600, `paused_max_s`
  43200 (12 h), `dup_fraction` 0.02, `max_clients` 40, `workers_max` 32, `job_target_s` 1200,
  `min_client_version` 3.0.0, `min_protocol` 3, `release` "campaign2-v3.0.0", client parameters at their
  defaults (§3.2). The server's windows (§3.6) then give 1,800-s windows with 120 s of absorption while the pool
  is deep, 120-s ramp windows with 60 s, and 15-120-s endgame windows with none.
- **Why 1,800-s windows** (§1.7, review M10): a longer or on-demand split would save about 3 % of the states on
  the split subtrees, but every forced split bounds what a crash, kill or sleep loses to one window and spreads
  the heavy exit-7 roots (about 25 CPU-hours each on average, by the estimate) over many workers.
- Memory: 0.5 GB per worker (§2.9). Capacity: no UNKNOWN on 60-s runs from every exit root (§2.5).

### 6.2 Campaign #1: 5x5 with at most 3 holes (complete, published 2026-09-24)

grid 5x5; extra `--allow-exit-transit --num-holes 3`; exits 0, 1, 2, 6, 7, 12; layer 4 (2,761 roots,
42/355/233/810/972/349, in `roots_h3/`); max_clients 40; workers_max 32; job_target 20 min; split_after 1800 s
(ramp 120 s); lease 3600 s; paused_max 12 h; dup_fraction 0.02. Result per exit: 127/149/112/138/115/80, so
**149 is the longest 5x5 level with at most 3 holes**; 597 CPU-hours, 166,733 jobs and 36.4 billion states (plus
1,033 fingerprint re-runs, 4.8 CPU-hours), about 11 hours of wall clock. The proofs are attributed "Collective".
Caveat (review 2026-09-24, §1 and C1): nothing found shows a published maximum is wrong, but exhaustiveness is
not fully proven. The campaign-#1 worker pruned a state silently when a single shortcut check hit its
1,048,576-entry pending cap (solver -2), with no counter in SUMMARY. In a spot check (the release candidate, 120 s
on each of the 12 heaviest campaign-#1 seeds) the largest single check held at most 52,499 pending entries
(§2.5), so such a prune is implausible but not ruled out. The review's two other residual risks are the 64-bit
keys (§6.3) and prunes that were checked by argument rather than against an independent enumerator (M89;
`v2/test_oracle.py` now does that for the current worker). The proofs stay published (published proofs are
never revoked or edited).
(This section first said layer 8. Campaign #1 was seeded at layer 4, and until worker 27443a8 a deeper layer
lost subtrees, §2.8.)

### 6.3 The live release, and the rollout of the release candidate

The live campaign runs worker 406e5500 (6991199), client 3.0.0 and server 914fa20 + ee3aebf. That worker already
has the protocol-3 core (UNKNOWN candidates, bounded solves, 1024-token paths, the bad_seed / path_overflow /
error / unknown_chain statuses, the split fixes), but it:
- packs wide states into 128 bits, so any 6x6 0-hole state with more than 20 blocks ends the job with status
  `error` (exit 5) on every client (such jobs come back as failures and can be quarantined);
- decides states wider than 64 bits on a slower solver (§2.9);
- checks the multi-start label table and the dedup table with 64-bit keys only (review M7, N13);
- prints no LAYERINFO header (the live roots were listed at layer 4 and given a hand-written header).

The rollout (the owner decides and deploys; nothing here changes the live data):
1. Back up `jobs.db`, then deploy the server from PathologyRecords `campaign2` (`server/ops/README.md`;
   additive: 3.0.0 clients keep working).
2. Publish the release sources. Volunteers clone and `git pull` the searcher repository's GitHub `main`, and
   the client's update advice is `git pull && ./build_pgo.sh ...`. GitHub `main` is still 6991199 (the live
   406e5500), an ancestor of project45 `campaign2`, so until the owner fast-forwards `main` to `campaign2` a
   pull delivers neither the c3ee5d2b worker nor client 3.1.0. The PathologyRecords README's link to this file
   (on `main`) also shows the pre-protocol-3 version until then.
3. Build the release worker from that checkout with `./build_pgo.sh -o backsearch_worker_nt --no-torch`, check
   its hash with `--version` (c3ee5d2b… for the current sources), and `v2_hashes add` it. Both hashes are then
   accepted; fingerprints are compared within one hash only.
4. **Keep 406e5500 whitelisted until campaign #2 is finished and published.** A done or split job counts as
   covered only while its hash is whitelisted (§3.5), and every job of campaign #2 finished so far was finished
   under 406e5500. `v2_hashes remove` at any time un-covers all of that work until the hash is added back: every
   exit goes unclean, the campaign cannot turn complete, and v2_finish refuses to publish (§3.8). `remove
   --reissue` would search it all again (thousands of CPU-hours). Neither is called for. 406e5500's finished
   results are exact, and the review's risk estimate for its 64-bit keys (about 0.01 false dedup matches in a
   campaign-#1-sized search, about 1e-10 chance of touching a maximum) does not justify a re-run.
5. Once most clients run the new worker, clear the jobs the old one could not do. `v2_jobs quarantined` lists
   each quarantined job with its last failures (reason and client version). Release with `v2_jobs release JOB`
   only the jobs whose failures are status `error` from 406e5500 (states wider than 128 bits, which the new
   worker packs). Look at any other reason first (`v2_jobs failures JOB`): a job quarantined for `bad_seed` or
   `path_overflow` would fail the same way on the new worker (same replay, same 1,024-token limit), and one
   quarantined for `unknown_chain` may. `release --all` reopens every quarantined job, whatever its reason.
6. Client 3.1.0 arrives with the same `git pull`; `min_client_version` can stay 3.0.0.

## 7. Load rules (added 2026-09-23 after the load analysis; these override §3/§4 where they differ)

Server load must depend on wall-clock only, never on how many tree nodes a job turns out to have. Job durations
range from 1 ms to hours; a worker on trivial jobs completes ~20/s (process spawn), so "one job = one server
event" is not acceptable.

1. **Shallow roots.** Campaigns are seeded at layer 3 or 4 (hundreds to a few thousand roots per exit), never at
   a layer where most roots are trivial. The listing is exact at every K (§2.8), so a deeper layer is safe with
   the current worker, but it only adds trivial roots.
2. **One report per worker per window.** A worker leases a job J and works a LOCAL stack for a window of
   `split_after_s` seconds: run J; if it splits, push its REMAINING seeds locally (LIFO) and keep popping, each
   run capped at the time left in the window. Trivial children finish locally and never reach the server. At
   the window's end the worker reports J once with the whole local tree (§7.5). A window therefore contains at
   most ~2 splits and a report at most a few hundred nodes.
3. **Absorb trivial children before hand-back.** Before reporting, spend up to `absorb_total_s` probing the
   still-open local seeds for `absorb_probe_s` (10) seconds each, **deepest first** (REMAINING lists the cursor
   and the top of the stack first: the cheap entries; review M18). A probe that exhausts is reported done. A
   probe that splits is kept as a split node with its summary and its REMAINING children (none of its time is
   lost), and the pass then continues with that probe's own children only: every seed still to come is no
   deeper than the one that split, i.e. larger. A probe that cannot expand its root in the probe time ends the
   pass. When the budget (or a stop) ends the pass inside the children of the last kept split, that split stays
   a split only if the work kept under it (its probe plus its children's probes) is at least 1 s per extra open
   job it creates (children never probed, minus one); otherwise it goes back to one `open` node like a
   fold-back (its children dropped, its deepest level kept). This matters for the budget-cut last probe, which
   gets only what is left, nearly always splits, and would otherwise hand its whole REMAINING (mostly trivial)
   to the pool; the work dropped instead is less than 1 s per job avoided. The open pool still gets the
   unprobed siblings at each level of the chain (no deeper than a child that did not finish in a probe, so
   mostly large, a few trivial) and any child the budget did not reach under a split that saved enough.
   The budget comes with the lease (§3.6: 120 s in a full window, 60 s in a ramp window, 0 in the endgame).
4. **The server sets the window.** The lease response carries `split_after_s`, `absorb_total_s` and
   `window_mode` (`full`, `ramp` or `endgame`, §3.6). The client takes a window's length and absorption budget
   when the window **starts**, from the latest lease response (or a heartbeat `window`, when the server sends
   one), not from the lease of that job (review M108); a fingerprint re-run keeps the window its grant carries.
   A campaign may set `min_window_s` (default 1 s, lower for tests) and `split_after_nodes` (adds
   `--split-after-nodes N`, tests).
5. **Tree reports.** `POST /api/v2/report` body:
   `{token, job_id, src_hash, nodes:[{seed, parent, status, summary?, level?, unresolved?}]}` where every `seed`
   extends the job's seed (token prefix, may equal it only for the job itself), `parent` is the job's seed or
   the seed of an earlier node in the list, `status` ∈ done|split|open; `done` requires a `summary` with
   `status:"exhausted"` and the fingerprint fields; `split` requires its summary and at least one node in the
   list whose parent it is; `open` has no summary but MAY carry a `level` (with its `depth`): a probe that ran
   out of time or a run that split still found real levels, and the deepest one belongs to the explored part,
   which no child will revisit — dropping it would lose a champion (found by the chaos test: status best 53 vs
   true 58). The job itself is the first node. All inserted in one transaction; the job's status becomes done
   or split accordingly. ≤ 5,000 nodes per report. The old `summary`+`remaining` shape stays accepted as the
   one-node case.
6. **Client-level batching.** A client sends at most one lease request and one
   `POST /api/v2/reports {token, reports:[…]}` (an array of §7.5 bodies and failure reports) per
   `batch_interval_s` (10), plus one heartbeat per 30 s. Lease requests are sized by expected work (review
   M108): at most 2 × workers jobs are held locally (running + queued; the job of a worker that retires after
   the worker count was lowered does not count), and a worker gets its next job when its current window has
   less than min(`lease_ahead_s`, window) left; a job's expected work is the mean window time at its depth (a
   depth nothing is known about counts as a whole window), and an idle worker always gets one. An idle worker
   with nothing queued may lease sooner than `batch_interval_s` (at least 2 s apart); a failed lease request
   waits a whole batch interval. The local queue runs the server's lease order: largest `est_s` first when the
   grant carries it (then shallowest, then grant order), and grant order for grants without it (the server
   sent them in its order). The server caps at `lease_cap` (200) per request and 3 × workers + 200 held.
   Failed batches stay in the outbox.
7. **No throttling between friends.** No per-token rate limit. The per-IP limit is a sanity cap honest clients
   cannot reach (600/min).
8. **Nothing scales with rows on the request path.** Status, audits, pools and the campaigns summary come from
   the stats worker's cache (§3.7); audits of campaigns nobody runs are refreshed hourly; level re-verification
   happens only for a new per-exit best.
9. **Server write path.** `PRAGMA synchronous = NORMAL` (WAL; a power loss can drop the last committed reports,
   which only returns those jobs to the pool when their leases expire, so it is safe), one transaction per
   report batch (validate every element first, apply all valid ones together), prepared statements only, no
   per-node JSON re-parsing on the hot path.

Worst-case rates under these rules (40 clients, 1,280 workers): 9.3 req/s; rows/s ≤ 128 one-node reports + 0.7
splits × 300 children ≈ 340 rows/s; ≤ 100 KB per report; ≤ 2M rows for a 5,000 CPU-hour campaign. Realistic
deployment (owner, 2026-09-23): 10 clients × 16 workers = 160 workers; the 40 × 32 cap is a configuration
ceiling, not a load target. Acceptance by `v2/test_load.sh` before launch: at 10 × 16 with 10-s jobs, report p95
latency < 1 s and server CPU < 30 %; at 30 × 16 with 2-s jobs (3× stress) zero errors or timeouts. Measured on
the protocol-3 server (2026-09-25): 10 × 16 p95 lease 77 ms, reports 90 ms, heartbeat 55 ms, 0 errors, server CPU
about 6 % on average; 30 × 16 p95 540 ms (all clients fire together from one test process), 0 errors. (The first
measurement, 2026-09-23 before optimisation, saturated one core at ~1,000 rows/s with report p95 25 s; §7.9
fixed it.)

## 8. Control-panel GUI (added 2026-09-23 from the owner's brief)

The local page is a control panel, not a status readout. It must let a volunteer understand the system and
steer it without reading any docs.

Server support:
- `lease {token, n, exit?}`: `exit` is a preference. Grant from that exit's open jobs first; if it has none,
  grant from any exit and set `exit_fallback: true` in the response so the panel can say so.
- The `heartbeat` response carries `me`: `{jobs_done, splits, nodes_done, cpu_s, states, solver_calls, best:
  {moves, code, exit, at} | null, rank, contributors, first_seen}` computed for this token's *name* (a person may
  re-register), and `exits`: per exit `{roots, roots_covered, open, leased, done, split, cpu_s, best: {moves,
  code, by, at} | null, clean, exact, progress}`. "Done" for an exit is reported as covered roots / roots plus
  CPU hours; the total work is unknown, so no percentage of work is shown (the est_s forecast is labelled as
  one).

Panel sections (top to bottom):
1. **Controls**: name (fixed), workers (a number input; Enter or Apply; a clamp to `workers_max` is explained),
   the exit selector (Any, or one of the campaign's exits, with each exit's covered roots and open jobs in the
   label), Pause/Resume, Stop. Every control that is not instantaneous shows a "working…" state until `/state`
   reflects it: a workers change ("2 workers finishing their jobs…"), an exit change ("current jobs finish
   first; new leases use exit 7"), pause ("freezing…" until all workers show T), stop ("waiting for N workers to
   print their remaining work…").
2. **What is happening**: per worker: job seed, exit, window time left, local stack size, nodes done this
   window, phase (search / absorbing / reporting), "retires after this job" when the count was lowered, and the
   current board (grey = undecided cells).
3. **Levels**: four boards with Pathology codes and copy buttons: the best on this machine in the last second,
   in the last hour and this session (fed by STATUS samples and LEVEL lines, so short runs count too), and the
   best known globally for the selected exit, or the longest over all exits with "Any exit" (from `exits`; the
   server joins code rows with newlines, the panel shows and copies them joined with `/`). Each with its depth,
   and for the global one who found it and when. A board is redrawn only when it changes, never under the
   pointer or a text selection.
4. **Your stats** (from `me`): jobs done, splits handed back, trivial children absorbed locally, CPU hours,
   states searched, solver calls, your deepest level, rank among contributors, first seen, session uptime, jobs
   this session, mean job length, longest job this session, unresolved candidates sent, failed runs reported.
5. **Campaign**: the per-exit table (roots covered / roots, open, leased, CPU hours, best, status exact /
   covered / candidates open), total CPU hours, active volunteers, the worker version and hash.
6. **How it works**: short explanations of a job (a subtree named by a seed path), windows and splitting (a job
   that outlives its window prints the exact remaining subtrees, which become new jobs; nothing is lost),
   pausing (freezes processes, keeps memory, jobs reserved for up to `lease_s` + `paused_max_s` after each lease
   while the client stays online), stopping (each worker hands back its remaining work, in seconds), changing
   workers (extra workers finish their current job; new ones start at once), leases (a job left unfinished
   returns to the pool after an hour), what a "done" fraction means and why there is no percentage of the total
   work.
7. **Completion**: when the campaign is complete, a banner "Campaign complete" with the closing summary and
   "The client has finished and exits now. You can close this window." (or "waiting for the next campaign" with
   `--keep-going`).

Rules: the page never talks to workers or to the campaign server directly; it only reads `/state` (one request
at a time with a 2-s timeout; after 3 s without an answer it greys out and says so) and posts to the client's
control endpoints. The client fetches `/api/v2/status` at most every 30 s for the campaign table; `me` and
`exits` arrive with each heartbeat. All text from the server is escaped.

## 9. Work stealing and lease hygiene (added 2026-09-24; current rules)

Leases are advisory; coverage is decided by reports (first report wins). When a lease request finds the pool
empty, the server re-leases jobs held by other clients:
- from a **live** holder, only jobs it has queued: granted at least 60 s ago, granted before the holder's last
  `running` list and not in it (so no progress is lost), and not stolen from a live holder within the last
  max(300 s, 5 heartbeats);
- from a holder that is **paused or silent** for max(300 s, 5 heartbeats), any job, running ones included;
- at most max(1, the thief's idle workers) per request; duplicates are never stolen.

The previous holder sees the ids in `drop` on its next heartbeat and removes them (killing the worker if it was
running one). The counters `empty_leases` and `stolen` are per campaign in the `/api/v2/status` totals, and every
steal is an event. The client sends `running` (the running job ids) alongside `holding`.

Release by omission: when a heartbeat carries `holding`, every other lease on that token older than 120 s is
returned to the pool (`released`). A restarted client reuses its token but not its old queue; without this its
abandoned queue was renewed forever. (So a client that is SIGKILLed and restarted keeps its dead process's
queued leases for up to those 120 s.) The Stop heartbeat (`stopping:true`) releases everything at once.

## 10. Where the implementation differs from PROTOCOL3.md, and known gaps

Differences (deliberate; the stage reports give the reasons):
- **Worker.** An extra void status `unknown_chain` (exit 6) and the chain guard (§2.6); REMAINING may list a
  chain head, and a split may be forced early or deferred; no split before the second expansion, and a due split
  with an empty stack ends exhausted (§2.7); `debug`, `time_cap` and `dedup_full` exist outside protocol use.
  There is no `unresolved_overflow` status: the worker keeps the 1,000 deepest candidates and reports
  `unresolved_dropped_max`, and the server treats a dropped depth above the best as not exact. SUMMARY has the
  extra fields of §2.3, and `accepted` excludes UNKNOWN states. LAYERINFO has the extra keys `protocol`, `seed`,
  `above` and `root_depth`, there is one header per exit, roots can be deeper than K, and the refuse-K≥5 fallback
  was not needed (§2.8). The solver widths are 64-bit, 16-byte and 32-byte (§2.9).
- **Client.** It sends `client_version` 3.1.0 (the gate is ≥ 3.0.0). A job whose own REMAINING cannot fit is
  handed back with failure `too_big`, not a silent release. The watchdog also fires on an overrun of
  max(600 s, split_after), and it moves its baselines after a clock gap. It voids more than PROTOCOL3 lists (a
  candidate depth ≠ path length, a bad cause, a count mismatch, a SUMMARY without flags or unresolved), and
  REMAINING equal to the seed is a failure only on the worker's own timer. Completion delivery is bounded by
  120 s. The worker is pinned to a private copy. It holds at most 2 jobs per worker, takes the window when it
  starts, uses the lease's `absorb_total_s`, and handles Ctrl-Z.
- **Server.** Versionless clients get 400/403 instead of 426. A complete campaign's tokens get 410 once a newer
  campaign is open. Every split node needs its SUMMARY, and deterministic 400 refusals count as failures.
  Candidates are deduplicated by code, and `cand_dropped_max` and contradicted candidates also block exact.
  v2_finish refuses `--min-walls` campaigns and solves the champion before it adds a proof. The job statuses
  `superseded` and `cancelled`, the full/ramp/endgame windows, the events table, the registration cap and the
  duplicate rules are additions. Without estimates the lease order is shallowest first. Stealing may take any
  job of a paused or silent holder.

Known gaps (not done; none affects exactness):
- The server does not send a heartbeat `window` yet (the client supports one), so a queued job starts with the
  window of the latest lease response, at most about `lease_ahead_s` old. Grants carry no `est_s`.
- status() is still a set of GROUP BYs (in the stats worker, never on the request path); there is no per-exit
  "published" marker (M69) and no `verify_status` column (N5); `verifyLater` only logs a definitive
  "unsolvable".
- The single-solve table and the reference table match on the first 64-bit key only (a collision can only
  cause a false accept, which the server's re-solve of reported levels catches).
- Not implemented after measuring: cross-job transposition dedup (M106, below 0.15 % on 6x6) and the
  mandatory-hole prune for campaigns (M104, no effect with 0 holes).
- The M42 eviction rule keeps a key-hash fraction of the threshold bucket, but the canonical key's top bits are
  biased low (it is a minimum over symmetry images): harmless for ≤ 2 images (every 6x6 exit), a possible
  slowdown at a 5x5 centre exit; to be fixed with the next worker change.
- The client's default worker count is not capped by RAM (plan 0.5 GB per worker).
- Nothing checks the worker's KNOBS line. The client reads only PROTOCOL and LIMITS, and the server only the
  hash. A hand `cc` build that embeds the release hash and adds `-D` table-size overrides would be accepted as
  the release, although it is not the tested build: its fingerprints can differ from the release's (a false
  mismatch), and so can its capacity and limits. The docs say to add no `-D` flag to a plain build (§1.6,
  §2.1). A client check of KNOBS against the release defaults is a candidate for the next client.

## 11. History

- 2026-09-23: design frozen for the restart; §7 load rules and §8 panel added.
- 2026-09-24: lease renewal, release by omission and work stealing (§9). Campaign #1 complete and published.
  The full review (`REVIEW-2026-09-24.md`) and the protocol-3 contract (`PROTOCOL3.md`). Campaign #2 started on
  the first protocol-3 release (406e5500).
- 2026-09-25: the protocol-3 stages (worker W1-W4, client C1-C2, server S1-S3, tests T1-T2) folded into this
  file; release candidate c3ee5d2b.
