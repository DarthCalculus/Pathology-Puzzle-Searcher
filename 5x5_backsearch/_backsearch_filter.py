#!/usr/bin/env python3
"""Stream filter for parallel backsearch output.

Reads merged worker output (lines prefixed with `[task N] `) on stdin,
keeps only blocks corresponding to NEW GLOBAL best depths, debounces them
by DEBOUNCE_S seconds (the worker's old behavior), and strips the
`[task N] ` prefix on emitted blocks so the stream looks like a single
search.  All other `[task N]` chatter is suppressed.  Lines without that
prefix (e.g. wrapper notices) pass through.

Semantics: when a new-best event arrives that exceeds the last-printed
depth, a 1 s timer starts (anchored to the *first* such event since the
last print — not extended by subsequent events).  When the timer fires,
the CURRENT best (highest-depth completed block across all tasks) is
printed, which is not necessarily the block that triggered the timer.
This mirrors the worker's old single-process debounce.

Implementation note: stdin is consumed with raw os.read() to bypass
Python's TextIOWrapper buffering — otherwise select() can report "not
ready" while a chunk is sitting in Python's internal buffer with unread
lines, stalling the filter indefinitely.
"""
import os, sys, re, time, select

DEBOUNCE_S = 1.0
PREFIX_RE  = re.compile(r'^\[task (\d+)\] (.*)$')
NEWBEST_RE = re.compile(r'^(\d+)\s+\([\d.]+[smh]\)\s*$')

STDIN_FD = sys.stdin.fileno()

per_task = {}     # task_id -> {"depth": int, "lines": [...], "complete": bool}
last_printed_depth = 0
deadline = None


def process_line(line):
    """Handle one complete line (no trailing newline)."""
    global last_printed_depth, deadline
    m = PREFIX_RE.match(line)
    if m is None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        return
    task_id, content = m.group(1), m.group(2)
    nb = NEWBEST_RE.match(content)
    if nb is not None:
        depth = int(nb.group(1))
        per_task[task_id] = {"depth": depth, "lines": [content + "\n"], "complete": False}
        if depth > last_printed_depth and deadline is None:
            deadline = time.monotonic() + DEBOUNCE_S
        return
    blk = per_task.get(task_id)
    if blk is not None and not blk["complete"]:
        blk["lines"].append(content + "\n")
        if content == "":
            blk["complete"] = True


def maybe_flush():
    global per_task, last_printed_depth, deadline
    best = None
    for b in per_task.values():
        if not b["complete"]:
            continue
        if b["depth"] <= last_printed_depth:
            continue
        if best is None or b["depth"] > best["depth"]:
            best = b
    if best is not None:
        sys.stdout.writelines(best["lines"])
        sys.stdout.flush()
        last_printed_depth = best["depth"]
    per_task = {}
    deadline = None


buf = ""
while True:
    timeout = None
    if deadline is not None:
        timeout = max(0.0, deadline - time.monotonic())
    r, _, _ = select.select([STDIN_FD], [], [], timeout)
    if not r:
        maybe_flush()
        continue
    chunk = os.read(STDIN_FD, 65536)
    if not chunk:
        break
    buf += chunk.decode("utf-8", errors="replace")
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        process_line(line)

# Drain any tail (no trailing newline) and pending state.
if buf:
    process_line(buf)
maybe_flush()
