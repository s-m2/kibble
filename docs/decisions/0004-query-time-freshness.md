# 0004. Freshness is guaranteed at query time

- Status: accepted
- Date: 2026-10-31

## Context
Agents edit files quickly and trust tool output. A stale index gives confident wrong
answers. Watchers and hooks drop events and are agent-specific.

## Decision
Every query does a cheap stat sweep of the manifest plus new-file and delete
detection, refreshes dirty files through `replace_file`, and returns freshness
metadata. Idle refresh, git-aware refresh, watchers and agent hooks are optimizations
layered on top, never the source of correctness. Above a dirty-file cap, return
current results flagged `stale` and refresh in the background.

## Consequences
Correctness does not depend on any event source. Each query pays a small sweep cost
(mitigate with `git ls-files` enumeration and mtime/size short-circuits). Evals must
include edit-then-query, new-file-then-query, and "optimizations disabled" cases.
