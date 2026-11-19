# 0011. Freshness: scan, stat sweep, windowed indexing

- Status: accepted
- Date: 2026-11-15

## Context
ADR 0004 made freshness a query-time guarantee. This records how it is built, and how
files get from disk into the index without burning quota or leaving half-written state.

## Decision
- **Discovery** follows git when available (`ls-files --cached --others
  --exclude-standard`), so `.gitignore` is honoured without re-implementing it; otherwise
  a directory walk prunes junk directories. Stat-level filters drop binaries by
  extension, lockfiles, minified bundles, symlinks, empty and oversized files. Files that
  turn out to be binary by content are recorded with no chunks, so sweeps don't re-read them.
- **The sweep** compares (mtime, size) to the manifest and reads a file only when they
  differ. Same content with new stat data is a *touch* (no embedding). To close the
  racy-timestamp hole (an edit within the filesystem's timestamp granularity that keeps
  mtime and size), a file whose mtime is within two seconds of its recorded index time is
  hashed anyway, as git does.
- **The stat is taken before the content is read**, so an edit racing the indexer is
  never recorded as indexed; the next sweep sees a different mtime.
- **One refresh path** (`Refresher.refresh`) serves every trigger. Callers choose the
  paths (all, or the few they know changed) and an inline budget (`max_files`); what
  isn't done is reported as `pending` and the report says whether the index is `stale`.
  Refreshes are serialised by a lock.
- **Embeddings are batched across files** in windows (by chunk count and file count),
  because quota-limited endpoints count requests. Each file is committed separately and
  atomically (`replace_file`), so a crash or failure never leaves a half-written file.
- **Failure semantics are conservative.** A rejected request is isolated to the file that
  caused it. An error that will keep happening (auth, credits, rate limit, outage) stops
  the run; nothing in flight is written, old entries stay, and the files are `pending`.
- **Session** wires config, `.env` (read under the real environment, never into
  `os.environ`), embedder probe, index open, drift canary and refresher for the CLI and
  later the API.

## Consequences
Queries pay one scan and a stat per file (no reads on the fast path); a changed file costs
one read plus its embedding. Stale-but-consistent beats fresh-but-inconsistent during an
outage: keyword search for a new file doesn't work until it can be embedded (a possible
later improvement is storing chunks without vectors). Edits that keep size and mtime *and*
happened long before indexing are, as in git, not detectable by stat alone. Tags
(`test`, `vendor`, `generated`) are recorded for later down-weighting but not yet used.
