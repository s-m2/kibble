# 0008. Embedding cache is canonical; vector search has two backends

- Status: accepted
- Date: 2026-11-02

## Context
sqlite-vec is pre-1.0, brute-force, and is a loadable extension that some Python
builds refuse to load. Embeddings cost money and time, so they must survive file
edits, branch switches and model changes. We also want a clean answer to "what
happens when the embedder changes?".

## Decision
- `embedding_cache(chunk_hash, embedder_id, embedding)` is the single source of truth
  for vectors. Vectors are unit-normalised float32 blobs, so dot product, cosine and
  L2 rankings agree.
- `chunk_vecs` (sqlite-vec `vec0`) is only an accelerator over the cache. It is
  dropped and rebuilt whenever it might be stale (embedder change, or writes made
  while the extension was unavailable, tracked by `meta.vec_index_valid`).
- If the extension can't be loaded, `vector_backend="auto"` falls back to a
  pure-Python scan over the cache. Rankings are identical (tested); only speed
  differs. `vector_backend="vec0"` makes a missing extension a hard error.
- The schema version lives in `PRAGMA user_version`, written in the same
  transaction as the DDL, rather than a row in `meta`.
- Changing embedder: `reset_embedder()` swaps `embedder_id`/dim, rebuilds the
  accelerator, and reports how many chunks lack a vector for the new embedder.
  Other embedders' cached vectors are kept, so switching back is free.

## Consequences
Model swaps and re-indexing are cheap and safe; the tool works on Pythons without
extension loading. Vectors are stored twice when vec0 is active (cache + index),
which is fine at our scale. The pure-Python fallback is O(chunks x dim) per query and
is not meant for very large indexes. There are no schema migrations yet: a version
mismatch means delete `.kibble/` and re-index.
