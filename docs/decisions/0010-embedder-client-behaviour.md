# 0010. How the embedder client behaves

- Status: accepted
- Date: 2026-11-02

## Context
ADR 0005 chose one OpenAI-compatible client for hosted, LAN and local servers. Those
servers disagree on batch limits, input limits, prefixes, auth, rate limiting and
cold starts, and the rest of the engine should not have to care.

## Decision
- **No default endpoint or model.** Required settings; code never goes to a hosted
  provider by default. Settings come from `.kibble/config.toml` and
  `KIBBLE_EMBED_*` variables; secrets only from the environment (`api_key_env`).
- **Plain validated dataclass**, not a settings framework: every rule is visible code.
- **Adapt, don't configure around servers.** A rejected batch is halved recursively;
  a single rejected input is shrunk and retried a few times; over-long inputs are
  truncated up front with the prefix preserved; result order comes from each item's
  `index`; the output size is learned from the first response and enforced after.
- **Retry only what is transient** (connection errors, timeouts, 429, 5xx) with
  exponential backoff and jitter, honouring `Retry-After`. Auth, credits and
  not-found errors are raised immediately, each with a message naming what to check.
- **Typed errors** (`EmbedderAuthError`, `EmbedderNotFoundError`, ...) so the API layer
  can turn them into useful responses instead of "embedding failed".
- **`embedder_id`** covers model, output size, document prefix and `dimensions`; not
  the base URL or the query prefix.
- **`doctor` command** probes a real endpoint and reports fixable problems before a
  long index run starts.
- **One HTTP library switch point** (`embedders/http.py`), and a fake
  OpenAI-compatible server shipped in `kibble_engine.testing` so tests and the eval
  harness need no network and no API keys.

## Consequences
Hosted, LAN and local setups share one tested code path, and failures say what to do
next. The adaptive behaviour can hide a misconfiguration (extra requests when
`batch_size` exceeds the server's limit), which `doctor` and `EmbedderStats` surface.
Length handling is character-based, an approximation of tokens; models with very tight
token limits may need a smaller `max_input_chars`. The client is synchronous; if the
API layer later needs async embedding, the same logic can be ported behind the same
interface.
