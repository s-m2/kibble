# Configuration

Kibble talks to any server that speaks the OpenAI embeddings API: hosted providers,
local servers, or a gateway on your LAN. There is **no default endpoint or model**;
code is never sent anywhere you didn't choose.

## Where settings live

`.kibble/config.toml` in the indexed repository (gitignored), plus environment
variables named `KIBBLE_EMBED_<SETTING>` which win over the file:

```toml
[embedder]
base_url    = "http://localhost:11434/v1"
model       = "nomic-embed-text"
doc_prefix  = "search_document: "
query_prefix = "search_query: "
```

**Secrets never go in the file.** `api_key_env` names the environment variable that
holds the key:

```toml
api_key_env = "OPENROUTER_API_KEY"
```

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `base_url` | required | Endpoint root, usually ending in `/v1` (`/embeddings` is appended) |
| `model` | required | Model name as the server knows it |
| `api_key_env` | none | Environment variable holding the key; omit for unauthenticated servers |
| `batch_size` | 64 | Texts per request; rejected batches are split automatically |
| `concurrency` | 4 | Parallel requests; use 1-2 for a single local GPU, 1 on a quota-limited tier |
| `requests_per_minute` | none | Self-throttle: spaces requests evenly so a per-minute quota is never hit |
| `max_input_chars` | 8000 | Longer inputs are truncated (the prefix is kept). Keep the chunker's `max_chars` below this |
| `timeout_s` | 30 | Per-request timeout |
| `probe_timeout_s` | 120 | Timeout for the first request, which may include model load |
| `max_retries` / `backoff_base_s` | 4 / 1.0 | Retry policy for connection errors, 429 and 5xx |
| `dimensions` | none | Ask the *server* to shorten vectors (only if it honours the parameter) |
| `slice_dimensions` | none | Keep the first N dimensions *client-side* and re-normalise (for Matryoshka-style models). Use instead of `dimensions` |
| `doc_prefix` / `query_prefix` | empty | Instruction prefixes some models need; see below |
| `verify_tls` / `ca_bundle` | true / none | For LAN servers with a private CA, set `ca_bundle`; disabling verification is a loud opt-in |

## Typical setups

Defaults differ by version and install, so verify the URL and model name for yours.

| Setup | `base_url` | Notes |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | set `api_key_env`; e.g. model `openai/text-embedding-3-small` |
| Ollama | `http://localhost:11434/v1` | no key; pull the model first |
| LM Studio | `http://localhost:1234/v1` | no key; load an embedding model in the app |
| llama.cpp server | `http://localhost:8080/v1` | start it with embeddings enabled |
| LAN gateway | `https://gpu-box.lan/v1` | use https; add `ca_bundle` for a private CA |

## Prefixes

Many open models are trained with different instructions for documents and queries
(for example E5 uses `passage: ` / `query: `, nomic-embed uses `search_document: ` /
`search_query: `). Skipping them silently lowers retrieval quality. Check the model card.
Prefixes are part of the vector's identity: changing `doc_prefix` re-embeds.

## Check your setup

```bash
uv run kibble-engine doctor
```

It makes real requests (about 7 in a healthy run, which matters on a quota) and reports,
with a suggested fix for anything that is off: reachability and auth, the vector
dimension and first-request latency, whether your batch size and input length are
accepted, whether repeat calls give the same vector, a small retrieval smoke test that
also detects prefixes that hurt (for example because the endpoint adds its own), missing
prefixes for well-known model families, and keys sent over cleartext http to a non-local
host. The retrieval test is a smoke test, not a benchmark.

## Switching models

Vectors are keyed by an `embedder_id` made from the model, output size, document prefix
and `dimensions`. Changing any of them re-embeds; the old vectors stay cached, so
switching back is free. The base URL is not part of the id, so moving the same model
between providers reuses the cache (quantised local variants differ slightly).

## Example: NVIDIA Nemotron-3-Embed-8B (self-hosted with vLLM)

Facts below are from the model card (huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16);
re-check it, since versions and recommendations change.

- 8B parameters, 4096-dimensional L2-normalised vectors, 32k-token context (the card's
  own evaluations used 4096 tokens; Kibble's chunks are far smaller).
- Documents need the prefix `passage: ` and queries `query: `.
- vLLM serves it, including the OpenAI-compatible `/v1/embeddings` endpoint, which is
  what Kibble uses (the card notes you add the prefixes yourself there). Supported
  hardware per the card: NVIDIA Ampere, Hopper or Blackwell on Linux; tested on A100 and
  H100 80GB.

```bash
vllm serve nvidia/Nemotron-3-Embed-8B-BF16      # default port 8000
```

```toml
[embedder]
base_url     = "http://localhost:8000/v1"
model        = "nvidia/Nemotron-3-Embed-8B-BF16"   # must match the served model name
doc_prefix   = "passage: "
query_prefix = "query: "
batch_size   = 16
concurrency  = 2
# slice_dimensions = 1024     # optional, see below
```

Wait for vLLM to finish loading (it only accepts connections once the model is ready),
then run `uv run kibble-engine doctor`. If you serve a local checkpoint, pass
`--served-model-name nvidia/Nemotron-3-Embed-8B-BF16` so the names match.

**Index size.** At 4096 dimensions a vector is 16 KiB, stored in the embedding cache and
again in the sqlite-vec index: roughly 330 MB per 10,000 chunks, and every query scans
all of it. The card says the first 2048 or 1024 dimensions "remain highly functional"
when re-normalised, so `slice_dimensions = 1024` cuts that 4x. How much retrieval quality
that costs on *code* is exactly what the eval harness should measure.

**Smaller GPU?** The card also lists a 1B variant (`Nemotron-3-Embed-1B-BF16`) with lower
benchmark scores (72.38 vs 78.46 on its RTEB-16 table).

## Example: OpenRouter free tier (Nemotron-3-Embed-1B)

Checked against OpenRouter's pages and docs in November 2026; re-check them.

- The free embedding model there is `nvidia/nemotron-3-embed-1b:free`: the **1B** model
  (2048-dim, 32k context, `query: ` / `passage: ` prefixes per its card). I did not find
  an 8B listing; `GET /api/v1/embeddings/models` shows what your account can use.
- It has a single provider (NVIDIA), and OpenRouter lists its quantization as
  "unknown", so you cannot verify it is BF16.
- **Data policy: do not send private code.** The model page says the free endpoint is for
  non-confidential data, that usage is logged, and that NVIDIA uses it to improve its
  products. Use it for open-source repositories (such as the eval repos). Use a local
  endpoint for anything private.
- **Limits are per request, not per token:** 20 requests/minute, plus 50 requests/day
  (1,000/day once you have bought at least $10 of credits). Large batches are therefore
  the efficient way to spend the quota: at `batch_size = 64`, 50 requests is about 3,200
  chunks. Run `doctor` first to confirm the batch size is accepted; a rejected batch is
  split automatically, but every split costs extra requests.

```toml
# .kibble/openrouter.toml   (use with: kibble-engine doctor --config .kibble/openrouter.toml)
[embedder]
base_url            = "https://openrouter.ai/api/v1"
model               = "nvidia/nemotron-3-embed-1b:free"
api_key_env         = "OPENROUTER_API_KEY"
doc_prefix          = "passage: "
query_prefix        = "query: "
batch_size          = 64
concurrency         = 1
requests_per_minute = 15
# slice_dimensions  = 1024     # optional: 2048 -> 1024 halves index size
```

**Unknown quantization.** Vectors from two different builds of "the same" model are close
but not identical, and mixing them in one index lowers quality without any error.
`check_canary` (in `kibble_engine.embedders`) embeds one fixed text when an index is first
used with an embedder, records the vector, and on later runs reports `drift` if the
endpoint's output has moved (cosine below 0.999), which is the cue to re-embed. `doctor`
also checks that repeated calls agree. The canary is not yet called automatically; it
will be wired into indexing.

## Example: local Ollama with nomic-embed-text

Private code stays on your machine and there are no quotas.

```bash
ollama pull nomic-embed-text
```

```toml
# .kibble/ollama.toml
[embedder]
base_url        = "http://localhost:11434/v1"
model           = "nomic-embed-text"
doc_prefix      = "search_document: "
query_prefix    = "search_query: "
concurrency     = 2
max_input_chars = 4000
```

nomic-embed-text produces 768-dimensional vectors and expects those prefixes. Ollama's
default context window may be smaller than the model's maximum, so keep inputs modest
(`max_input_chars`) and let `doctor` confirm the length it accepts.

## Choosing a model

| Situation | Suggested model |
|---|---|
| Private or confidential code | Local: nomic-embed-text on Ollama, or Nemotron-3-Embed self-hosted |
| Open-source repos, evals, trying things out | OpenRouter free `nemotron-3-embed-1b:free`, paced and batched |
| Best quality, you own a large GPU | Nemotron-3-Embed-8B via vLLM |
| Text-only code retrieval | Prefer Nemotron-3-Embed-1B over 3.5-1B: the 3.5 model adds image support you won't use, and its card documents serving through `/v2/embed` or a chat template rather than plain `/v1/embeddings` with prefixes |

The embedder choice is itself an eval dimension: the same question set against
nomic (768), Nemotron-1B (2048), and Nemotron-1B sliced to 1024 will show what the extra
dimensions buy on code.
