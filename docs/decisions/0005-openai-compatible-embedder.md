# 0005. OpenAI-compatible embeddings from day one

- Status: accepted
- Date: 2026-10-31

## Context
The PoC uses OpenRouter; the product should also work with local inference, LAN
servers and any OpenAI-shaped provider.

## Decision
A single embedder speaking the OpenAI embeddings wire format, configured by base URL,
model, optional key env var, batch size, concurrency, token limit, optional dimension
truncation, and document/query prefixes. Vectors are identified by an `embedder_id`
hash of the settings that change them. Dimension is probed at startup. No silent
downloads at query time.

## Consequences
Hosted, LAN and local setups share one code path. We must handle uneven server
behavior (batch limits, truncation, prefixes, TLS, cold starts) and provide a
`doctor` command. Chunk text leaves the machine when a hosted endpoint is used;
document it. A fake OpenAI-compatible server backs all tests.
