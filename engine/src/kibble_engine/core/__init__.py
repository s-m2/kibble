"""Pure-Python core: chunking, retrieval, fusion, refresh, freshness.

Rule: nothing in here imports FastAPI or any web code. The API layer wraps
these functions; the CLI and the eval harness call them directly.
"""
