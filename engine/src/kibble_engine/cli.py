"""Command line entry point. Plain argparse on purpose.

Exit codes: 0 ok; 1 something failed or the index is stale; 2 configuration error;
3 the embedding endpoint no longer matches the index (drift).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from kibble_engine import __version__

if TYPE_CHECKING:
    from kibble_engine.core.session import Session
    from kibble_engine.store.models import StoredChunk


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from kibble_engine.api.app import create_app

    token = None if args.no_auth else os.environ.get("KIBBLE_TOKEN")
    if token is None and not args.no_auth:
        print(
            "KIBBLE_TOKEN is not set; refusing to start (use --no-auth for local dev).",
            file=sys.stderr,
        )
        return 2

    port = args.port or _pick_free_port("127.0.0.1")
    # Single handshake line on stdout so a launcher can discover the port.
    # Everything else (logs, diagnostics) goes to stderr.
    print(json.dumps({"event": "listening", "host": "127.0.0.1", "port": port}), flush=True)
    uvicorn.run(create_app(token), host="127.0.0.1", port=port, log_level="warning")
    return 0


def _repo(args: argparse.Namespace) -> Path:
    return Path(args.repo).resolve()


# ---------------------------------------------------------------------- doctor


def _doctor(args: argparse.Namespace) -> int:
    from kibble_engine.core.session import environment_for
    from kibble_engine.embedders import (
        ConfigError,
        build_embedder,
        load_config,
        resolve_api_key,
        run_doctor,
    )

    repo = _repo(args)
    env = environment_for(repo)
    try:
        config = load_config(Path(args.config) if args.config else None, env=env, cwd=repo)
        has_key = resolve_api_key(config, env) is not None
        embedder = build_embedder(config, env)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    print(f"checking {config.endpoint} (model {config.model})")
    with embedder:
        checks = run_doctor(embedder, config, has_key=has_key)
    labels = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}
    for check in checks:
        print(f"{labels[check.status]} {check.name}: {check.detail}")
    return 1 if any(c.status == "fail" for c in checks) else 0


# ----------------------------------------------------------------------- index


def _open(args: argparse.Namespace) -> Session | int:
    """Open a Session, or print the problem and return the exit code to use."""
    from kibble_engine.core.session import open_session
    from kibble_engine.embedders import ConfigError, EmbedderError
    from kibble_engine.store.sqlite import StoreError

    try:
        return open_session(_repo(args), config_path=Path(args.config) if args.config else None)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except EmbedderError as exc:
        print(f"embedder error: {exc}", file=sys.stderr)
        return 1
    except StoreError as exc:
        print(f"index error: {exc}", file=sys.stderr)
        return 1


def _index(args: argparse.Namespace) -> int:
    session = _open(args)
    if isinstance(session, int):
        return session
    with session:
        print(f"indexing {session.root}  (embedder {session.embedder.info.embedder_id})")
        if session.canary.status == "drift":
            if not args.ignore_drift:
                print(f"DRIFT: {session.canary.message}", file=sys.stderr)
                print(
                    "Delete .kibble/index.db to rebuild from scratch, or pass --ignore-drift "
                    "to continue anyway (new vectors will not match the old ones).",
                    file=sys.stderr,
                )
                return 3
            print(f"warning: {session.canary.message}", file=sys.stderr)

        last = [0]

        def progress(done: int, total: int) -> None:
            if done == total or done - last[0] >= 10:
                last[0] = done
                print(f"  {done}/{total} files", flush=True)

        report = session.refresher.refresh(max_files=args.max_files, on_progress=progress)

    print(
        f"scanned {report.scanned} files: {report.new} new, {report.changed} changed, "
        f"{len(report.removed)} removed, {report.unchanged} unchanged "
        f"({report.skipped_by_scan} skipped as non-indexable)"
    )
    print(
        f"indexed {len(report.refreshed)} files: {report.embedded_chunks} chunks embedded, "
        f"{report.cached_chunks} from cache, in {report.duration_s:.1f}s"
    )
    for path, reason in report.failed[:10]:
        print(f"FAILED {path}: {reason}", file=sys.stderr)
    if report.aborted:
        print(f"STOPPED: {report.aborted}", file=sys.stderr)
    if report.pending:
        print(
            f"{len(report.pending)} files still pending; run `index` again to continue",
            file=sys.stderr,
        )
    return 1 if report.stale else 0


# ---------------------------------------------------------------------- search


def _first_line(text: str, width: int = 90) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= width else line[: width - 1] + "…"


def _label(chunk: StoredChunk) -> str:
    if chunk.symbol:
        return f"{chunk.parent_symbol}.{chunk.symbol}" if chunk.parent_symbol else chunk.symbol
    return chunk.kind


def _search(args: argparse.Namespace) -> int:
    from kibble_engine.core.search import SearchConfig, SearchFilters

    session = _open(args)
    if isinstance(session, int):
        return session
    with session:
        response = session.searcher.search(
            args.query,
            config=SearchConfig(k=args.k, mode=args.mode),
            filters=SearchFilters(
                path_prefix=args.path,
                exclude_tags=frozenset(args.exclude_tag or ()),
            ),
            fresh=not args.no_refresh,
        )

    stale = bool(response.freshness and response.freshness["stale"])
    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
        return 1 if stale else 0

    if response.freshness is not None:
        f = response.freshness
        note = f"{f['refreshed_files']} refreshed, {f['removed_files']} removed"
        if stale:
            note += f"; STALE ({f['pending_files']} pending, {len(f['failed_files'])} failed)"
        print(f"freshness: {note} ({f['duration_ms']} ms)")
    for reason in response.degraded:
        print(f"DEGRADED: {reason}", file=sys.stderr)
    v_weight, k_weight = response.weights
    print(
        f"query: {response.shape}  weights vector x{v_weight:.2f}, keyword x{k_weight:.2f}  "
        f"keyword terms: {response.keyword_query or '-'}"
    )
    if not response.hits:
        print("no results")
    for i, hit in enumerate(response.hits, start=1):
        chunk = hit.chunk
        sources = " ".join(
            f"{name}#{rank}"
            for name, rank in (("v", hit.vector_rank), ("k", hit.keyword_rank))
            if rank is not None
        )
        where = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        print(f"{i:2d}. {hit.score:.4f}  {where}  {_label(chunk)}  [{sources}]")
        print(f"      {_first_line(chunk.content)}")
    return 1 if stale else 0


# ---------------------------------------------------------------------- status


def _status(args: argparse.Namespace) -> int:
    from kibble_engine.core.refresh import plan_changes
    from kibble_engine.core.session import INDEX_RELATIVE_PATH
    from kibble_engine.store.sqlite import SqliteStore, StoreError

    root = _repo(args)
    index_path = root / INDEX_RELATIVE_PATH
    meta = SqliteStore.peek_meta(index_path)
    if meta is None or "embedder_id" not in meta:
        print(f"no index at {index_path} (run `kibble-engine index`)")
        return 1
    try:
        store = SqliteStore.open(
            index_path, embedder_id=meta["embedder_id"], embed_dim=int(meta["embed_dim"])
        )
    except StoreError as exc:
        print(f"index error: {exc}", file=sys.stderr)
        return 1
    with store:
        counts = store.counts()
        scan, result = plan_changes(root, store)
    print(f"index: {index_path}")
    print(f"embedder: {meta['embedder_id']}  (vector backend: {store.vector_backend})")
    print(
        f"contents: {counts['files']} files, {counts['chunks']} chunks, "
        f"{counts['symbols']} symbols, {counts['embedding_cache']} cached embeddings"
    )
    print(
        f"working tree: {len(result.new)} new, {len(result.changed)} changed, "
        f"{len(result.deleted)} deleted, {result.unchanged} unchanged "
        f"({'git' if scan.used_git else 'directory walk'})"
    )
    dirty = len(result.new) + len(result.changed) + len(result.deleted)
    return 1 if dirty else 0


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kibble-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_repo(p: argparse.ArgumentParser, *, config: bool = True) -> argparse.ArgumentParser:
        p.add_argument("--repo", default=".", help="repository root (default: current directory)")
        if config:
            p.add_argument("--config", help="config TOML (default: <repo>/.kibble/config.toml)")
        return p

    sub.add_parser("version", help="print the engine version")

    serve = sub.add_parser("serve", help="run the local HTTP engine (127.0.0.1 only)")
    serve.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    serve.add_argument("--no-auth", action="store_true", help="disable the token check (dev only)")

    with_repo(sub.add_parser("doctor", help="check the embedder configuration and endpoint"))

    index = with_repo(sub.add_parser("index", help="index or refresh a repository"))
    index.add_argument("--max-files", type=int, help="index at most this many files this run")
    index.add_argument(
        "--ignore-drift", action="store_true", help="continue even if the endpoint output changed"
    )

    search = with_repo(sub.add_parser("search", help="hybrid search: vector + keyword, fused"))
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10, help="number of results (default 10)")
    search.add_argument(
        "--mode",
        choices=["hybrid", "vector", "keyword"],
        default="hybrid",
        help="use only one signal (for debugging and ablations)",
    )
    search.add_argument("--path", help="only files whose path starts with this prefix")
    search.add_argument(
        "--exclude-tag",
        action="append",
        choices=["test", "vendor", "generated"],
        help="drop files with this tag (repeatable)",
    )
    search.add_argument(
        "--no-refresh", action="store_true", help="skip the freshness check (not recommended)"
    )
    search.add_argument("--json", action="store_true", help="machine-readable output")

    status = sub.add_parser("status", help="show index contents and pending changes")
    with_repo(status, config=False)

    args = parser.parse_args(argv)
    if args.command == "version":
        print(__version__)
        return 0
    handlers = {"doctor": _doctor, "index": _index, "search": _search, "status": _status}
    if args.command in handlers:
        return int(handlers[args.command](args))
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
