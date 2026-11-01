#!/usr/bin/env python3
"""Build the single-file vector index shipped in the container image.

Reads ``data/chunks.jsonl`` (or ``--chunks``), embeds it with the model in
``OPENROUTER_EMBEDDING_MODEL``, and writes::

    data/index.json        chunks + vectors
    data/index.meta.json   the embedding model id and counters

The runtime loads the index only when the recorded model matches the current
``OPENROUTER_EMBEDDING_MODEL``; otherwise it logs a warning and re-indexes from
the corpus.  Changing the embedding model therefore requires rebuilding.

Usage::

    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --chunks data/index/chunks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantrag.config import load_settings  # noqa: E402
from quantrag.embeddings import OpenRouterEmbedder, load_embedding_model  # noqa: E402
from quantrag.errors import ConfigurationError, QuantRagError, RetrievalError  # noqa: E402
from quantrag.index_file import save_index_file  # noqa: E402
from quantrag.retriever import Retriever, load_chunks_jsonl  # noqa: E402

DEFAULT_CHUNKS = Path("data/chunks.jsonl")
FALLBACK_CHUNKS = Path("data/index/chunks.jsonl")
DEFAULT_OUT = Path("data/index.json")
DEFAULT_META = Path("data/index.meta.json")
DEFAULT_BATCH_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_index.py",
        description="Embed a chunks.jsonl file into data/index.json + data/index.meta.json.",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help=(
            "Input chunks.jsonl (default: $QUANTRAG_CORPUS_PATH, then "
            "data/chunks.jsonl, then data/index/chunks.jsonl)"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output index file")
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_META,
        help="Output metadata sidecar",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser


def _default_chunks() -> Path:
    configured = (os.environ.get("QUANTRAG_CORPUS_PATH") or "").strip()
    if configured:
        return Path(configured)
    if DEFAULT_CHUNKS.is_file():
        return DEFAULT_CHUNKS
    return FALLBACK_CHUNKS


async def build_index_file(
    *,
    chunks_path: str | Path,
    out_path: str | Path,
    meta_path: str | Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    """Embed ``chunks_path`` with OpenRouter and write the single-file index."""
    source = Path(chunks_path)
    if not source.is_file():
        raise ConfigurationError(
            f"Chunks file {source} does not exist. Build it with the ingestion pipeline first."
        )
    if batch_size <= 0:
        raise ConfigurationError(f"batch_size must be greater than 0, got {batch_size}.")

    chunks = load_chunks_jsonl(source)
    if not chunks:
        raise ConfigurationError(f"No chunks found in {source}.")

    settings = load_settings(load_env_file=True)
    model = load_embedding_model(os.environ)
    embedder = OpenRouterEmbedder(settings, model)
    retriever = Retriever(embedder)
    indexed = await retriever.index(chunks, batch_size=batch_size)
    if indexed == 0:
        raise RetrievalError(
            f"None of the {len(chunks)} chunks in {source} are indexable; "
            "refusing to write an empty index."
        )

    return save_index_file(
        retriever.store,
        out_path,
        embedder_model=model,
        meta_path=meta_path,
        source=source,
        batch_size=batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks_path = args.chunks or _default_chunks()

    try:
        meta = asyncio.run(
            build_index_file(
                chunks_path=chunks_path,
                out_path=args.out,
                meta_path=args.meta,
                batch_size=args.batch_size,
            )
        )
    except QuantRagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"indexed {meta['chunks_indexed']} chunks with embedder {meta['embedder_model']} "
        f"-> {args.out} (metadata: {args.meta})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
