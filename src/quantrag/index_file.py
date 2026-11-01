"""Single-file vector index used by the container deployment.

``InMemoryVectorStore.save`` already writes a portable ``{chunks, vectors}``
JSON document.  This module adds the small metadata sidecar that records which
embedding model produced those vectors, so a container can decide whether the
prebuilt index is still usable or whether it must fall back to re-indexing.

Artifacts::

    data/index.json        chunks + vectors (InMemoryVectorStore format)
    data/index.meta.json   embedder model id, chunk count, creation time

The directory format built by ``quantrag.index_build`` remains supported; this
is simply a flatter alternative for deployments that ship a single index file.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import RetrievalError
from .retriever import InMemoryVectorStore

INDEX_FILE_FORMAT_VERSION = 1


def save_index_file(
    store: InMemoryVectorStore,
    path: str | Path,
    *,
    embedder_model: str,
    meta_path: str | Path,
    source: str | Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Write ``store`` to ``path`` plus a metadata sidecar to ``meta_path``.

    ``embedder_model`` is mandatory: without it the runtime cannot tell whether
    a prebuilt index matches the configured ``OPENROUTER_EMBEDDING_MODEL``.
    """
    model = str(embedder_model or "").strip()
    if not model:
        raise RetrievalError("embedder_model is required to write index metadata.")

    index_path = Path(path)
    meta_file = Path(meta_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    store.save(index_path)

    meta: dict[str, Any] = {
        "format": INDEX_FILE_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedder_model": model,
        "chunks_indexed": len(store),
        "source": str(source) if source is not None else None,
        "batch_size": batch_size,
    }
    meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def read_index_meta(meta_path: str | Path) -> dict[str, Any]:
    """Read and validate the index metadata sidecar."""
    path = Path(meta_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RetrievalError(f"Index metadata {path} does not exist.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Cannot read index metadata {path}: {exc!r}") from exc

    if not isinstance(data, dict) or data.get("format") != INDEX_FILE_FORMAT_VERSION:
        raise RetrievalError(
            f"{path} is not a version-{INDEX_FILE_FORMAT_VERSION} index metadata file."
        )
    return data


def index_model(meta: dict[str, Any]) -> str:
    """Return the embedding model recorded in ``meta`` (empty string if absent)."""
    return str(meta.get("embedder_model") or "").strip()


def load_index_file(path: str | Path) -> InMemoryVectorStore:
    """Load a single-file index written by :func:`save_index_file`."""
    store = InMemoryVectorStore.load(path)
    if len(store) == 0:
        raise RetrievalError(f"Index {path} contains no chunks.")
    return store
