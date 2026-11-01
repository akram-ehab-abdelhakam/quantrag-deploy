"""Tests for cloud-deployment configuration (transport, auth, prebuilt index)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from quantrag.config import load_auth_settings, load_rag_settings, load_transport_settings
from quantrag.errors import ConfigurationError
from quantrag.index_file import index_model, load_index_file, read_index_meta, save_index_file
from quantrag.retriever import Retriever, load_chunks_jsonl
from quantrag.service import RagService, build_embedder, build_service

pytestmark = pytest.mark.anyio

FIXTURE = Path(__file__).parent / "fixtures" / "mini_chunks.jsonl"


def anyio_backend() -> str:
    return "asyncio"


class FakeChat:
    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        return "The answer is [1]."


# ------------------------------------------------------------------
# MCP_TRANSPORT / PORT
# ------------------------------------------------------------------


def test_transport_defaults_to_stdio() -> None:
    assert load_transport_settings({}, load_env_file=False) == ("stdio", 8000)


def test_transport_and_port_read_from_env() -> None:
    assert load_transport_settings(
        {"MCP_TRANSPORT": "streamable-http", "PORT": "9090"}, load_env_file=False
    ) == ("streamable-http", 9090)


def test_invalid_transport_raises() -> None:
    with pytest.raises(ConfigurationError, match="MCP_TRANSPORT"):
        load_transport_settings({"MCP_TRANSPORT": "carrier-pigeon"}, load_env_file=False)


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_invalid_port_raises(port: str) -> None:
    with pytest.raises(ConfigurationError, match="PORT"):
        load_transport_settings({"PORT": port}, load_env_file=False)


def test_main_uses_env_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP_TRANSPORT=streamable-http starts the combined HTTP app on PORT."""
    import quantrag.server as server_module

    called: dict[str, int] = {}

    def fake_run_http(port: int) -> int:
        called["port"] = port
        return 0

    monkeypatch.setattr(server_module, "configure_logging", lambda: None)
    monkeypatch.setattr(server_module, "load_rag_settings", lambda **_: None)
    monkeypatch.setattr(
        server_module,
        "load_transport_settings",
        lambda **_: ("streamable-http", 8123),
    )
    monkeypatch.setattr(server_module, "_run_http_app", fake_run_http)

    assert server_module.main([]) == 0
    assert called["port"] == 8123


# ------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------


def test_singular_api_key_is_loaded() -> None:
    keys, _require = load_auth_settings({"QUANTRAG_API_KEY": "secret"}, load_env_file=False)
    assert keys == frozenset({"secret"})

    settings = load_rag_settings({"QUANTRAG_API_KEY": "secret"}, load_env_file=False)
    assert settings.api_keys == frozenset({"secret"})


def test_plural_and_singular_keys_merge() -> None:
    keys, _require = load_auth_settings(
        {"QUANTRAG_API_KEYS": "a,b", "QUANTRAG_API_KEY": "c"}, load_env_file=False
    )
    assert keys == frozenset({"a", "b", "c"})


async def _install_fixture_service(settings) -> object:
    from quantrag import api

    chunks = load_chunks_jsonl(FIXTURE)
    service = await RagService.from_chunks(
        chunks,
        chat=FakeChat(),
        settings=settings,
        embedder=build_embedder("local", chunks=chunks),
        corpus_label="fixture",
    )
    api._SERVICE = service
    api._RAG_SETTINGS = settings
    api._API_KEYS = settings.api_keys
    return api


async def test_bearer_token_authorizes_rest() -> None:
    from quantrag import api

    previous = (api._SERVICE, api._RAG_SETTINGS, api._API_KEYS)
    settings = load_rag_settings({"QUANTRAG_API_KEY": "secret-123"}, load_env_file=False)
    try:
        await _install_fixture_service(settings)
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            # /health stays public
            assert (await client.get("/health")).status_code == 200

            ok = await client.post(
                "/ask",
                json={"question": "Brownian motion with drift"},
                headers={"Authorization": "Bearer secret-123"},
            )
            assert ok.status_code == 200

            wrong = await client.post(
                "/ask",
                json={"question": "Brownian motion with drift"},
                headers={"Authorization": "Bearer wrong"},
            )
            assert wrong.status_code == 401

            missing = await client.post("/ask", json={"question": "Brownian motion with drift"})
            assert missing.status_code == 401
    finally:
        api._SERVICE, api._RAG_SETTINGS, api._API_KEYS = previous


async def test_x_api_key_accepts_singular_key() -> None:
    from quantrag import api

    previous = (api._SERVICE, api._RAG_SETTINGS, api._API_KEYS)
    settings = load_rag_settings({"QUANTRAG_API_KEY": "secret-123"}, load_env_file=False)
    try:
        await _install_fixture_service(settings)
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            ok = await client.post(
                "/search",
                json={"query": "Brownian motion"},
                headers={"X-API-Key": "secret-123"},
            )
            assert ok.status_code == 200
    finally:
        api._SERVICE, api._RAG_SETTINGS, api._API_KEYS = previous


async def test_overlong_question_is_rejected() -> None:
    from quantrag import api

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        response = await client.post("/ask", json={"question": "x" * 8001})
        assert response.status_code == 422


# ------------------------------------------------------------------
# Single-file index
# ------------------------------------------------------------------


async def _write_index(tmp_path: Path, model: str) -> tuple[Path, Path]:
    chunks = load_chunks_jsonl(FIXTURE)
    embedder = build_embedder("local", chunks=chunks)
    retriever = Retriever(embedder)
    await retriever.index(chunks)
    index_path = tmp_path / "index.json"
    meta_path = tmp_path / "index.meta.json"
    save_index_file(
        retriever.store,
        index_path,
        embedder_model=model,
        meta_path=meta_path,
        source=FIXTURE,
        batch_size=8,
    )
    return index_path, meta_path


async def test_single_file_index_round_trip(tmp_path: Path) -> None:
    index_path, meta_path = await _write_index(tmp_path, "test-embedder")

    assert index_path.is_file()
    assert meta_path.is_file()

    meta = read_index_meta(meta_path)
    assert index_model(meta) == "test-embedder"
    assert meta["chunks_indexed"] == 30

    store = load_index_file(index_path)
    assert len(store) == 30
    assert store.chunks[0].id == load_chunks_jsonl(FIXTURE)[0].id


async def test_build_service_loads_matching_index(tmp_path: Path) -> None:
    index_path, meta_path = await _write_index(tmp_path, "test-embedder")
    env = {
        "OPENROUTER_API_KEY": "test-key-not-a-real-secret",
        "OPENROUTER_MODEL": "test/model",
        "OPENROUTER_EMBEDDING_MODEL": "test-embedder",
    }
    settings = load_rag_settings(
        {
            **env,
            "QUANTRAG_INDEX_FILE": str(index_path),
            "QUANTRAG_INDEX_META": str(meta_path),
            "QUANTRAG_INDEX_DIR": str(tmp_path / "missing-index"),
            "QUANTRAG_CORPUS_PATH": str(tmp_path / "missing.jsonl"),
        },
        load_env_file=False,
    )

    service = await build_service(chat=FakeChat(), settings=settings, env=env)

    assert service.corpus_label == str(index_path)
    assert service.has_vectors is True
    assert service.size == 30


async def test_build_service_falls_back_when_index_model_differs(tmp_path: Path) -> None:
    index_path, meta_path = await _write_index(tmp_path, "old-embedder")
    env = {
        "OPENROUTER_API_KEY": "test-key-not-a-real-secret",
        "OPENROUTER_MODEL": "test/model",
        "OPENROUTER_EMBEDDING_MODEL": "new-embedder",
    }
    settings = load_rag_settings(
        {
            **env,
            "QUANTRAG_INDEX_FILE": str(index_path),
            "QUANTRAG_INDEX_META": str(meta_path),
            "QUANTRAG_INDEX_DIR": str(tmp_path / "missing-index"),
            "QUANTRAG_CORPUS_PATH": str(FIXTURE),
            "QUANTRAG_EMBEDDER": "local",
        },
        load_env_file=False,
    )

    service = await build_service(chat=FakeChat(), settings=settings, env=env)

    assert service.corpus_label == str(FIXTURE)
    assert service.has_vectors is True
    assert service.size == 30


async def test_build_service_requires_a_knowledge_base(tmp_path: Path) -> None:
    settings = load_rag_settings(
        {
            "QUANTRAG_INDEX_FILE": str(tmp_path / "missing.json"),
            "QUANTRAG_INDEX_META": str(tmp_path / "missing.meta.json"),
            "QUANTRAG_INDEX_DIR": str(tmp_path / "missing-index"),
            "QUANTRAG_CORPUS_PATH": str(tmp_path / "missing.jsonl"),
        },
        load_env_file=False,
    )

    with pytest.raises(ConfigurationError, match="No knowledge base"):
        await build_service(chat=FakeChat(), settings=settings, env={})


def test_load_chunks_accepts_utf8_bom(tmp_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = tmp_path / "bom.jsonl"
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    target.write_text("\ufeff" + body + "\n", encoding="utf-8")

    assert len(load_chunks_jsonl(target)) == len(rows)


def test_build_index_script_reports_missing_chunks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = Path(__file__).parent.parent / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("quantrag_build_index_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    capsys.readouterr()
    code = module.main(
        [
            "--chunks",
            str(tmp_path / "missing.jsonl"),
            "--out",
            str(tmp_path / "index.json"),
            "--meta",
            str(tmp_path / "index.meta.json"),
        ]
    )

    assert code == 1
    assert capsys.readouterr().err.startswith("error:")
