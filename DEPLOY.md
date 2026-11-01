# Deploying QuantRAG

One container serves all four endpoints from a single process:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe. Public, never calls OpenRouter. |
| `POST /ask` | REST RAG answer (`{"question": "..."}`). |
| `POST /search` | REST retrieval (`{"query": "..."}`). |
| `/mcp` | MCP Streamable HTTP transport (`ask_quant_question`, `search_quant_kb`). |

`MCP_TRANSPORT=streamable-http` selects the combined HTTP server; `stdio`
remains the default for local MCP clients.

## 1. Build the index (once, before building the image)

The image never embeds the corpus at startup. Build the index locally with the
same embedding model the deployed service will use:

```bash
uv sync --extra api
uv run python scripts/build_index.py
```

This reads `data/chunks.jsonl` (or `--chunks <path>`) and writes:

- `data/index.json` - chunks + vectors
- `data/index.meta.json` - the embedding model id and counters

`OPENROUTER_API_KEY`, `OPENROUTER_MODEL` and `OPENROUTER_EMBEDDING_MODEL` must
be set in the environment (or `.env`) for the build command. Changing
`OPENROUTER_EMBEDDING_MODEL` requires rebuilding the index: the runtime refuses
to use an index whose recorded model
does not match and falls back to re-indexing the corpus.

The directory index (`data/index/` built by `python -m quantrag.index_build`)
is still supported as a fallback, so either layout works.

## 2. Build and run the container locally

```bash
docker build -t quantrag .

docker run --rm -p 8080:8080 \
  --env-file .env \
  -e MCP_TRANSPORT=streamable-http \
  -e QUANTRAG_REQUIRE_AUTH=1 \
  -e QUANTRAG_API_KEY=change-me \
  quantrag
```

Then:

```bash
curl http://localhost:8080/health
# {"status":"ok"}

curl -X POST http://localhost:8080/ask \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the Black-Scholes formula?"}'

# Without the key: 401
curl -i -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the Black-Scholes formula?"}'
```

`/health` stays public. `/ask`, `/search` and `/mcp` accept either
`X-API-Key: <key>` or `Authorization: Bearer <key>`.

## 3. Environment variables to set in the cloud

Set these through the platform's secret manager / environment configuration,
never in the image or the repository:

| Variable | Required | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | Secret. |
| `OPENROUTER_MODEL` | yes | Chat model id. |
| `OPENROUTER_EMBEDDING_MODEL` | yes (vector retrieval) | Must match the built index. |
| `QUANTRAG_API_KEY` | yes in shared deployments | Secret; protects `/ask`, `/search`, `/mcp`. |
| `QUANTRAG_REQUIRE_AUTH` | recommended `1` | Startup fails if no key is configured. |
| `MCP_TRANSPORT` | yes | `streamable-http` for the combined container. |
| `PORT` | platform-injected | Defaults to `8080` in the image; Cloud Run sets it. |
| `QUANTRAG_INDEX_FILE` | no | Default `data/index.json`. |
| `QUANTRAG_INDEX_META` | no | Default `data/index.meta.json`. |
| `QUANTRAG_INDEX_DIR` | no | Default `data/index` (directory fallback). |
| `QUANTRAG_CORPUS_PATH` | no | Default `data/chunks.jsonl` (fallback re-index). |

Optional tuning: `OPENROUTER_BASE_URL`, `OPENROUTER_TIMEOUT_SECONDS`,
`OPENROUTER_MAX_TOKENS`, `QUANTRAG_RETRIEVAL_MODE`, `QUANTRAG_HYBRID_ALPHA`,
`QUANTRAG_TOP_K`, `QUANTRAG_MIN_SCORE`, `QUANTRAG_MAX_CONTEXT_CHARS`.

## 4. Google Cloud Run

Create the secrets first (values are never written to the image):

```bash
printf '%s' "$OPENROUTER_API_KEY" | gcloud secrets create openrouter-api-key --data-file=-
printf '%s' "$QUANTRAG_API_KEY"   | gcloud secrets create quantrag-api-key   --data-file=-
```

Deploy from this directory (Cloud Run detects the `Dockerfile`; it injects
`PORT=8080`):

```bash
gcloud run deploy quantrag \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MCP_TRANSPORT=streamable-http,QUANTRAG_REQUIRE_AUTH=1,OPENROUTER_MODEL=YOUR_MODEL_ID,OPENROUTER_EMBEDDING_MODEL=YOUR_EMBEDDING_MODEL_ID \
  --set-secrets OPENROUTER_API_KEY=openrouter-api-key:latest,QUANTRAG_API_KEY=quantrag-api-key:latest
```

`--allow-unauthenticated` is appropriate because the service enforces its own
API key. Remove it if you prefer Cloud Run IAM authentication.

For Azure Container Apps or Render, use the same `docker build` image and set
the same environment variables/secrets; expose port `8080` (or let the platform
inject `PORT`).

## 5. Startup behaviour

- If `data/index.json` exists and its `embedder_model` matches
  `OPENROUTER_EMBEDDING_MODEL`, it is loaded without any embedding calls.
- Otherwise the service logs a warning and falls back to `data/index/`, then to
  re-indexing `data/chunks.jsonl`.
- If none of those exist, startup fails with a clear message.
- If no API key is configured, startup logs a warning that the service is open
  (and `QUANTRAG_REQUIRE_AUTH=1` turns that into a hard failure).
