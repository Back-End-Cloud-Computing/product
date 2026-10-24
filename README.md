# product

Product microservice for an e-commerce platform (GANJJ), built with **FastAPI** + **MongoDB**. Owns product CRUD and orchestrates the product lifecycle: it builds the business prompts (description drafting, agentic search strategy/evaluation/justification) and delegates the actual embedding/vector-store/LLM work to sibling microservices — `embedding-reranking`, `vector-db`, and `llm-provider` — over HTTP/WebSocket, keeping this service free of ML models and direct vector-store access.

## Scope

- CRUD for products (name, SKU, sale type, brand, category, free-form attributes).
- LLM-assisted commercial description: `llm-provider` only **suggests** text (over a WebSocket, streamed); the user edits/approves it before it's persisted.
- Triggers `embedding-reranking` to (re)generate and index a product's embedding whenever it's created or its description is approved; triggers a direct delete on `vector-db` when a product is removed (no embedding needs generating just to delete one).
- Lexical (own MongoDB text index), semantic (via `embedding-reranking`), and agentic product search (an LLM, via `llm-provider`, decides per iteration whether to run lexical, semantic, or both).
- `POST /products/batch` to hydrate full product data for the thin `product_id` + `score` results returned by `vector-db`'s recommendation endpoints (recommendations themselves live in `vector-db` now, since it's the service that owns vector similarity and behavioral signals).

Out of scope for this service: authentication, embeddings/vector storage (→ `embedding-reranking` / `vector-db`), LLM provider access (→ `llm-provider`), recommendations (→ `vector-db`), and the real Cart/Order/Customer services (mocked inside `vector-db`, which is where recommendation signals are consumed).

## Stack

| Concern | Choice |
|---|---|
| API framework | FastAPI (async), Pydantic v2 for schemas/validation |
| Structured data store | MongoDB, via PyMongo's native async API (`pymongo.AsyncMongoClient`) |
| Calls to sibling services | `httpx` (REST) for `embedding-reranking`/`vector-db`/`llm-provider`; `websockets` (client) for the description-generation stream from `llm-provider` |
| Tests | pytest, pytest-asyncio, mongomock-motor, respx (mocks the outbound HTTP calls) |
| Runtime | Docker + Docker Compose (orchestrated from the repo root, alongside the other services) |

> **Note on the MongoDB driver:** this service uses `pymongo.AsyncMongoClient` rather than Motor. MongoDB is deprecating Motor in favor of async support built directly into PyMongo (stable since PyMongo 4.9), so a new service has no reason to take the extra dependency.

## Architecture

```
                                docker-compose (repo root)
                                          │
   ┌───────────────┬──────────────────────┼───────────────────┬──────────────┐
   ▼               ▼                      ▼                   ▼              ▼
product-api      mongodb           embedding-reranking      vector-db     llm-provider
  :8000          :27017                  :8003                :8002          :8004
   │        (source of truth                │                   ▲              │
   │           for products)                 └── HTTPS ─────────►│              │
   ├── HTTPS (index/search) ────────────────────────────────────►│              │
   ├── HTTPS (delete vector directly, no embedding needed) ──────►│              │
   ├── WS  (description generation, streamed) ──────────────────────────────────►│
   └── REST (agentic search: strategy/evaluation/justification) ─────────────────►│
```

Coupling rule: this service knows `embedding-reranking` (index + semantic search) and `llm-provider` (description + agentic search prompts), plus a single direct call to `vector-db` for vector deletion. It never talks to ChromaDB or an LLM provider directly, and never generates embeddings itself.

Internally, layered by responsibility: `routes` (HTTP contracts) → `services` (business logic/orchestration, including the prompts sent to `llm-provider`) → `clients` (HTTP/WS clients to the sibling services) → `database` (MongoDB). `schemas` define the Pydantic request/response contracts; `models` define the MongoDB document shape (including internal lifecycle fields the API contracts don't expose).

## Embedding & description consistency

**MongoDB stays the single source of truth.** Vectors in `vector-db` are only ever a derived representation, kept eventually consistent:

- Every product has an `embedding_status` (`pending` / `synced` / `failed`) and `embedding_sync_error`.
- Confirming a description (`PATCH /products/{id}/description`) writes to MongoDB first, then asks `embedding-reranking` to embed and index it. If that call fails, the failure is recorded on the product instead of failing the whole request — the product still exists and is searchable lexically.
- `POST /products/{id}/embedding/sync` lets a client (or a future retry job) re-attempt a failed sync at any time.
- Deleting a product removes it from MongoDB first, then calls `vector-db` directly to remove the vector — best-effort, logged but never blocking the deletion.

`/search/agentic_search` runs an LLM-driven agent loop: each iteration the model (via `llm-provider`) decides whether to run lexical, semantic, or both, results are merged across iterations via **Reciprocal Rank Fusion**, and the model evaluates whether coverage is sufficient or the query should be reformulated. Available as `POST /search/agentic_search` (single response) and `WS /search/agentic_search/ws` (streams each iteration's progress).

## API endpoints

Interactive docs (Swagger UI) are always available at **`/docs`** once the service is running; raw OpenAPI JSON at `/openapi.json`.

### Products

| Method | Path | Description |
|---|---|---|
| `POST` | `/products` | Create a product from its metadata (name, SKU, sale type, brand, category, `price`, `stock`, attributes). Answers `201` as soon as it is stored; AI description, summary and vector indexing run in the background — `GET /products/{id}` shows `description`/`summary` once done, or `error` if a step failed. A repeated SKU answers `409`. |
| `GET` | `/products` | List products, paginated (`skip`, `limit`), filterable by `category`/`brand`. |
| `GET` | `/products/{id}` | Fetch a single product. |
| `PATCH` | `/products/{id}` | Partially update product metadata. |
| `DELETE` | `/products/{id}` | Delete a product from MongoDB, then remove its vector from `vector-db`. |
| `POST` | `/products/{id}/description/generate` | Ask `llm-provider` for a description **suggestion** (streamed internally over WS) — nothing is persisted yet. |
| `PATCH` | `/products/{id}/description` | Persist the final (edited or as-is) description; triggers `embedding-reranking` indexing. |
| `POST` | `/products/{id}/embedding/sync` | Manually retry a failed embedding sync. |
| `POST` | `/products/batch` | Hydrate a batch of product ids (`{"ids": [...]}`) — for enriching `vector-db` recommendation results. |

### Search

| Method | Path | Description |
|---|---|---|
| `GET` | `/search/lexical?q=` | Keyword search over MongoDB's text index. |
| `GET` | `/search/semantic?q=` | Meaning-based search via `embedding-reranking` — matches even without keyword overlap. |
| `POST` | `/search/agentic_search` | Agentic search: an LLM decides per iteration between lexical, semantic, or both, until coverage is sufficient or a max iteration count is hit. |
| `WS` | `/search/agentic_search/ws` | Same agentic search, streaming each iteration's progress before the final result. |

### Internal (service-to-service)

Called by `order` during checkout. Authenticated by the `x-internal-secret` header (env `INTERNAL_SECRET`) instead of a user JWT, and not exposed on the Ingress.

| Method | Path | Description |
|---|---|---|
| `POST` | `/internal/products/{id}/stock/decrement` | Take `quantity` units out of stock. |
| `POST` | `/internal/products/{id}/stock/restore` | Put `quantity` units back (saga compensation). |

Both take the `Idempotency-Key` header and a body `{"quantity": 2, "reason": "...", "reference_id": "...", "compensates_operation_id": "..."}` (only `quantity` is required), and answer `{"operation_id", "product_id", "remaining_stock"}`. `{id}` is the product's string id, the same one `GET /products/{id}` uses. Repeating an `Idempotency-Key` with the same request returns the original result without touching stock again.

| HTTP | `error_type` | Meaning |
|---|---|---|
| 401 | `authentication_error` | Missing or wrong `x-internal-secret` |
| 404 | `product_not_found` | Unknown product id |
| 409 | `insufficient_stock` | Not enough stock; the body carries `available_stock` |
| 409 | `operation_conflict` | `Idempotency-Key` reused for a different request |
| 422 | — | Missing `Idempotency-Key` or `quantity` not positive |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, also used by the Docker healthcheck. |
| `GET` | `/metrics` | Prometheus metrics (request count, errors, latency per route). No token; not exposed on the Ingress. |

## Running with Docker

From a clean clone of this repository:

```bash
cp .env.example .env   # set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY for real LLM output
docker compose up -d
```

`docker-compose.yml` builds this service from the local code and brings up everything it depends on: MongoDB, `authorization` (+ Oracle, built straight from its GitHub repo), and `llm-provider`, `embedding-reranking`, `vector-db` (+ ChromaDB) from their Docker Hub images. The first `up` takes a few minutes: Oracle has to initialize and `authorization` is compiled.

- API: `http://localhost:8000` (docs at `http://localhost:8000/docs`)
- Tokens: `POST http://localhost:8081/auth/login` (default admin `admin@ganjj.com` / `adminSegura123`)
- MongoDB: `localhost:27017`

Notes:
- `authorization` runs with a temporary RSA key pair here, regenerated on every restart: tokens stop being valid after it restarts.
- Without `OPENROUTER_API_KEY`, `llm-provider` answers in mock mode — this service never hard-depends on the LLM being available.
- `docker compose down` removes the containers but keeps the named volumes; add `-v` to wipe them too.

## Running locally without Docker

```bash
cd product
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # point *_BASE_URL vars at wherever the sibling services are running
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
```

Unit and integration tests run fully offline: MongoDB is faked with `mongomock-motor`, and every outbound call to `embedding-reranking`, `vector-db`, and `llm-provider` is stubbed at the client boundary (`respx` for HTTP, a fake WebSocket for the description stream) — no network access or real services are required.
