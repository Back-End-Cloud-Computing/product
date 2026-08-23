# product

Product microservice for an e-commerce platform, built with **FastAPI** + **MongoDB**, featuring LLM-assisted product descriptions, semantic search via a vector database, and behavior-aware recommendations. Designed to run and be tested completely on its own, with clear seams for future integration with Cart, Order, and Customer microservices.

## Scope

- CRUD for products (name, SKU, sale type, brand, category, free-form attributes).
- LLM-assisted commercial description: the model only **suggests** text; the user edits/approves it before it's persisted.
- Semantic embeddings generated from the approved product data and stored in a vector database, kept in sync with MongoDB.
- Lexical, semantic, and hybrid product search.
- Recommendations combining vector similarity with (currently mocked) behavioral signals — purchase history, viewed items, cart contents, similar users.

Out of scope for now: authentication, the actual Cart/Order/Customer services (mocked behind interfaces), and real payment/checkout flows.

## Stack

| Concern | Choice |
|---|---|
| API framework | FastAPI (async), Pydantic v2 for schemas/validation |
| Structured data store | MongoDB, via PyMongo's native async API (`pymongo.AsyncMongoClient`) |
| Vector store | ChromaDB (HTTP client, separate service) |
| Embeddings | `sentence-transformers` — `paraphrase-multilingual-MiniLM-L12-v2` |
| LLM (description drafting) | OpenRouter (`qwen/qwen3-8b:free` by default), pluggable, with an offline mock fallback |
| Tests | pytest, pytest-asyncio, mongomock-motor, httpx |
| Runtime | Docker + Docker Compose |

> **Note on the MongoDB driver:** this service uses `pymongo.AsyncMongoClient` rather than Motor. MongoDB is deprecating Motor in favor of async support built directly into PyMongo (stable since PyMongo 4.9), so a new service has no reason to take the extra dependency. The public API is essentially the same (`await collection.find_one(...)`, `async for doc in collection.find(...)`, etc.), so this is a driver choice, not an architectural one.

## Why a vector database?

Keyword search only finds products whose *exact words* match the query. A customer searching "celular com bastante armazenamento" won't match a product named "Smartphone Galaxy Z, 128GB" through keywords alone, even though it's exactly what they want.

To solve that, every approved product description (plus name, brand, category, and attributes) is converted into a numeric vector — an **embedding** — that captures its meaning rather than its exact wording. **ChromaDB** stores these vectors, indexed by the product's MongoDB `_id`, and lets us ask "which products are semantically closest to *this* query vector?" via cosine similarity. That's what powers `/search/semantic` and half of `/search/hybrid`.

**MongoDB stays the single source of truth** for structured data; ChromaDB only ever holds a derived representation. The two are not updated in a single transaction, so consistency between them is handled explicitly:

- Every product has an `embedding_status` (`pending` / `synced` / `failed`) and `embedding_sync_error`.
- Confirming a description (`PATCH /products/{id}/description`) writes to MongoDB first, then attempts to embed and upsert into ChromaDB. If that upsert fails (network hiccup, ChromaDB down, etc.), the failure is recorded on the product instead of failing the whole request — the product still exists and is searchable lexically.
- `POST /products/{id}/embedding/sync` lets a client (or a future retry job) re-attempt a failed sync at any time, giving eventual consistency without distributed transactions.

`/search/hybrid` merges the lexical (MongoDB text index) and semantic (ChromaDB) result lists using **Reciprocal Rank Fusion** — items are scored by their rank position in each list rather than by raw score, since a MongoDB text score and a cosine similarity aren't on comparable scales.

## Architecture

```
                         Docker Compose
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    product-api           mongodb              chromadb
      :8000                :27017                :8001
   (FastAPI app)     (structured data,      (vector store,
                       source of truth)     embeddings by
          │                                  product id)
          │
          ├──── LLM API (OpenRouter) ── description suggestions
          │
          └──── Embedding model (sentence-transformers, local/CPU)
```

Internally, the app is layered by responsibility: `routes` (HTTP contracts) → `services` (business logic, orchestration) → `database` (MongoDB/ChromaDB clients). `schemas` define the Pydantic request/response contracts; `models` define the MongoDB document shape (including internal lifecycle fields the API contracts don't expose). `integrations/` holds interfaces (with mock implementations) for the future Cart/Order/Customer clients, so `recommendation_service` doesn't need to change when those become real HTTP services.

## API endpoints

Interactive docs (Swagger UI) are always available at **`/docs`** once the service is running; raw OpenAPI JSON at `/openapi.json`.

### Products

| Method | Path | Description |
|---|---|---|
| `POST` | `/products` | Create a product from its metadata (name, SKU, sale type, brand, category, attributes). |
| `GET` | `/products` | List products, paginated (`skip`, `limit`), filterable by `category`/`brand`. |
| `GET` | `/products/{id}` | Fetch a single product. |
| `PATCH` | `/products/{id}` | Partially update product metadata. |
| `DELETE` | `/products/{id}` | Delete a product and its embedding. |
| `POST` | `/products/{id}/description/generate` | Ask the LLM for a description **suggestion** — nothing is persisted yet. |
| `PATCH` | `/products/{id}/description` | Persist the final (edited or as-is) description; triggers embedding generation + ChromaDB sync. |
| `POST` | `/products/{id}/embedding/sync` | Manually retry a failed MongoDB → ChromaDB embedding sync. |

### Search

| Method | Path | Description |
|---|---|---|
| `GET` | `/search/lexical?q=` | Keyword search over MongoDB's text index. |
| `GET` | `/search/semantic?q=` | Meaning-based search via embeddings + ChromaDB — matches even without keyword overlap. |
| `GET` | `/search/hybrid?q=` | Lexical + semantic results merged via Reciprocal Rank Fusion. |

### Recommendations

| Method | Path | Description |
|---|---|---|
| `GET` | `/recommendations/products/{id}/similar` | Content-based: nearest neighbors of a product's own embedding. |
| `GET` | `/recommendations/users/{user_id}` | Personalized: weighted mix of purchase history, viewed products, cart, similar users (mocked today) plus content similarity; falls back to recently-approved products when no signal exists. |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check, also used by the Docker healthcheck. |

## Running with Docker

The stack has three containers: `product-api` (this service), `mongodb`, and `chromadb`.

```bash
cd product
cp .env.example .env      # fill in OPENROUTER_API_KEY if you want real LLM output
docker compose up -d
```

- API: `http://localhost:8000` (docs at `http://localhost:8000/docs`)
- MongoDB: `localhost:27017`
- ChromaDB: `localhost:8001`

Notes:
- If `OPENROUTER_API_KEY` is left empty (or the provider is unreachable), description generation **automatically falls back** to an offline mock instead of failing — the service never hard-depends on the LLM being available.
- The embedding model is downloaded from Hugging Face on first use and cached in the `hf_cache` volume, so only the very first description confirmation is slow; subsequent ones are fast.
- `docker compose down` removes the containers but keeps the named volumes (Mongo data, Chroma data, model cache); add `-v` to wipe them too.

To stop:

```bash
docker compose down
```

## Running locally without Docker

```bash
cd product
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # point MONGODB_URI / CHROMA_HOST at wherever those are running
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
```

Unit and integration tests run fully offline: MongoDB is faked with `mongomock-motor` (its async collection/cursor interface is method-compatible with PyMongo's native async API, so it works as a mock even though production no longer uses Motor), and the embedding model / ChromaDB calls / LLM provider are stubbed at the boundary — no network access or real services are required.
