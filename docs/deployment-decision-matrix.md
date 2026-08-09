# Deployment Decision Matrix

Updated: 2026-06-03

## Goal

Deploy the system as a usable web product, not just "make it run somewhere".

The deployment must protect the maintainer's model quota, keep conversation/image history durable, preserve the existing GraphRAG behavior, and avoid a hosting choice that forces a rewrite later.

## Non-Negotiables

- Public users must not fall back to the maintainer `.env` model key.
- Django must not run with `runserver` or `DEBUG=True` in production.
- Frontend and API should be same-origin through a reverse proxy to keep session/CSRF behavior simple.
- Uploaded images need a persistent media strategy; development `web_backend/media/` is not enough unless it is a backed-up production volume.
- SQLite is acceptable for local development only. Production account/team/conversation records should move to Postgres or another durable managed database.
- Self-hosted Milvus is not a 2C4G-friendly choice. Official Milvus standalone requirements list 8 GB RAM as the minimum and 16 GB recommended.

## Recommended Baseline

For the first real deployment:

```text
Single VPS or ECS, preferably 4C8G
  nginx :80/:443
    /api/*   -> Gunicorn/Django
    /media/* -> persistent media volume, later object storage
    /*       -> Vue dist static files

  Docker Compose services:
    django
    postgres
    nginx

External managed services:
    Neo4j AuraDB Free or self-hosted Neo4j only if cloud graph limits block us
    Zilliz Cloud Free for Milvus-compatible vectors, or keep Milvus external
    DashScope/OpenAI/Ollama-compatible model endpoint via BYOK
```

Why this baseline:

- It keeps the operational surface small: one server plus two managed knowledge stores.
- It avoids running Milvus, etcd, MinIO, Django, torch, nginx, and Neo4j on a tiny box.
- It matches the current code shape: Django session auth, uploaded media, SSE, local frontend build, and existing Neo4j/Milvus connection manager.
- It gives an obvious upgrade path: local media volume -> object storage, local Postgres -> managed Postgres, single Django instance -> multi-instance after request-scoped model config is refactored.

## Hosting Options

| Option | What It Means | Pros | Cons | User Impact | Decision |
|---|---|---|---|---|---|
| Single VPS + managed graph/vector | VPS runs nginx, Django, Vue static, Postgres; Neo4j/Zilliz external | Best balance of control, cost, and stability | Need server ops and backups | Stable enough for demo/beta; predictable same-origin UX | Recommended |
| PaaS such as Railway/Render | Platform builds/runs app; use platform DB/storage where possible | Faster initial deploy, less Linux work | Heavy Python/torch images, persistent media, long SSE/image calls can hit platform limits; costs become less transparent | Faster first URL, but more deployment surprises | Viable only after a small proof |
| All-in-one VPS | VPS runs nginx, Django, Vue, Postgres, Neo4j, Milvus/MinIO/etcd | Maximum control, no managed DB dependency | Needs more RAM/CPU; Milvus alone wants 8 GB minimum | Higher chance of slow first request and downtime under load | Avoid for 2C4G; possible on 8C16G+ |
| Static frontend on Vercel/Netlify + API elsewhere | Frontend CDN, backend separately hosted | Good static delivery | Cross-origin cookies/CSRF/CORS become more complex; not needed yet | Login/sync failures become easier to introduce | Not first choice |

## Model Key Strategy

| Strategy | Backend Stores User API Key? | Pros | Cons | User Impact | Decision |
|---|---:|---|---|---|---|
| BYOK local-only | No | Lowest security/cost risk; maintainer quota protected | User must configure each browser/device; key is sent with each request | Clear ownership: users pay for their own model usage | Recommended for public deployment |
| BYOK encrypted server-side | Yes, encrypted | Better cross-device UX | Requires encryption key management, masking, rotation, audit, and breach handling | Users configure once, but trust server with secret | Later phase |
| Admin-managed tenant key | Yes, maintainer/tenant key | Best demo UX | Maintainer pays; abuse/quota risk; needs tenant quotas/rate limits | Users can try without setup | Only private demo or controlled users |

Important current-code implication:

`frontend/src/stores/chat.ts` currently sends `settings.value` to `updateRemoteProfile()`, and `web_backend/users/views.py` normalizes and stores `settings.modelConfig` in `UserProfile.settings`. If we choose `BYOK local-only`, deployment work must strip API keys from profile sync and backend profile responses while still sending keys in per-request `model_config`.

## Data Store Choices

| Component | Option | Fit | Notes |
|---|---|---|---|
| Django app database | Postgres in Compose | Recommended first deployment | Durable account/team/conversation data without adding another vendor. Backups are mandatory. |
| Django app database | Managed Postgres | Better production choice | Less ops, more cost; clean upgrade path from local Postgres. |
| Django app database | SQLite | Local only | Unsafe for production writes, backups, concurrency, and future migration discipline. |
| Neo4j | AuraDB Free | Recommended first check | Official free tier exists, but has node/relationship limits. We need import-size validation before committing. |
| Neo4j | Self-hosted Neo4j | Fallback | Works if Aura limits block us; adds server memory/storage and backup burden. |
| Vector store | Zilliz Cloud Free | Recommended first check | Official free cluster currently lists 5 GB storage, enough for small knowledge indexes. |
| Vector store | Self-hosted Milvus | Avoid on small VPS | Official standalone minimum is 8 GB RAM; current local compose also pulls etcd and MinIO. |
| Vector store | FAISS | Possible simplification | Removes Milvus service, but requires code changes and persistence/rebuild policy. Good if Zilliz is blocked. |
| Media | VPS persistent volume + nginx | Good MVP | Simple, same-origin, works with current `ConversationAttachment`. Needs backup and size limits. |
| Media | Object storage | Better public production | More setup, but cleaner for multi-instance and large uploads. |

## Server Size Guidance

| Size | Fit |
|---|---|
| 1C1G / 1C2G | Too small for this project. Django + torch dependencies + uploads + SSE will be fragile. |
| 2C4G | Possible only if Neo4j and Milvus are external and traffic is tiny. First image request and embedding model loading may still feel slow. |
| 4C8G | Recommended first VPS size. Enough headroom for Django, torch/BGE, Postgres, nginx, and long requests. |
| 8C16G+ | Needed if we insist on self-hosting Milvus and Neo4j on the same machine. |

## Recommended Implementation Order

1. Lock deployment policy before code: choose hosting, key strategy, graph/vector services, media storage.
2. Patch model key boundary for the selected strategy. For public beta, implement `BYOK local-only` by preventing API keys from syncing into `UserProfile.settings`.
3. Add production settings: `DEBUG=False`, strict `ALLOWED_HOSTS`, CSRF origins, proxy SSL header, static/media roots, Postgres config.
4. Add Dockerfile, `docker-compose.prod.yml`, nginx config, and a production start command using Gunicorn.
5. Export/import or rebuild Neo4j and vector indexes against the selected services.
6. Run local production smoke: `docker compose -f docker-compose.prod.yml up --build`, then verify login, model test, text chat, image chat, SSE, media preview, stats, team permissions.
7. Deploy manually once. Add GitHub Actions SSH deploy only after the manual path is repeatable.

## Open Decisions

| Decision | Recommended Default | Why |
|---|---|---|
| First hosting target | 4C8G VPS/ECS + Docker Compose | Best fit for current Django/SSE/media/torch shape. |
| Model key policy | BYOK local-only | Protects maintainer quota and avoids server-side secret liability. |
| App database | Postgres in Compose | Small ops cost, real production DB semantics. |
| Neo4j | AuraDB Free first, self-host only if limits block import | Reduces server pressure. |
| Vector DB | Zilliz Cloud Free first, FAISS fallback if cloud blocked | Avoids Milvus 8 GB RAM burden on the app server. |
| Media | Persistent VPS volume first, object storage later | Fastest working production path; document backups and limits. |
| CI/CD | Manual first, then GitHub Actions SSH | Avoid automating a path before it is proven. |

## Sources Checked

- Neo4j pricing and AuraDB tiers: https://neo4j.com/pricing/
- Zilliz Cloud free trial/free cluster: https://docs.zilliz.com/docs/free-trials
- Milvus standalone requirements: https://milvus.io/docs/prerequisite-docker.md
- Django deployment checklist: https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/
- Railway pricing limits: https://railway.com/pricing
- Render free instances: https://render.com/docs/free
- Gunicorn deployment notes: https://gunicorn.org/deploy/
