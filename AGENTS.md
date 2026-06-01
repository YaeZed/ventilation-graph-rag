# AGENTS.md - ventilation-graph-rag

## Project Snapshot

This repo is a coal-mine ventilation safety GraphRAG system. It now includes:

- Python RAG core under `agent/rag_system/`
- shared Neo4j/Milvus connection management in `agent/connection_manager.py`
- deterministic Cypher template retrieval in `agent/rag_system/cypher_templates/`
- two-pass Qwen3.5-Omni image extraction, multi-image joint analysis, and concept retrieval in `agent/rag_system/ventilation_vision_extractor.py`
- ventilation concept lookup in `agent/rag_system/ventilation_concept_retriever.py`
- Django API and SSE backend under `web_backend/`
- Vue 3 + TypeScript + Pinia frontend under `frontend/`

The active Python environment name is `ventilation-identify-system`.

## Source Of Truth

- Human quick start: `README.md`
- Architecture and data flow: `docs/architecture.md`
- API contract: `docs/api.md`
- Runbook and troubleshooting: `docs/runbook.md`
- Current implementation status: `docs/status.md`
- Original planning interview and phase logs: `docs/grill-me-interview/`

Do not treat `docs/grill-me-interview/` as the runtime manual; it is planning/history.

## Runtime Services

Docker services:

- Neo4j: `127.0.0.1:7474` HTTP, `127.0.0.1:7687` Bolt
- Milvus: `127.0.0.1:19530`
- MinIO: `127.0.0.1:9001`

Web services:

- Django backend: `http://127.0.0.1:8000`
- Vite frontend: `http://127.0.0.1:5173`

## Important Commands

```powershell
# Activate Python env
conda activate ventilation-identify-system

# Install Python deps
D:\Miniconda\envs\ventilation-identify-system\python.exe -m pip install -r requirements.txt

# Start data services
docker compose up -d

# Django checks and server
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py check
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py migrate
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py runserver 127.0.0.1:8000 --noreload

# RAG CLI smoke test
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\ventilation_rag_pipeline.py -q "掘进中的岩巷最低风速要求是多少" --top-k 3

# Unit-style smoke tests
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\test_ventilation_cypher_templates.py
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\test_ventilation_vision_extractor.py

# Build or refresh the concept knowledge layer
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\data_pipeline\build_concept_knowledge.py

# Frontend
cd frontend
npm install
npm run build
npm run dev
```

## Coding Boundaries

- Reuse `ConnectionManager` for Neo4j/Milvus. Do not create new long-lived drivers in feature modules.
- Modules that accept injected Neo4j/Milvus clients must only close resources they own.
- RAG retrieval outputs should remain `langchain_core.documents.Document`.
- `page_content` can contain Markdown; the frontend renders assistant messages with `markdown-it` and raw HTML disabled.
- Django initializes `VentilationRAGPipeline` lazily through `web_backend/chat/pipeline_service.py`.
- SSE endpoint emits plain Server-Sent Events. Text flows use `status`, `token`, `done`, `error`; image/multi-image/sensor flows may also emit `step` events for the frontend agent timeline.
- Image upload flow is: Django temp file(s) -> `VentilationRAGPipeline.query(image_path=.../image_paths=..., sensor_data=...)` -> Qwen3.5-Omni observation -> concept retrieval -> Qwen3.5-Omni single-image or multi-image analysis -> Cypher template retrieval -> hybrid fallback -> answer generation.
- Sensor data flow is: frontend `SensorInputPanel` manual/CSV input -> structured `sensorData` on the user message -> Django `sensor_data` passthrough -> pipeline sensor-enhanced retrieval -> `generate_multimodal_answer(_stream)` cross-validates image evidence, sensor values, and regulation context.
- Concept knowledge build flow is: create Neo4j `Concept` nodes, then populate Milvus collection `ventilation_concepts`. If `Concept` nodes already exist, `build_concept_knowledge.py` skips LLM generation and refreshes Milvus from Neo4j.
- Frontend conversation isolation is intentional: sending state, SSE message updates, input drafts, pending image queues, and pending sensor payloads are keyed by `conversationId`.
- Frontend user-layer persistence is local-first in `frontend/src/stores/chat.ts`: guests use `ventilation-graph-rag:user-module:v2:guest`, logged-in users use `ventilation-graph-rag:user-module:v2:user:<userId>`, and snapshots sync to `web_backend/users` through Django session APIs. Existing-account login must not inherit guest/other-account conversations; only new registration migrates guest conversations.
- Frontend image persistence is split by destination: browser localStorage may keep compressed message `images[]` data URLs for refresh resilience, including logged-in scoped caches; remote sync/export strips data URLs and logged-in uploads create backend `ConversationAttachment` records. Multi-image messages store `images[]` for display plus legacy `imageUrl` for compatibility, and attachment upload retries with the compressed preview if the original file is over the backend limit. Development media files live under `web_backend/media/`; production still needs an object-storage or static-media policy.
- Team support is explicit: `Team`/`TeamMembership` provide `owner/admin/member` roles, and `ConversationRecord.team` is nullable. Personal conversations are not auto-shared; only conversations with a `teamId` enter team stats.
- Team conversation browsing is separate from personal conversation sync. `GET /api/users/teams/<teamId>/conversations/` returns membership-gated team conversations for read-only sidebar browsing; do not merge other users' team conversations into the current user's local `conversations` array.
- `/stats` uses layered statistics: guests use local Pinia aggregation, logged-in personal scope uses backend `ConversationRecord` aggregation through `GET /api/users/stats/summary/?days=7`, and team scope uses the same endpoint with `teamId`.
- User-module write APIs are CSRF-protected. Frontend calls `GET /api/users/auth/csrf/` and sends `X-CSRFToken`; login/register rotate the token, so the frontend API wrapper refreshes the cached token from the cookie after responses.
- P5 account security uses Django password validators, cache-backed login throttling, session cookie settings, and persisted `SecurityEvent` audit records shown in `/settings`.
- Settings and stats dropdowns use `frontend/src/components/SettingsSelect.vue` instead of native `select`; keep this component's light styling isolated from global page-header button rules.

## Environment Variables

`.env` is local-only. Important keys:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `LLM_MODEL`
- `QWEN_VL_MODEL` or `VL_MODEL`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_URI`, `MILVUS_COLLECTION`
- `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`
- `DJANGO_CSRF_TRUSTED_ORIGINS`, `DJANGO_SESSION_COOKIE_AGE`, `DJANGO_SESSION_COOKIE_SECURE`, `DJANGO_CSRF_COOKIE_SECURE`
- `ACCOUNT_LOGIN_FAILURE_LIMIT`, `ACCOUNT_LOGIN_LOCKOUT_SECONDS`, `ACCOUNT_REGISTER_RATE_LIMIT`, `ACCOUNT_REGISTER_WINDOW_SECONDS`
- `VENTILATION_PIPELINE_FORCE_REBUILD`
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`

## Known Pitfalls

- PowerShell may render Chinese or symbols as mojibake; verify API strings with UTF-8 or `unicode_escape` before assuming backend corruption.
- `node` inside the Codex app sandbox may be denied. Escalated shell access has used system Node `v20.19.5` and npm `10.8.2`.
- Frontend SSE must update messages through the reactive Pinia store. Updating a stale object reference can leave the UI stuck at "正在生成...".
- Frontend streaming callbacks must update messages by captured `conversationId`; using the current active conversation can break replies after the user switches chats.
- `frontend/node_modules/` and `frontend/dist/` are generated and ignored.
- `.env`, local model files, Docker volumes, `.hf-cache/`, and `.codex_runtime/` are not source artifacts.
