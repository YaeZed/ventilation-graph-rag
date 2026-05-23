# AGENTS.md - ventilation-graph-rag

## Project Snapshot

This repo is a coal-mine ventilation safety GraphRAG system. It now includes:

- Python RAG core under `agent/rag_system/`
- shared Neo4j/Milvus connection management in `agent/connection_manager.py`
- deterministic Cypher template retrieval in `agent/rag_system/cypher_templates/`
- two-pass Qwen3.5-Omni image extraction with concept retrieval in `agent/rag_system/ventilation_vision_extractor.py`
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
- SSE endpoint emits plain Server-Sent Events. Text flows use `status`, `token`, `done`, `error`; image flows may also emit `step` events for the frontend agent timeline.
- Image upload flow is: Django temp file -> `VentilationRAGPipeline.query(image_path=...)` -> Qwen3.5-Omni observation -> concept retrieval -> Qwen3.5-Omni analysis -> Cypher template retrieval -> hybrid fallback -> answer generation.
- Concept knowledge build flow is: create Neo4j `Concept` nodes, then populate Milvus collection `ventilation_concepts`. If `Concept` nodes already exist, `build_concept_knowledge.py` skips LLM generation and refreshes Milvus from Neo4j.
- Frontend conversation isolation is intentional: sending state, SSE message updates, input drafts, and pending image previews are keyed by `conversationId`.

## Environment Variables

`.env` is local-only. Important keys:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `LLM_MODEL`
- `QWEN_VL_MODEL` or `VL_MODEL`
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_URI`, `MILVUS_COLLECTION`
- `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`
- `VENTILATION_PIPELINE_FORCE_REBUILD`
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`

## Known Pitfalls

- PowerShell may render Chinese or symbols as mojibake; verify API strings with UTF-8 or `unicode_escape` before assuming backend corruption.
- `node` inside the Codex app sandbox may be denied. Escalated shell access has used system Node `v20.19.5` and npm `10.8.2`.
- Frontend SSE must update messages through the reactive Pinia store. Updating a stale object reference can leave the UI stuck at "正在生成...".
- Frontend streaming callbacks must update messages by captured `conversationId`; using the current active conversation can break replies after the user switches chats.
- `frontend/node_modules/` and `frontend/dist/` are generated and ignored.
- `.env`, local model files, Docker volumes, `.hf-cache/`, and `.codex_runtime/` are not source artifacts.
