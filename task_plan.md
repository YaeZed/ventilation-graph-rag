# Task Plan: Model Configuration Runtime

## Goal

Implement the usable part of `docs/plan-model-config-deploy.md`: users can choose a model provider/preset in `/settings`, store model configuration locally or in their account profile, and have each chat request use that configuration for text generation and vision analysis without rebuilding the knowledge indexes.

## Constraints

- Follow the existing user-module storage split: guests use scoped localStorage, authenticated users sync preferences through `UserProfile.settings`.
- Do not store or expose secrets outside the user's explicit model config payload.
- Preserve existing text-only, image, multi-image, sensor, and SSE request flows.
- Reuse the existing global RAG pipeline, Neo4j, and Milvus resources; model changes must not force index rebuilds.
- Do not revert unrelated dirty worktree changes from the previous sensor/multi-image phase.
- Deployment scaffolding remains a later phase until runtime model configuration is verified.

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | complete | Inspect model call sites, settings persistence, chat transport, and request routing |
| 2 | complete | Add shared model-config types, presets, normalization, and frontend persistence |
| 3 | complete | Add `/settings` model configuration UI with provider presets, restore default, and connection test |
| 4 | complete | Pass model config through text, upload, and SSE chat APIs |
| 5 | complete | Add backend model-config validation and request-scoped pipeline overrides |
| 6 | complete | Validate backend, frontend build, and smoke request behavior |
| 7 | complete | Update docs/status/runbook/planning notes |
| 8 | pending | Plan deployment scaffolding as a separate implementation pass |
| 9 | complete | Fix local settings-page CSRF failure and avoid rendering backend HTML errors in the UI |
| 10 | complete | Fix model-test probe token limit for DashScope Omni-compatible vision models |
| 11 | complete | Reconcile docs and deployment plan around BYOK/user-key policy |

## Decisions

- A user model config is a preference inside `settings.modelConfig`, not a separate domain model. This keeps guest/authenticated behavior aligned with existing preference sync.
- The frontend always sends a sanitized `model_config` with chat requests. Backend still has env-based defaults when the client sends nothing.
- Request-scoped overrides temporarily swap text/VL clients and model names on the initialized pipeline, then restore the original objects. Because those objects are shared on the singleton pipeline, all query execution enters the same model-state lock, including no-config requests. This prevents default requests from observing another user's temporary client; the tradeoff is lower in-process chat concurrency.
- Provider presets should fill endpoint/model defaults but allow the user to edit model names, endpoints, and API keys.
- `/api/chat/model/test/` should validate text and vision OpenAI-compatible chat clients without touching RAG retrieval or rebuilding indexes. This gives users immediate feedback before spending time on a full chat/image request.
- Streamed requests must hold the model override context for the full generator iteration; otherwise SSE would fall back to the default model after `query()` returns.
- Deployment scaffolding is intentionally not marked complete. It needs its own pass after choosing target hosting, secret handling, and media/static policy.
- Deployment must not let ordinary users consume the maintainer's `.env` model key. The deployment pass needs a BYOK or tenant-key policy before public exposure.
- Local Vite may fall back from 5173 to 5174 when another dev server is still running. The dev CSRF defaults trust both ports, and frontend API parsing now converts Django HTML debug pages into short actionable errors instead of rendering raw HTML in settings.
- The model-test probe must use a token limit accepted by all configured providers. DashScope `qwen3.5-omni-plus` rejects `max_tokens < 10`, so the probe uses `MODEL_TEST_MAX_TOKENS = 16`.

## Error Log

| Error | Attempt | Resolution |
|---|---|---|
| `docs/plan-model-config-deploy.md` printed as mojibake in PowerShell output | Initial plan read | Treat the file as the design source but verify behavior from code; avoid relying on garbled rendered text for exact UI copy |
| Existing worktree is dirty from previous docs/UI work | Initial status check | Preserve unrelated changes and only edit files needed for this feature |
| Smoke test patched `web_backend.chat.views`, but Django URL imported `chat.views` | First model-config API smoke | Re-run smoke with the actual `chat.views` module path so the fake pipeline replaces the real Neo4j-backed service |
| Django test client used `testserver`, which is not in local `ALLOWED_HOSTS` | First model-test endpoint smoke | Re-run the smoke with `SERVER_NAME='127.0.0.1'` without changing deployment/settings config |
| `/settings` displayed a full Django CSRF debug HTML page after Vite used port 5174 | User-reported settings screenshot | Add 5174 to local CSRF defaults, return JSON for CSRF failures, and normalize non-JSON/HTML frontend API errors into concise messages |
| DashScope `qwen3.5-omni-plus` rejected model-test vision probe with `Range of max_tokens should be [10, 65536]` | User-reported settings test | Raise the shared model-test probe limit from 8 to 16 tokens |
