# Findings: User Module P1 Backend Image Attachments

## Initial Scope

- Current plan in `docs/plan-user-module.md` defines P1 as moving image persistence from compressed data URL in `localStorage` and conversation JSON to backend attachment references.
- Required acceptance: refresh keeps image preview, another logged-in browser can see thumbnail/reference, `localStorage` no longer stores large data URLs for logged-in uploads, PDF still includes images, deleting conversation makes attachments inaccessible.

## Constraints From Project Docs

- Frontend user-layer persistence lives in `frontend/src/stores/chat.ts`.
- Guest key: `ventilation-graph-rag:user-module:v2:guest`.
- Logged-in key: `ventilation-graph-rag:user-module:v2:user:<userId>`.
- Existing-account login must not inherit guest/other-account conversations; only registration migrates guest conversations.
- User backend is `web_backend/users` with Django session APIs.

## Code Findings

- `ChatMessage.imageUrl` is currently the persisted preview source and PDF image source.
- `sanitizeConversationForStorage(conversation, includeImages)` keeps/removes `imageUrl` and `previewImageUrl`; this is the right hook to strip data URLs for logged-in snapshots.
- `submit()` currently converts every uploaded image to resized data URL before sending, then stores it on the user message and conversation preview.
- `web_backend/users` has `UserProfile` and `ConversationRecord`; conversation sync upserts by `(user, client_id)`.
- `settings.py` already defines `MEDIA_URL = "/media/"` and `MEDIA_ROOT = BASE_DIR / "media"`, but `ventilation_web/urls.py` does not expose media URLs in development yet.
- Use `FileField` for P1 attachments to avoid requiring Pillow through `ImageField`; validate MIME type starts with `image/`.

## Implementation Findings

- Vite originally only proxied `/api`; `/media` proxy was added for local development convenience.
- Backend returns absolute media URLs using `request.build_absolute_uri(...)`, so previews work from the Vite dev app even without same-origin `/media`.
- Test client needs `HTTP_HOST=127.0.0.1` because project `ALLOWED_HOSTS` does not include Django's default `testserver`.
- `thumbnailUrl` currently equals the original file URL. Real thumbnail generation remains a future optimization.

## Markdown Rendering Findings

- `markdown-it` runs with `html: false`, so model text containing `<br>` is safely escaped to `&lt;br&gt;` and can appear literally in table cells.
- Keep raw HTML disabled. The accepted fix is to restore only escaped `<br>` / `<br />` text tokens to `<br />` through the shared safe Markdown renderer used by both chat display and PDF export.

## Brand UI Findings

- The user chose the product name `矿风眼`.
- The brand mark is implemented as a reusable Vue SVG component instead of adding an icon dependency. This keeps the sidebar/auth visuals consistent and avoids extra bundle weight for one custom logo.
- The logo combines three semantic cues: mine tunnel outline, recognition eye, and ventilation wind-flow lines.

## P2 Stats Findings

- `StatsView.vue` currently shows four metric cards, seven-day activity bars, scene counts, and JSON export.
- `chat.ts` already computes stats locally from non-archived conversations, so P2 can remain frontend-only and respect guest/logged-in storage isolation.
- Current `ChatStats` lacks risk distribution, completion rate, and active-day summary. These are low-risk additions because they derive from existing `Conversation.hazardLevel`, `messages`, and timestamps.

## Neat-Freak Sync Findings

- `AGENTS.md` still described backend image attachments as a future phase; that was stale after P1 and has been corrected.
- `README.md` still implied every upload is compressed into data URL; current truth is guest data URL, logged-in backend `ConversationAttachment`, fallback data URL only when upload fails.
- `docs/plan-user-module.md` had P2 both as next phase and as completed section; it now treats P2 as completed and points next work at P3 backend/team aggregation or production account hardening.
- `/stats` was local-first before P3. P3 now adds backend stats for logged-in users while guests remain local-first.

## P3 Backend Stats Findings

- P3 can derive stats directly from `users.ConversationRecord`; no migration is needed for the first backend aggregation step.
- The backend summary response intentionally mirrors frontend `ChatStats`, so `StatsView.vue` can keep the same rendering logic.
- Logged-in stats only include the current authenticated user's records. Archived conversations are excluded from primary counts and included as `archivedCount`.
- The first PowerShell `manage.py shell -c` smoke command failed due to argument quoting, not backend behavior. Python stdin execution validated the stats endpoints.
- Chinese output in PowerShell/stdin may mismatch in direct literal assertions; use Unicode escape when validating exact Chinese labels from shell scripts.

## P3 Neat-Freak Findings

- `AGENTS.md` still described `/stats` as a P2 local-only panel and backend stats as a future P3 item. That was stale after P3 and has been corrected to the layered guest/local and logged-in/backend model.

## P4 Team Layer Findings

- P4 should not auto-share personal history. Team membership and conversation team assignment must be explicit to avoid exposing old records.
- The smallest useful backend model is `Team`, `TeamMembership`, and nullable `ConversationRecord.team`.
- Team statistics can reuse the P3 `ChatStats` shape and add a `teamId` query parameter instead of creating a separate chart contract.
- Current frontend has no separate team page; the lowest-friction UI is team management in `/settings` and a compact personal/team selector in `/stats`.
- Existing conversation sync already round-trips optional fields, so `teamId` can be added to the `Conversation` payload without changing the core chat send flow.

## P5 Security Findings

- `web_backend/users/views.py` currently marks most mutating account endpoints with `@csrf_exempt`; P5 should restore Django CSRF protection and make the frontend send `X-CSRFToken`.
- `frontend/src/api/users.ts` already centralizes user-module fetch calls, so CSRF can be added once through shared request helpers instead of touching every store action.
- Registration currently accepts weak passwords with only a short length check. Django password validators are the direct production baseline.
- Login currently has no retry limit. A small cache-backed throttle by IP plus username is enough for this app's current scale and avoids new infrastructure.
- There is no account security audit trail. A simple `SecurityEvent` table can capture login success/failure, registration, logout, and throttling without changing the chat data model.
- Team P4 permissions cover basic membership checks, but P5 should verify that non-members cannot see team stats and that member management cannot modify the owner.
- Django rotates the CSRF token during `login()`. Frontend must refresh its cached token after login/register responses; otherwise the next mutating request can fail with 403.
- Password rejection during registration happens before the account exists, so those `SecurityEvent` rows are stored by username without a user FK. Failed logins for an existing username are linked to that user.
- Login throttling is cache-backed. It is correct for local/single-process development; multi-instance production needs a shared cache backend.

## P4+ Team Conversation Browser Findings

- Current P4 lets members contribute conversations to team stats, but `GET /api/users/conversations/` only returns the current user's records. Team members cannot open each other's team conversations.
- The correct backend boundary is a team-scoped list endpoint requiring membership, not widening the personal conversation endpoint.
- Current `/settings` contains "当前会话归属", but the user wants ownership assignment in each conversation item's action menu; settings should stop mutating the active chat.
- Sidebar already has a collapsible "已归档" pattern; "团队对话" should reuse that structure for alignment and predictable scanning.

## P4+/P5 UI Polish Findings

- The conversation "归属团队" submenu should be treated as part of the same open layer as the main menu. Teleporting it to `body`, positioning it beside the trigger, and closing only on outside pointer-down avoids both sidebar overflow clipping and hover-gap disappearance.
- The first `/settings` team panel render needs immediate member loading even when the selected team id does not change after `refreshTeams()`.
- Account security can keep the API's recent 20 events, but the settings UI should constrain the visible list to about five rows with a scrollbar to preserve card alignment.
- Team name editing belongs inline beside the selected team title for owner/admin users; after team mutations, refresh teams and members from the backend instead of relying only on optimistic state.
- Native `select` controls are not reliable for this UI because OS option styling can override colors. Use `SettingsSelect.vue` in settings/team/stat controls and keep global `.page-header` button rules scoped to direct action buttons.

## Team-Space Remark/Delete Polish Findings

- `Team.description` already exists in `web_backend/users/models.py`, is serialized by `_serialize_team()`, and is accepted by create/update APIs, so remark editing is a frontend-only Settings UI change.
- Inline edit should submit both `name` and `description` together. Auto-saving on first input blur would interrupt editing the second field, so this flow needs explicit save/cancel controls.
- Delete confirmation should be local UI state scoped to the selected team id, and it should reset when selection changes to avoid confirming deletion for a stale team.

## Sensor Multi-Image Module Findings

- `docs/plan-sensor-multiimage.md` defines the active scope: sensor readings, multi-image draft upload, and fused image/data/regulation reporting.
- Current frontend only stores one pending `File`, one `imageUrl`, and the first attachment per message. Multi-image support needs a separate `images[]` display model while keeping legacy fields.
- Current Django chat endpoints accept only `image`; multipart handlers need `images` lists plus legacy fallback for one-file callers.
- Current pipeline branches on a single `image_path`. Multi-image and sensor-enhanced requests need explicit routing so legacy text-only and single-image behavior remains stable.
- Current generation has `_build_image_prompt`; sensor fusion should be a new multimodal prompt rather than weakening the existing single-image path.
- Tooling note: `rg.exe` is access-denied in this sandbox, and `session-catchup.py` can crash on GBK output. Use Git/PowerShell-native inspection and log failures instead of repeating the same commands.
- The implemented frontend keeps legacy `imageUrl` for compatibility, but message display/PDF export prefer `attachments[]` and `images[]` so multi-image messages render all evidence.
- Django accepts repeated multipart `images` while also appending legacy `image` for old clients; backend `_uploaded_images()` intentionally reads `images` first to avoid duplicating the first file.
- Pipeline routing is explicit: text-only uses existing adaptive answer, sensor-only uses multimodal prompt without a vision block, single image without sensors uses existing image prompt, and multi-image or sensor-enhanced paths use multimodal generation.
- Multi-image analysis should remain sequential for now. It is slower, but it preserves per-image observations for auditability and avoids mixing visual evidence before concept retrieval.
- The sensor manual type dropdown was the remaining native `select` in the multimodal input panel. Reusing `SettingsSelect.vue` requires scoping `.sensor-row` button styles; otherwise the shared select trigger/options inherit the circular delete-button styling.
- The reported scenario 2 failure is in the Django SSE adapter, not the sensor UI or RAG pipeline. `_stream_pipeline_events()` normalizes images into `image_path_texts`, but its no-image completion check still referenced the removed `image_path_text` single-image variable.
- `.multi-image-list` originally stretched all children to the tallest thumbnail card. Since `.multi-image-add` had width but no height, it became a tall dashed rectangle after the first image was added. Fix it by centering list items and giving the add button fixed 44px dimensions.
- Sent-image disappearance can happen when authenticated remote sync returns a conversation payload whose messages do not include attachment/image fields. Backend serialization should attach `ConversationAttachment` rows back to their `messageClientId`, and frontend merge should preserve local media fields when remote messages lack media.
- The screenshot retest showed the backend had already received the image (`观察图片` step) while the user card only showed sensor data. The immediate UI path must not depend on attachment URL availability. User messages now receive local data-URL thumbnails before upload/sync, and successful authenticated uploads replace those thumbnails with backend attachment URLs for persistence.
- Opening a message image in a new tab breaks the chat workflow. The thumbnail should act as an in-place zoom control, using a fixed overlay teleported to `body` so it is not clipped by the scrollable message area.
- Refresh loss after the local-preview fix can still happen for authenticated users because `saveToStorage()` previously removed data URLs for logged-in scopes. The correct split is: keep compressed local previews in browser storage for refresh resilience, but strip data URLs only for remote sync/export-to-backend. Attachment uploads also need a compressed-preview retry path because the backend rejects original files above 8 MB.
- Neat-freak sync clarified the durable boundary: `saveToStorage()` keeps compressed message `images[]` previews locally, `syncWithRemote()` calls `sanitizeConversationForStorage(..., false)` before backend sync, and `ConversationAttachment.messageClientId` is the cross-refresh join key that restores backend media onto user messages.
- Image preview should be treated as a gallery for the current message, not as a standalone image URL. `HomeView.vue` keeps `previewImages[]` plus `previewIndex`, constrains preview media to the dialog content box, and exposes arrows plus keyboard left/right navigation only when the message has multiple images.

## Model Configuration Runtime Findings

- `docs/plan-model-config-deploy.md` mixes two scopes: runtime model configuration and production deployment scaffolding. The smallest user-visible value is the runtime loop, so this pass implemented `/settings` configuration, persistence, request payload propagation, and request-scoped backend overrides first.
- Model config belongs in `settings.modelConfig`, not a standalone model table. It is a user preference like stream mode and temperature; this preserves the existing guest localStorage and authenticated `UserProfile.settings` sync behavior.
- Frontend chat APIs now send `model_config` for JSON requests and multipart uploads/SSE. Text model settings affect route analysis, graph/hybrid LLM helper calls, and final generation. Vision model settings affect image observation and single/multi-image analysis.
- `/settings` includes a connection test before real chat use. The backend `POST /api/chat/model/test/` endpoint performs minimal OpenAI-compatible `chat.completions` probes for text and vision model configs and masks submitted API keys in returned error messages. It intentionally does not initialize or rebuild RAG indexes.
- The Django API must patch/use the actual `chat.views` module in tests. Patching `web_backend.chat.views` does not replace the URL-loaded module and can accidentally initialize the real Neo4j-backed pipeline.
- `VentilationRAGPipeline.query(..., stream=True)` returns a generator, so model override context must live for the whole iteration. The implementation wraps stream iteration in `_query_stream_with_model_config()` instead of exiting the context before SSE consumes the generator.
- Request-scoped model override uses a lock and temporarily swaps model names/client objects, then restores them. The lock must cover no-config requests too; otherwise a default request could run while another request has installed a custom client and leak another user's model/API key. The tradeoff is that in-process chat requests serialize while the shared pipeline state is protected.
- Deployment remains unfinished. Runtime model config does not solve production key storage, secret encryption/masking, multi-instance coordination, reverse proxy, static/media hosting, or formal deployment scripts.
- The settings-page 403 screenshot was caused by Vite serving the frontend at `http://127.0.0.1:5174` while Django's default `CSRF_TRUSTED_ORIGINS` only trusted 5173. The blocked endpoint was authenticated conversation sync, so the frontend showed the raw Django HTML debug response through `syncError`.
- Frontend user/chat API clients should never surface a full HTML backend debug page in product UI. Non-JSON responses are now compacted, with a special CSRF Origin message that points to the trusted-origin setting or backend restart. Django CSRF failures also return JSON through `CSRF_FAILURE_VIEW`.
- DashScope `qwen3.5-omni-plus` validates `max_tokens` with a lower bound of 10. The model-test endpoint originally used `max_tokens=8`, which made the vision probe fail even though the endpoint/key/model were valid.
- For public deployment, the current `.env` fallback is a cost/security risk: users without their own key would consume the maintainer's model quota. The deployment plan now treats BYOK or explicit tenant-key policy as a required step before public exposure.
