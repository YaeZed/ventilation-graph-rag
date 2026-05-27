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
- `/stats` remains local-first. No backend stats API exists yet.
