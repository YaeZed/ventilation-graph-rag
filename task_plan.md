# Task Plan: Sensor Data + Multi-Image Recognition

## Goal

Complete `docs/plan-sensor-multiimage.md` as a usable product module: users can submit sensor readings and one or more site images in the chat flow, and the backend RAG pipeline uses image evidence, sensor evidence, and retrieved regulation context to generate a fused safety report.

## Constraints

- Preserve existing guest/local-first and authenticated/backend-attachment behavior.
- Keep team-shared conversations read-only; multimodal submission belongs to personal conversations.
- Keep legacy single-image and text-only requests working.
- Remote logged-in conversation snapshots must not persist data URLs; browser-local scoped caches may keep compressed message `images[]` previews for refresh resilience.
- Do not revert unrelated dirty worktree changes in settings/CSS/planning files.
- Update docs/planning before and after implementation because this module changes product behavior.

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | complete | Inspect current chat, upload, SSE, image extraction, and generation paths |
| 2 | complete | Add shared frontend types and UI components for sensor input, sensor badges, and multi-image draft previews |
| 3 | complete | Update chat API/store/HomeView for multi-file uploads, sensor payloads, message persistence, PDF export, and reactive streaming |
| 4 | complete | Extend Django chat endpoints and SSE worker to accept `sensor_data` plus multiple images |
| 5 | complete | Extend vision extractor, RAG pipeline, and generation prompts for multi-image and sensor fusion |
| 6 | complete | Update module/docs/status and run backend/frontend validation |
| 7 | complete | Reuse `SettingsSelect` for the sensor type dropdown and align the manual sensor row styling |
| 8 | complete | Fix sensor-only SSE completion path and re-test the reported scenario |
| 9 | complete | Keep the multi-image add button square after thumbnails are added |
| 10 | complete | Restore sent user-message image display in the conversation window |
| 11 | complete | Make sent user-message images render immediately from local previews and keep thumbnails clickable |
| 12 | complete | Show sent-message images in an in-page centered preview instead of opening a new tab |
| 13 | complete | Preserve sent-message images after refresh with local fallback and compressed attachment retry |
| 14 | complete | Reconcile project docs after the image persistence and preview fixes |

## Decisions

- Sensor data will travel as structured JSON (`sensor_data`) rather than being flattened into user text, so the UI can preserve/re-render it and the backend can format it consistently.
- Multi-image messages will use a new `images[]` display field while preserving existing `imageUrl` and `attachments[]` for backward compatibility.
- For authenticated users, each selected image reuses the existing attachment upload API; browser-local scoped caches may keep compressed previews, while remote sync strips data URLs and stores attachment references.
- Single-image requests without sensor data will stay on the existing image path. Multi-image or sensor-enhanced requests use the new multimodal generation prompt.
- Multi-image vision analysis performs independent per-image observation before joint reasoning, so the final report can distinguish per-image evidence from cross-image findings.
- Sensor manual entry uses the shared `SettingsSelect` control instead of native `select`, so OS-level blue option styling does not leak into the product UI. Sensor-row CSS must scope delete-button styling to its own class because `SettingsSelect` also renders buttons internally.

## Error Log

| Error | Attempt | Resolution |
|---|---|---|
| `rg.exe` returned access denied in the Codex sandbox | Initial repo search | Switched to `git grep`, `Get-ChildItem`, and `Select-String` for this session |
| `session-catchup.py` crashed on Windows GBK output while printing an emoji | Planning-with-files catchup | Used the partial catchup output plus `git diff --stat` as the factual recovery source |
| Direct Django smoke could not import `ventilation_web` from repo root | First inline smoke | Added `web_backend` to `sys.path`, matching `manage.py` import behavior |
| Fake pipeline returned a generator for non-streaming calls | Second inline smoke | Moved stream yields into an inner generator so non-streaming returns a plain string |
| `node.exe` returned access denied inside the Codex sandbox | First frontend validation attempts | Re-ran `vue-tsc` and `vite build` with approved escalated execution |
| Sensor-only stream returned `name 'image_path_text' is not defined` | User test scenario 2 | Replaced the stale single-image variable check with the normalized `image_path_texts` list |
| Multi-image add button stretched vertically after upload | User UI test | Centered the image list items and fixed `.multi-image-add` to a 44px square |
| Sent user messages could lose image display after sync | User UI test | Merge backend attachments into serialized messages and preserve local message media when remote snapshots lack media fields |
| Sent user images still did not appear in the chat window | User screenshot retest | Store local data-URL thumbnails on the user message before attachment upload/backend sync, then replace with persisted attachment URLs when available; wrap thumbnails in links for click-to-view |
| Image click opened a new page/tab | User UI feedback | Replaced external image links with an in-page modal preview centered in the viewport; background click, close button, and Escape dismiss it |
| Images disappeared again after page refresh | User refresh test | Local storage now preserves compressed message thumbnails even for authenticated users, while remote sync still strips data URLs; authenticated attachment upload retries with the compressed preview if the original file is rejected |
