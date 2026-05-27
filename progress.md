# Progress: User Module P1 Backend Image Attachments

## 2026-05-27

- Started P1 implementation after user requested next phase.
- Created planning files per project instruction because this phase spans backend, frontend, UI, docs, and validation.
- Inspected current backend user module and frontend chat store/rendering. Confirmed P1 can be implemented by adding attachment records and preserving `imageUrl` as display URL while stripping data URLs for authenticated storage.
- Added `ConversationAttachment`, migration `users.0002`, upload/list/delete APIs, and development media routing.
- Updated frontend user API, chat store, PDF export, message image rendering, and image alignment styles.
- Updated `docs/api.md`, `docs/architecture.md`, `docs/runbook.md`, `docs/status.md`, and `docs/plan-user-module.md` to mark P1 implemented.
- Validation passed: Django check, migration check, attachment upload/list/delete smoke, `vue-tsc --build`, and `vite build`.
- Fixed assistant Markdown rendering for model-generated `<br>` line breaks inside tables by adding a shared safe Markdown renderer. Validation passed: `vue-tsc --build` and `vite build`.
- Renamed the product brand to `矿风眼` across the sidebar, home header, auth pages, and browser title. Added a reusable SVG `BrandMark` with mine-tunnel, eye, and wind-flow motifs. Validation passed: `vue-tsc --build` and `vite build`.
- Added fade transition for in-chat conversation switching by reusing the existing `.fade` transition around the active conversation pane. Validation passed: `vue-tsc --build` and `vite build`.
- Added per-conversation in-memory scroll position restore for the chat message window. Switching conversations restores the previous scroll position; new messages in the same conversation still scroll to bottom. Validation passed: `vue-tsc --build` and `vite build`.
- Reworked the chat bottom input bar from absolute overlay to the normal `home-view` grid footer, so the message window reserves real layout space instead of being covered by the input bar. Validation passed: `vue-tsc --build` and `vite build`.
- Started P2 stats panel enhancement. Updated `task_plan.md` from P1 attachment scope to P2 stats scope after confirming `docs/plan-user-module.md` defines P2 as `/stats` local statistics enhancement.
- Completed P2 stats panel enhancement: extended `ChatStats`, redesigned `/stats` with completion overview, risk distribution, trend bars, and scene distribution, and updated docs. Validation passed: `vue-tsc --build` and `vite build`.
- Ran neat-freak doc reconciliation after P2. Updated `AGENTS.md`, `README.md`, `docs/plan-user-module.md`, `docs/runbook.md`, `docs/architecture.md`, and planning files so P1/P2 status, attachment persistence, and `/stats` data ownership match the current code. No build was run because this sync only changed documentation.
