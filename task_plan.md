# Task Plan: User Module P3 Backend Stats Aggregation

## Goal

Move logged-in `/stats` data from purely local Pinia aggregation to backend user-scoped aggregation, while keeping guest statistics local-first.

## Constraints

- Keep guest mode local-first; unauthenticated users should not call backend stats APIs.
- Logged-in stats must be computed only from the authenticated user's `ConversationRecord` rows.
- Keep UI alignment consistent with the existing Gemini-style sidebar and current chat layout.
- Avoid new tables/migrations unless aggregation requires persisted rollups; P3 can derive from conversation snapshots.
- Do not revert unrelated dirty worktree changes.

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | complete | Inspect P3 plan, backend user models/views/routes, frontend stats store/view, and user API client |
| 2 | complete | Add authenticated backend stats aggregation endpoints under `web_backend/users` |
| 3 | complete | Add frontend stats API client and make `/stats` prefer backend stats for logged-in users |
| 4 | complete | Update docs/API/status/runbook/plan for P3 |
| 5 | complete | Run Django and frontend validation |

## Decisions

- P3 summary endpoint should return the same shape as frontend `ChatStats` so the stats page stays simple.
- Keep `/api/users/stats/summary/`, `/trends/`, and `/hazards/` because they were already documented as the P3 direction; `summary` can include all fields needed by the current UI.
- Risk buckets use stored `hazard_level` when present and fall back to `未分级`; backend should match frontend normalization.
- Completion means assistant messages with `role === "assistant"` and `status === "done"`.
- Completion rate means conversations with at least one completed report divided by active conversations; this avoids rates above 100% when one conversation has multiple reports.
- Visual charts use CSS so the page remains dependency-light and consistent with the existing UI.
- Backend stats should ignore archived conversations for primary counts, with `archivedCount` reported separately.

## Error Log

| Error | Attempt | Resolution |
|---|---|---|
| PowerShell renders Chinese as mojibake in shell output | Read Vue/CSS files through shell | Treat shell output as diagnostic only; preserve actual file content through targeted patches |
| `rg` denied by sandbox | Search docs for stale phrases | Used PowerShell `Select-String` instead |
| `manage.py shell -c` broke on PowerShell quoting | Backend stats smoke test | Re-ran the same smoke test through Python stdin |
| Chinese literal assertion mismatch in stdin smoke test | Backend stats smoke test | Used Unicode escape for `高风险` assertion |
