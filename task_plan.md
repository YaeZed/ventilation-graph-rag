# Task Plan: User Module P2 Stats Panel Enhancement and Neat-Freak Sync

## Goal

Improve the `/stats` page so users can quickly understand conversation volume, report completion, recent activity, scene distribution, and risk distribution from their current isolated conversation scope.

## Constraints

- Keep P2 frontend/local-first. Do not add backend stats APIs until P3.
- Compute stats only from the current store scope so guest/user isolation remains intact.
- Keep UI alignment consistent with the existing Gemini-style sidebar and current chat layout.
- Avoid chart libraries for P2; use lightweight CSS bars/segments because the data is small.
- Do not revert unrelated dirty worktree changes.

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | complete | Inspect P2 scope in `docs/plan-user-module.md`, `StatsView.vue`, `chat.ts`, and current CSS |
| 2 | complete | Extend `ChatStats` with completion rate, active-day count, risk distribution, and top risk |
| 3 | complete | Redesign `/stats` layout with aligned summary cards, trend bars, scene list, and risk distribution |
| 4 | complete | Update docs/status/progress for P2 |
| 5 | complete | Run frontend validation |
| 6 | complete | Reconcile project docs after P1/P2 so AGENTS, README, runbook, architecture, status, and user-module plan match code |

## Decisions

- P2 should not depend on server aggregation. The frontend already has the current user's scoped conversations after login/sync.
- Risk buckets use stored `hazardLevel` when present and fall back to `未分级`; inference improvements can stay in the store.
- Completion means assistant messages with `role === "assistant"` and `status === "done"`.
- Completion rate means conversations with at least one completed report divided by active conversations; this avoids rates above 100% when one conversation has multiple reports.
- Visual charts use CSS so the page remains dependency-light and consistent with the existing UI.
- Neat-freak sync should not run builds for docs-only changes; the latest P2 code validation remains `vue-tsc --build` and `vite build`.

## Error Log

| Error | Attempt | Resolution |
|---|---|---|
| PowerShell renders Chinese as mojibake in shell output | Read Vue/CSS files through shell | Treat shell output as diagnostic only; preserve actual file content through targeted patches |
| `rg` denied by sandbox | Search docs for stale phrases | Used PowerShell `Select-String` instead |
