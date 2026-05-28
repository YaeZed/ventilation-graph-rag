# Task Plan: User Module P4+ Team Conversation Browser

## Goal

Turn teams from "statistics grouping" into a usable team workspace: users can assign a conversation to a team from the conversation action menu, and team members can browse/open team-owned conversations from a sidebar "团队对话" section.

## Constraints

- Keep personal conversations private by default. Only conversations with explicit `teamId` are visible to team members.
- Preserve existing local-first guest behavior; guests should not see team controls.
- Move conversation team assignment out of `/settings`; the settings page should keep team/member/account management only.
- Sidebar alignment must match the existing "最近对话" and "已归档" visual system.
- Do not auto-edit or delete another user's team conversation in this phase; viewing shared team conversations can be read-only.
- Do not revert unrelated dirty worktree changes.

## Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | complete | Inspect sidebar/conversation item/store/API structure for team conversation integration |
| 2 | complete | Add backend team conversation list endpoint with membership-gated serialization |
| 3 | complete | Add frontend API/store state for team conversations and selection behavior |
| 4 | complete | Move team assignment to conversation menu submenu with confirm/cancel flow |
| 5 | complete | Add sidebar "团队对话" collapsible section and remove settings "当前会话归属" |
| 6 | complete | Update docs/planning and run backend/frontend validation |
| 7 | complete | Polish P4+/P5 UI alignment: team submenu hover, settings team panel, account security list, and reusable dropdowns |

## Decisions

- The sidebar "团队对话" section should show conversations from teams the user joined, grouped as one collapsible section to avoid adding a new route or full team workspace page.
- Team conversations created by other users should initially be read-only in the chat pane. This prevents accidental edits/deletes and keeps permissions simple.
- Conversation assignment should remain explicit and reversible: selecting no team returns the conversation to personal space.
- The menu flow should require confirmation after selecting a team because it changes visibility for other members.
- Dropdowns in settings and stats should use the shared `SettingsSelect` component so option colors, hover states, and typography stay consistent across pages.

## Error Log

| Error | Attempt | Resolution |
|---|---|---|
| Team assignment submenu disappeared before the pointer reached it | Hover-only parent row close logic | Teleported submenu to `body`, positioned it to the right of the main menu, and used outside-click closing |
| `/settings` first open showed incomplete team state until refresh | Team list refresh without immediate member reload | Load selected-team members immediately and after team-list refreshes |
| Settings page crashed from an immediate watcher | Watcher called `cancelTeamNameEdit` before a const function initializer ran | Converted team-name edit handlers to hoisted function declarations |
| Native dropdowns showed system-blue or blank hover/selected states | Reused browser `select` in team/stat controls | Introduced `SettingsSelect.vue` and narrowed global page-header button styles |
