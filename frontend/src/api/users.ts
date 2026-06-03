import type { ChatAttachment, ChatStats, Conversation, UserProfile, UserSettings } from '@/stores/chat'
import { friendlyHttpError } from '@/api/errors'

export type RemoteUser = {
  id: number
  username: string
  nickname: string
  avatarText: string
  settings: UserSettings
}

export type TeamRole = 'owner' | 'admin' | 'member'

export type RemoteTeam = {
  id: string
  name: string
  description: string
  role: TeamRole
  memberCount: number
  createdAt: string
  updatedAt: string
}

export type RemoteTeamMember = {
  id: number
  username: string
  nickname: string
  avatarText: string
  role: TeamRole
  joinedAt: string
}

export type RemoteSecurityEvent = {
  id: number
  type: string
  username: string
  ipAddress: string
  userAgent: string
  metadata: Record<string, unknown>
  createdAt: string
}

export type TeamConversation = Conversation & {
  owner?: {
    id: number
    username: string
    nickname: string
    avatarText: string
  }
  isOwnedByCurrentUser?: boolean
}

type UserResponse = {
  ok: boolean
  user?: RemoteUser | null
  error?: string
}

type ConversationsResponse = {
  ok: boolean
  conversations?: Conversation[]
  conversation?: Conversation
  error?: string
}

type TeamsResponse = {
  ok: boolean
  teams?: RemoteTeam[]
  team?: RemoteTeam
  error?: string
}

type TeamMembersResponse = {
  ok: boolean
  members?: RemoteTeamMember[]
  member?: RemoteTeamMember
  error?: string
}

type TeamConversationsResponse = {
  ok: boolean
  team?: RemoteTeam
  conversations?: TeamConversation[]
  error?: string
}

type AttachmentResponse = {
  ok: boolean
  attachment?: ChatAttachment
  error?: string
}

type StatsResponse = {
  ok: boolean
  stats?: ChatStats
  error?: string
}

type CsrfResponse = {
  ok: boolean
  csrfToken?: string
  error?: string
}

type SecurityEventsResponse = {
  ok: boolean
  events?: RemoteSecurityEvent[]
  error?: string
}

const API_BASE = import.meta.env.VITE_API_BASE || ''
const CSRF_COOKIE_NAME = 'csrftoken'
const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE'])
let csrfTokenCache = ''

async function apiFetch(path: string, options: RequestInit = {}) {
  const method = (options.method || 'GET').toUpperCase()
  const headers = new Headers(options.headers || {})
  const isUnsafe = UNSAFE_METHODS.has(method)
  if (isUnsafe) {
    const token = await ensureCsrfToken()
    headers.set('X-CSRFToken', token)
  }
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'include',
    headers,
  })
  const nextToken = readCookie(CSRF_COOKIE_NAME)
  if (nextToken) {
    csrfTokenCache = nextToken
  } else if (isUnsafe && response.status === 403) {
    csrfTokenCache = ''
  }
  return response
}

async function ensureCsrfToken() {
  csrfTokenCache = csrfTokenCache || readCookie(CSRF_COOKIE_NAME)
  if (csrfTokenCache) return csrfTokenCache

  const response = await fetch(`${API_BASE}/api/users/auth/csrf/`, {
    credentials: 'include',
  })
  const payload = (await parseJson(response)) as CsrfResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  csrfTokenCache = payload.csrfToken || readCookie(CSRF_COOKIE_NAME)
  if (!csrfTokenCache) {
    throw new Error('无法获取 CSRF token')
  }
  return csrfTokenCache
}

function readCookie(name: string) {
  if (typeof document === 'undefined') return ''
  const prefix = `${name}=`
  const item = document.cookie
    .split(';')
    .map((value) => value.trim())
    .find((value) => value.startsWith(prefix))
  return item ? decodeURIComponent(item.slice(prefix.length)) : ''
}

export async function getCurrentUser() {
  const response = await apiFetch('/api/users/me/')
  return parseUserResponse(response)
}

export async function registerUser(payload: {
  username: string
  password: string
  nickname: string
  avatarText: string
  settings: UserSettings
}) {
  const response = await apiFetch('/api/users/auth/register/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseUserResponse(response)
}

export async function loginUser(username: string, password: string) {
  const response = await apiFetch('/api/users/auth/login/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  return parseUserResponse(response)
}

export async function logoutUser() {
  const response = await apiFetch('/api/users/auth/logout/', {
    method: 'POST',
  })
  await parseOkResponse(response)
}

export async function updateRemoteProfile(payload: {
  nickname?: string
  avatarText?: string
  settings?: UserSettings
}) {
  const response = await apiFetch('/api/users/profile/', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseUserResponse(response)
}

export async function fetchRemoteConversations() {
  const response = await apiFetch('/api/users/conversations/')
  return parseConversationsResponse(response)
}

export async function syncRemoteConversations(conversations: Conversation[]) {
  const response = await apiFetch('/api/users/conversations/sync/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversations }),
  })
  return parseConversationsResponse(response)
}

export async function deleteRemoteConversation(id: string) {
  const response = await apiFetch(
    `/api/users/conversations/${encodeURIComponent(id)}/delete/`,
    {
      method: 'DELETE',
    },
  )
  await parseOkResponse(response)
}

export async function assignRemoteConversationTeam(id: string, teamId: string | null) {
  const response = await apiFetch(
    `/api/users/conversations/${encodeURIComponent(id)}/team/`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ teamId }),
    },
  )
  return parseConversationResponse(response)
}

export async function fetchTeams() {
  const response = await apiFetch('/api/users/teams/')
  return parseTeamsResponse(response)
}

export async function createTeam(payload: { name: string; description?: string }) {
  const response = await apiFetch('/api/users/teams/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseTeamResponse(response)
}

export async function updateTeam(
  teamId: string,
  payload: { name?: string; description?: string },
) {
  const response = await apiFetch(`/api/users/teams/${encodeURIComponent(teamId)}/`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return parseTeamResponse(response)
}

export async function deleteTeam(teamId: string) {
  const response = await apiFetch(`/api/users/teams/${encodeURIComponent(teamId)}/`, {
    method: 'DELETE',
  })
  await parseOkResponse(response)
}

export async function fetchTeamMembers(teamId: string) {
  const response = await apiFetch(`/api/users/teams/${encodeURIComponent(teamId)}/members/`)
  return parseTeamMembersResponse(response)
}

export async function fetchTeamConversations(teamId: string) {
  const response = await apiFetch(`/api/users/teams/${encodeURIComponent(teamId)}/conversations/`)
  return parseTeamConversationsResponse(response)
}

export async function addTeamMember(
  teamId: string,
  payload: { username: string; role: TeamRole },
) {
  const response = await apiFetch(
    `/api/users/teams/${encodeURIComponent(teamId)}/members/`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
  )
  return parseTeamMemberResponse(response)
}

export async function updateTeamMemberRole(teamId: string, userId: number, role: TeamRole) {
  const response = await apiFetch(
    `/api/users/teams/${encodeURIComponent(teamId)}/members/${userId}/`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role }),
    },
  )
  return parseTeamMemberResponse(response)
}

export async function removeTeamMember(teamId: string, userId: number) {
  const response = await apiFetch(
    `/api/users/teams/${encodeURIComponent(teamId)}/members/${userId}/`,
    {
      method: 'DELETE',
    },
  )
  await parseOkResponse(response)
}

export async function uploadConversationAttachment(
  conversationId: string,
  file: File,
  messageClientId: string,
) {
  const formData = new FormData()
  formData.append('image', file)
  formData.append('messageClientId', messageClientId)
  const response = await apiFetch(
    `/api/users/conversations/${encodeURIComponent(conversationId)}/attachments/upload/`,
    {
      method: 'POST',
      body: formData,
    },
  )
  return parseAttachmentResponse(response)
}

export async function fetchUserStatsSummary(days = 7, teamId?: string) {
  const params = new URLSearchParams({ days: String(days) })
  if (teamId) params.set('teamId', teamId)
  const response = await apiFetch(`/api/users/stats/summary/?${params.toString()}`)
  return parseStatsResponse(response)
}

export async function fetchSecurityEvents() {
  const response = await apiFetch('/api/users/security/events/')
  return parseSecurityEventsResponse(response)
}

async function parseUserResponse(response: Response) {
  const payload = (await parseJson(response)) as UserResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.user || null
}

async function parseConversationsResponse(response: Response) {
  const payload = (await parseJson(response)) as ConversationsResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.conversations || []
}

async function parseConversationResponse(response: Response) {
  const payload = (await parseJson(response)) as ConversationsResponse
  if (!response.ok || !payload.ok || !payload.conversation) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.conversation
}

async function parseTeamsResponse(response: Response) {
  const payload = (await parseJson(response)) as TeamsResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.teams || []
}

async function parseTeamResponse(response: Response) {
  const payload = (await parseJson(response)) as TeamsResponse
  if (!response.ok || !payload.ok || !payload.team) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.team
}

async function parseTeamMembersResponse(response: Response) {
  const payload = (await parseJson(response)) as TeamMembersResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.members || []
}

async function parseTeamConversationsResponse(response: Response) {
  const payload = (await parseJson(response)) as TeamConversationsResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.conversations || []
}

async function parseTeamMemberResponse(response: Response) {
  const payload = (await parseJson(response)) as TeamMembersResponse
  if (!response.ok || !payload.ok || !payload.member) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.member
}

async function parseOkResponse(response: Response) {
  const payload = (await parseJson(response)) as { ok: boolean; error?: string }
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
}

async function parseAttachmentResponse(response: Response) {
  const payload = (await parseJson(response)) as AttachmentResponse
  if (!response.ok || !payload.ok || !payload.attachment) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.attachment
}

async function parseStatsResponse(response: Response) {
  const payload = (await parseJson(response)) as StatsResponse
  if (!response.ok || !payload.ok || !payload.stats) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.stats
}

async function parseSecurityEventsResponse(response: Response) {
  const payload = (await parseJson(response)) as SecurityEventsResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.events || []
}

async function parseJson(response: Response) {
  const rawText = await response.text()
  try {
    return JSON.parse(rawText)
  } catch {
    throw new Error(friendlyHttpError(response, rawText))
  }
}
