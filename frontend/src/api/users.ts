import type { Conversation, UserProfile, UserSettings } from '@/stores/chat'

export type RemoteUser = {
  id: number
  username: string
  nickname: string
  avatarText: string
  settings: UserSettings
}

type UserResponse = {
  ok: boolean
  user?: RemoteUser | null
  error?: string
}

type ConversationsResponse = {
  ok: boolean
  conversations?: Conversation[]
  error?: string
}

const API_BASE = import.meta.env.VITE_API_BASE || ''

export async function getCurrentUser() {
  const response = await fetch(`${API_BASE}/api/users/me/`, {
    credentials: 'include',
  })
  return parseUserResponse(response)
}

export async function registerUser(payload: {
  username: string
  password: string
  nickname: string
  avatarText: string
  settings: UserSettings
}) {
  const response = await fetch(`${API_BASE}/api/users/auth/register/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  })
  return parseUserResponse(response)
}

export async function loginUser(username: string, password: string) {
  const response = await fetch(`${API_BASE}/api/users/auth/login/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ username, password }),
  })
  return parseUserResponse(response)
}

export async function logoutUser() {
  const response = await fetch(`${API_BASE}/api/users/auth/logout/`, {
    method: 'POST',
    credentials: 'include',
  })
  await parseOkResponse(response)
}

export async function updateRemoteProfile(payload: {
  nickname?: string
  avatarText?: string
  settings?: UserSettings
}) {
  const response = await fetch(`${API_BASE}/api/users/profile/`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  })
  return parseUserResponse(response)
}

export async function fetchRemoteConversations() {
  const response = await fetch(`${API_BASE}/api/users/conversations/`, {
    credentials: 'include',
  })
  return parseConversationsResponse(response)
}

export async function syncRemoteConversations(conversations: Conversation[]) {
  const response = await fetch(`${API_BASE}/api/users/conversations/sync/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ conversations }),
  })
  return parseConversationsResponse(response)
}

export async function deleteRemoteConversation(id: string) {
  const response = await fetch(
    `${API_BASE}/api/users/conversations/${encodeURIComponent(id)}/delete/`,
    {
      method: 'DELETE',
      credentials: 'include',
    },
  )
  await parseOkResponse(response)
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

async function parseOkResponse(response: Response) {
  const payload = (await parseJson(response)) as { ok: boolean; error?: string }
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
}

async function parseJson(response: Response) {
  const rawText = await response.text()
  try {
    return JSON.parse(rawText)
  } catch {
    throw new Error(rawText || `HTTP ${response.status}`)
  }
}
