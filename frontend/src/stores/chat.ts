import { sendImageMessage, sendTextMessage, streamMessage, type StreamStepEvent } from '@/api/chat'
import {
  deleteRemoteConversation,
  fetchUserStatsSummary,
  fetchRemoteConversations,
  getCurrentUser,
  loginUser,
  logoutUser,
  registerUser,
  syncRemoteConversations,
  updateRemoteProfile,
  uploadConversationAttachment,
  type RemoteUser,
} from '@/api/users'
import { createSafeMarkdownRenderer } from '@/utils/markdown'
import { defineStore } from 'pinia'
import { computed, nextTick, ref, watch } from 'vue'

export type MessageRole = 'user' | 'assistant'
export type AgentStepStatus = 'pending' | 'active' | 'done' | 'error'

export type AgentStep = {
  key: string
  label: string
  message: string
  status: AgentStepStatus
  data?: Record<string, unknown>
}

export type ChatAttachment = {
  id: string
  kind: 'image'
  name: string
  url: string
  thumbnailUrl?: string
  size: number
  mimeType: string
  createdAt: string
  messageClientId?: string | null
}

export type ChatMessage = {
  id: string
  role: MessageRole
  content: string
  imageUrl?: string
  sourceFileName?: string
  attachments?: ChatAttachment[]
  createdAt: string
  status?: 'streaming' | 'done' | 'error'
  steps?: AgentStep[]
  currentStatus?: string
}

export type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
  sceneType?: string
  hazardLevel?: string
  isArchived?: boolean
  previewImageUrl?: string
  previewAttachmentId?: string
  isTitleManual?: boolean
}

export type UserProfile = {
  nickname: string
  avatarText: string
}

export type UserSettings = {
  useStream: boolean
  autoExpandSteps: boolean
  temperature: number
}

export type AuthStatus = 'checking' | 'guest' | 'authenticated'
export type SyncStatus = 'idle' | 'syncing' | 'synced' | 'error'
export type StatsStatus = 'idle' | 'loading' | 'ready' | 'error'

type StoredChatState = {
  conversations?: Conversation[]
  activeId?: string
  userProfile?: Partial<UserProfile>
  settings?: Partial<UserSettings>
  ownerUserId?: number | null
}

export type ChatStats = {
  totalConversations: number
  totalMessages: number
  completedReports: number
  archivedCount: number
  completionRate: number
  activeDays: number
  latestActivity: string
  recentSevenDays: Array<{ date: string; count: number }>
  sceneCounts: Array<{ label: string; count: number }>
  hazardCounts: Array<{ label: string; count: number; tone: 'danger' | 'warning' | 'success' | 'neutral' }>
  topHazardLabel: string
}

type AppendMessageOptions = {
  id?: string
  imageUrl?: string
  status?: ChatMessage['status']
  sourceFileName?: string
  attachments?: ChatAttachment[]
}

const LEGACY_STORAGE_KEY = 'ventilation-graph-rag:user-module:v1'
const STORAGE_KEY_PREFIX = 'ventilation-graph-rag:user-module:v2'
const GUEST_STORAGE_KEY = `${STORAGE_KEY_PREFIX}:guest`
const DEFAULT_TITLE = '新建辨识'

const STEP_LABELS: Record<string, string> = {
  vision_observe: '观察图片',
  vision_observe_done: '观察完成',
  concept_search: '学习概念',
  concept_search_done: '概念完成',
  vision_analyze: '重新分析',
  vision_analyze_done: '分析完成',
  cypher_match: '匹配规程',
  cypher_match_done: '规程完成',
  generating: '生成报告',
}

const DONE_STEPS: Record<string, string> = {
  vision_observe_done: 'vision_observe',
  concept_search_done: 'concept_search',
  vision_analyze_done: 'vision_analyze',
  cypher_match_done: 'cypher_match',
}

const DEFAULT_PROFILE: UserProfile = {
  nickname: '安全工程师',
  avatarText: '安',
}

const DEFAULT_SETTINGS: UserSettings = {
  useStream: true,
  autoExpandSteps: true,
  temperature: 0.2,
}

const printableMarkdown = createSafeMarkdownRenderer()

const createId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const toTimestamp = (value?: string) => {
  const time = value ? Date.parse(value) : Number.NaN
  return Number.isFinite(time) ? time : 0
}

const sortByUpdatedAt = (items: Conversation[]) =>
  [...items].sort((left, right) => toTimestamp(right.updatedAt) - toTimestamp(left.updatedAt))

export const useChatStore = defineStore('chat', () => {
  const conversations = ref<Conversation[]>([])
  const activeId = ref('')
  const sendingByConversation = ref<Record<string, boolean>>({})
  const error = ref('')
  const searchQuery = ref('')
  const userProfile = ref<UserProfile>({ ...DEFAULT_PROFILE })
  const settings = ref<UserSettings>({ ...DEFAULT_SETTINGS })
  const remoteUser = ref<RemoteUser | null>(null)
  const authStatus = ref<AuthStatus>('checking')
  const authError = ref('')
  const syncStatus = ref<SyncStatus>('idle')
  const syncError = ref('')
  const statsStatus = ref<StatsStatus>('idle')
  const statsError = ref('')
  const remoteStats = ref<ChatStats | null>(null)
  const lastSyncedAt = ref('')
  let saveTimer: number | undefined
  let syncTimer: number | undefined
  let isApplyingRemote = false
  let activeStorageKey = GUEST_STORAGE_KEY

  const visibleConversations = computed(() =>
    sortByUpdatedAt(conversations.value.filter((conversation) => !conversation.isArchived)),
  )
  const archivedConversations = computed(() =>
    sortByUpdatedAt(conversations.value.filter((conversation) => conversation.isArchived)),
  )
  const activeConversation = computed<Conversation | null>(() => {
    const conversation = findConversation(activeId.value)
    return conversation && !conversation.isArchived ? conversation : null
  })
  const isSending = computed(() =>
    activeId.value ? Boolean(sendingByConversation.value[activeId.value]) : false,
  )
  const filteredConversations = computed(() => searchConversations(searchQuery.value))
  const localStats = computed<ChatStats>(() => buildStats())
  const stats = computed<ChatStats>(() =>
    authStatus.value === 'authenticated' && remoteStats.value ? remoteStats.value : localStats.value,
  )

  function loadFromStorage() {
    if (typeof window === 'undefined') return
    const raw = readStoredState(activeStorageKey)
    if (!raw) {
      resetLocalState()
      return
    }

    try {
      const stored = JSON.parse(raw) as StoredChatState
      conversations.value = normalizeConversations(stored.conversations || [])
      userProfile.value = normalizeProfile(stored.userProfile)
      settings.value = normalizeSettings(stored.settings)
      const storedActiveId = stored.activeId || ''
      activeId.value = conversations.value.some((conversation) => conversation.id === storedActiveId)
        ? storedActiveId
        : visibleConversations.value[0]?.id || ''
    } catch {
      conversations.value = []
      activeId.value = ''
    }
  }

  function switchStorageScope(user: RemoteUser | null) {
    if (typeof window === 'undefined') return
    if (saveTimer) {
      window.clearTimeout(saveTimer)
      saveTimer = undefined
    }
    if (syncTimer) {
      window.clearTimeout(syncTimer)
      syncTimer = undefined
    }
    activeStorageKey = storageKeyForUser(user)
    remoteStats.value = null
    statsStatus.value = 'idle'
    statsError.value = ''
    isApplyingRemote = true
    loadFromStorage()
    isApplyingRemote = false
  }

  function resetLocalState() {
    conversations.value = []
    activeId.value = ''
    userProfile.value = { ...DEFAULT_PROFILE }
    settings.value = { ...DEFAULT_SETTINGS }
  }

  function saveToStorage() {
    if (typeof window === 'undefined') return
    const payload: StoredChatState = {
      conversations: conversations.value.map((conversation) =>
        sanitizeConversationForStorage(conversation, authStatus.value !== 'authenticated'),
      ),
      activeId: activeId.value,
      userProfile: userProfile.value,
      settings: settings.value,
      ownerUserId: remoteUser.value?.id || null,
    }
    try {
      window.localStorage.setItem(activeStorageKey, JSON.stringify(payload))
    } catch {
      const compactPayload: StoredChatState = {
        ...payload,
        conversations: conversations.value.map((conversation) =>
          sanitizeConversationForStorage(conversation, false),
        ),
      }
      window.localStorage.setItem(activeStorageKey, JSON.stringify(compactPayload))
    }
  }

  function scheduleSave() {
    if (typeof window === 'undefined') return
    if (isApplyingRemote) return
    if (saveTimer) window.clearTimeout(saveTimer)
    saveTimer = window.setTimeout(() => {
      saveToStorage()
      saveTimer = undefined
    }, 250)
  }

  function scheduleRemoteSync() {
    if (typeof window === 'undefined' || authStatus.value !== 'authenticated' || isApplyingRemote) {
      return
    }
    if (syncTimer) window.clearTimeout(syncTimer)
    syncTimer = window.setTimeout(() => {
      void syncWithRemote()
      syncTimer = undefined
    }, 800)
  }

  function createConversation(title = DEFAULT_TITLE) {
    const now = new Date().toISOString()
    const conversation: Conversation = {
      id: createId(),
      title,
      messages: [],
      createdAt: now,
      updatedAt: now,
    }
    conversations.value.unshift(conversation)
    activeId.value = conversation.id
    return conversation.id
  }

  function newConversation() {
    return createConversation()
  }

  function selectConversation(id: string) {
    const conversation = findConversation(id)
    if (!conversation || conversation.isArchived) return false
    activeId.value = id
    return true
  }

  function deleteConversation(id: string) {
    if (sendingByConversation.value[id]) return activeId.value
    const index = conversations.value.findIndex((conversation) => conversation.id === id)
    if (index === -1) return activeId.value
    conversations.value.splice(index, 1)
    void deleteRemoteConversationIfNeeded(id)
    if (activeId.value === id) {
      activeId.value = visibleConversations.value[0]?.id || ''
    }
    return activeId.value
  }

  function archiveConversation(id: string) {
    if (sendingByConversation.value[id]) return activeId.value
    const conversation = findConversation(id)
    if (!conversation) return activeId.value
    conversation.isArchived = true
    conversation.updatedAt = new Date().toISOString()
    if (activeId.value === id) {
      activeId.value = visibleConversations.value[0]?.id || ''
    }
    return activeId.value
  }

  function restoreConversation(id: string) {
    const conversation = findConversation(id)
    if (!conversation) return false
    conversation.isArchived = false
    conversation.updatedAt = new Date().toISOString()
    activeId.value = id
    return true
  }

  function renameConversation(id: string, title: string) {
    const conversation = findConversation(id)
    const nextTitle = title.trim()
    if (!conversation || !nextTitle) return false
    conversation.title = nextTitle
    conversation.isTitleManual = true
    conversation.updatedAt = new Date().toISOString()
    return true
  }

  async function submit(question: string, image: File | null, useStream: boolean) {
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion && !image) return
    const conversationId = activeConversation.value?.id || createConversation()
    const conversation = findConversation(conversationId)
    if (!conversation || sendingByConversation.value[conversationId]) return

    setConversationSending(conversationId, true)
    error.value = ''

    const messageId = createId()
    const uploadedAttachment = image ? await uploadAttachmentForMessage(conversationId, image, messageId) : undefined
    const fallbackImageUrl =
      image && (!uploadedAttachment || authStatus.value !== 'authenticated')
        ? await fileToImageDataUrl(image)
        : undefined
    const messageImageUrl = uploadedAttachment?.url || fallbackImageUrl
    const userMessage = appendMessage(
      conversationId,
      'user',
      trimmedQuestion || '请判断图片中的通风安全隐患',
      {
        id: messageId,
        imageUrl: messageImageUrl,
        status: 'done',
        sourceFileName: image?.name,
        attachments: uploadedAttachment ? [uploadedAttachment] : undefined,
      },
    )
    if (messageImageUrl && !conversation.previewImageUrl) {
      conversation.previewImageUrl = messageImageUrl
    }
    if (uploadedAttachment && !conversation.previewAttachmentId) {
      conversation.previewAttachmentId = uploadedAttachment.id
    }
    updateAutoTitle(conversation, userMessage.content)
    const assistantMessage = appendMessage(conversationId, 'assistant', '', { status: 'streaming' })
    let hasReceivedToken = false

    try {
      if (useStream) {
        await streamMessage(userMessage.content, image, {
          onStatus(message) {
            const current = findMessage(conversationId, assistantMessage.id)
            if (!current) return
            current.currentStatus = normalizeStatusMessage(message)
            if (!hasReceivedToken && !current.steps?.length) {
              current.content = current.currentStatus
            }
            touchConversation(conversationId)
          },
          onStep(step) {
            applyStep(conversationId, assistantMessage.id, step)
          },
          onToken(content) {
            const current = findMessage(conversationId, assistantMessage.id)
            if (!current) return
            if (!hasReceivedToken) {
              current.content = ''
              hasReceivedToken = true
              markActiveStepsDone(current)
            }
            current.content += content
            current.status = 'streaming'
            touchConversation(conversationId)
          },
          onError(message) {
            const current = findMessage(conversationId, assistantMessage.id)
            if (current?.steps) {
              current.steps.forEach((step) => {
                if (step.status === 'active') step.status = 'error'
              })
            }
            updateMessage(conversationId, assistantMessage.id, { content: message, status: 'error' })
            error.value = message
          },
          onDone() {
            const current = findMessage(conversationId, assistantMessage.id)
            if (!current || current.status === 'error') return
            markActiveStepsDone(current)
            if (!current.content.trim()) current.content = '未收到有效回答'
            current.status = 'done'
            updateConversationMeta(conversationId, current)
          },
        })
        const current = findMessage(conversationId, assistantMessage.id)
        if (current?.status === 'streaming') {
          markActiveStepsDone(current)
          current.status = 'done'
          updateConversationMeta(conversationId, current)
        }
      } else {
        const answer = image
          ? await sendImageMessage(userMessage.content, image)
          : await sendTextMessage(userMessage.content)
        updateMessage(conversationId, assistantMessage.id, { content: answer, status: 'done' })
        const current = findMessage(conversationId, assistantMessage.id)
        if (current) updateConversationMeta(conversationId, current)
      }
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : '请求失败'
      updateMessage(conversationId, assistantMessage.id, { content: message, status: 'error' })
      error.value = message
    } finally {
      touchConversation(conversationId)
      setConversationSending(conversationId, false)
    }
  }

  function appendMessage(
    conversationId: string,
    role: MessageRole,
    content: string,
    options: AppendMessageOptions = {},
  ): ChatMessage {
    const message: ChatMessage = {
      id: options.id || createId(),
      role,
      content,
      imageUrl: options.imageUrl,
      sourceFileName: options.sourceFileName,
      attachments: options.attachments,
      createdAt: new Date().toISOString(),
      status: options.status || 'done',
    }
    const conversation = findConversation(conversationId)
    if (!conversation) {
      throw new Error('未找到当前对话')
    }
    conversation.messages.push(message)
    conversation.updatedAt = new Date().toISOString()
    return conversation.messages[conversation.messages.length - 1]!
  }

  async function uploadAttachmentForMessage(conversationId: string, image: File, messageId: string) {
    if (authStatus.value !== 'authenticated') return undefined
    try {
      return await uploadConversationAttachment(conversationId, image, messageId)
    } catch (exc) {
      syncStatus.value = 'error'
      syncError.value = exc instanceof Error ? exc.message : '图片附件上传失败'
      return undefined
    }
  }

  function findConversation(id: string): Conversation | undefined {
    return conversations.value.find((conversation) => conversation.id === id)
  }

  function hasConversation(id: string) {
    return Boolean(findConversation(id))
  }

  function findMessage(conversationId: string, id: string): ChatMessage | undefined {
    return findConversation(conversationId)?.messages.find((message) => message.id === id)
  }

  function updateMessage(
    conversationId: string,
    id: string,
    updates: Partial<Pick<ChatMessage, 'content' | 'status'>>,
  ) {
    const message = findMessage(conversationId, id)
    if (!message) return
    Object.assign(message, updates)
    touchConversation(conversationId)
  }

  function applyStep(conversationId: string, messageId: string, event: StreamStepEvent) {
    const message = findMessage(conversationId, messageId)
    if (!message) return
    if (!message.steps) message.steps = []

    const doneTarget = DONE_STEPS[event.step]
    if (doneTarget) {
      const target = message.steps.find((step) => step.key === doneTarget)
      if (target) {
        target.status = 'done'
        target.message = event.message || target.message
        target.data = event.data
        message.currentStatus = event.message
        message.status = 'streaming'
        updateMetaFromStep(conversationId, event)
        touchConversation(conversationId)
        return
      }
    }

    markActiveStepsDone(message)
    const existing = message.steps.find((step) => step.key === event.step)
    const status: AgentStepStatus = event.step.endsWith('_done') ? 'done' : 'active'
    if (existing) {
      existing.message = event.message
      existing.status = status
      existing.data = event.data
    } else {
      message.steps.push({
        key: event.step,
        label: STEP_LABELS[event.step] || event.step,
        message: event.message,
        status,
        data: event.data,
      })
    }
    message.currentStatus = event.message
    message.status = 'streaming'
    updateMetaFromStep(conversationId, event)
    touchConversation(conversationId)
  }

  function setConversationSending(conversationId: string, value: boolean) {
    if (value) {
      sendingByConversation.value = {
        ...sendingByConversation.value,
        [conversationId]: true,
      }
      return
    }

    const next = { ...sendingByConversation.value }
    delete next[conversationId]
    sendingByConversation.value = next
  }

  function markActiveStepsDone(message: ChatMessage) {
    message.steps?.forEach((step) => {
      if (step.status === 'active') step.status = 'done'
    })
  }

  function touchConversation(conversationId: string) {
    const conversation = findConversation(conversationId)
    if (conversation) conversation.updatedAt = new Date().toISOString()
  }

  function updateAutoTitle(conversation: Conversation, content: string) {
    if (conversation.isTitleManual) return
    const hasEarlierUserMessage =
      conversation.messages.filter((message) => message.role === 'user').length > 1
    if (hasEarlierUserMessage) return
    conversation.title = buildTitle(content)
  }

  function buildTitle(content: string) {
    const normalized = content.replace(/\s+/g, ' ').trim()
    if (!normalized) return DEFAULT_TITLE
    return normalized.length > 18 ? `${normalized.slice(0, 18)}...` : normalized
  }

  function normalizeStatusMessage(message: string) {
    if (!message || message === 'started') {
      return '已连接，正在检索规程知识库...'
    }
    return message
  }

  function updateMetaFromStep(conversationId: string, event: StreamStepEvent) {
    const conversation = findConversation(conversationId)
    const data = event.data || {}
    if (!conversation) return
    if (typeof data.scene_name === 'string' && data.scene_name.trim()) {
      conversation.sceneType = data.scene_name.trim()
    }
    const hazard = data.hazard_level || data.risk_level || data.level
    if (typeof hazard === 'string' && hazard.trim()) {
      conversation.hazardLevel = hazard.trim()
    }
  }

  function updateConversationMeta(conversationId: string, message: ChatMessage) {
    const conversation = findConversation(conversationId)
    if (!conversation) return
    const inferredHazard = inferHazardLevel(message.content)
    if (inferredHazard) conversation.hazardLevel = inferredHazard
    touchConversation(conversationId)
  }

  function inferHazardLevel(content: string) {
    if (/重大隐患|重大风险|高风险|严重/.test(content)) return '高风险'
    if (/一般隐患|中等风险|中风险/.test(content)) return '中风险'
    if (/低风险|轻微|基本合规|未发现/.test(content)) return '低风险'
    return ''
  }

  function searchConversations(query: string) {
    const tokens = tokenizeSearchQuery(query)
    const activeItems = visibleConversations.value
    if (!tokens.length) return activeItems

    return activeItems
      .map((conversation) => ({
        conversation,
        score: scoreConversationSearch(conversation, tokens),
      }))
      .filter((item) => item.score > 0)
      .sort((left, right) => {
        if (right.score !== left.score) return right.score - left.score
        return toTimestamp(right.conversation.updatedAt) - toTimestamp(left.conversation.updatedAt)
      })
      .map((item) => item.conversation)
  }

  function scoreConversationSearch(conversation: Conversation, tokens: string[]) {
    const fields = buildConversationSearchFields(conversation)
    let totalScore = 0

    for (const token of tokens) {
      const expandedTokens = expandSearchToken(token)
      const tokenScore = Math.max(
        ...expandedTokens.map((candidate) =>
          fields.reduce((score, field) => {
            if (!field.text.includes(candidate)) return score
            const exactBoost = candidate === token ? 2 : 1
            return score + field.weight * exactBoost
          }, 0),
        ),
      )
      if (tokenScore <= 0) return 0
      totalScore += tokenScore
    }

    return totalScore
  }

  function buildConversationSearchFields(conversation: Conversation) {
    const updatedAt = new Date(conversation.updatedAt)
    const date = updatedAt.toLocaleDateString('zh-CN')
    const isoDate = conversation.updatedAt.slice(0, 10)
    const messageText = conversation.messages
      .map((message) => `${message.role === 'assistant' ? '报告' : '提问'} ${message.content} ${message.sourceFileName || ''}`)
      .join(' ')

    return [
      { text: normalizeSearchText(conversation.title), weight: 10 },
      { text: normalizeSearchText(conversation.sceneType || ''), weight: 8 },
      { text: normalizeSearchText(conversation.hazardLevel || ''), weight: 7 },
      { text: normalizeSearchText(`${date} ${isoDate}`), weight: 4 },
      { text: normalizeSearchText(messageText), weight: 3 },
    ]
  }

  function updateUserProfile(updates: Partial<UserProfile>) {
    const nickname = updates.nickname?.trim() || userProfile.value.nickname
    userProfile.value = {
      nickname,
      avatarText: (updates.avatarText?.trim() || nickname.slice(0, 1) || '安').slice(0, 2),
    }
    void pushRemoteProfile()
  }

  function updateSettings(updates: Partial<UserSettings>) {
    settings.value = normalizeSettings({ ...settings.value, ...updates })
    void pushRemoteProfile()
  }

  async function initializeUserSession() {
    authStatus.value = 'checking'
    authError.value = ''
    try {
      const user = await getCurrentUser()
      if (!user) {
        remoteUser.value = null
        authStatus.value = 'guest'
        remoteStats.value = null
        statsStatus.value = 'idle'
        switchStorageScope(null)
        return
      }
      applyRemoteUser(user)
      authStatus.value = 'authenticated'
      await loadAccountConversations(user)
    } catch (exc) {
      remoteUser.value = null
      authStatus.value = 'guest'
      remoteStats.value = null
      statsStatus.value = 'idle'
      switchStorageScope(null)
      authError.value = exc instanceof Error ? exc.message : '无法读取登录状态'
    }
  }

  async function registerAccount(username: string, password: string, nickname: string) {
    authError.value = ''
    const cleanUsername = username.trim()
    const cleanNickname = nickname.trim() || userProfile.value.nickname || cleanUsername
    try {
      const user = await registerUser({
        username: cleanUsername,
        password,
        nickname: cleanNickname,
        avatarText: cleanNickname.slice(0, 2),
        settings: settings.value,
      })
      if (user) {
        applyRemoteUser(user)
        authStatus.value = 'authenticated'
        await migrateGuestConversationsToNewAccount(user)
      }
      return true
    } catch (exc) {
      authError.value = exc instanceof Error ? exc.message : '注册失败'
      return false
    }
  }

  async function loginAccount(username: string, password: string) {
    authError.value = ''
    try {
      const user = await loginUser(username.trim(), password)
      if (user) {
        applyRemoteUser(user)
        authStatus.value = 'authenticated'
        await loadAccountConversations(user)
      }
      return true
    } catch (exc) {
      authError.value = exc instanceof Error ? exc.message : '登录失败'
      return false
    }
  }

  async function logoutAccount() {
    authError.value = ''
    try {
      await logoutUser()
      saveToStorage()
      remoteUser.value = null
      authStatus.value = 'guest'
      syncStatus.value = 'idle'
      remoteStats.value = null
      statsStatus.value = 'idle'
      switchStorageScope(null)
      return true
    } catch (exc) {
      authError.value = exc instanceof Error ? exc.message : '退出失败'
      return false
    }
  }

  async function syncWithRemote() {
    if (authStatus.value !== 'authenticated') return false
    if (syncTimer) {
      window.clearTimeout(syncTimer)
      syncTimer = undefined
    }
    syncStatus.value = 'syncing'
    syncError.value = ''
    try {
      const remoteConversations = await syncRemoteConversations(
        conversations.value.map((conversation) => sanitizeConversationForStorage(conversation, false)),
      )
      isApplyingRemote = true
      conversations.value = mergeConversations(conversations.value, remoteConversations)
      if (!activeId.value || !conversations.value.some((conversation) => conversation.id === activeId.value)) {
        activeId.value = visibleConversations.value[0]?.id || ''
      }
      lastSyncedAt.value = new Date().toISOString()
      syncStatus.value = 'synced'
      await nextTick()
      saveToStorage()
      void refreshStats()
      return true
    } catch (exc) {
      syncStatus.value = 'error'
      syncError.value = exc instanceof Error ? exc.message : '同步失败'
      return false
    } finally {
      await nextTick()
      isApplyingRemote = false
    }
  }

  async function loadAccountConversations(user: RemoteUser) {
    switchStorageScope(user)
    const scopedLocalConversations = normalizeConversations(conversations.value)
    applyRemoteUser(user)
    syncStatus.value = 'syncing'
    syncError.value = ''
    try {
      const remoteConversations = await fetchRemoteConversations()
      isApplyingRemote = true
      conversations.value = mergeConversations(scopedLocalConversations, remoteConversations)
      activeId.value = visibleConversations.value[0]?.id || ''
      lastSyncedAt.value = new Date().toISOString()
      syncStatus.value = 'synced'
      await nextTick()
      saveToStorage()
      clearLegacyStorage()
      void refreshStats()
      return true
    } catch (exc) {
      syncStatus.value = 'error'
      syncError.value = exc instanceof Error ? exc.message : '同步失败'
      return false
    } finally {
      await nextTick()
      isApplyingRemote = false
    }
  }

  async function migrateGuestConversationsToNewAccount(user: RemoteUser) {
    const guestConversations = normalizeConversations(conversations.value)
    switchStorageScope(user)
    applyRemoteUser(user)
    isApplyingRemote = true
    conversations.value = guestConversations
    activeId.value = visibleConversations.value[0]?.id || ''
    isApplyingRemote = false
    const ok = await syncWithRemote()
    if (ok) clearGuestStorage()
    void refreshStats()
    return ok
  }

  async function refreshStats() {
    if (authStatus.value !== 'authenticated') {
      remoteStats.value = null
      statsStatus.value = 'idle'
      statsError.value = ''
      return localStats.value
    }

    statsStatus.value = 'loading'
    statsError.value = ''
    try {
      const nextStats = await fetchUserStatsSummary(7)
      remoteStats.value = normalizeStats(nextStats, localStats.value)
      statsStatus.value = 'ready'
      return remoteStats.value
    } catch (exc) {
      statsStatus.value = 'error'
      statsError.value = exc instanceof Error ? exc.message : '统计加载失败'
      return localStats.value
    }
  }

  function applyRemoteUser(user: RemoteUser) {
    remoteUser.value = user
    userProfile.value = normalizeProfile({
      nickname: user.nickname,
      avatarText: user.avatarText,
    })
    settings.value = normalizeSettings(user.settings)
  }

  async function pushRemoteProfile() {
    if (authStatus.value !== 'authenticated') return
    try {
      const user = await updateRemoteProfile({
        nickname: userProfile.value.nickname,
        avatarText: userProfile.value.avatarText,
        settings: settings.value,
      })
      if (user) remoteUser.value = user
    } catch (exc) {
      syncStatus.value = 'error'
      syncError.value = exc instanceof Error ? exc.message : '用户资料同步失败'
    }
  }

  async function deleteRemoteConversationIfNeeded(id: string) {
    if (authStatus.value !== 'authenticated') return
    try {
      await deleteRemoteConversation(id)
    } catch (exc) {
      syncStatus.value = 'error'
      syncError.value = exc instanceof Error ? exc.message : '远端删除失败'
    }
  }

  function exportConversationAsPDF(id: string) {
    if (typeof window === 'undefined') return false
    const conversation = findConversation(id)
    if (!conversation) return false
    const printWindow = window.open('', '_blank', 'width=980,height=760')
    if (!printWindow) return false
    printWindow.document.write(buildPrintableReport(conversation))
    printWindow.document.close()
    printWindow.focus()
    window.setTimeout(() => printWindow.print(), 300)
    return true
  }

  function exportAllAsJson() {
    if (typeof window === 'undefined') return false
    const payload = {
      exportedAt: new Date().toISOString(),
      conversations: conversations.value.map((conversation) =>
        sanitizeConversationForStorage(conversation, authStatus.value !== 'authenticated'),
      ),
      userProfile: userProfile.value,
      settings: settings.value,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `ventilation-conversations-${new Date().toISOString().slice(0, 10)}.json`
    link.click()
    URL.revokeObjectURL(url)
    return true
  }

  function buildStats(): ChatStats {
    const activeItems = conversations.value.filter((conversation) => !conversation.isArchived)
    const sceneMap = new Map<string, number>()
    const hazardMap = new Map<string, number>()
    activeItems.forEach((conversation) => {
      const label = conversation.sceneType || conversation.hazardLevel || '未分类'
      sceneMap.set(label, (sceneMap.get(label) || 0) + 1)
      const hazardLabel = normalizeHazardLabel(conversation.hazardLevel)
      hazardMap.set(hazardLabel, (hazardMap.get(hazardLabel) || 0) + 1)
    })
    const completedReports = activeItems.reduce(
      (sum, item) =>
        sum +
        item.messages.filter((message) => message.role === 'assistant' && message.status === 'done')
          .length,
      0,
    )
    const completedConversations = activeItems.filter((conversation) =>
      conversation.messages.some((message) => message.role === 'assistant' && message.status === 'done'),
    ).length
    const recentSevenDays = buildRecentSevenDays(activeItems)
    const hazardCounts = [...hazardMap.entries()]
      .map(([label, count]) => ({ label, count, tone: hazardTone(label) }))
      .sort((left, right) => right.count - left.count || hazardRank(left.label) - hazardRank(right.label))

    return {
      totalConversations: activeItems.length,
      totalMessages: activeItems.reduce((sum, item) => sum + item.messages.length, 0),
      completedReports,
      archivedCount: archivedConversations.value.length,
      completionRate: activeItems.length ? Math.round((completedConversations / activeItems.length) * 100) : 0,
      activeDays: recentSevenDays.filter((item) => item.count > 0).length,
      latestActivity: activeItems[0]?.updatedAt || '',
      recentSevenDays,
      sceneCounts: [...sceneMap.entries()]
        .map(([label, count]) => ({ label, count }))
        .sort((left, right) => right.count - left.count),
      hazardCounts,
      topHazardLabel: hazardCounts.find((item) => item.label !== '未分级')?.label || '未分级',
    }
  }

  loadFromStorage()
  if (typeof window !== 'undefined') {
    void initializeUserSession()
  }

  watch(conversations, scheduleSave, { deep: true })
  watch(conversations, scheduleRemoteSync, { deep: true })
  watch(activeId, scheduleSave)
  watch(userProfile, scheduleSave, { deep: true })
  watch(settings, scheduleSave, { deep: true })

  return {
    conversations,
    visibleConversations,
    archivedConversations,
    filteredConversations,
    activeId,
    activeConversation,
    sendingByConversation,
    isSending,
    error,
    searchQuery,
    userProfile,
    settings,
    remoteUser,
    authStatus,
    authError,
    syncStatus,
    syncError,
    statsStatus,
    statsError,
    lastSyncedAt,
    localStats,
    stats,
    loadFromStorage,
    saveToStorage,
    initializeUserSession,
    createConversation,
    newConversation,
    selectConversation,
    deleteConversation,
    archiveConversation,
    restoreConversation,
    renameConversation,
    submit,
    findConversation,
    hasConversation,
    searchConversations,
    updateUserProfile,
    updateSettings,
    registerAccount,
    loginAccount,
    logoutAccount,
    syncWithRemote,
    refreshStats,
    exportConversationAsPDF,
    exportAllAsJson,
  }
})

function normalizeConversations(items: Conversation[]) {
  const safeItems = Array.isArray(items) ? items : []
  return sortByUpdatedAt(
    safeItems
      .filter((item) => item && typeof item.id === 'string')
      .map((item) => {
        const now = new Date().toISOString()
        return {
          id: item.id,
          title: item.title?.trim() || DEFAULT_TITLE,
          messages: normalizeMessages(item.messages || []),
          createdAt: item.createdAt || item.updatedAt || now,
          updatedAt: item.updatedAt || item.createdAt || now,
          sceneType: item.sceneType,
          hazardLevel: item.hazardLevel,
          isArchived: Boolean(item.isArchived),
          previewImageUrl: isPersistableImageUrl(item.previewImageUrl) ? item.previewImageUrl : undefined,
          previewAttachmentId: item.previewAttachmentId,
          isTitleManual: Boolean(item.isTitleManual),
        }
      }),
  )
}

function normalizeMessages(items: ChatMessage[]): ChatMessage[] {
  const safeItems = Array.isArray(items) ? items : []
  return safeItems
    .filter((item) => item && typeof item.id === 'string')
    .map((item) => ({
      id: item.id,
      role: item.role === 'assistant' ? 'assistant' : 'user',
      content:
        item.status === 'streaming' && !item.content
          ? '上次响应在刷新前中断'
          : String(item.content || ''),
      imageUrl: isPersistableImageUrl(item.imageUrl) ? item.imageUrl : undefined,
      sourceFileName: item.sourceFileName,
      attachments: normalizeAttachments(item.attachments || []),
      createdAt: item.createdAt || new Date().toISOString(),
      status: item.status === 'streaming' ? 'error' : item.status || 'done',
      steps: item.steps?.map((step) => ({
        key: step.key,
        label: step.label,
        message: step.message,
        status: step.status === 'active' ? 'error' : step.status,
        data: step.data,
      })),
      currentStatus: item.currentStatus,
    }))
}

function normalizeAttachments(items: ChatAttachment[]): ChatAttachment[] | undefined {
  const attachments = (Array.isArray(items) ? items : [])
    .filter((item) => item && item.kind === 'image' && typeof item.url === 'string')
    .map((item) => ({
      id: String(item.id || ''),
      kind: 'image' as const,
      name: String(item.name || '现场图片'),
      url: item.url,
      thumbnailUrl: item.thumbnailUrl || item.url,
      size: Number(item.size || 0),
      mimeType: String(item.mimeType || 'image/*'),
      createdAt: item.createdAt || new Date().toISOString(),
      messageClientId: item.messageClientId || null,
    }))
    .filter((item) => item.id && isPersistableImageUrl(item.url))
  return attachments.length ? attachments : undefined
}

function normalizeProfile(profile?: Partial<UserProfile>) {
  const nickname = profile?.nickname?.trim() || DEFAULT_PROFILE.nickname
  return {
    nickname,
    avatarText: (profile?.avatarText?.trim() || nickname.slice(0, 1) || '安').slice(0, 2),
  }
}

function normalizeSettings(next?: Partial<UserSettings>) {
  const temperature = Number(next?.temperature ?? DEFAULT_SETTINGS.temperature)
  return {
    useStream: next?.useStream ?? DEFAULT_SETTINGS.useStream,
    autoExpandSteps: next?.autoExpandSteps ?? DEFAULT_SETTINGS.autoExpandSteps,
    temperature: Number.isFinite(temperature) ? Math.min(1, Math.max(0, temperature)) : 0.2,
  }
}

function sanitizeConversationForStorage(conversation: Conversation, includeImages: boolean): Conversation {
  const shouldKeepImageUrl = (value?: string) =>
    includeImages && isPersistableImageUrl(value) && !isDataUrl(value)
  return {
    ...conversation,
    previewImageUrl: shouldKeepImageUrl(conversation.previewImageUrl)
      ? conversation.previewImageUrl
      : undefined,
    messages: conversation.messages.map((message) => ({
      ...message,
      imageUrl: shouldKeepImageUrl(message.imageUrl) ? message.imageUrl : undefined,
      attachments: normalizeAttachments(message.attachments || []),
    })),
  }
}

function mergeConversations(localItems: Conversation[], remoteItems: Conversation[]) {
  const merged = new Map<string, Conversation>()
  normalizeConversations(localItems).forEach((conversation) => {
    merged.set(conversation.id, conversation)
  })
  normalizeConversations(remoteItems).forEach((remoteConversation) => {
    const localConversation = merged.get(remoteConversation.id)
    if (
      !localConversation ||
      toTimestamp(remoteConversation.updatedAt) >= toTimestamp(localConversation.updatedAt)
    ) {
      merged.set(remoteConversation.id, remoteConversation)
    }
  })
  return sortByUpdatedAt([...merged.values()])
}

function normalizeStats(next: Partial<ChatStats>, fallback: ChatStats): ChatStats {
  const recentSevenDays = normalizeCountItems(next.recentSevenDays, fallback.recentSevenDays).map((item) => ({
    date: item.date || item.label,
    count: item.count,
  }))
  const sceneCounts = normalizeCountItems(next.sceneCounts, fallback.sceneCounts).map((item) => ({
    label: item.label || item.date,
    count: item.count,
  }))
  const hazardCounts = normalizeCountItems(next.hazardCounts, fallback.hazardCounts).map((item) => {
    const label = item.label || '未分级'
    return {
      label,
      count: item.count,
      tone: item.tone || hazardTone(label),
    }
  })

  return {
    totalConversations: safeNumber(next.totalConversations, fallback.totalConversations),
    totalMessages: safeNumber(next.totalMessages, fallback.totalMessages),
    completedReports: safeNumber(next.completedReports, fallback.completedReports),
    archivedCount: safeNumber(next.archivedCount, fallback.archivedCount),
    completionRate: safeNumber(next.completionRate, fallback.completionRate),
    activeDays: safeNumber(next.activeDays, fallback.activeDays),
    latestActivity: String(next.latestActivity || fallback.latestActivity || ''),
    recentSevenDays,
    sceneCounts,
    hazardCounts,
    topHazardLabel: String(next.topHazardLabel || fallback.topHazardLabel || '未分级'),
  }
}

function normalizeCountItems<T extends { label?: string; date?: string; count?: number; tone?: ChatStats['hazardCounts'][number]['tone'] }>(
  next: T[] | undefined,
  fallback: T[],
) {
  const items = Array.isArray(next) ? next : fallback
  return items.map((item) => ({
    ...item,
    label: String(item.label || ''),
    date: String(item.date || ''),
    count: safeNumber(item.count, 0),
  }))
}

function safeNumber(value: unknown, fallback: number) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : fallback
}

function tokenizeSearchQuery(query: string) {
  return normalizeSearchText(query)
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean)
}

function normalizeSearchText(value: string) {
  return String(value || '')
    .toLowerCase()
    .normalize('NFKC')
    .replace(/[\u200b-\u200f\uFEFF]/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function expandSearchToken(token: string) {
  const synonyms: Record<string, string[]> = {
    违规: ['违规', '违章', '不合规', '不符合', '不满足', '超标', '超限', '隐患', '风险'],
    合规: ['合规', '符合', '满足', '正常', '允许', '达标'],
    隐患: ['隐患', '风险', '危险', '问题', '异常', '违规', '不符合'],
    风险: ['风险', '隐患', '危险', '异常', '问题'],
    超标: ['超标', '超限', '超过', '高于', '违规', '不符合'],
    瓦斯: ['瓦斯', '甲烷', 'ch4'],
    一氧化碳: ['一氧化碳', 'co'],
    二氧化碳: ['二氧化碳', 'co2'],
    氧气: ['氧气', '氧浓度', 'o2'],
    风速: ['风速', '最低风速', '最高风速', '通风速度'],
    风量: ['风量', '供风量', '需风量'],
    风筒: ['风筒', '风管', '局部通风'],
    局扇: ['局扇', '局部通风机', '通风机'],
    归档: ['归档', '已归档'],
  }
  return [token, ...(synonyms[token] || [])].map(normalizeSearchText).filter(Boolean)
}

function storageKeyForUser(user: RemoteUser | null) {
  return user ? `${STORAGE_KEY_PREFIX}:user:${user.id}` : GUEST_STORAGE_KEY
}

function readStoredState(key: string) {
  if (typeof window === 'undefined') return null
  const scopedState = window.localStorage.getItem(key)
  if (scopedState) return scopedState
  if (key === GUEST_STORAGE_KEY) {
    return window.localStorage.getItem(LEGACY_STORAGE_KEY)
  }
  return null
}

function clearGuestStorage() {
  if (typeof window === 'undefined') return
  window.localStorage.removeItem(GUEST_STORAGE_KEY)
  clearLegacyStorage()
}

function clearLegacyStorage() {
  if (typeof window === 'undefined') return
  window.localStorage.removeItem(LEGACY_STORAGE_KEY)
}

function isPersistableImageUrl(value?: string) {
  return Boolean(value && !value.startsWith('blob:'))
}

function isDataUrl(value?: string) {
  return Boolean(value?.startsWith('data:'))
}

function buildRecentSevenDays(conversations: Conversation[]) {
  const days = Array.from({ length: 7 }, (_, index) => {
    const date = new Date()
    date.setDate(date.getDate() - (6 - index))
    return date.toISOString().slice(0, 10)
  })
  return days.map((date) => ({
    date,
    count: conversations.filter((conversation) => conversation.updatedAt.slice(0, 10) === date).length,
  }))
}

function normalizeHazardLabel(value?: string) {
  const label = String(value || '').trim()
  if (!label) return '未分级'
  if (/高|重大|严重|danger|high/i.test(label)) return '高风险'
  if (/中|较大|warning|medium/i.test(label)) return '中风险'
  if (/低|一般|轻微|success|low/i.test(label)) return '低风险'
  return label
}

function hazardRank(label: string) {
  if (label === '高风险') return 1
  if (label === '中风险') return 2
  if (label === '低风险') return 3
  if (label === '未分级') return 9
  return 4
}

function hazardTone(label: string): 'danger' | 'warning' | 'success' | 'neutral' {
  if (label === '高风险') return 'danger'
  if (label === '中风险') return 'warning'
  if (label === '低风险') return 'success'
  return 'neutral'
}

function buildPrintableReport(conversation: Conversation) {
  const messages = conversation.messages
    .map((message) => {
      const content = message.content || message.currentStatus || ''
      const body =
        message.role === 'assistant'
          ? renderPrintableMarkdown(content)
          : renderPrintablePlainText(content)
      const imageUrl = getMessageImageUrl(message)
      const image = imageUrl
        ? `<img class="report-image" src="${escapeAttribute(imageUrl)}" alt="现场图片" />`
        : ''
      const imageName = getMessageImageName(message)
      const source = imageName
        ? `<div class="source-file">图片文件：${escapeHtml(imageName)}</div>`
        : ''
      return `<section class="message ${message.role}">
        <div class="message-meta">${message.role === 'assistant' ? '辨识报告' : '现场输入'} · ${formatDateTime(
          message.createdAt,
        )}</div>
        ${image}
        ${source}
        <div class="message-content markdown-body">${body}</div>
      </section>`
    })
    .join('')

  return `<!doctype html>
  <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <title>${escapeHtml(conversation.title)} - 通风辨识报告</title>
      <style>
        body { margin: 0; padding: 32px; color: #1f2d28; font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif; }
        h1 { margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }
        .subtitle { margin-bottom: 24px; color: #5f6d68; font-size: 13px; }
        .message { break-inside: avoid; margin: 0 0 18px; padding: 16px; border: 1px solid #dfe8e4; border-radius: 8px; }
        .message-meta { margin-bottom: 10px; color: #155e75; font-size: 13px; font-weight: 700; }
        .message.user { background: #f6fbf8; }
        .message.assistant { background: #ffffff; }
        .message-content { white-space: normal; font-size: 14px; line-height: 1.75; }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 { margin: 18px 0 8px; color: #18352e; line-height: 1.35; page-break-after: avoid; }
        .markdown-body h1 { font-size: 22px; }
        .markdown-body h2 { font-size: 18px; border-bottom: 1px solid #dfe8e4; padding-bottom: 5px; }
        .markdown-body h3 { font-size: 15px; }
        .markdown-body p { margin: 8px 0; }
        .markdown-body ul, .markdown-body ol { margin: 8px 0 10px 22px; padding: 0; }
        .markdown-body li { margin: 4px 0; }
        .markdown-body strong { color: #12342e; font-weight: 700; }
        .markdown-body blockquote { margin: 10px 0; padding: 8px 12px; border-left: 3px solid #3b8f82; background: #f4faf7; color: #40534e; }
        .markdown-body code { padding: 2px 5px; border-radius: 4px; background: #edf3f0; font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }
        .markdown-body pre { overflow: auto; margin: 10px 0; padding: 12px; border-radius: 8px; background: #edf3f0; white-space: pre-wrap; }
        .markdown-body pre code { padding: 0; background: transparent; }
        .markdown-body table { width: 100%; margin: 12px 0; border-collapse: collapse; break-inside: avoid; }
        .markdown-body th, .markdown-body td { padding: 8px 10px; border: 1px solid #d7e2dd; text-align: left; vertical-align: top; }
        .markdown-body th { background: #eff6f3; color: #173b34; font-weight: 700; }
        .markdown-body tr:nth-child(even) td { background: #fbfdfc; }
        .report-image { display: block; max-width: 520px; max-height: 360px; margin: 8px 0 10px; border-radius: 8px; object-fit: contain; }
        .source-file { margin-bottom: 8px; color: #697873; font-size: 12px; }
        @media print { body { padding: 18mm; } .message { border-color: #cfdad5; } }
      </style>
    </head>
    <body>
      <h1>${escapeHtml(conversation.title)}</h1>
      <div class="subtitle">创建时间：${formatDateTime(conversation.createdAt)} · 最后更新：${formatDateTime(
        conversation.updatedAt,
      )}</div>
      ${messages}
    </body>
  </html>`
}

function getMessageImageUrl(message: ChatMessage) {
  return message.attachments?.[0]?.url || message.imageUrl || ''
}

function getMessageImageName(message: ChatMessage) {
  return message.attachments?.[0]?.name || message.sourceFileName || ''
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function renderPrintableMarkdown(value: string) {
  return printableMarkdown.render(value || '')
}

function renderPrintablePlainText(value: string) {
  return escapeHtml(value || '').replace(/\n/g, '<br />')
}

function escapeAttribute(value: string) {
  return escapeHtml(value).replace(/`/g, '&#96;')
}

function formatDateTime(value: string) {
  if (!value) return '暂无'
  return new Date(value).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function fileToImageDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const rawDataUrl = String(reader.result || '')
      resizeDataUrl(rawDataUrl).then(resolve).catch(() => resolve(rawDataUrl))
    }
    reader.onerror = () => reject(reader.error || new Error('图片读取失败'))
    reader.readAsDataURL(file)
  })
}

function resizeDataUrl(dataUrl: string) {
  return new Promise<string>((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      const maxSize = 960
      const scale = Math.min(1, maxSize / Math.max(image.width, image.height))
      if (scale >= 1) {
        resolve(dataUrl)
        return
      }

      const canvas = document.createElement('canvas')
      canvas.width = Math.round(image.width * scale)
      canvas.height = Math.round(image.height * scale)
      const context = canvas.getContext('2d')
      if (!context) {
        reject(new Error('无法创建图片缩略图'))
        return
      }
      context.drawImage(image, 0, 0, canvas.width, canvas.height)
      resolve(canvas.toDataURL('image/jpeg', 0.82))
    }
    image.onerror = () => reject(new Error('图片缩放失败'))
    image.src = dataUrl
  })
}
