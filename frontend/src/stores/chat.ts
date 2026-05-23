import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import {
  sendImageMessage,
  sendTextMessage,
  streamMessage,
  type StreamStepEvent,
} from '@/api/chat'

export type MessageRole = 'user' | 'assistant'
export type AgentStepStatus = 'pending' | 'active' | 'done' | 'error'

export type AgentStep = {
  key: string
  label: string
  message: string
  status: AgentStepStatus
  data?: Record<string, unknown>
}

export type ChatMessage = {
  id: string
  role: MessageRole
  content: string
  imageUrl?: string
  sourceFileName?: string
  createdAt: string
  status?: 'streaming' | 'done' | 'error'
  steps?: AgentStep[]
  currentStatus?: string
}

export type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
  updatedAt: string
}

const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`

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

export const useChatStore = defineStore('chat', () => {
  const initialConversation: Conversation = {
    id: createId(),
    title: '通风隐患辨识',
    messages: [],
    updatedAt: new Date().toISOString(),
  }
  const conversations = ref<Conversation[]>([initialConversation])
  const activeId = ref(initialConversation.id)
  const sendingByConversation = ref<Record<string, boolean>>({})
  const error = ref('')

  const activeConversation = computed<Conversation>(() => ensureActiveConversation())
  const isSending = computed(() => Boolean(sendingByConversation.value[activeId.value]))

  function ensureActiveConversation(): Conversation {
    const existing = conversations.value.find((item) => item.id === activeId.value)
    if (existing) return existing

    if (conversations.value.length === 0) {
      const conversation: Conversation = {
        id: createId(),
        title: '新的辨识会话',
        messages: [],
        updatedAt: new Date().toISOString(),
      }
      conversations.value.unshift(conversation)
      activeId.value = conversation.id
      return conversation
    }

    const fallback = conversations.value[0]!
    activeId.value = fallback.id
    return fallback
  }

  function newConversation() {
    const conversation: Conversation = {
      id: createId(),
      title: '新的辨识会话',
      messages: [],
      updatedAt: new Date().toISOString(),
    }
    conversations.value.unshift(conversation)
    activeId.value = conversation.id
  }

  function selectConversation(id: string) {
    activeId.value = id
  }

  async function submit(question: string, image: File | null, useStream: boolean) {
    if (!question.trim() && !image) return
    const conversation = ensureActiveConversation()
    const conversationId = conversation.id
    if (sendingByConversation.value[conversationId]) return

    setConversationSending(conversationId, true)
    error.value = ''

    const imageUrl = image ? URL.createObjectURL(image) : undefined
    const userMessage = appendMessage(
      conversationId,
      'user',
      question || '请判断图片中的通风安全隐患',
      imageUrl,
      'done',
      image?.name,
    )
    const assistantMessage = appendMessage(conversationId, 'assistant', '', undefined, 'streaming')
    let hasReceivedToken = false

    try {
      conversation.title = buildTitle(userMessage.content)
      if (useStream) {
        await streamMessage(userMessage.content, image, {
          onStatus(message) {
            const current = findMessage(conversationId, assistantMessage.id)
            if (!current) return
            current.currentStatus = normalizeStatusMessage(message)
            if (!hasReceivedToken && !current.steps?.length) {
              current.content = current.currentStatus
            }
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
          },
        })
        const current = findMessage(conversationId, assistantMessage.id)
        if (current?.status === 'streaming') {
          markActiveStepsDone(current)
          current.status = 'done'
        }
      } else {
        const answer = image
          ? await sendImageMessage(userMessage.content, image)
          : await sendTextMessage(userMessage.content)
        updateMessage(conversationId, assistantMessage.id, { content: answer, status: 'done' })
      }
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : '请求失败'
      updateMessage(conversationId, assistantMessage.id, { content: message, status: 'error' })
      error.value = message
    } finally {
      conversation.updatedAt = new Date().toISOString()
      setConversationSending(conversationId, false)
    }
  }

  function appendMessage(
    conversationId: string,
    role: MessageRole,
    content: string,
    imageUrl?: string,
    status: ChatMessage['status'] = 'done',
    sourceFileName?: string,
  ): ChatMessage {
    const message: ChatMessage = {
      id: createId(),
      role,
      content,
      imageUrl,
      sourceFileName,
      createdAt: new Date().toISOString(),
      status,
    }
    const conversation = findConversation(conversationId) || ensureActiveConversation()
    conversation.messages.push(message)
    conversation.updatedAt = new Date().toISOString()
    return conversation.messages[conversation.messages.length - 1]!
  }

  function findConversation(id: string): Conversation | undefined {
    return conversations.value.find((conversation) => conversation.id === id)
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

  function buildTitle(content: string) {
    return content.length > 16 ? `${content.slice(0, 16)}...` : content
  }

  function normalizeStatusMessage(message: string) {
    if (!message || message === 'started') {
      return '已连接，正在检索规程知识库...'
    }
    return message
  }

  return {
    conversations,
    activeId,
    activeConversation,
    isSending,
    error,
    newConversation,
    selectConversation,
    submit,
  }
})
