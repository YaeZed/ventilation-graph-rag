import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { sendImageMessage, sendTextMessage, streamMessage } from '@/api/chat'

export type MessageRole = 'user' | 'assistant'

export type ChatMessage = {
  id: string
  role: MessageRole
  content: string
  imageUrl?: string
  sourceFileName?: string
  createdAt: string
  status?: 'streaming' | 'done' | 'error'
}

export type Conversation = {
  id: string
  title: string
  messages: ChatMessage[]
  updatedAt: string
}

const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`

export const useChatStore = defineStore('chat', () => {
  const initialConversation: Conversation = {
    id: createId(),
    title: '通风隐患辨识',
    messages: [],
    updatedAt: new Date().toISOString(),
  }
  const conversations = ref<Conversation[]>([initialConversation])
  const activeId = ref(initialConversation.id)
  const isSending = ref(false)
  const error = ref('')

  const activeConversation = computed<Conversation>(() => ensureActiveConversation())

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
    isSending.value = true
    error.value = ''

    const imageUrl = image ? URL.createObjectURL(image) : undefined
    const userMessage = appendMessage(
      'user',
      question || '请判断图片中的通风安全隐患',
      imageUrl,
      'done',
      image?.name,
    )
    const assistantMessage = appendMessage('assistant', '', undefined, 'streaming')
    const conversation = ensureActiveConversation()
    let hasReceivedToken = false

    try {
      conversation.title = buildTitle(userMessage.content)
      if (useStream) {
        await streamMessage(userMessage.content, image, {
          onStatus(message) {
            if (!hasReceivedToken) {
              updateMessage(assistantMessage.id, {
                content: normalizeStatusMessage(message),
                status: 'streaming',
              })
            }
          },
          onToken(content) {
            const current = findMessage(assistantMessage.id)
            if (!current) return
            if (!hasReceivedToken) {
              current.content = ''
              hasReceivedToken = true
            }
            current.content += content
            current.status = 'streaming'
          },
          onError(message) {
            updateMessage(assistantMessage.id, { content: message, status: 'error' })
            error.value = message
          },
          onDone() {
            const current = findMessage(assistantMessage.id)
            if (!current || current.status === 'error') return
            if (!current.content.trim()) current.content = '未收到有效回答'
            current.status = 'done'
          },
        })
        const current = findMessage(assistantMessage.id)
        if (current?.status === 'streaming') {
          current.status = 'done'
        }
      } else {
        const answer = image
          ? await sendImageMessage(userMessage.content, image)
          : await sendTextMessage(userMessage.content)
        updateMessage(assistantMessage.id, { content: answer, status: 'done' })
      }
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : '请求失败'
      updateMessage(assistantMessage.id, { content: message, status: 'error' })
      error.value = message
    } finally {
      conversation.updatedAt = new Date().toISOString()
      isSending.value = false
    }
  }

  function appendMessage(
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
    const conversation = ensureActiveConversation()
    conversation.messages.push(message)
    conversation.updatedAt = new Date().toISOString()
    return conversation.messages[conversation.messages.length - 1]!
  }

  function findMessage(id: string): ChatMessage | undefined {
    return ensureActiveConversation().messages.find((message) => message.id === id)
  }

  function updateMessage(id: string, updates: Partial<Pick<ChatMessage, 'content' | 'status'>>) {
    const message = findMessage(id)
    if (!message) return
    Object.assign(message, updates)
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
