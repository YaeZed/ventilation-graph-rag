<template>
  <section class="home-view">
    <header class="header-section">
      <div>
        <h1>矿风眼</h1>
        <p>{{ headerDescription }}</p>
      </div>
      <div class="header-actions">
        <button
          v-if="chat.activeConversation"
          class="report-action"
          type="button"
          title="导出当前对话 PDF"
          @click="exportCurrent"
        >
          <span class="menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 4v10" />
              <path d="m8 10 4 4 4-4" />
              <path d="M5 19h14" />
            </svg>
          </span>
          <span>导出</span>
        </button>
        <label class="stream-switch">
          <input
            :checked="chat.settings.useStream"
            type="checkbox"
            @change="
              chat.updateSettings({ useStream: ($event.target as HTMLInputElement).checked })
            "
          />
          <span>{{ chat.settings.useStream ? 'SSE 流式' : '普通响应' }}</span>
        </label>
      </div>
    </header>

    <div ref="messagesEl" class="result-area custom-scrollbar" @scroll.passive="rememberActiveScroll">
      <Transition name="fade" mode="out-in" @after-enter="restoreActiveScroll">
        <div :key="chat.activeId || 'empty-conversation'" class="conversation-pane">
          <EmptyState
            v-if="!chat.activeConversation || chat.activeConversation.messages.length === 0"
            @pick="applyPrompt"
          />

          <template v-if="chat.activeConversation">
            <article
              v-for="message in chat.activeConversation.messages"
              :key="message.id"
              class="message-card"
              :class="[message.role, message.status]"
            >
              <div class="card-header">
                <span
                  class="tag"
                  :class="message.role === 'assistant' ? tagClass(message) : 'neutral'"
                >
                  {{ message.role === 'assistant' ? '辨识报告' : '现场输入' }}
                </span>
                <span class="timestamp">{{ formatTime(message.createdAt) }}</span>
              </div>

              <figure v-if="messageImageUrl(message)" class="image-frame">
                <img class="message-image" :src="messageImageUrl(message)" alt="上传的现场图片" />
                <figcaption v-if="imageCaption(message)">{{ imageCaption(message) }}</figcaption>
              </figure>

              <div v-if="message.role === 'assistant' && message.steps?.length" class="agent-steps">
                <button class="steps-summary" type="button" @click="toggleSteps(message.id)">
                  <span>{{ stepSummary(message) }}</span>
                  <span class="chevron">{{ collapsedSteps[message.id] ? '展开' : '收起' }}</span>
                </button>
                <ol v-if="!collapsedSteps[message.id]" class="step-list">
                  <li
                    v-for="step in message.steps"
                    :key="step.key"
                    class="step-item"
                    :class="step.status"
                  >
                    <span class="step-dot"></span>
                    <div>
                      <strong>{{ step.label }}</strong>
                      <p>{{ step.message }}</p>
                      <small v-if="stepMeta(step)">{{ stepMeta(step) }}</small>
                    </div>
                  </li>
                </ol>
              </div>

              <MarkdownRenderer
                v-if="message.role === 'assistant' && message.content && hasReportContent(message)"
                :content="message.content"
              />
              <div v-else class="ai-text">
                {{ message.content || message.currentStatus || '正在生成...' }}
              </div>
            </article>
          </template>
        </div>
      </Transition>
    </div>

    <form class="bottom-bar-container" @submit.prevent="submit">
      <div class="input-capsule">
        <button class="upload-btn" type="button" title="上传图片" @click="openFilePicker">
          <span>+</span>
        </button>
        <div v-if="activeDraft.preview" class="image-pill">
          <img :src="activeDraft.preview" alt="待上传图片" />
          <button type="button" title="移除图片" @click="clearImage">×</button>
        </div>
        <input v-model="activeDraft.text" type="text" :placeholder="inputPlaceholder" />
        <button class="send-btn" type="submit" :disabled="!canSend" title="发送">
          <span>➤</span>
        </button>
      </div>
      <input ref="fileInput" hidden type="file" accept="image/*" @change="handleFileChange" />
    </form>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import EmptyState from '@/components/EmptyState.vue'
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import { useChatStore, type AgentStep, type ChatMessage } from '@/stores/chat'

type InputDraft = {
  text: string
  file: File | null
  preview: string
}

const chat = useChatStore()
const route = useRoute()
const router = useRouter()
const drafts = reactive<Record<string, InputDraft>>({})
const fileInput = ref<HTMLInputElement | null>(null)
const messagesEl = ref<HTMLElement | null>(null)
const collapsedSteps = reactive<Record<string, boolean>>({})
const scrollPositions = new Map<string, number>()
const pendingScrollRestoreId = ref('')
let scrollRestoreTimer = 0

const draftKey = computed(() => chat.activeId || 'new')
const activeDraft = computed(() => getDraft(draftKey.value))
const canSend = computed(
  () => (activeDraft.value.text.trim().length > 0 || activeDraft.value.file) && !chat.isSending,
)
const inputPlaceholder = computed(() =>
  activeDraft.value.file
    ? '补充现场描述或检查重点，例如：局部通风机距回风口约 8 米'
    : '输入检查项，或上传图片后补充现场描述',
)
const headerDescription = computed(() =>
  chat.activeConversation
    ? '上传现场图像并补充描述，系统会生成规程依据明确的辨识报告'
    : '先新建或选择一个对话，所有辨识记录会自动保存到本机',
)

watch(
  () => route.params.conversationId,
  (conversationId) => {
    if (typeof conversationId !== 'string') return
    if (chat.hasConversation(conversationId)) {
      chat.selectConversation(conversationId)
      return
    }
    router.replace('/chat')
  },
  { immediate: true },
)

watch(
  () => chat.activeId,
  (activeId, previousId) => {
    if (previousId) rememberConversationScroll(previousId)
    if (activeId) queueScrollRestore(activeId)
    if (!activeId) return
    if (route.name === 'chat' || route.params.conversationId !== activeId) {
      router.replace(`/chat/${activeId}`)
    }
  },
)

watch(
  () => ({
    conversationId: chat.activeId,
    messageCount: chat.activeConversation?.messages.length || 0,
  }),
  async (current, previous) => {
    if (!current.conversationId) return
    if (current.conversationId !== previous?.conversationId) return
    if (current.messageCount <= (previous?.messageCount || 0)) return
    await nextTick()
    const container = messagesEl.value
    if (!container) return
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
    scrollPositions.set(current.conversationId, container.scrollHeight)
  },
)

watch(
  () => chat.settings.autoExpandSteps,
  (autoExpand) => {
    if (autoExpand) {
      Object.keys(collapsedSteps).forEach((key) => {
        collapsedSteps[key] = false
      })
    }
  },
)

const openFilePicker = () => fileInput.value?.click()

const getDraft = (conversationId: string) => {
  if (!drafts[conversationId]) {
    drafts[conversationId] = {
      text: '',
      file: null,
      preview: '',
    }
  }
  return drafts[conversationId]
}

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  const draft = activeDraft.value
  if (draft.preview) URL.revokeObjectURL(draft.preview)
  draft.file = file
  draft.preview = URL.createObjectURL(file)
  target.value = ''
}

const clearImage = () => {
  const draft = activeDraft.value
  if (draft.preview) URL.revokeObjectURL(draft.preview)
  draft.file = null
  draft.preview = ''
  if (fileInput.value) fileInput.value.value = ''
}

const applyPrompt = (prompt: string) => {
  activeDraft.value.text = prompt
}

const submit = async () => {
  if (!canSend.value) return
  const draft = activeDraft.value
  const question = draft.text.trim()
  const image = draft.file
  draft.text = ''
  clearImage()
  await chat.submit(question, image, chat.settings.useStream)
}

const exportCurrent = () => {
  if (!chat.activeId) return
  chat.exportConversationAsPDF(chat.activeId)
}

onBeforeUnmount(() => {
  rememberActiveScroll()
  if (scrollRestoreTimer) window.clearTimeout(scrollRestoreTimer)
  Object.values(drafts).forEach((draft) => {
    if (draft.preview) URL.revokeObjectURL(draft.preview)
  })
})

const tagClass = (message: ChatMessage) => {
  if (message.status === 'error') return 'danger'
  if (message.status === 'streaming') return 'processing'
  return 'success'
}

const formatTime = (value: string) =>
  new Date(value).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

const toggleSteps = (messageId: string) => {
  collapsedSteps[messageId] = !collapsedSteps[messageId]
}

const stepSummary = (message: ChatMessage) => {
  const active = message.steps?.find((step) => step.status === 'active')
  if (active) return active.message
  const last = message.steps?.[message.steps.length - 1]
  return last?.message || message.currentStatus || '正在处理'
}

const stepMeta = (step: AgentStep) => {
  const data = step.data || {}
  if (Array.isArray(data.concepts) && data.concepts.length)
    return `概念：${data.concepts.join('、')}`
  if (typeof data.doc_count === 'number') return `命中条文：${data.doc_count}`
  if (typeof data.scene_name === 'string') return `场景：${data.scene_name}`
  return ''
}

const messageImageUrl = (message: ChatMessage) =>
  message.attachments?.[0]?.url || message.imageUrl || ''

const imageCaption = (message: ChatMessage) => {
  const attachment = message.attachments?.[0]
  return attachment?.name || message.sourceFileName || ''
}

const hasReportContent = (message: ChatMessage) => {
  if (!message.steps?.length) return true
  return message.content.trim() !== message.currentStatus?.trim()
}

const rememberActiveScroll = () => {
  rememberConversationScroll(chat.activeId)
}

const rememberConversationScroll = (conversationId: string) => {
  const container = messagesEl.value
  if (!conversationId || !container) return
  scrollPositions.set(conversationId, container.scrollTop)
}

const queueScrollRestore = (conversationId: string) => {
  pendingScrollRestoreId.value = conversationId
  if (scrollRestoreTimer) window.clearTimeout(scrollRestoreTimer)
  scrollRestoreTimer = window.setTimeout(() => {
    void nextTick(() => {
      if (pendingScrollRestoreId.value === conversationId) {
        restoreConversationScroll(conversationId)
        pendingScrollRestoreId.value = ''
      }
    })
  }, 230)
}

const restoreActiveScroll = () => {
  const conversationId = pendingScrollRestoreId.value || chat.activeId
  if (!conversationId) return
  if (scrollRestoreTimer) {
    window.clearTimeout(scrollRestoreTimer)
    scrollRestoreTimer = 0
  }
  void nextTick(() => {
    restoreConversationScroll(conversationId)
    if (pendingScrollRestoreId.value === conversationId) {
      pendingScrollRestoreId.value = ''
    }
  })
}

const restoreConversationScroll = (conversationId: string) => {
  const container = messagesEl.value
  if (!conversationId || !container) return
  const savedTop = scrollPositions.get(conversationId)
  const targetTop = savedTop ?? container.scrollHeight
  container.scrollTo({ top: targetTop, behavior: 'auto' })
}
</script>
