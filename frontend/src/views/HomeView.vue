<template>
  <section class="home-view">
    <header class="header-section">
      <div>
        <h1>矿风眼</h1>
        <p>{{ headerDescription }}</p>
      </div>
      <div class="header-actions">
        <button
          v-if="chat.activeConversation && !chat.isViewingTeamConversation"
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

              <div v-if="messageImages(message).length" class="message-image-grid">
                <figure
                  v-for="(image, imageIndex) in messageImages(message)"
                  :key="image.id"
                  class="image-frame"
                >
                  <button
                    class="message-image-button"
                    type="button"
                    :title="`查看 ${image.name}`"
                    @click="openImagePreview(messageImages(message), imageIndex)"
                  >
                    <img class="message-image" :src="image.url" :alt="image.name" />
                  </button>
                  <figcaption>{{ image.name }}</figcaption>
                </figure>
              </div>

              <SensorDataBadge v-if="message.sensorData" :sensor-data="message.sensorData" />

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
      <div
        v-if="!chat.isViewingTeamConversation && hasDraftContext"
        class="draft-context-panel"
      >
        <MultiImageBar
          v-if="activeDraft.images.length"
          :images="activeDraft.images"
          @add="openFilePicker"
          @remove="removeDraftImage"
        />
        <SensorDataBadge
          v-if="activeDraft.sensorData && !activeDraft.showSensorPanel"
          :sensor-data="activeDraft.sensorData"
          removable
          @remove="clearSensorData"
        />
        <SensorInputPanel
          v-if="activeDraft.showSensorPanel"
          :initial-data="activeDraft.sensorData"
          @apply="applySensorData"
          @close="activeDraft.showSensorPanel = false"
        />
      </div>
      <div class="input-capsule">
        <button
          class="upload-btn"
          type="button"
          title="上传图片"
          :disabled="chat.isViewingTeamConversation"
          @click="openFilePicker"
        >
          <span>+</span>
        </button>
        <button
          class="upload-btn sensor-toggle-btn"
          :class="{ active: activeDraft.sensorData || activeDraft.showSensorPanel }"
          type="button"
          title="添加传感器数据"
          :disabled="chat.isViewingTeamConversation"
          @click="toggleSensorPanel"
        >
          <span>数</span>
        </button>
        <div v-if="firstDraftImage" class="image-pill">
          <img :src="firstDraftImage.preview" alt="待上传图片" />
          <span>{{ activeDraft.images.length }}</span>
          <button type="button" title="移除全部图片" @click="clearDraftImages">×</button>
        </div>
        <input
          v-model="activeDraft.text"
          type="text"
          :disabled="chat.isViewingTeamConversation"
          :placeholder="inputPlaceholder"
        />
        <button class="send-btn" type="submit" :disabled="!canSend" title="发送">
          <span>➤</span>
        </button>
      </div>
      <input ref="fileInput" hidden multiple type="file" accept="image/*" @change="handleFileChange" />
    </form>

    <Teleport to="body">
      <div
        v-if="previewImage"
        class="image-preview-overlay"
        role="dialog"
        aria-modal="true"
        :aria-label="`查看图片：${previewImage.name}`"
        @click.self="closeImagePreview"
      >
        <figure class="image-preview-dialog">
          <button
            v-if="hasPreviewNavigation"
            class="image-preview-nav previous"
            type="button"
            title="上一张"
            aria-label="上一张图片"
            @click.stop="showPreviousPreview"
          >
            ‹
          </button>
          <button
            class="image-preview-close"
            type="button"
            title="关闭预览"
            @click="closeImagePreview"
          >
            ×
          </button>
          <img :src="previewImage.url" :alt="previewImage.name" />
          <figcaption>{{ previewCaption }}</figcaption>
          <button
            v-if="hasPreviewNavigation"
            class="image-preview-nav next"
            type="button"
            title="下一张"
            aria-label="下一张图片"
            @click.stop="showNextPreview"
          >
            ›
          </button>
        </figure>
      </div>
    </Teleport>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import EmptyState from '@/components/EmptyState.vue'
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import MultiImageBar from '@/components/MultiImageBar.vue'
import SensorDataBadge from '@/components/SensorDataBadge.vue'
import SensorInputPanel from '@/components/SensorInputPanel.vue'
import { useChatStore, type AgentStep, type ChatMessage } from '@/stores/chat'
import type { ChatMessageImage, DraftImage, SensorData } from '@/types/multimodal'

type InputDraft = {
  text: string
  images: DraftImage[]
  sensorData: SensorData | null
  showSensorPanel: boolean
}

const chat = useChatStore()
const route = useRoute()
const router = useRouter()
const drafts = reactive<Record<string, InputDraft>>({})
const fileInput = ref<HTMLInputElement | null>(null)
const messagesEl = ref<HTMLElement | null>(null)
const previewImages = ref<ChatMessageImage[]>([])
const previewIndex = ref(0)
const collapsedSteps = reactive<Record<string, boolean>>({})
const scrollPositions = new Map<string, number>()
const pendingScrollRestoreId = ref('')
let scrollRestoreTimer = 0

const draftKey = computed(() => chat.activeId || 'new')
const activeDraft = computed(() => getDraft(draftKey.value))
const firstDraftImage = computed(() => activeDraft.value.images[0] || null)
const previewImage = computed(() => previewImages.value[previewIndex.value] || null)
const hasPreviewNavigation = computed(() => previewImages.value.length > 1)
const previewCaption = computed(() => {
  const image = previewImage.value
  if (!image) return ''
  if (!hasPreviewNavigation.value) return image.name
  return `${previewIndex.value + 1} / ${previewImages.value.length} · ${image.name}`
})
const hasDraftContext = computed(
  () =>
    activeDraft.value.images.length > 0 ||
    Boolean(activeDraft.value.sensorData) ||
    activeDraft.value.showSensorPanel,
)
const canSend = computed(
  () =>
    !chat.isViewingTeamConversation &&
    (activeDraft.value.text.trim().length > 0 ||
      activeDraft.value.images.length > 0 ||
      Boolean(activeDraft.value.sensorData)) &&
    !chat.isSending,
)
const inputPlaceholder = computed(() =>
  chat.isViewingTeamConversation
    ? '团队对话为只读浏览，继续辨识请回到自己的对话或新建对话'
    : activeDraft.value.images.length || activeDraft.value.sensorData
    ? '补充现场描述，例如：检查掘进工作面整体通风状况'
    : '输入检查项，或上传图片后补充现场描述',
)
const headerDescription = computed(() =>
  chat.isViewingTeamConversation && chat.activeTeamConversation
    ? `${chat.activeTeamConversation.teamName || '团队'} · ${chat.activeTeamConversation.owner?.nickname || '团队成员'} 的共享辨识记录`
    : chat.activeConversation
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
    if (chat.selectTeamConversation(conversationId)) return
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
      images: [],
      sensorData: null,
      showSensorPanel: false,
    }
  }
  return drafts[conversationId]
}

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files || []).filter((file) => file.type.startsWith('image/'))
  if (!files.length) return
  const draft = activeDraft.value
  const availableSlots = Math.max(0, MAX_DRAFT_IMAGES - draft.images.length)
  files.slice(0, availableSlots).forEach((file) => {
    draft.images.push({
      id: createDraftImageId(),
      file,
      preview: URL.createObjectURL(file),
    })
  })
  target.value = ''
}

const removeDraftImage = (id: string) => {
  const draft = activeDraft.value
  const image = draft.images.find((item) => item.id === id)
  if (image) URL.revokeObjectURL(image.preview)
  draft.images = draft.images.filter((item) => item.id !== id)
  if (fileInput.value) fileInput.value.value = ''
}

const clearDraftImages = () => {
  const draft = activeDraft.value
  draft.images.forEach((image) => URL.revokeObjectURL(image.preview))
  draft.images = []
  if (fileInput.value) fileInput.value.value = ''
}

const toggleSensorPanel = () => {
  activeDraft.value.showSensorPanel = !activeDraft.value.showSensorPanel
}

const applySensorData = (sensorData: SensorData) => {
  activeDraft.value.sensorData = sensorData
  activeDraft.value.showSensorPanel = false
}

const clearSensorData = () => {
  activeDraft.value.sensorData = null
}

const applyPrompt = (prompt: string) => {
  activeDraft.value.text = prompt
}

const submit = async () => {
  if (!canSend.value) return
  const draft = activeDraft.value
  const question = draft.text.trim()
  const images = draft.images.map((image) => image.file)
  const sensorData = draft.sensorData
  draft.text = ''
  draft.sensorData = null
  draft.showSensorPanel = false
  clearDraftImages()
  await chat.submit(question, images, chat.settings.useStream, sensorData)
}

const exportCurrent = () => {
  if (!chat.activeId) return
  chat.exportConversationAsPDF(chat.activeId)
}

const openImagePreview = (images: ChatMessageImage[], index: number) => {
  previewImages.value = images
  previewIndex.value = Math.min(Math.max(index, 0), images.length - 1)
}

const closeImagePreview = () => {
  previewImages.value = []
  previewIndex.value = 0
}

const showPreviousPreview = () => {
  if (!hasPreviewNavigation.value) return
  previewIndex.value =
    (previewIndex.value - 1 + previewImages.value.length) % previewImages.value.length
}

const showNextPreview = () => {
  if (!hasPreviewNavigation.value) return
  previewIndex.value = (previewIndex.value + 1) % previewImages.value.length
}

const handlePreviewKeydown = (event: KeyboardEvent) => {
  if (!previewImage.value) return
  if (event.key === 'Escape') {
    event.preventDefault()
    closeImagePreview()
  }
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    showPreviousPreview()
  }
  if (event.key === 'ArrowRight') {
    event.preventDefault()
    showNextPreview()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handlePreviewKeydown)
})

onBeforeUnmount(() => {
  rememberActiveScroll()
  if (scrollRestoreTimer) window.clearTimeout(scrollRestoreTimer)
  window.removeEventListener('keydown', handlePreviewKeydown)
  Object.values(drafts).forEach((draft) => {
    draft.images.forEach((image) => URL.revokeObjectURL(image.preview))
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

const messageImages = (message: ChatMessage): ChatMessageImage[] => {
  const attachmentImages =
    message.attachments?.map((attachment) => ({
      id: attachment.id,
      name: attachment.name,
      url: attachment.url,
      size: attachment.size,
      mimeType: attachment.mimeType,
      createdAt: attachment.createdAt,
    })) || []
  const legacyImage =
    !message.images?.length && !attachmentImages.length && message.imageUrl
      ? [
          {
            id: message.id,
            name: message.sourceFileName || '现场图片',
            url: message.imageUrl,
            size: 0,
            mimeType: 'image/*',
          },
        ]
      : []
  const seen = new Set<string>()
  return [...attachmentImages, ...(message.images || []), ...legacyImage].filter((image) => {
    if (!image.url || seen.has(image.url)) return false
    seen.add(image.url)
    return true
  })
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

const MAX_DRAFT_IMAGES = 6

const createDraftImageId = () => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}
</script>
