<template>
  <section class="home-view">
    <header class="header-section">
      <div>
        <h1>煤矿通风隐患智能辨识</h1>
        <p>上传现场图像并补充描述，系统会先分析图片和描述，再生成规程依据明确的辨识报告</p>
      </div>
      <label class="stream-switch">
        <input v-model="useStream" type="checkbox" />
        <span>{{ useStream ? 'SSE 流式' : '普通响应' }}</span>
      </label>
    </header>

    <div ref="messagesEl" class="result-area custom-scrollbar">
      <div v-if="chat.activeConversation.messages.length === 0" class="empty-state">
        <div class="empty-icon">+</div>
        <p>上传局部通风机、风筒、风门或巷道风速场景图片</p>
        <div class="sample-prompts">
          <button type="button" @click="activeDraft.text = '检查图片中的局部通风机安装是否合规'">
            局部通风机安装
          </button>
          <button type="button" @click="activeDraft.text = '判断掘进工作面风筒与通风状态是否存在隐患'">
            掘进工作面通风
          </button>
          <button type="button" @click="activeDraft.text = '这处风门设施是否满足规程要求'">
            风门设施核查
          </button>
        </div>
      </div>

      <article
        v-for="message in chat.activeConversation.messages"
        :key="message.id"
        class="message-card"
        :class="[message.role, message.status]"
      >
        <div class="card-header">
          <span class="tag" :class="message.role === 'assistant' ? tagClass(message) : 'neutral'">
            {{ message.role === 'assistant' ? '辨识报告' : '现场输入' }}
          </span>
          <span class="timestamp">{{ formatTime(message.createdAt) }}</span>
        </div>

        <figure v-if="message.imageUrl" class="image-frame">
          <img class="message-image" :src="message.imageUrl" alt="上传的现场图片" />
          <figcaption v-if="message.sourceFileName">{{ message.sourceFileName }}</figcaption>
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
        <div v-else class="ai-text">{{ message.content || message.currentStatus || '正在生成...' }}</div>
      </article>
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
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import { useChatStore, type AgentStep, type ChatMessage } from '@/stores/chat'

type InputDraft = {
  text: string
  file: File | null
  preview: string
}

const chat = useChatStore()
const drafts = reactive<Record<string, InputDraft>>({})
const useStream = ref(true)
const fileInput = ref<HTMLInputElement | null>(null)
const messagesEl = ref<HTMLElement | null>(null)
const collapsedSteps = reactive<Record<string, boolean>>({})

const activeDraft = computed(() => getDraft(chat.activeId))
const canSend = computed(
  () => (activeDraft.value.text.trim().length > 0 || activeDraft.value.file) && !chat.isSending,
)
const inputPlaceholder = computed(() =>
  activeDraft.value.file
    ? '补充现场描述或检查重点，例如：局部通风机距回风口约 8 米'
    : '输入检查项，或上传图片后补充现场描述',
)

watch(
  () => chat.activeConversation.messages.length,
  async () => {
    await nextTick()
    messagesEl.value?.scrollTo({ top: messagesEl.value.scrollHeight, behavior: 'smooth' })
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

const submit = async () => {
  if (!canSend.value) return
  const draft = activeDraft.value
  const question = draft.text.trim()
  const image = draft.file
  draft.text = ''
  clearImage()
  await chat.submit(question, image, useStream.value)
}

onBeforeUnmount(() => {
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
  if (Array.isArray(data.concepts) && data.concepts.length) return `概念：${data.concepts.join('、')}`
  if (typeof data.doc_count === 'number') return `命中条文：${data.doc_count}`
  if (typeof data.scene_name === 'string') return `场景：${data.scene_name}`
  return ''
}

const hasReportContent = (message: ChatMessage) => {
  if (!message.steps?.length) return true
  return message.content.trim() !== message.currentStatus?.trim()
}
</script>
