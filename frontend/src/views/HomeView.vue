<template>
  <section class="home-view">
    <header class="header-section">
      <div>
        <h1>煤矿通风隐患智能辨识</h1>
        <p>上传现场图像或输入检查项，生成规程依据明确的核合报告</p>
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
          <button type="button" @click="inputText = '检查图片中的局部通风机安装是否合规'">
            局部通风机安装
          </button>
          <button type="button" @click="inputText = '判断掘进工作面风筒与通风状态是否存在隐患'">
            掘进工作面通风
          </button>
          <button type="button" @click="inputText = '这处风门设施是否满足规程要求'">
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
        <MarkdownRenderer
          v-if="message.role === 'assistant' && message.content"
          :content="message.content"
        />
        <div v-else class="ai-text">{{ message.content || '正在生成...' }}</div>
      </article>
    </div>

    <form class="bottom-bar-container" @submit.prevent="submit">
      <div class="input-capsule">
        <button class="upload-btn" type="button" title="上传图片" @click="openFilePicker">
          <span>+</span>
        </button>
        <div v-if="selectedPreview" class="image-pill">
          <img :src="selectedPreview" alt="待上传图片" />
          <button type="button" title="移除图片" @click="clearImage">×</button>
        </div>
        <input
          v-model="inputText"
          type="text"
          placeholder="输入检查项，例如：这张图中的局部通风机距回风口是否合规"
        />
        <button class="send-btn" type="submit" :disabled="!canSend" title="发送">
          <span>➤</span>
        </button>
      </div>
      <input ref="fileInput" hidden type="file" accept="image/*" @change="handleFileChange" />
    </form>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import { useChatStore, type ChatMessage } from '@/stores/chat'

const chat = useChatStore()
const inputText = ref('')
const selectedFile = ref<File | null>(null)
const selectedPreview = ref('')
const useStream = ref(true)
const fileInput = ref<HTMLInputElement | null>(null)
const messagesEl = ref<HTMLElement | null>(null)

const canSend = computed(() => (inputText.value.trim().length > 0 || selectedFile.value) && !chat.isSending)

watch(
  () => chat.activeConversation.messages.length,
  async () => {
    await nextTick()
    messagesEl.value?.scrollTo({ top: messagesEl.value.scrollHeight, behavior: 'smooth' })
  },
)

const openFilePicker = () => fileInput.value?.click()

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  if (selectedPreview.value) URL.revokeObjectURL(selectedPreview.value)
  selectedFile.value = file
  selectedPreview.value = URL.createObjectURL(file)
}

const clearImage = () => {
  if (selectedPreview.value) URL.revokeObjectURL(selectedPreview.value)
  selectedFile.value = null
  selectedPreview.value = ''
  if (fileInput.value) fileInput.value.value = ''
}

const submit = async () => {
  if (!canSend.value) return
  const question = inputText.value.trim()
  const image = selectedFile.value
  inputText.value = ''
  clearImage()
  await chat.submit(question, image, useStream.value)
}

const tagClass = (message: ChatMessage) => {
  if (message.status === 'error') return 'danger'
  if (message.status === 'streaming') return 'processing'
  return 'success'
}

const formatTime = (value: string) =>
  new Date(value).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
</script>
