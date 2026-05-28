<template>
  <article
    ref="itemEl"
    class="conversation-item"
    :class="{ active: conversation.id === activeId, sending: isSending }"
  >
    <button
      v-if="!isEditing"
      class="conversation-select"
      type="button"
      :title="conversation.title"
      @click="$emit('select', conversation.id)"
      @dblclick.stop="startEditing"
    >
      <span class="conversation-title">{{ conversation.title }}</span>
      <span class="conversation-meta">
        <span>{{ formatRelativeTime(conversation.updatedAt) }}</span>
        <span v-if="conversation.hazardLevel">{{ conversation.hazardLevel }}</span>
        <span v-else-if="conversation.sceneType">{{ conversation.sceneType }}</span>
      </span>
    </button>

    <form v-else class="rename-form" @submit.prevent="save">
      <input
        ref="inputEl"
        v-model="draftTitle"
        maxlength="40"
        @blur="save"
        @keydown.esc.prevent="cancel"
      />
    </form>

    <div class="conversation-menu-wrap">
      <button
        class="conversation-more"
        type="button"
        title="更多"
        :aria-expanded="isMenuOpen"
        @click.stop="toggleMenu"
      >
        ⋮
      </button>

      <div v-if="isMenuOpen" ref="menuEl" class="conversation-menu" @click.stop>
        <div
          v-if="canAssignTeam"
          class="conversation-menu-submenu"
          @mouseenter="openTeamMenu"
        >
          <button ref="teamMenuTriggerEl" type="button" @click="openTeamMenu">
            <span class="menu-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                <path d="M16 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                <path d="M3.5 20a4.5 4.5 0 0 1 9 0" />
                <path d="M12.5 20a4.5 4.5 0 0 1 8 0" />
              </svg>
            </span>
            <span>归属团队</span>
          </button>
          <Teleport to="body">
            <div
              v-if="isTeamMenuOpen"
              ref="teamSubmenuEl"
              class="team-submenu"
              :style="teamSubmenuStyle"
              @mouseenter="openTeamMenu"
              @mouseleave="scheduleCloseTeamMenu"
              @click.stop
            >
            <label class="team-option">
              <input v-model="selectedTeamId" type="radio" value="" />
              <span>个人空间</span>
            </label>
            <label v-for="team in teams" :key="team.id" class="team-option">
              <input v-model="selectedTeamId" type="radio" :value="team.id" />
              <span>{{ team.name }}</span>
            </label>
            <p v-if="!teams.length" class="team-option-empty">暂无可选团队</p>
            <div class="team-option-actions">
              <button type="button" class="team-option-cancel" @click="cancelTeamAssign">取消</button>
              <button type="button" class="team-option-confirm" @click="confirmTeamAssign">确认</button>
            </div>
            </div>
          </Teleport>
        </div>
        <button type="button" @click="shareConversation">
          <span class="menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <path d="M8.6 10.7 15.4 6.3M8.6 13.3l6.8 4.4" />
            </svg>
          </span>
          <span>分享对话内容</span>
        </button>
        <button type="button" :disabled="isSending" @click="archive">
          <span class="menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M4 7.5h16M6 7.5l1.2 11h9.6l1.2-11M9.5 11.2h5M10.5 4.5h3l1.2 3h-5.4l1.2-3Z" />
            </svg>
          </span>
          <span>归档对话</span>
        </button>
        <button type="button" :disabled="isSending" @click="startEditingFromMenu">
          <span class="menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M4 20h4.5L19 9.5 14.5 5 4 15.5V20Z" />
              <path d="m13.5 6 4.5 4.5" />
            </svg>
          </span>
          <span>重命名</span>
        </button>
        <button type="button" @click="exportPdf">
          <span class="menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 4v10" />
              <path d="m8 10 4 4 4-4" />
              <path d="M5 19h14" />
            </svg>
          </span>
          <span>导出 PDF</span>
        </button>
        <button class="danger-action" type="button" :disabled="isSending" @click="deleteItem">
          <span class="menu-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M5 7h14" />
              <path d="M9 7V5h6v2" />
              <path d="M8 10v8M12 10v8M16 10v8" />
              <path d="M7 7l1 13h8l1-13" />
            </svg>
          </span>
          <span>删除</span>
        </button>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import type { Conversation, Team } from '@/stores/chat'

const props = defineProps<{
  conversation: Conversation
  activeId: string
  isSending: boolean
  teams: Team[]
  canAssignTeam?: boolean
}>()

const emit = defineEmits<{
  select: [id: string]
  rename: [id: string, title: string]
  archive: [id: string]
  delete: [id: string]
  export: [id: string]
  assignTeam: [id: string, teamId: string]
}>()

const isEditing = ref(false)
const isMenuOpen = ref(false)
const isTeamMenuOpen = ref(false)
const selectedTeamId = ref(props.conversation.teamId || '')
const draftTitle = ref(props.conversation.title)
const itemEl = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLInputElement | null>(null)
const menuEl = ref<HTMLElement | null>(null)
const teamMenuTriggerEl = ref<HTMLButtonElement | null>(null)
const teamSubmenuEl = ref<HTMLElement | null>(null)
const teamSubmenuPosition = ref({ top: 0, left: 0 })
let teamMenuCloseTimer: number | undefined

const teamSubmenuStyle = computed(() => ({
  left: `${teamSubmenuPosition.value.left}px`,
  top: `${teamSubmenuPosition.value.top}px`,
}))

watch(
  () => props.conversation.teamId,
  (teamId) => {
    selectedTeamId.value = teamId || ''
  },
)

const startEditing = async () => {
  if (props.isSending) return
  draftTitle.value = props.conversation.title
  isEditing.value = true
  await nextTick()
  inputEl.value?.select()
}

const startEditingFromMenu = async () => {
  closeMenu()
  await startEditing()
}

const save = () => {
  if (!isEditing.value) return
  const nextTitle = draftTitle.value.trim()
  if (nextTitle) emit('rename', props.conversation.id, nextTitle)
  isEditing.value = false
}

const cancel = () => {
  isEditing.value = false
}

const toggleMenu = () => {
  if (isMenuOpen.value) {
    closeMenu()
    return
  }
  isMenuOpen.value = true
}

const clearTeamMenuCloseTimer = () => {
  if (teamMenuCloseTimer === undefined) return
  window.clearTimeout(teamMenuCloseTimer)
  teamMenuCloseTimer = undefined
}

const openTeamMenu = () => {
  clearTeamMenuCloseTimer()
  updateTeamSubmenuPosition()
  isTeamMenuOpen.value = true
  void nextTick(updateTeamSubmenuPosition)
}

const scheduleCloseTeamMenu = () => {
  clearTeamMenuCloseTimer()
  teamMenuCloseTimer = window.setTimeout(() => {
    isTeamMenuOpen.value = false
    teamMenuCloseTimer = undefined
  }, 180)
}

const closeMenu = () => {
  clearTeamMenuCloseTimer()
  isMenuOpen.value = false
  isTeamMenuOpen.value = false
}

const handlePointerDown = (event: PointerEvent) => {
  if (!isMenuOpen.value) return
  const target = event.target
  if (!(target instanceof Node)) return
  if (itemEl.value?.contains(target) || teamSubmenuEl.value?.contains(target)) return
  closeMenu()
}

const updateTeamSubmenuPosition = () => {
  const triggerRect = teamMenuTriggerEl.value?.getBoundingClientRect()
  if (!triggerRect) return
  const menuRect = menuEl.value?.getBoundingClientRect()
  const submenuWidth = 190
  const gap = 0
  const viewportPadding = 8
  const preferredLeft = (menuRect?.right || triggerRect.right) + gap
  const left = Math.min(
    preferredLeft,
    Math.max(viewportPadding, window.innerWidth - submenuWidth - viewportPadding),
  )
  const top = Math.max(viewportPadding, triggerRect.top)
  teamSubmenuPosition.value = { left, top }
}

const cancelTeamAssign = () => {
  clearTeamMenuCloseTimer()
  selectedTeamId.value = props.conversation.teamId || ''
  isTeamMenuOpen.value = false
}

const confirmTeamAssign = () => {
  const targetTeam = props.teams.find((team) => team.id === selectedTeamId.value)
  const nextLabel = targetTeam?.name || '个人空间'
  if (selectedTeamId.value !== (props.conversation.teamId || '')) {
    const confirmed = window.confirm(`将“${props.conversation.title}”归属到“${nextLabel}”？`)
    if (!confirmed) {
      cancelTeamAssign()
      return
    }
    emit('assignTeam', props.conversation.id, selectedTeamId.value)
  }
  closeMenu()
}

const archive = () => {
  closeMenu()
  emit('archive', props.conversation.id)
}

const deleteItem = () => {
  closeMenu()
  emit('delete', props.conversation.id)
}

const exportPdf = () => {
  closeMenu()
  emit('export', props.conversation.id)
}

const shareConversation = async () => {
  closeMenu()
  const shareText = `${props.conversation.title}\n${props.conversation.messages
    .map((message) => `${message.role === 'user' ? '现场输入' : '辨识报告'}：${message.content}`)
    .join('\n\n')}`
  if (navigator.share) {
    await navigator.share({ title: props.conversation.title, text: shareText })
    return
  }
  await navigator.clipboard?.writeText(shareText)
}

const formatRelativeTime = (value: string) => {
  const time = Date.parse(value)
  if (!Number.isFinite(time)) return '刚刚'
  const diffMinutes = Math.floor((Date.now() - time) / 60_000)
  if (diffMinutes < 1) return '刚刚'
  if (diffMinutes < 60) return `${diffMinutes} 分钟前`
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours} 小时前`
  return new Date(value).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}

if (typeof window !== 'undefined') {
  window.addEventListener('resize', updateTeamSubmenuPosition)
  window.addEventListener('scroll', updateTeamSubmenuPosition, true)
  window.addEventListener('pointerdown', handlePointerDown, true)
}

onBeforeUnmount(() => {
  clearTeamMenuCloseTimer()
  if (typeof window !== 'undefined') {
    window.removeEventListener('resize', updateTeamSubmenuPosition)
    window.removeEventListener('scroll', updateTeamSubmenuPosition, true)
    window.removeEventListener('pointerdown', handlePointerDown, true)
  }
})
</script>
