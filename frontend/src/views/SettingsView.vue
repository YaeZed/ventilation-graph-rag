<template>
  <section class="settings-view">
    <header class="page-header">
      <div>
        <h1>偏好设置</h1>
        <p>{{ statusCopy }}</p>
      </div>
      <button
        v-if="chat.authStatus === 'authenticated'"
        type="button"
        class="ghost-button"
        @click="logout"
      >
        退出登录
      </button>
    </header>

    <div class="settings-panel account-panel">
      <div class="account-status">
        <span class="avatar-button account-avatar">{{ chat.userProfile.avatarText }}</span>
        <div>
          <strong>{{ chat.userProfile.nickname }}</strong>
          <p>{{ accountCopy }}</p>
        </div>
      </div>

      <div class="account-actions">
        <button
          v-if="chat.authStatus !== 'authenticated'"
          type="button"
          class="auth-inline-button"
          @click="router.push('/login')"
        >
          登录同步
        </button>
        <button
          v-if="chat.authStatus !== 'authenticated'"
          type="button"
          class="auth-inline-button secondary"
          @click="router.push('/register')"
        >
          注册账号
        </button>
        <button
          v-else
          type="button"
          class="auth-inline-button"
          :disabled="chat.syncStatus === 'syncing'"
          @click="syncNow"
        >
          {{ chat.syncStatus === 'syncing' ? '同步中...' : '立即同步' }}
        </button>
      </div>

      <p v-if="chat.authError || chat.syncError" class="settings-error">
        {{ chat.authError || chat.syncError }}
      </p>
    </div>

    <div v-if="chat.authStatus === 'authenticated'" class="settings-panel security-panel">
      <div class="settings-section-header">
        <div>
          <strong>账号安全</strong>
          <p>{{ securityStatusCopy }}</p>
        </div>
        <button
          type="button"
          class="auth-inline-button secondary"
          :disabled="chat.securityStatus === 'loading'"
          @click="chat.refreshSecurityEvents"
        >
          刷新
        </button>
      </div>

      <div v-if="chat.securityEvents.length" class="security-event-list custom-scrollbar">
        <div v-for="event in chat.securityEvents" :key="event.id" class="security-event-row">
          <span class="security-event-type">{{ securityEventLabel(event.type) }}</span>
          <div>
            <strong>{{ formatSecurityTime(event.createdAt) }}</strong>
            <small>{{ event.ipAddress || '未知 IP' }}</small>
          </div>
        </div>
      </div>
      <p v-else class="muted-placeholder">暂无安全记录</p>
      <p v-if="chat.securityError" class="settings-error">{{ chat.securityError }}</p>
    </div>

    <div v-if="chat.authStatus === 'authenticated'" class="settings-panel team-panel">
      <div class="settings-section-header">
        <div>
          <strong>团队空间</strong>
          <p>{{ teamStatusCopy }}</p>
        </div>
        <button
          type="button"
          class="auth-inline-button secondary"
          :disabled="chat.teamStatus === 'loading'"
          @click="chat.refreshTeams"
        >
          刷新
        </button>
      </div>

      <form class="team-create-row" @submit.prevent="createTeam">
        <input v-model="newTeamName" type="text" maxlength="40" placeholder="团队名称" />
        <input
          v-model="newTeamDescription"
          type="text"
          maxlength="120"
          placeholder="备注"
        />
        <button type="submit" class="auth-inline-button">创建</button>
      </form>

      <label v-if="chat.teams.length" class="team-select-row">
        <span>管理团队</span>
        <SettingsSelect
          v-model="selectedTeamId"
          aria-label="选择要管理的团队"
          :options="teamSelectOptions"
        />
      </label>

      <div v-if="selectedTeam" class="team-members-block">
        <div class="settings-section-header compact">
          <div>
            <form
              v-if="editingTeamId === selectedTeam.id"
              class="team-title-edit-form"
              @submit.prevent="saveTeamName"
            >
              <input
                ref="teamNameInputEl"
                v-model="teamNameDraft"
                type="text"
                maxlength="40"
                @blur="saveTeamName"
                @keydown.esc.prevent="cancelTeamNameEdit"
              />
            </form>
            <div v-else class="team-title-row">
              <strong>{{ selectedTeam.name }}</strong>
              <button
                v-if="canManageSelectedTeam"
                type="button"
                class="team-title-edit-button"
                title="编辑团队名称"
                @click="startTeamNameEdit"
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M4 20h4.5L19 9.5 14.5 5 4 15.5V20Z" />
                  <path d="m13.5 6 4.5 4.5" />
                </svg>
              </button>
            </div>
            <p>{{ selectedTeam.memberCount }} 名成员</p>
          </div>
          <button
            v-if="selectedTeam.role === 'owner'"
            type="button"
            class="ghost-button danger"
            @click="deleteSelectedTeam"
          >
            删除团队
          </button>
        </div>

        <div class="team-member-list">
          <div v-for="member in selectedMembers" :key="member.id" class="team-member-row">
            <span class="avatar-button team-member-avatar">{{ member.avatarText }}</span>
            <div>
              <strong>{{ member.nickname }}</strong>
              <small>{{ member.username }}</small>
            </div>
            <SettingsSelect
              :model-value="member.role"
              aria-label="调整成员角色"
              :disabled="!canManageSelectedTeam || member.role === 'owner'"
              :options="memberRoleOptions"
              @change="updateMemberRole(member.id, $event)"
            />
            <button
              type="button"
              class="ghost-button"
              :disabled="member.role === 'owner' || (!canManageSelectedTeam && member.id !== chat.remoteUser?.id)"
              @click="removeMember(member.id)"
            >
              移除
            </button>
          </div>
        </div>

        <form v-if="canManageSelectedTeam" class="team-create-row" @submit.prevent="addMember">
          <input v-model="memberUsername" type="text" maxlength="80" placeholder="用户名" />
          <SettingsSelect
            v-model="memberRole"
            aria-label="选择新成员角色"
            :options="newMemberRoleOptions"
          />
          <button type="submit" class="auth-inline-button">添加</button>
        </form>
      </div>

      <p v-if="chat.teamError" class="settings-error">{{ chat.teamError }}</p>
    </div>

    <div class="settings-panel">
      <label>
        <span>昵称</span>
        <input v-model="nickname" type="text" maxlength="18" @change="saveProfile" />
      </label>

      <label class="toggle-row">
        <span>SSE 流式响应</span>
        <input
          :checked="chat.settings.useStream"
          type="checkbox"
          @change="chat.updateSettings({ useStream: ($event.target as HTMLInputElement).checked })"
        />
      </label>

      <label class="toggle-row">
        <span>自动展开 Agent 步骤</span>
        <input
          :checked="chat.settings.autoExpandSteps"
          type="checkbox"
          @change="
            chat.updateSettings({ autoExpandSteps: ($event.target as HTMLInputElement).checked })
          "
        />
      </label>

      <label>
        <span>默认 temperature</span>
        <input
          :value="chat.settings.temperature"
          type="range"
          min="0"
          max="1"
          step="0.1"
          @input="
            chat.updateSettings({ temperature: Number(($event.target as HTMLInputElement).value) })
          "
        />
        <small>{{ chat.settings.temperature.toFixed(1) }}</small>
      </label>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import SettingsSelect from '@/components/SettingsSelect.vue'
import { useChatStore } from '@/stores/chat'

type SettingsSelectOption = {
  value: string
  label: string
  disabled?: boolean
}

const chat = useChatStore()
const router = useRouter()
const nickname = ref(chat.userProfile.nickname)
const newTeamName = ref('')
const newTeamDescription = ref('')
const selectedTeamId = ref('')
const memberUsername = ref('')
const memberRole = ref<'admin' | 'member'>('member')
const editingTeamId = ref('')
const teamNameDraft = ref('')
const teamNameInputEl = ref<HTMLInputElement | null>(null)

const statusCopy = computed(() => {
  if (chat.authStatus === 'checking') return '正在检查账号状态，本地会话会先安全保留。'
  if (chat.authStatus === 'authenticated') return '账号已登录，会话、偏好设置会同步到后端。'
  return '未登录时使用本机浏览器保存；登录后可同步到后端账号。'
})

const accountCopy = computed(() => {
  if (chat.authStatus === 'authenticated') {
    const syncedAt = chat.lastSyncedAt
      ? new Date(chat.lastSyncedAt).toLocaleString('zh-CN')
      : '等待首次同步'
    return `已登录 ${chat.remoteUser?.username || ''} · ${syncedAt}`
  }
  return '当前为本地模式，浏览器清理数据后会话可能丢失。'
})

const selectedTeam = computed(() => chat.teams.find((team) => team.id === selectedTeamId.value) || null)
const selectedMembers = computed(() => (selectedTeamId.value ? chat.teamMembers[selectedTeamId.value] || [] : []))
const canManageSelectedTeam = computed(() =>
  selectedTeam.value ? ['owner', 'admin'].includes(selectedTeam.value.role) : false,
)
const teamSelectOptions = computed<SettingsSelectOption[]>(() =>
  chat.teams.map((team) => ({
    value: team.id,
    label: `${team.name} · ${roleLabel(team.role)}`,
  })),
)
const memberRoleOptions = computed<SettingsSelectOption[]>(() => [
  { value: 'owner', label: roleLabel('owner'), disabled: true },
  { value: 'admin', label: roleLabel('admin') },
  { value: 'member', label: roleLabel('member') },
])
const newMemberRoleOptions = computed<SettingsSelectOption[]>(() => [
  { value: 'member', label: roleLabel('member') },
  { value: 'admin', label: roleLabel('admin') },
])
const teamStatusCopy = computed(() => {
  if (chat.teamStatus === 'loading') return '正在同步团队信息'
  if (!chat.teams.length) return '暂无团队'
  return `${chat.teams.length} 个团队`
})

const securityStatusCopy = computed(() => {
  if (chat.securityStatus === 'loading') return '正在读取最近账号事件'
  if (chat.securityStatus === 'error') return '安全记录暂时不可用'
  if (!chat.securityEvents.length) return '暂无登录或注册记录'
  return `最近 ${chat.securityEvents.length} 条账号事件`
})

onMounted(() => {
  if (chat.authStatus === 'authenticated') {
    void chat.refreshTeams()
    void chat.refreshSecurityEvents()
  }
})

function cancelTeamNameEdit() {
  editingTeamId.value = ''
  teamNameDraft.value = ''
}

async function startTeamNameEdit() {
  if (!selectedTeam.value || !canManageSelectedTeam.value) return
  editingTeamId.value = selectedTeam.value.id
  teamNameDraft.value = selectedTeam.value.name
  await nextTick()
  teamNameInputEl.value?.select()
}

async function saveTeamName() {
  if (!selectedTeam.value || editingTeamId.value !== selectedTeam.value.id) return
  const teamId = selectedTeam.value.id
  const nextName = teamNameDraft.value.trim()
  if (!nextName || nextName === selectedTeam.value.name) {
    cancelTeamNameEdit()
    return
  }
  const ok = await chat.updateTeamSpace(teamId, { name: nextName })
  if (ok) {
    cancelTeamNameEdit()
    await refreshTeamSpace(teamId)
  }
}

async function refreshTeamSpace(preferredTeamId = selectedTeamId.value) {
  await chat.refreshTeams()
  const nextTeamId = chat.teams.some((team) => team.id === preferredTeamId)
    ? preferredTeamId
    : chat.teams[0]?.id || ''
  selectedTeamId.value = nextTeamId
  if (nextTeamId) await chat.loadTeamMembers(nextTeamId)
}

watch(
  () => chat.userProfile.nickname,
  (value) => {
    nickname.value = value
  },
)

watch(
  () => chat.authStatus,
  (status) => {
    if (status === 'authenticated') {
      void chat.refreshTeams()
      void chat.refreshSecurityEvents()
    }
  },
)

watch(
  () => chat.teams,
  (teams) => {
    if (!teams.length) {
      selectedTeamId.value = ''
      return
    }
    const nextTeamId = teams.some((team) => team.id === selectedTeamId.value)
      ? selectedTeamId.value
      : teams[0]?.id || ''
    if (selectedTeamId.value !== nextTeamId) {
      selectedTeamId.value = nextTeamId
      return
    }
    if (nextTeamId && !chat.teamMembers[nextTeamId]?.length) {
      void chat.loadTeamMembers(nextTeamId)
    }
  },
  { deep: true, immediate: true },
)

watch(selectedTeamId, (teamId) => {
  cancelTeamNameEdit()
  if (teamId) void chat.loadTeamMembers(teamId)
}, { immediate: true })

const saveProfile = () => {
  chat.updateUserProfile({
    nickname: nickname.value,
    avatarText: nickname.value.slice(0, 1),
  })
}

const syncNow = () => {
  void chat.syncWithRemote()
}

const roleLabel = (role: string) => {
  if (role === 'owner') return '所有者'
  if (role === 'admin') return '管理员'
  return '成员'
}

const securityEventLabel = (type: string) => {
  const labels: Record<string, string> = {
    register: '注册',
    register_throttled: '注册限制',
    password_rejected: '密码拒绝',
    login_success: '登录成功',
    login_failure: '登录失败',
    login_throttled: '登录限制',
    logout: '退出登录',
  }
  return labels[type] || '账号事件'
}

const formatSecurityTime = (value: string) =>
  value
    ? new Date(value).toLocaleString('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '暂无时间'

const createTeam = async () => {
  const ok = await chat.createTeamSpace(newTeamName.value, newTeamDescription.value)
  if (!ok) return
  const createdTeamId = chat.statsScopeTeamId || chat.teams[0]?.id || ''
  newTeamName.value = ''
  newTeamDescription.value = ''
  await refreshTeamSpace(createdTeamId)
}

const addMember = async () => {
  if (!selectedTeamId.value) return
  const teamId = selectedTeamId.value
  const ok = await chat.addMemberToTeam(selectedTeamId.value, memberUsername.value, memberRole.value)
  if (!ok) return
  memberUsername.value = ''
  await refreshTeamSpace(teamId)
}

const updateMemberRole = async (userId: number, role: string) => {
  if (!selectedTeamId.value || role === 'owner') return
  const teamId = selectedTeamId.value
  const ok = await chat.updateMemberRole(teamId, userId, role === 'admin' ? 'admin' : 'member')
  if (ok) await refreshTeamSpace(teamId)
}

const removeMember = async (userId: number) => {
  if (!selectedTeamId.value) return
  const teamId = selectedTeamId.value
  const ok = await chat.removeMemberFromTeam(teamId, userId)
  if (ok) await refreshTeamSpace(teamId)
}

const deleteSelectedTeam = async () => {
  if (!selectedTeamId.value) return
  const ok = await chat.deleteTeamSpace(selectedTeamId.value)
  if (ok) await refreshTeamSpace()
}

const logout = async () => {
  const ok = await chat.logoutAccount()
  if (ok) router.push('/login')
}
</script>
