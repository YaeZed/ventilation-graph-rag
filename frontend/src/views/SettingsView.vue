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
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'

const chat = useChatStore()
const router = useRouter()
const nickname = ref(chat.userProfile.nickname)

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

watch(
  () => chat.userProfile.nickname,
  (value) => {
    nickname.value = value
  },
)

const saveProfile = () => {
  chat.updateUserProfile({
    nickname: nickname.value,
    avatarText: nickname.value.slice(0, 1),
  })
}

const syncNow = () => {
  void chat.syncWithRemote()
}

const logout = async () => {
  const ok = await chat.logoutAccount()
  if (ok) router.push('/login')
}
</script>
