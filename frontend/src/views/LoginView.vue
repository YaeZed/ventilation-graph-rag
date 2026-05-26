<template>
  <main class="auth-page auth-page-login">
    <div class="auth-orbit" aria-hidden="true"></div>
    <section class="auth-card">
      <div class="auth-brand">
        <span class="brand-mark">✦</span>
        <div>
          <h1>煤矿通风隐患智能辨识</h1>
          <p>登录后同步会话、偏好设置和辨识记录。</p>
        </div>
      </div>

      <form class="auth-form" @submit.prevent="submit">
        <label>
          <span>用户名</span>
          <input v-model="username" autocomplete="username" type="text" placeholder="请输入用户名" />
        </label>
        <label>
          <span>密码</span>
          <input
            v-model="password"
            autocomplete="current-password"
            type="password"
            placeholder="请输入密码"
          />
        </label>

        <p v-if="feedback" class="auth-feedback" :class="{ error: !isSuccess }">{{ feedback }}</p>

        <button class="auth-primary" type="submit" :disabled="isSubmitting || !canSubmit">
          {{ isSubmitting ? '登录中...' : '立即登录' }}
        </button>
        <button class="auth-secondary" type="button" @click="router.push('/register')">
          注册账号
        </button>
      </form>

      <RouterLink class="auth-skip" to="/chat">暂不登录，继续本地模式</RouterLink>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'

const router = useRouter()
const chat = useChatStore()
const username = ref('')
const password = ref('')
const feedback = ref('')
const isSuccess = ref(false)
const isSubmitting = ref(false)

const canSubmit = computed(() => Boolean(username.value.trim() && password.value))

const submit = async () => {
  if (!canSubmit.value || isSubmitting.value) return
  isSubmitting.value = true
  feedback.value = ''
  const ok = await chat.loginAccount(username.value, password.value)
  isSubmitting.value = false
  isSuccess.value = ok
  if (!ok) {
    feedback.value = chat.authError || '请检查用户名和密码'
    return
  }
  feedback.value = buildGreeting(chat.userProfile.nickname)
  window.setTimeout(() => router.push('/chat'), 450)
}

function buildGreeting(nickname: string) {
  const hour = new Date().getHours()
  if (hour <= 6) return `深夜了，${nickname}，记得早点休息。`
  if (hour <= 11) return `早上好，${nickname}，欢迎回来。`
  if (hour <= 13) return `中午好，${nickname}，继续辨识前也可以歇一下。`
  if (hour <= 17) return `下午好，${nickname}，会话已准备同步。`
  if (hour <= 21) return `晚上好，${nickname}，祝你工作顺利。`
  return `欢迎回来，${nickname}。`
}
</script>

