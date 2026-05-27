<template>
  <main class="auth-page auth-page-register">
    <div class="auth-orbit" aria-hidden="true"></div>
    <section class="auth-card">
      <div class="auth-brand">
        <BrandMark />
        <div>
          <h1>创建矿风眼账号</h1>
          <p>注册后会自动把当前浏览器里的本地会话同步到账号。</p>
        </div>
      </div>

      <form class="auth-form" @submit.prevent="submit">
        <label>
          <span>用户名</span>
          <input v-model="username" autocomplete="username" type="text" placeholder="用户名" />
        </label>
        <label>
          <span>昵称</span>
          <input v-model="nickname" autocomplete="nickname" type="text" placeholder="安全工程师" />
        </label>
        <label>
          <span>密码</span>
          <input
            v-model="password"
            autocomplete="new-password"
            type="password"
            placeholder="至少 6 位"
          />
        </label>
        <label>
          <span>确认密码</span>
          <input
            v-model="passwordConfirm"
            autocomplete="new-password"
            type="password"
            placeholder="请再次输入密码"
          />
        </label>

        <p v-if="feedback" class="auth-feedback error">{{ feedback }}</p>

        <button class="auth-primary" type="submit" :disabled="isSubmitting || !canSubmit">
          {{ isSubmitting ? '注册中...' : '注册' }}
        </button>
      </form>

      <p class="auth-footnote">
        已经有账号了？
        <RouterLink to="/login">立即登录</RouterLink>
      </p>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import BrandMark from '@/components/BrandMark.vue'
import { useChatStore } from '@/stores/chat'

const router = useRouter()
const chat = useChatStore()
const username = ref('')
const nickname = ref('')
const password = ref('')
const passwordConfirm = ref('')
const feedback = ref('')
const isSubmitting = ref(false)

const canSubmit = computed(() =>
  Boolean(username.value.trim() && password.value && passwordConfirm.value),
)

const submit = async () => {
  if (!canSubmit.value || isSubmitting.value) return
  feedback.value = ''
  if (password.value !== passwordConfirm.value) {
    feedback.value = '两次密码输入不一致，请重新输入'
    return
  }
  isSubmitting.value = true
  const ok = await chat.registerAccount(username.value, password.value, nickname.value)
  isSubmitting.value = false
  if (!ok) {
    feedback.value = chat.authError || '注册失败，请检查输入信息'
    return
  }
  router.push('/chat')
}
</script>
