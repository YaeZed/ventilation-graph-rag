<template>
  <section class="user-mini-card">
    <button class="avatar-button" type="button" :title="buttonTitle" @click="handleAvatarClick">
      {{ chat.userProfile.avatarText }}
    </button>
    <div class="user-copy">
      <template v-if="isEditing">
        <input
          ref="inputEl"
          v-model="draftName"
          class="nickname-input"
          maxlength="18"
          @blur="save"
          @keydown.enter.prevent="save"
          @keydown.esc.prevent="cancel"
        />
      </template>
      <template v-else>
        <strong>{{ chat.userProfile.nickname }}</strong>
        <button type="button" @click="handleProfileAction">{{ actionLabel }}</button>
      </template>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { nextTick, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useChatStore } from '@/stores/chat'

const chat = useChatStore()
const router = useRouter()
const isEditing = ref(false)
const draftName = ref(chat.userProfile.nickname)
const inputEl = ref<HTMLInputElement | null>(null)

const isAuthenticated = computed(() => chat.authStatus === 'authenticated')
const actionLabel = computed(() => (isAuthenticated.value ? '编辑身份' : '登录同步'))
const buttonTitle = computed(() => (isAuthenticated.value ? '修改昵称' : '登录账号'))

const handleAvatarClick = () => {
  if (!isAuthenticated.value) {
    router.push('/login')
    return
  }
  void startEditing()
}

const handleProfileAction = () => {
  if (!isAuthenticated.value) {
    router.push('/login')
    return
  }
  void startEditing()
}

const startEditing = async () => {
  draftName.value = chat.userProfile.nickname
  isEditing.value = true
  await nextTick()
  inputEl.value?.select()
}

const save = () => {
  const nickname = draftName.value.trim()
  if (nickname) {
    chat.updateUserProfile({
      nickname,
      avatarText: nickname.slice(0, 1),
    })
  }
  isEditing.value = false
}

const cancel = () => {
  isEditing.value = false
}
</script>
