<template>
  <div class="app-container">
    <Sidebar
      :collapsed="isCollapsed"
      @toggle="toggleSidebar"
      @create="createConversation"
      @select="selectConversation"
      @select-team="selectTeamConversation"
      @archive="archiveConversation"
      @delete="deleteConversation"
      @restore="restoreConversation"
    />

    <main class="main-content">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import Sidebar from '@/components/Sidebar.vue'
import { useChatStore } from '@/stores/chat'

const router = useRouter()
const chat = useChatStore()
const isCollapsed = ref(false)

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const createConversation = () => {
  const id = chat.createConversation()
  router.push(`/chat/${id}`)
}

const selectConversation = (id: string) => {
  if (chat.selectConversation(id)) {
    router.push(`/chat/${id}`)
  }
}

const selectTeamConversation = (id: string) => {
  if (chat.selectTeamConversation(id)) {
    router.push(`/chat/${id}`)
  }
}

const archiveConversation = (id: string) => {
  const nextId = chat.archiveConversation(id)
  router.push(nextId ? `/chat/${nextId}` : '/chat')
}

const deleteConversation = (id: string) => {
  const conversation = chat.findConversation(id)
  if (!conversation) return
  const confirmed = window.confirm(`删除“${conversation.title}”？此操作不可恢复。`)
  if (!confirmed) return
  const nextId = chat.deleteConversation(id)
  router.push(nextId ? `/chat/${nextId}` : '/chat')
}

const restoreConversation = (id: string) => {
  if (chat.restoreConversation(id)) {
    router.push(`/chat/${id}`)
  }
}
</script>
