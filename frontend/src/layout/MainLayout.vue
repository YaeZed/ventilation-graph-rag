<template>
  <div class="app-container">
    <aside class="sidebar" :class="{ collapsed: isCollapsed }">
      <div class="top-bar">
        <button class="menu-btn" type="button" title="展开/收起菜单" @click="toggleSidebar">
          <span class="menu-lines" aria-hidden="true"></span>
        </button>
      </div>

      <div class="action-area">
        <button class="new-chat-btn" type="button" title="发起新对话" @click="createConversation">
          <span class="plus-icon">+</span>
          <span class="btn-text">发起新对话</span>
        </button>
      </div>

      <nav class="nav-menu custom-scrollbar">
        <div class="nav-group-title">最近对话</div>
        <button
          v-for="conversation in chat.conversations"
          :key="conversation.id"
          class="nav-item"
          :class="{ active: conversation.id === chat.activeId }"
          type="button"
          :title="conversation.title"
          @click="chat.selectConversation(conversation.id)"
        >
          <span class="nav-text">{{ conversation.title }}</span>
        </button>
      </nav>

      <div class="bottom-area">
        <div class="nav-item info-item" title="规程知识库">
          <span class="status-dot"></span>
          <span class="nav-text">规程知识库已连接</span>
        </div>
        <div class="status-chip">
          <span class="dot"></span>
          <span class="loc-text">开发版本</span>
        </div>
      </div>
    </aside>

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
import { useChatStore } from '@/stores/chat'

const router = useRouter()
const chat = useChatStore()
const isCollapsed = ref(false)

const toggleSidebar = () => {
  isCollapsed.value = !isCollapsed.value
}

const createConversation = () => {
  chat.newConversation()
  router.push('/')
}
</script>
