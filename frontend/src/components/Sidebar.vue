<template>
  <aside class="sidebar" :class="{ collapsed }">
    <div class="sidebar-top">
      <button class="brand-lockup" type="button" title="展开/收起菜单" @click="$emit('toggle')">
        <BrandMark />
        <span class="sidebar-brand">矿风眼</span>
      </button>
      <button class="sidebar-toggle" type="button" title="展开/收起侧栏" @click="$emit('toggle')">
        <span aria-hidden="true">{{ collapsed ? '☰' : '☷' }}</span>
      </button>
    </div>

    <nav class="quick-actions">
      <NewChatButton @create="$emit('create')" />

      <label class="quick-action search-action" title="搜索对话内容">
        <span class="quick-icon" aria-hidden="true">⌕</span>
        <input v-model="chat.searchQuery" type="text" placeholder="搜索对话内容" />
        <button
          v-if="chat.searchQuery"
          type="button"
          class="search-clear"
          title="清空搜索"
          aria-label="清空搜索"
          @click="chat.searchQuery = ''"
        >
          ×
        </button>
      </label>

      <RouterLink class="quick-action" to="/stats" title="库">
        <span class="quick-icon" aria-hidden="true">▦</span>
        <span>库</span>
      </RouterLink>
    </nav>

    <ConversationList
      @select="$emit('select', $event)"
      @select-team="$emit('selectTeam', $event)"
      @archive="$emit('archive', $event)"
      @delete="$emit('delete', $event)"
      @restore="$emit('restore', $event)"
    />

    <div class="bottom-area">
      <RouterLink class="settings-entry-button" to="/settings" title="偏好设置">
        <span aria-hidden="true">⚙</span>
        <span>偏好设置</span>
      </RouterLink>
      <UserMiniCard />
    </div>
  </aside>
</template>

<script setup lang="ts">
import { RouterLink } from 'vue-router'
import BrandMark from '@/components/BrandMark.vue'
import ConversationList from '@/components/ConversationList.vue'
import NewChatButton from '@/components/NewChatButton.vue'
import UserMiniCard from '@/components/UserMiniCard.vue'
import { useChatStore } from '@/stores/chat'

const chat = useChatStore()

defineProps<{
  collapsed: boolean
}>()

defineEmits<{
  toggle: []
  create: []
  select: [id: string]
  selectTeam: [id: string]
  archive: [id: string]
  delete: [id: string]
  restore: [id: string]
}>()
</script>
