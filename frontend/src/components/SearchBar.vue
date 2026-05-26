<template>
  <label class="conversation-search">
    <span aria-hidden="true">⌕</span>
    <input v-model="chat.searchQuery" type="text" placeholder="搜索场景、隐患或日期" />
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
</template>

<script setup lang="ts">
import { useChatStore } from '@/stores/chat'

const chat = useChatStore()
</script>
