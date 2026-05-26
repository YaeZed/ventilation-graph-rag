<template>
  <button class="quick-action new-chat-btn" type="button" title="发起新对话" @click="$emit('create')">
    <span class="quick-icon" aria-hidden="true">◉</span>
    <span class="btn-text">发起新对话</span>
  </button>
</template>

<script setup lang="ts">
defineEmits<{
  create: []
}>()
</script>
