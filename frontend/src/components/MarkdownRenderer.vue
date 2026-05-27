<template>
  <div class="markdown-body" v-html="renderedHtml"></div>
</template>

<script setup lang="ts">
import { createSafeMarkdownRenderer } from '@/utils/markdown'
import { computed } from 'vue'

const props = defineProps<{
  content: string
}>()

const markdown = createSafeMarkdownRenderer()

const renderedHtml = computed(() => markdown.render(props.content || ''))
</script>
