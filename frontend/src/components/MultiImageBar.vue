<template>
  <div class="multi-image-bar">
    <div class="multi-image-list">
      <figure v-for="image in images" :key="image.id" class="multi-image-thumb">
        <img :src="image.preview" :alt="image.file.name" />
        <figcaption>{{ image.file.name }}</figcaption>
        <button type="button" title="移除图片" @click="$emit('remove', image.id)">×</button>
      </figure>
      <button
        v-if="images.length < maxImages"
        class="multi-image-add"
        type="button"
        title="继续添加图片"
        @click="$emit('add')"
      >
        +
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { DraftImage } from '@/types/multimodal'

withDefaults(
  defineProps<{
    images: DraftImage[]
    maxImages?: number
  }>(),
  {
    maxImages: 6,
  },
)

defineEmits<{
  add: []
  remove: [id: string]
}>()
</script>
