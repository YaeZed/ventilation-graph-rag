<template>
  <div v-if="sensorData?.entries?.length" class="sensor-data-badge">
    <div class="sensor-badge-header">
      <span>传感器数据</span>
      <small>{{ sensorData.location || '未标注地点' }} · {{ sourceLabel }}</small>
      <button
        v-if="removable"
        type="button"
        title="移除传感器数据"
        @click="$emit('remove')"
      >
        ×
      </button>
    </div>
    <div class="sensor-chip-row">
      <span
        v-for="entry in visibleEntries"
        :key="`${entry.type}-${entry.label}-${entry.value}-${entry.timestamp || ''}`"
        class="sensor-chip"
        :title="entryTitle(entry)"
      >
        <strong>{{ entry.label }}</strong>
        {{ formatValue(entry.value) }} {{ entry.unit }}
      </span>
      <span v-if="hiddenCount > 0" class="sensor-chip muted">+{{ hiddenCount }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { SensorData, SensorEntry } from '@/types/multimodal'
import { computed } from 'vue'

const props = defineProps<{
  sensorData?: SensorData | null
  removable?: boolean
}>()

defineEmits<{
  remove: []
}>()

const visibleEntries = computed(() => props.sensorData?.entries.slice(0, 5) || [])
const hiddenCount = computed(() => Math.max(0, (props.sensorData?.entries.length || 0) - 5))
const sourceLabel = computed(() => (props.sensorData?.source === 'csv' ? 'CSV' : '手动'))

const formatValue = (value: number) =>
  Number.isInteger(value) ? String(value) : String(Number(value.toFixed(3)))

const entryTitle = (entry: SensorEntry) => {
  const parts = [`${entry.label}: ${formatValue(entry.value)} ${entry.unit}`]
  if (entry.location) parts.push(`地点: ${entry.location}`)
  if (entry.timestamp) parts.push(`时间: ${entry.timestamp}`)
  return parts.join('\n')
}
</script>
