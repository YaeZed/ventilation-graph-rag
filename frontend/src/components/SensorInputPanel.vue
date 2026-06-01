<template>
  <section class="sensor-input-panel">
    <div class="sensor-panel-header">
      <div>
        <strong>传感器数据</strong>
        <span>{{ mode === 'manual' ? '手动录入' : 'CSV 导入' }}</span>
      </div>
      <button type="button" title="关闭" @click="$emit('close')">×</button>
    </div>

    <div class="sensor-mode-tabs" role="tablist" aria-label="传感器数据输入方式">
      <button
        type="button"
        :class="{ active: mode === 'manual' }"
        @click="mode = 'manual'"
      >
        手动
      </button>
      <button type="button" :class="{ active: mode === 'csv' }" @click="mode = 'csv'">
        CSV
      </button>
    </div>

    <label class="sensor-field">
      <span>检测地点</span>
      <input v-model="location" type="text" maxlength="80" placeholder="掘进工作面" />
    </label>

    <div v-if="mode === 'manual'" class="sensor-manual-list">
      <div v-for="row in rows" :key="row.id" class="sensor-row">
        <SettingsSelect
          class="sensor-type-select"
          :model-value="row.type"
          aria-label="选择传感器类型"
          :options="sensorTypeOptions"
          @update:model-value="(value) => updateRowType(row, value)"
        />
        <input
          v-model="row.label"
          type="text"
          maxlength="24"
          placeholder="参数"
          :disabled="row.type !== 'custom'"
        />
        <input v-model.number="row.value" type="number" step="0.001" placeholder="数值" />
        <input v-model="row.unit" type="text" maxlength="12" placeholder="单位" />
        <button
          type="button"
          class="sensor-row-remove"
          title="删除数据项"
          @click="removeRow(row.id)"
        >
          ×
        </button>
      </div>
      <button type="button" class="sensor-add-row" @click="addRow()">+ 添加数据项</button>
    </div>

    <div v-else class="sensor-csv-box">
      <textarea
        v-model="rawCsv"
        rows="6"
        placeholder="时间, 风速(m/s), 瓦斯(%), 温度(℃)
08:00, 0.25, 0.05, 22.1
08:05, 0.12, 0.08, 22.5"
      ></textarea>
      <small v-if="csvPreviewCount">将解析 {{ csvPreviewCount }} 条数值记录</small>
    </div>

    <div class="sensor-panel-actions">
      <button type="button" @click="$emit('close')">取消</button>
      <button type="button" class="primary" :disabled="!canApply" @click="apply">确认</button>
    </div>
  </section>
</template>

<script setup lang="ts">
import SettingsSelect from '@/components/SettingsSelect.vue'
import type { SensorData, SensorEntry, SensorEntryType } from '@/types/multimodal'
import { computed, ref, watch } from 'vue'

type SensorPreset = {
  type: SensorEntryType
  label: string
  unit: string
}

type SensorRow = SensorEntry & {
  id: string
}

const SENSOR_PRESETS: SensorPreset[] = [
  { type: 'wind_speed', label: '风速', unit: 'm/s' },
  { type: 'methane', label: '瓦斯浓度', unit: '%' },
  { type: 'co', label: 'CO 浓度', unit: 'ppm' },
  { type: 'temperature', label: '温度', unit: '℃' },
  { type: 'oxygen', label: '氧气浓度', unit: '%' },
  { type: 'custom', label: '其他', unit: '' },
]
const DEFAULT_SENSOR_PRESET: SensorPreset = { type: 'wind_speed', label: '风速', unit: 'm/s' }
const sensorTypeOptions = SENSOR_PRESETS.map((preset) => ({
  value: preset.type,
  label: preset.label,
}))

const props = defineProps<{
  initialData?: SensorData | null
}>()

const emit = defineEmits<{
  apply: [data: SensorData]
  close: []
}>()

const mode = ref<'manual' | 'csv'>('manual')
const location = ref('掘进工作面')
const rows = ref<SensorRow[]>([])
const rawCsv = ref('')

const canApply = computed(() =>
  mode.value === 'manual' ? buildManualEntries().length > 0 : parseCsvEntries().length > 0,
)
const csvPreviewCount = computed(() => parseCsvEntries().length)

watch(
  () => props.initialData,
  (value) => {
    loadInitialData(value)
  },
  { immediate: true },
)

function loadInitialData(value?: SensorData | null) {
  location.value = value?.location || '掘进工作面'
  rawCsv.value = value?.rawCsv || ''
  mode.value = value?.source || 'manual'
  rows.value = (value?.entries?.length ? value.entries : [defaultEntry()]).map((entry) => ({
    id: createId(),
    type: entry.type,
    label: entry.label,
    value: Number(entry.value || 0),
    unit: entry.unit,
    location: entry.location,
    timestamp: entry.timestamp,
    thresholdRef: entry.thresholdRef,
  }))
}

function defaultEntry(): SensorEntry {
  const preset = findPreset('wind_speed')
  return {
    type: preset.type,
    label: preset.label,
    value: 0,
    unit: preset.unit,
    location: location.value,
  }
}

function addRow(type: SensorEntryType = 'wind_speed') {
  const preset = findPreset(type)
  rows.value.push({
    id: createId(),
    type: preset.type,
    label: preset.label,
    value: 0,
    unit: preset.unit,
    location: location.value,
  })
}

function removeRow(id: string) {
  rows.value = rows.value.filter((row) => row.id !== id)
  if (!rows.value.length) addRow()
}

function applyPreset(row: SensorRow) {
  const preset = findPreset(row.type)
  if (preset.type !== 'custom') {
    row.label = preset.label
    row.unit = preset.unit
  }
}

function updateRowType(row: SensorRow, value: string) {
  const preset = findPreset(value as SensorEntryType)
  row.type = preset.type
  applyPreset(row)
}

function apply() {
  const entries = mode.value === 'manual' ? buildManualEntries() : parseCsvEntries()
  if (!entries.length) return
  emit('apply', {
    entries,
    location: location.value.trim() || '未标注地点',
    source: mode.value,
    rawCsv: mode.value === 'csv' ? rawCsv.value.trim() : undefined,
  })
}

function buildManualEntries(): SensorEntry[] {
  const fallbackLocation = location.value.trim()
  return rows.value
    .map((row) => ({
      type: row.type,
      label: row.label.trim() || findPreset(row.type).label,
      value: Number(row.value),
      unit: row.unit.trim(),
      location: row.location?.trim() || fallbackLocation,
      timestamp: row.timestamp,
      thresholdRef: row.thresholdRef,
    }))
    .filter((entry) => entry.label && Number.isFinite(entry.value))
}

function parseCsvEntries(): SensorEntry[] {
  const lines = rawCsv.value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  if (lines.length < 2) return []

  const headers = splitCsvLine(lines[0] || '')
  const timeIndex = headers.findIndex((header) => /^(时间|time|timestamp)$/i.test(header.trim()))
  const entries: SensorEntry[] = []
  for (const line of lines.slice(1)) {
    const cells = splitCsvLine(line)
    const timestamp = timeIndex >= 0 ? cells[timeIndex]?.trim() : ''
    headers.forEach((header, index) => {
      if (index === timeIndex) return
      const value = Number(cells[index] || '')
      if (!Number.isFinite(value)) return
      const inferred = inferSensor(header)
      entries.push({
        type: inferred.type,
        label: inferred.label,
        unit: inferred.unit,
        value,
        location: location.value.trim() || '未标注地点',
        timestamp,
      })
    })
  }
  return entries.slice(0, 60)
}

function splitCsvLine(line: string) {
  return line.split(',').map((cell) => cell.trim())
}

function inferSensor(header: string): SensorPreset {
  const text = header.toLowerCase()
  const unitMatch = header.match(/\(([^)]+)\)/)
  if (/风速|wind/.test(text)) return { type: 'wind_speed', label: '风速', unit: unitMatch?.[1] || 'm/s' }
  if (/瓦斯|甲烷|methane|ch4/.test(text))
    return { type: 'methane', label: '瓦斯浓度', unit: unitMatch?.[1] || '%' }
  if (/\bco\b|一氧化碳/.test(text)) return { type: 'co', label: 'CO 浓度', unit: unitMatch?.[1] || 'ppm' }
  if (/温度|temperature|temp/.test(text))
    return { type: 'temperature', label: '温度', unit: unitMatch?.[1] || '℃' }
  if (/氧气|oxygen|o2/.test(text)) return { type: 'oxygen', label: '氧气浓度', unit: unitMatch?.[1] || '%' }
  return { type: 'custom', label: header.replace(/\([^)]*\)/g, '').trim() || '自定义数据', unit: unitMatch?.[1] || '' }
}

function findPreset(type: SensorEntryType) {
  return SENSOR_PRESETS.find((preset) => preset.type === type) || DEFAULT_SENSOR_PRESET
}

function createId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID()
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}
</script>
