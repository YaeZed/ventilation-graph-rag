<template>
  <div ref="rootEl" class="settings-select" :class="{ open: isOpen, disabled }">
    <button
      class="settings-select-trigger"
      type="button"
      :aria-expanded="isOpen"
      :aria-label="ariaLabel"
      :disabled="disabled"
      @click="toggleOpen"
    >
      <span>{{ selectedLabel }}</span>
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m6 9 6 6 6-6" />
      </svg>
    </button>

    <div v-if="isOpen" class="settings-select-menu">
      <button
        v-for="option in options"
        :key="option.value"
        class="settings-select-option"
        :class="{ selected: option.value === modelValue }"
        type="button"
        :disabled="option.disabled"
        @click="selectOption(option.value, option.disabled)"
      >
        {{ option.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

type SettingsSelectOption = {
  value: string
  label: string
  disabled?: boolean
}

const props = defineProps<{
  modelValue: string
  options: SettingsSelectOption[]
  disabled?: boolean
  ariaLabel?: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: [value: string]
}>()

const rootEl = ref<HTMLElement | null>(null)
const isOpen = ref(false)

const selectedLabel = computed(
  () => props.options.find((option) => option.value === props.modelValue)?.label || props.modelValue,
)

const close = () => {
  isOpen.value = false
}

const toggleOpen = () => {
  if (props.disabled) return
  isOpen.value = !isOpen.value
}

const selectOption = (value: string, disabled?: boolean) => {
  if (disabled) return
  emit('update:modelValue', value)
  if (value !== props.modelValue) emit('change', value)
  close()
}

const handlePointerDown = (event: PointerEvent) => {
  const target = event.target
  if (!(target instanceof Node)) return
  if (rootEl.value?.contains(target)) return
  close()
}

watch(
  () => props.disabled,
  (disabled) => {
    if (disabled) close()
  },
)

onMounted(() => {
  window.addEventListener('pointerdown', handlePointerDown, true)
})

onBeforeUnmount(() => {
  window.removeEventListener('pointerdown', handlePointerDown, true)
})
</script>
