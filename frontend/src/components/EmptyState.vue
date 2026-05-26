<template>
  <section class="empty-state-panel">
    <div class="empty-state-copy">
      <span class="eyebrow">开始一次辨识</span>
      <h2>选择典型场景，或者直接上传现场图片</h2>
      <p>每个对话都会单独保存，后续可搜索、重命名、导出报告。</p>
    </div>

    <div class="scenario-grid">
      <button
        v-for="scenario in scenarios"
        :key="scenario.title"
        type="button"
        @click="$emit('pick', scenario.prompt)"
      >
        <strong>{{ scenario.title }}</strong>
        <span>{{ scenario.description }}</span>
      </button>
    </div>
  </section>
</template>

<script setup lang="ts">
defineEmits<{
  pick: [prompt: string]
}>()

const scenarios = [
  {
    title: '局部通风机安装',
    description: '检查位置、供风状态和回风侧安全距离',
    prompt: '检查图片中的局部通风机安装是否合规',
  },
  {
    title: '掘进工作面通风',
    description: '判断风筒末端、风速和瓦斯积聚风险',
    prompt: '判断掘进工作面风筒与通风状态是否存在隐患',
  },
  {
    title: '风门设施核查',
    description: '核对风门闭锁、漏风和设施完好性',
    prompt: '这处风门设施是否满足规程要求',
  },
]
</script>
