<template>
  <section class="stats-view">
    <header class="page-header">
      <div>
        <h1>辨识统计</h1>
        <p>先把会话沉淀起来，这里会逐步扩展成风险分布和趋势看板。</p>
      </div>
      <button type="button" @click="chat.exportAllAsJson">导出 JSON</button>
    </header>

    <div class="stats-grid">
      <article class="metric-card">
        <span>历史对话</span>
        <strong>{{ chat.stats.totalConversations }}</strong>
      </article>
      <article class="metric-card">
        <span>生成报告</span>
        <strong>{{ chat.stats.completedReports }}</strong>
      </article>
      <article class="metric-card">
        <span>消息总数</span>
        <strong>{{ chat.stats.totalMessages }}</strong>
      </article>
      <article class="metric-card">
        <span>已归档</span>
        <strong>{{ chat.stats.archivedCount }}</strong>
      </article>
    </div>

    <section class="stats-panel">
      <div class="stats-panel-header">
        <h2>近七天趋势</h2>
        <span>{{ latestActivityText }}</span>
      </div>
      <div class="trend-bars">
        <div v-for="item in chat.stats.recentSevenDays" :key="item.date" class="trend-bar-item">
          <span class="trend-bar" :style="{ height: `${barHeight(item.count)}%` }"></span>
          <small>{{ item.date.slice(5) }}</small>
        </div>
      </div>
    </section>

    <section class="stats-panel">
      <div class="stats-panel-header">
        <h2>场景分布</h2>
        <span>来自会话元数据和报告推断</span>
      </div>
      <div v-if="chat.stats.sceneCounts.length" class="scene-list">
        <div v-for="scene in chat.stats.sceneCounts" :key="scene.label">
          <span>{{ scene.label }}</span>
          <strong>{{ scene.count }}</strong>
        </div>
      </div>
      <p v-else class="muted-placeholder">暂无可统计的辨识记录。</p>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useChatStore } from '@/stores/chat'

const chat = useChatStore()

const maxCount = computed(() =>
  Math.max(1, ...chat.stats.recentSevenDays.map((item) => item.count)),
)
const latestActivityText = computed(() =>
  chat.stats.latestActivity
    ? `最后更新 ${new Date(chat.stats.latestActivity).toLocaleString('zh-CN')}`
    : '暂无活动',
)

const barHeight = (count: number) => Math.max(8, Math.round((count / maxCount.value) * 100))
</script>
