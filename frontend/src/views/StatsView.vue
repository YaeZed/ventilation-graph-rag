<template>
  <section class="stats-view">
    <header class="page-header">
      <div>
        <h1>统计</h1>
        <p>按当前账号的会话记录汇总辨识数量、完成度、风险等级和近期活跃情况。</p>
      </div>
      <button type="button" @click="chat.exportAllAsJson">导出 JSON</button>
    </header>

    <div class="stats-grid">
      <article class="metric-card">
        <span>会话数</span>
        <strong>{{ chat.stats.totalConversations }}</strong>
        <small>归档 {{ chat.stats.archivedCount }}</small>
      </article>
      <article class="metric-card">
        <span>完成报告</span>
        <strong>{{ chat.stats.completedReports }}</strong>
        <small>{{ chat.stats.completionRate }}% 完成率</small>
      </article>
      <article class="metric-card">
        <span>风险重点</span>
        <strong>{{ chat.stats.topHazardLabel }}</strong>
        <small>{{ hazardSummary }}</small>
      </article>
      <article class="metric-card">
        <span>近 7 天活跃</span>
        <strong>{{ chat.stats.activeDays }} 天</strong>
        <small>{{ chat.stats.totalMessages }} 条消息</small>
      </article>
    </div>

    <section class="stats-panel">
      <div class="stats-panel-header">
        <h2>完成概览</h2>
        <span>{{ latestActivityText }}</span>
      </div>
      <div class="completion-layout">
        <div class="completion-meter" :style="{ '--completion': `${chat.stats.completionRate}%` }">
          <strong>{{ chat.stats.completionRate }}%</strong>
          <span>报告完成率</span>
        </div>
        <div class="completion-details">
          <div>
            <span>有效会话</span>
            <strong>{{ chat.stats.totalConversations }}</strong>
          </div>
          <div>
            <span>完成报告</span>
            <strong>{{ chat.stats.completedReports }}</strong>
          </div>
          <div>
            <span>消息总数</span>
            <strong>{{ chat.stats.totalMessages }}</strong>
          </div>
        </div>
      </div>
    </section>

    <section class="stats-panel">
      <div class="stats-panel-header">
        <h2>风险等级分布</h2>
        <span>来自会话辨识结果的风险标记</span>
      </div>
      <div v-if="chat.stats.hazardCounts.length" class="hazard-list">
        <div
          v-for="hazard in chat.stats.hazardCounts"
          :key="hazard.label"
          class="hazard-row"
          :class="hazard.tone"
        >
          <div>
            <span>{{ hazard.label }}</span>
            <strong>{{ hazard.count }}</strong>
          </div>
          <span class="hazard-track">
            <span
              class="hazard-fill"
              :style="{ width: `${distributionWidth(hazard.count)}%` }"
            ></span>
          </span>
        </div>
      </div>
      <p v-else class="muted-placeholder">暂无可统计的风险等级。</p>
    </section>

    <section class="stats-panel">
      <div class="stats-panel-header">
        <h2>近 7 天趋势</h2>
        <span>{{ latestActivityText }}</span>
      </div>
      <div class="trend-bars">
        <div v-for="item in chat.stats.recentSevenDays" :key="item.date" class="trend-bar-item">
          <span class="trend-bar" :style="{ height: `${barHeight(item.count)}%` }">
            <em>{{ item.count }}</em>
          </span>
          <small>{{ item.date.slice(5) }}</small>
        </div>
      </div>
    </section>

    <section class="stats-panel">
      <div class="stats-panel-header">
        <h2>场景分布</h2>
        <span>按场景或风险标记聚合</span>
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
const maxHazardCount = computed(() =>
  Math.max(1, ...chat.stats.hazardCounts.map((item) => item.count)),
)
const latestActivityText = computed(() =>
  chat.stats.latestActivity
    ? `最近更新 ${new Date(chat.stats.latestActivity).toLocaleString('zh-CN')}`
    : '暂无活动',
)
const hazardSummary = computed(() => {
  const total = chat.stats.hazardCounts.reduce((sum, item) => sum + item.count, 0)
  if (!total) return '暂无风险标记'
  return `${total} 条风险记录`
})

const barHeight = (count: number) => Math.max(8, Math.round((count / maxCount.value) * 100))
const distributionWidth = (count: number) =>
  Math.max(6, Math.round((count / maxHazardCount.value) * 100))
</script>
