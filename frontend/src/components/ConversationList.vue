<template>
  <section class="conversation-list">
    <div class="conversation-list-header">
      <span>最近对话</span>
      <small>{{ chat.filteredConversations.length }}</small>
    </div>

    <div v-if="chat.filteredConversations.length" class="conversation-items custom-scrollbar">
      <ConversationItem
        v-for="conversation in chat.filteredConversations"
        :key="conversation.id"
        :conversation="conversation"
        :active-id="chat.activeId"
        :is-sending="Boolean(chat.sendingByConversation[conversation.id])"
        :teams="chat.teams"
        :can-assign-team="chat.authStatus === 'authenticated'"
        @select="$emit('select', $event)"
        @rename="chat.renameConversation"
        @archive="$emit('archive', $event)"
        @delete="$emit('delete', $event)"
        @export="handleExport"
        @assign-team="chat.assignConversationToTeam"
      />
    </div>

    <div v-else class="conversation-list-empty">
      <strong>{{ chat.searchQuery ? '没有匹配结果' : '暂无历史对话' }}</strong>
      <span v-if="chat.searchQuery">换个关键词试试</span>
    </div>

    <section
      v-if="chat.authStatus === 'authenticated' && chat.flattenedTeamConversations.length"
      class="archived-section team-conversation-section"
    >
      <button
        class="archived-header"
        type="button"
        :aria-expanded="isTeamOpen"
        @click="isTeamOpen = !isTeamOpen"
      >
        <span class="archive-title">
          <span class="archive-arrow" aria-hidden="true">›</span>
          <span>团队对话</span>
        </span>
        <small>{{ chat.flattenedTeamConversations.length }}</small>
      </button>
      <div v-if="isTeamOpen" class="archived-items">
        <button
          v-for="conversation in chat.flattenedTeamConversations"
          :key="`${conversation.teamId}-${conversation.id}`"
          class="archived-conversation-item team-conversation-item"
          :class="{ active: conversation.id === chat.activeId && chat.isViewingTeamConversation }"
          type="button"
          :title="`${conversation.teamName || '团队'}：${conversation.title}`"
          @click="$emit('selectTeam', conversation.id)"
        >
          <span>{{ conversation.title }}</span>
          <small>{{ conversation.teamName || conversation.owner?.nickname || '团队对话' }}</small>
        </button>
      </div>
    </section>

    <section v-if="chat.archivedConversations.length" class="archived-section">
      <button
        class="archived-header"
        type="button"
        :aria-expanded="isArchiveOpen"
        @click="isArchiveOpen = !isArchiveOpen"
      >
        <span class="archive-title">
          <span class="archive-arrow" aria-hidden="true">›</span>
          <span>已归档</span>
        </span>
        <small>{{ chat.archivedConversations.length }}</small>
      </button>
      <div v-if="isArchiveOpen" class="archived-items">
        <button
          v-for="conversation in chat.archivedConversations"
          :key="conversation.id"
          class="archived-conversation-item"
          type="button"
          :title="`恢复：${conversation.title}`"
          @click="$emit('restore', conversation.id)"
        >
          <span>{{ conversation.title }}</span>
        </button>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ConversationItem from '@/components/ConversationItem.vue'
import { useChatStore } from '@/stores/chat'

const chat = useChatStore()
const isArchiveOpen = ref(false)
const isTeamOpen = ref(true)

defineEmits<{
  select: [id: string]
  selectTeam: [id: string]
  archive: [id: string]
  delete: [id: string]
  restore: [id: string]
}>()

const handleExport = (id: string) => {
  chat.exportConversationAsPDF(id)
}
</script>
