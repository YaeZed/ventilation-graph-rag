# Plan: Agent 式用户模块

> 状态：第一阶段、P1 后端图片附件存储、P2 统计面板增强、P3 后端账号级统计聚合已完成；下一阶段优先做团队级权限/统计，或进入生产账号安全硬化。本文保留设计思路、实现边界、验收记录和后续路线图；运行和接手优先看 `docs/architecture.md`、`docs/api.md`、`docs/runbook.md` 和 `docs/status.md`。

## Context

ventilation-graph-rag 的辨识引擎已完成（VL 两轮识别 + 概念知识层 + Agent 步骤展示 + 灵活生成）。现在需要构建支撑对话式交互的用户模块，对标现代 AI Agent 产品的用户体验。

实施前前端已有基础：单个对话窗口（`HomeView.vue`）、Pinia store（`chat.ts`）、图片上传、SSE 流式接收、Agent 步骤时间线。缺少对话管理、持久化、搜索、导出等用户层面的功能。

当前第一阶段已补齐前端多会话和后端账号同步。用户模块采用本地优先设计：未登录用户用浏览器 `localStorage` 保存会话和偏好；登录后通过 Django session API 同步到后端 `users.ConversationRecord`。

账号切换规则：已有账号登录时只加载该账号作用域下的本地缓存和后端会话，不把当前浏览器里的游客/其他账号会话自动上传到该账号；注册新账号时才把游客本地会话作为首次数据迁移，迁移成功后清理游客缓存。

当前存储策略：

- 游客本地缓存：`localStorage` key `ventilation-graph-rag:user-module:v2:guest`。
- 登录用户本地缓存：`localStorage` key `ventilation-graph-rag:user-module:v2:user:<userId>`。
- 登录用户后端数据：Django 内置 `auth_user`、`users.UserProfile`、`users.ConversationRecord`，当前开发数据库为 `web_backend/db.sqlite3`。
- 图片附件策略：游客仍压缩为 data URL 写入本地会话快照；登录用户图片写入后端 `users.ConversationAttachment`，会话快照只保存附件 URL/元数据。

## 功能分层

### 第一层：必须有的（答辩可用性底线）

| 功能 | 说明 | 当前状态 |
|------|------|---------|
| 对话列表 | 左侧边栏显示所有历史对话，按时间排序 | 已完成 |
| 新建对话 | 点击按钮新建空对话，切到新对话 | 已完成 |
| 对话重命名 | 双击标题改名，默认用第一条提问做标题 | 已完成 |
| 对话删除 | 右键/按钮删除对话 | 已完成 |
| 对话持久化 | 刷新页面不丢对话，切换对话隔离 | 已完成 |
| 简易身份 | localStorage 存昵称+头像，无需注册登录 | 已完成 |

### 第二层：让体验丝滑的

| 功能 | 说明 |
|------|------|
| 对话搜索 | 顶部搜索框，按场景名/隐患类型/日期搜历史对话 |
| 对话导出 PDF | 单个辨识报告导出为 PDF（图片 + 四段式报告） |
| 空状态引导 | 无对话时不展示空白区，展示示例场景卡片和引导文字 |
| 对话归档 | 不删除但隐藏，侧边栏保持清爽 |

### 第三层：锦上添花的

| 功能 | 说明 |
|------|------|
| 辨识统计面板 | 累计辨识数、隐患类型分布饼图、近一周趋势 |
| 快捷场景入口 | 首页预设场景卡片，一键切换对应检查模式 |
| 偏好设置 | 默认 temperature、是否自动展开步骤时间线 |
| 全量导出 JSON | 导出全量辨识记录为结构化 JSON 文件 |

当前实施结果：**第一层已完成 + 第二层搜索、归档和导出已完成 + 第三层统计面板和偏好设置已可用 + 登录/注册与后端同步已完成 + 登录用户图片附件后端存储已完成 + 登录用户统计已接入后端聚合**。

## 架构设计

### 前端路由

```
/                          → 重定向到 /chat
/chat                      → 主对话页（左侧对话列表 + 右侧对话区）
/chat/:conversationId      → 特定对话
/stats                     → 辨识统计面板与 JSON 导出
/settings                  → 偏好设置、账号同步状态、退出登录
/login                     → 登录
/register                  → 注册
```

### 组件树

```
MainLayout.vue
├── Sidebar.vue                     # 侧边栏
│   ├── UserMiniCard.vue            # 用户头像+昵称
│   ├── 内联搜索框                   # 对话搜索
│   ├── ConversationList.vue        # 对话列表
│   │   └── ConversationItem.vue    # 单个对话项（可重命名、删除）
│   ├── NewChatButton.vue           # 新建对话
│   └── StatsEntryButton.vue        # 统计入口
└── RouterView                      # 主区域
    ├── HomeView.vue                # 对话视图
    ├── EmptyState.vue              # 空状态引导
    ├── StatsView.vue               # 统计面板和 JSON 导出
    ├── SettingsView.vue            # 偏好设置和账号同步
    ├── LoginView.vue               # 登录页
    └── RegisterView.vue            # 注册页
```

### 数据流

```
Pinia Store (chat.ts)               localStorage
┌─────────────────────┐            ┌──────────────┐
│ conversations: Map  │ ←──持久化──→ │ conversations│
│ activeId: string    │            │ settings     │
│ isSending: bool     │            │ userProfile  │
│ streamingTargetId   │            └──────────────┘
│                     │
│ actions:            │
│  createConversation │
│  deleteConversation │
│  renameConversation │
│  searchConversations│
│  exportAsPDF        │
│  loadFromStorage    │
│  saveToStorage      │
│  syncConversations  │
│  login/register     │
└─────────────────────┘
```

### 对话数据结构演进

```typescript
interface Conversation {
  id: string                    // uuid
  title: string                 // 默认 "新建辨识"
  messages: ChatMessage[]       // 现有消息列表
  createdAt: string             // ISO timestamp
  updatedAt: string             // 最后活跃时间
  sceneType?: string            // 最后一次辨识场景
  hazardLevel?: string          // 最后一次风险等级
  isArchived?: boolean          // 归档标记
  previewImageUrl?: string      // 首张上传图片缩略图
}
```

登录用户图片已从 data URL 演进为附件引用，当前结构为：

```typescript
interface ChatAttachment {
  id: string
  kind: 'image'
  name: string
  url: string                  // 后端 media URL 或对象存储签名 URL
  thumbnailUrl?: string          // 当前等同原图 URL，后续可替换为真实缩略图
  size: number
  mimeType: string
  createdAt: string
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  attachments?: ChatAttachment[]
  createdAt: string
  status?: 'streaming' | 'done' | 'error'
}

interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  previewAttachmentId?: string
  createdAt: string
  updatedAt: string
}
```

## 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `frontend/src/stores/chat.ts` | **重写** | 多对话管理 + localStorage 持久化 + 搜索/导出 |
| `frontend/src/views/HomeView.vue` | **改造** | 适配多对话 store，空状态展示 |
| `frontend/src/components/Sidebar.vue` | **新建** | 侧边栏容器 |
| `frontend/src/components/ConversationList.vue` | **新建** | 对话列表 + 搜索 |
| `frontend/src/components/ConversationItem.vue` | **新建** | 单个对话项（重命名/删除） |
| `frontend/src/components/EmptyState.vue` | **新建** | 空状态引导页 |
| `frontend/src/components/UserMiniCard.vue` | **新建** | 用户信息卡片 |
| `frontend/src/router/index.ts` | **新建/改造** | 路由配置 |
| `frontend/src/layout/MainLayout.vue` | **新建/改造** | 侧边栏 + 主区域布局 |
| `frontend/src/views/LoginView.vue` | **新建** | 登录页 |
| `frontend/src/views/RegisterView.vue` | **新建** | 注册页 |
| `frontend/src/views/SettingsView.vue` | **新建** | 偏好设置和账号同步 |
| `frontend/src/views/StatsView.vue` | **新建** | 统计面板和 JSON 导出 |
| `frontend/src/api/users.ts` | **新建** | 用户和会话同步 API client |
| `web_backend/users/` | **新建/扩展** | Django session 用户模块、会话快照和图片附件 |
| `frontend/src/assets/main.css` | **追加** | 侧边栏、空状态、对话项样式 |

## 验证方式

1. **对话 CRUD**：新建 3 个对话，分别发消息，关闭浏览器重新打开，对话仍在
2. **对话隔离**：在对话 A 发送消息后切到对话 B，再切回 A，消息完整
3. **搜索**：搜索"局部通风"，过滤出相关对话
4. **导出 PDF**：点击导出，下载的 PDF 包含图片 + 四段式报告
5. **空状态**：首次打开或删除所有对话后，显示引导页而非空白

## 已完成验收记录

- `web_backend/manage.py check` 通过。
- `web_backend/manage.py makemigrations --check --dry-run` 通过。
- Django `users` 迁移已执行，`users.0001_initial` 已应用。
- 用户 API 烟测通过：注册、读取当前用户、同步会话、删除远端会话快照。
- `node node_modules/vue-tsc/bin/vue-tsc.js --build` 通过。
- `node node_modules/vite/bin/vite.js build` 通过。

## P1 完成记录：后端图片附件存储

P1 已完成。原因：图片以 data URL 存在 `localStorage` 和会话 JSON 里，适合游客演示，但不适合登录用户长期使用。后端附件引用解决了三个直接问题：浏览器容量容易满、跨设备同步成本高、后端会话快照 JSON 变大。

完成内容：

- 登录用户上传图片后，后端保存图片文件，并返回附件 ID、URL、缩略图 URL、文件名、大小和 MIME 类型。
- 前端消息保存附件引用，不再把完整 data URL 写进登录用户会话快照。
- PDF 导出通过附件 URL 拉取图片展示。
- 删除会话或删除附件时，后端清理附件文件。
- 游客模式仍保留 data URL，避免未登录用户被强制要求创建账号。

当前模型：

```python
class ConversationAttachment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    conversation = models.ForeignKey(ConversationRecord, on_delete=models.CASCADE, related_name="attachments")
    message_client_id = models.CharField(max_length=80, blank=True)
    file = models.FileField(upload_to="conversation_attachments/%Y/%m/%d/")
    original_name = models.CharField(max_length=160)
    mime_type = models.CharField(max_length=80)
    size = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
```

当前 API：

| API | 作用 |
|---|---|
| `POST /api/users/conversations/<conversationId>/attachments/upload/` | 上传图片附件并返回引用，最大 8MB |
| `GET /api/users/conversations/<conversationId>/attachments/` | 获取会话附件列表 |
| `DELETE /api/users/attachments/<attachmentId>/delete/` | 删除附件 |

验收结果：

1. 登录用户上传图片后刷新页面，图片仍可通过附件 URL 预览。
2. 同一账号同步会话时保留附件引用和图片文件名。
3. 登录用户 `localStorage` 中不再保存大体积 data URL；只保存附件 URL/元数据。
4. 导出 PDF 仍包含图片和 Markdown 渲染后的报告。
5. 删除会话或附件后，对应附件文件会被删除。
6. 已验证：`web_backend/manage.py check`、`makemigrations --check --dry-run`、附件上传/列表/删除烟测、`vue-tsc --build`、`vite build`。

剩余边界：

- 当前 `thumbnailUrl` 等同原图 URL，还没有生成真实缩略图。
- 当前 media 文件存储在开发期本地 `web_backend/media/`；生产部署需要对象存储或明确静态资源托管策略。
- 游客仍使用 data URL，因为游客没有后端账号所有者。

## P2 完成记录：统计面板增强

P2 已按前端本地统计落地，范围限定在 `/stats` 和 Pinia store：

- `frontend/src/stores/chat.ts`
  - `ChatStats` 增加 `completionRate`、`activeDays`、`hazardCounts`、`topHazardLabel`。
  - 统计只基于未归档会话；归档数单独展示。
  - 完成率按“至少有一份完成报告的有效会话 / 有效会话总数”计算。
  - 风险等级从 `Conversation.hazardLevel` 归一化为 `高风险`、`中风险`、`低风险`、`未分级` 或原始标签。
- `frontend/src/views/StatsView.vue`
  - 增加完成概览、风险等级分布、近 7 天趋势、场景分布。
  - 保留 JSON 导出入口。
- `frontend/src/assets/main.css`
  - 增加完成率圆环、风险分布条、趋势数值标签和移动端单列适配。

验收结果：

1. `/stats` 能在游客和登录用户各自的会话作用域内计算统计，不跨用户混数据。
2. 完成率、近 7 天活跃天数、风险分布、场景分布和归档数显示正常。
3. JSON 导出仍保留。
4. 已验证：`vue-tsc --build`、`vite build`。

P2 不新增后端 API，后端聚合已在 P3 完成。

## P3 完成记录：后端账号级统计聚合

P3 已完成账号级后端统计聚合。当前统计来源分层：

- 游客：仍使用前端 Pinia 本地统计。
- 登录用户：`/stats` 进入页面后调用 `GET /api/users/stats/summary/?days=7`，从后端 `ConversationRecord` 聚合。
- 后端不可用或统计接口失败时：前端回退到本地统计，并在页面右上角显示回退状态。

当前 API：

| API | 作用 |
|---|---|
| `GET /api/users/stats/summary/` | 当前用户统计汇总 |
| `GET /api/users/stats/trends/?days=7` | 趋势统计 |
| `GET /api/users/stats/hazards/` | 风险/隐患类型分布 |

完成内容：

- 后端新增统计聚合工具函数，直接从当前登录用户的 `ConversationRecord` 读取，不新增表和 migration。
- 统计口径与 P2 前端保持一致：未归档会话参与主统计，归档会话只计入 `archivedCount`；完成率按“至少有一份完成报告的有效会话 / 有效会话总数”计算。
- 前端新增 `fetchUserStatsSummary()` 和 `refreshStats()`；登录后、同步后、进入 `/stats` 时刷新后端统计。
- `/stats` 右上角显示统计来源：本地统计、后端统计、后端统计同步中或回退本地统计。

验收结果：

1. 后端 stats summary/trends/hazards 烟测通过。
2. `web_backend/manage.py check` 通过。
3. `vue-tsc --build` 和 `vite build` 通过。

剩余边界：

- 当前 P3 是单账号聚合，不包含团队空间、角色权限或跨用户统计。
- 后端统计读取 JSON 快照，未来数据量上来后可考虑增量统计表或缓存。

## 下一阶段路线图

### P4：团队级权限与统计

当前账号模块只有单用户数据所有权。团队级统计需要先定义团队、成员、角色和可见范围，再做跨用户聚合。

建议模型：

| 模型 | 作用 |
|---|---|
| `Team` | 团队/项目空间 |
| `TeamMembership` | 用户、团队和角色关系 |
| `ConversationRecord.team` | 可选团队归属 |

### P5：生产级账号安全

当前账号模块使用 Django session 和 SQLite，定位是本地演示/开发。生产部署前需要补齐：

- CSRF/CORS/session cookie 策略。
- 密码复杂度和重试限制。
- 权限审计和会话过期策略。
- 反向代理后的 secure cookie 配置。
- 数据库从 SQLite 切换到生产数据库。
