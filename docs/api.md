# API 说明

默认后端地址：`http://127.0.0.1:8000`

## 用户与会话同步 API

用户模块使用 Django session。通过 Vite 同源代理访问时，前端请求会携带 cookie。所有 `POST` / `PATCH` / `DELETE` 用户模块接口都启用 Django CSRF 校验，前端需要先获取 CSRF cookie，再在写请求中发送 `X-CSRFToken`。

### GET `/api/users/auth/csrf/`

设置 CSRF cookie，并返回当前可用 token。前端启动或首次写请求前调用。

```json
{
  "ok": true,
  "csrfToken": "..."
}
```

### POST `/api/users/auth/register/`

注册并自动登录。

```json
{
  "username": "yaezed",
  "password": "MineWindEye#2026A",
  "nickname": "安全工程师",
  "avatarText": "安",
  "settings": {
    "useStream": true,
    "autoExpandSteps": true,
    "temperature": 0.2
  }
}
```

注册密码使用 Django 内置 password validators，默认最少 8 位、拒绝常见密码、纯数字密码和与用户名过近的密码。

### POST `/api/users/auth/login/`

```json
{
  "username": "yaezed",
  "password": "MineWindEye#2026A"
}
```

登录失败会按 IP + 用户名做服务端限流；默认 5 次失败后锁定 5 分钟并返回 HTTP 429。

### POST `/api/users/auth/logout/`

退出当前 session。

### GET `/api/users/me/`

返回当前登录用户；未登录时 `user` 为 `null`。

### PATCH `/api/users/profile/`

更新昵称、头像文字或偏好设置。

### GET `/api/users/security/events/`

返回当前账号最近 20 条安全事件，用于 `/settings` 的账号安全记录。

```json
{
  "ok": true,
  "events": [
    {
      "id": 1,
      "type": "login_success",
      "username": "yaezed",
      "ipAddress": "127.0.0.1",
      "userAgent": "Mozilla/5.0 ...",
      "metadata": {},
      "createdAt": "2026-05-28T09:40:00+08:00"
    }
  ]
}
```

常见 `type`：`register`、`password_rejected`、`login_success`、`login_failure`、`login_throttled`、`logout`。

### GET `/api/users/teams/`

返回当前登录用户加入的团队列表。

```json
{
  "ok": true,
  "teams": [
    {
      "id": "1",
      "name": "通风一队",
      "description": "日常辨识协作",
      "role": "owner",
      "memberCount": 2,
      "createdAt": "2026-05-28T09:00:00+08:00",
      "updatedAt": "2026-05-28T09:00:00+08:00"
    }
  ]
}
```

### POST `/api/users/teams/`

创建团队。创建者自动成为 `owner`。

```json
{
  "name": "通风一队",
  "description": "日常辨识协作"
}
```

### PATCH `/api/users/teams/<teamId>/`

更新团队名称或备注。仅 `owner` / `admin` 可操作。

### DELETE `/api/users/teams/<teamId>/`

删除团队。仅 `owner` 可操作；团队删除后，会话的 `team` 外键置空。

### GET `/api/users/teams/<teamId>/members/`

返回团队成员列表。仅团队成员可访问。

### GET `/api/users/teams/<teamId>/conversations/`

返回该团队下所有显式归属团队的未归档会话。仅团队成员可访问。前端用于侧边栏“团队对话”只读浏览，不会把其他成员的会话写入当前用户个人会话列表。

```json
{
  "ok": true,
  "team": {
    "id": "1",
    "name": "通风一队",
    "role": "member"
  },
  "conversations": [
    {
      "id": "conversation-client-id",
      "title": "成员B团队会话",
      "teamId": "1",
      "teamName": "通风一队",
      "owner": {
        "id": 2,
        "username": "inspector_b",
        "nickname": "检查员B",
        "avatarText": "检"
      },
      "isOwnedByCurrentUser": false
    }
  ]
}
```

### POST `/api/users/teams/<teamId>/members/`

按用户名添加成员或更新已有成员角色。仅 `owner` / `admin` 可操作；可选角色为 `admin` 或 `member`。

```json
{
  "username": "inspector_a",
  "role": "member"
}
```

### PATCH `/api/users/teams/<teamId>/members/<userId>/`

修改成员角色。仅 `owner` / `admin` 可操作；不能修改 `owner`。

### DELETE `/api/users/teams/<teamId>/members/<userId>/`

移除成员。`owner` / `admin` 可移除成员；普通成员可退出团队；不能移除 `owner`。

### GET `/api/users/conversations/`

返回当前用户的后端会话快照。

### POST `/api/users/conversations/sync/`

批量上行前端会话快照，后端按 `(user, conversation.id)` upsert 后返回完整会话列表。P4 起支持可选 `teamId`；只有当前用户已加入的团队才能被写入，否则按个人会话处理。

前端同步到后端前应剥离消息中的 data URL 图片预览，只上传附件 URL/元数据；浏览器本地缓存可以保留压缩预览作为刷新兜底。

### DELETE `/api/users/conversations/<conversationId>/delete/`

删除当前用户指定会话的后端快照。前端删除已登录账号下的会话时会调用该接口，避免下次同步把已删除会话重新拉回。

### PATCH `/api/users/conversations/<conversationId>/team/`

显式修改当前用户某个会话的团队归属。`teamId` 为空时回到个人空间；非空时要求当前用户是该团队成员。

```json
{
  "teamId": "1"
}
```

当前前端入口在对话三点菜单的“归属团队”子菜单中；`/settings` 不再提供“当前会话归属”控件。

### POST `/api/users/conversations/<conversationId>/attachments/upload/`

上传当前用户某个会话下的图片附件。请求类型为 `multipart/form-data`，需要已登录。

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` / `file` | 是 | 图片文件，最大 8MB |
| `messageClientId` | 否 | 前端消息 ID，用于把附件关联到消息 |

后端序列化会话时会按 `messageClientId` 把 `ConversationAttachment` 回填到对应消息的 `attachments`、`images[]` 和兼容字段 `imageUrl`。前端原图上传失败时可用压缩预览重试，避免刷新后只剩文字消息。

响应：

```json
{
  "ok": true,
  "attachment": {
    "id": "1",
    "kind": "image",
    "messageClientId": "message-id",
    "name": "现场图片.png",
    "url": "http://127.0.0.1:8000/media/conversation_attachments/2026/05/27/a.png",
    "thumbnailUrl": "http://127.0.0.1:8000/media/conversation_attachments/2026/05/27/a.png",
    "size": 1024,
    "mimeType": "image/png",
    "createdAt": "2026-05-27T09:30:00+08:00"
  }
}
```

### GET `/api/users/conversations/<conversationId>/attachments/`

返回当前用户指定会话的附件列表。

### DELETE `/api/users/attachments/<attachmentId>/delete/`

删除当前用户指定附件记录和本地 media 文件。

### GET `/api/users/stats/summary/?days=7&teamId=1`

返回后端会话统计汇总。未传 `teamId` 时统计当前用户个人空间会话；传入 `teamId` 时统计该团队内所有成员显式分配到团队的会话，并要求当前用户是团队成员。未归档会话参与主统计，归档会话只计入 `archivedCount`。`days` 控制趋势天数，范围会在后端限制到 1-90。

响应：

```json
{
  "ok": true,
  "stats": {
    "totalConversations": 3,
    "totalMessages": 12,
    "completedReports": 3,
    "archivedCount": 1,
    "completionRate": 67,
    "activeDays": 2,
    "latestActivity": "2026-05-27T09:30:00+08:00",
    "recentSevenDays": [
      {"date": "2026-05-21", "count": 0},
      {"date": "2026-05-27", "count": 2}
    ],
    "sceneCounts": [
      {"label": "局部通风机与风筒", "count": 2}
    ],
    "hazardCounts": [
      {"label": "高风险", "count": 1, "tone": "danger"}
    ],
    "topHazardLabel": "高风险"
  }
}
```

### GET `/api/users/stats/trends/?days=7&teamId=1`

返回个人空间或团队空间未归档会话的日期趋势数组：

```json
{
  "ok": true,
  "trends": [
    {"date": "2026-05-27", "count": 2}
  ]
}
```

### GET `/api/users/stats/hazards/?teamId=1`

返回个人空间或团队空间未归档会话的风险等级分布：

```json
{
  "ok": true,
  "hazards": [
    {"label": "高风险", "count": 1, "tone": "danger"}
  ]
}
```

## POST `/api/chat/`

文字问答，非流式返回。

请求：

```json
{
  "question": "掘进中的岩巷最低风速要求是多少",
  "top_k": 3,
  "sensor_data": {
    "location": "掘进工作面",
    "source": "manual",
    "entries": [
      {
        "type": "wind_speed",
        "label": "风速",
        "value": 0.12,
        "unit": "m/s",
        "location": "掘进工作面",
        "timestamp": "08:10"
      }
    ]
  }
}
```

`sensor_data` 可选；提供后，生成 prompt 会增加“传感器实测数据”和“交叉验证分析”，并把参数名称、数值、地点用于规程阈值检索。

响应：

```json
{
  "ok": true,
  "answer": "Markdown 格式回答"
}
```

错误响应：

```json
{
  "ok": false,
  "error": "缺少 question/message"
}
```

## POST `/api/chat/upload/`

图片问答，非流式返回。请求类型为 `multipart/form-data`。兼容旧的单图字段 `image`，也支持多图字段 `images` 重复提交。

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` | 否 | 旧版单张现场图片字段 |
| `images` | 否 | 多张现场图片，可重复提交；`image` 和 `images` 至少提供一个 |
| `question` / `message` | 否 | 用户补充问题；默认“请判断图片中的通风安全隐患” |
| `top_k` | 否 | 检索数量，默认 5 |
| `sensor_data` / `sensorData` | 否 | JSON 字符串，结构同 `/api/chat/` 的 `sensor_data` |

响应同 `/api/chat/`。

## POST `/api/chat/stream/`

流式问答。支持 JSON 文字/传感器请求和 `multipart/form-data` 图片/多图片请求。

JSON 请求：

```json
{
  "question": "矿井有害气体最高允许浓度范围是什么",
  "top_k": 5,
  "sensor_data": {
    "location": "回风巷",
    "source": "csv",
    "entries": [
      {"type": "methane", "label": "瓦斯浓度", "value": 0.8, "unit": "%"}
    ],
    "rawCsv": "时间,瓦斯(%)\n08:10,0.8"
  }
}
```

multipart 请求字段同 `/api/chat/upload/`，可同时携带 `images` 与 `sensor_data`。

SSE 响应类型：

```text
Content-Type: text/event-stream; charset=utf-8
```

文字流式事件：

```text
event: status
data: {"message":"started"}

event: token
data: {"content":"1"}

event: done
data: {"message":"completed"}
```

错误事件：

```text
event: error
data: {"message":"错误信息"}
```

前端要求：`token.content` 应追加到当前助手消息；`done` 或流结束时将消息状态置为完成；`error` 时展示错误并结束等待态。

图片流式请求还会额外返回 `step` 事件，用于展示 Agent 当前处理阶段：

```text
event: step
data: {"step":"vision_observe","message":"正在初步观察图片...","data":{}}
```

常见 `step` 值：

| step | 含义 |
|---|---|
| `vision_observe` / `vision_observe_done` | 初步观察图片，提取不确定概念和原始观察 |
| `multi_image_observe` / `multi_image_observe_done` | 多图逐张观察，合并待确认概念 |
| `concept_search` / `concept_search_done` | 检索通风概念定义 |
| `vision_analyze` / `vision_analyze_done` | 结合概念定义复核图片并输出结构化字段 |
| `multi_image_analyze` / `multi_image_analyze_done` | 结合所有图片和概念卡片做联合分析，输出跨图发现 |
| `sensor_compare` / `sensor_compare_done` | 接入传感器数据并准备规程阈值比对 |
| `cypher_match` / `cypher_match_done` | 匹配规程 Cypher 模板和兜底检索 |
| `generating` | 生成 Markdown 辨识报告 |

## GET `/api/chat/vision/scenes/`

返回当前 Qwen-VL 场景分类可选项及每类结构化字段 schema，用于真实图片验证样本标注和字段参考。

响应：

```json
{
  "ok": true,
  "scenes": [
    {
      "id": "local_ventilation",
      "name": "局部通风机与风筒",
      "schema": {
        "facility_type": {"type": "string"},
        "has_backup_fan": {"type": "boolean"}
      },
      "aliases": ["局部通风机"]
    }
  ]
}
```

## POST `/api/chat/vision/evaluate/`

真实图片识别精度验证接口。请求类型为 `multipart/form-data`，适合一次上传多张现场样图和人工标注，后端会执行 Qwen3.5-Omni 观察、概念检索、概念增强分析，并统计场景准确率、字段准确率、综合准确率和 Cypher 模板检索命中率。

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `metadata` | 是 | JSON 字符串，包含 `samples` 数组 |
| `image_0`、`image_1`... | 是 | 与 `metadata.samples` 顺序对应的图片文件 |

`metadata` 示例：

```json
{
  "samples": [
    {
      "id": "sample-1",
      "question": "请识别图片中的局部通风设施",
      "expected_scene_id": "local_ventilation",
      "expected_fields": {
        "facility_type": "局部通风机",
        "has_backup_fan": true
      }
    }
  ]
}
```

响应：

```json
{
  "ok": true,
  "summary": {
    "total_samples": 1,
    "scene_accuracy": 1.0,
    "field_accuracy": 0.5,
    "overall_accuracy": 0.75,
    "retrieval_hit_rate": 1.0
  },
  "samples": [],
  "markdown_report": "# 真实图片识别精度验证报告"
}
```

## curl 示例

```bash
curl -X POST http://127.0.0.1:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"掘进中的岩巷最低风速要求是多少\",\"top_k\":3}"
```

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream/ \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"矿井有害气体最高允许浓度范围是什么\",\"top_k\":5}"
```
