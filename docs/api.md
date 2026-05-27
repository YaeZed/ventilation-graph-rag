# API 说明

默认后端地址：`http://127.0.0.1:8000`

## 用户与会话同步 API

用户模块使用 Django session。通过 Vite 同源代理访问时，前端请求会携带 cookie；跨域部署时需要额外配置 CSRF/CORS/session cookie。

### POST `/api/users/auth/register/`

注册并自动登录。

```json
{
  "username": "yaezed",
  "password": "160722",
  "nickname": "安全工程师",
  "avatarText": "安",
  "settings": {
    "useStream": true,
    "autoExpandSteps": true,
    "temperature": 0.2
  }
}
```

### POST `/api/users/auth/login/`

```json
{
  "username": "yaezed",
  "password": "160722"
}
```

### POST `/api/users/auth/logout/`

退出当前 session。

### GET `/api/users/me/`

返回当前登录用户；未登录时 `user` 为 `null`。

### PATCH `/api/users/profile/`

更新昵称、头像文字或偏好设置。

### GET `/api/users/conversations/`

返回当前用户的后端会话快照。

### POST `/api/users/conversations/sync/`

批量上行前端会话快照，后端按 `(user, conversation.id)` upsert 后返回完整会话列表。

### DELETE `/api/users/conversations/<conversationId>/delete/`

删除当前用户指定会话的后端快照。前端删除已登录账号下的会话时会调用该接口，避免下次同步把已删除会话重新拉回。

### POST `/api/users/conversations/<conversationId>/attachments/upload/`

上传当前用户某个会话下的图片附件。请求类型为 `multipart/form-data`，需要已登录。

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` / `file` | 是 | 图片文件，最大 8MB |
| `messageClientId` | 否 | 前端消息 ID，用于把附件关联到消息 |

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

## POST `/api/chat/`

文字问答，非流式返回。

请求：

```json
{
  "question": "掘进中的岩巷最低风速要求是多少",
  "top_k": 3
}
```

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

图片问答，非流式返回。请求类型为 `multipart/form-data`。

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `image` | 是 | 现场图片文件 |
| `question` / `message` | 否 | 用户补充问题；默认“请判断图片中的通风安全隐患” |
| `top_k` | 否 | 检索数量，默认 5 |

响应同 `/api/chat/`。

## POST `/api/chat/stream/`

流式问答。支持 JSON 文字请求和 `multipart/form-data` 图片请求。

JSON 请求：

```json
{
  "question": "矿井有害气体最高允许浓度范围是什么",
  "top_k": 5
}
```

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
| `concept_search` / `concept_search_done` | 检索通风概念定义 |
| `vision_analyze` / `vision_analyze_done` | 结合概念定义复核图片并输出结构化字段 |
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
