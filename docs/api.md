# API 说明

默认后端地址：`http://127.0.0.1:8000`

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

事件：

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

