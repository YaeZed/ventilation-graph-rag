export type ChatResponse = {
  ok: boolean
  answer?: string
  error?: string
}

export type StreamHandlers = {
  onStatus?: (message: string) => void
  onToken: (content: string) => void
  onError: (message: string) => void
  onDone: () => void
}

const API_BASE = import.meta.env.VITE_API_BASE || ''
const REQUEST_TIMEOUT_MS = 120_000

export async function sendTextMessage(question: string, topK = 5): Promise<string> {
  const controller = createTimeoutController()
  const response = await fetch(`${API_BASE}/api/chat/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
    signal: controller.signal,
  })
  controller.clear()
  return parseAnswer(response)
}

export async function sendImageMessage(question: string, image: File, topK = 5): Promise<string> {
  const controller = createTimeoutController()
  const formData = new FormData()
  formData.append('question', question)
  formData.append('top_k', String(topK))
  formData.append('image', image)

  const response = await fetch(`${API_BASE}/api/chat/upload/`, {
    method: 'POST',
    body: formData,
    signal: controller.signal,
  })
  controller.clear()
  return parseAnswer(response)
}

export async function streamMessage(
  question: string,
  image: File | null,
  handlers: StreamHandlers,
  topK = 5,
): Promise<void> {
  const controller = createTimeoutController()
  const init: RequestInit = { method: 'POST' }

  if (image) {
    const formData = new FormData()
    formData.append('question', question)
    formData.append('top_k', String(topK))
    formData.append('image', image)
    init.body = formData
  } else {
    init.headers = { 'Content-Type': 'application/json' }
    init.body = JSON.stringify({ question, top_k: topK })
  }

  init.signal = controller.signal

  try {
    const response = await fetch(`${API_BASE}/api/chat/stream/`, init)
    if (!response.ok || !response.body) {
      throw new Error(await response.text())
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split(/\r?\n\r?\n/)
      buffer = events.pop() || ''
      for (const eventText of events) {
        dispatchSseEvent(eventText, handlers)
      }
    }

    if (buffer.trim()) {
      dispatchSseEvent(buffer, handlers)
    }
  } finally {
    controller.clear()
  }
}

async function parseAnswer(response: Response): Promise<string> {
  const payload = (await response.json()) as ChatResponse
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.answer || ''
}

function dispatchSseEvent(eventText: string, handlers: StreamHandlers) {
  const eventLine = eventText.split('\n').find((line) => line.startsWith('event:'))
  const dataLines = eventText
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.replace('data:', '').trim())
  const event = eventLine?.replace('event:', '').trim() || 'message'
  const rawData = dataLines.join('\n') || '{}'

  let data: Record<string, string> = {}
  try {
    data = JSON.parse(rawData)
  } catch {
    data = { content: rawData }
  }

  if (event === 'token') handlers.onToken(data.content || '')
  if (event === 'status') handlers.onStatus?.(data.message || '')
  if (event === 'error') {
    handlers.onError(data.message || '流式响应失败')
    handlers.onDone()
  }
  if (event === 'done') handlers.onDone()
}

function createTimeoutController() {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  return {
    signal: controller.signal,
    clear: () => window.clearTimeout(timer),
  }
}
