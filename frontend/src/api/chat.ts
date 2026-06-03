import type { SensorData } from '@/types/multimodal'
import { modelConfigForRequest, type ModelConfig } from '@/types/modelConfig'
import { friendlyHttpError } from '@/api/errors'

export type ChatResponse = {
  ok: boolean
  answer?: string
  error?: string
}

export type StreamStepEvent = {
  step: string
  message: string
  data?: Record<string, unknown>
}

export type ModelTestPart = {
  ok: boolean
  model: string
  endpoint: string
  message: string
}

export type ModelTestResult = {
  ok: boolean
  results: {
    text: ModelTestPart
    vision: ModelTestPart
  }
}

export type StreamHandlers = {
  onStatus?: (message: string) => void
  onStep?: (step: StreamStepEvent) => void
  onToken: (content: string) => void
  onError: (message: string) => void
  onDone: () => void
}

const API_BASE = import.meta.env.VITE_API_BASE || ''
const REQUEST_TIMEOUT_MS = 120_000
const IMAGE_REQUEST_TIMEOUT_MS = 600_000
const STREAM_TIMEOUT_MS = 1_800_000
const MODEL_TEST_TIMEOUT_MS = 60_000

export async function sendTextMessage(
  question: string,
  topK = 5,
  sensorData?: SensorData | null,
  modelConfig?: ModelConfig | null,
): Promise<string> {
  const controller = createTimeoutController(REQUEST_TIMEOUT_MS)
  try {
    const modelPayload = modelConfig ? modelConfigForRequest(modelConfig) : undefined
    const response = await fetch(`${API_BASE}/api/chat/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        top_k: topK,
        sensor_data: sensorData || undefined,
        model_config: modelPayload,
      }),
      signal: controller.signal,
    })
    return parseAnswer(response)
  } catch (exc) {
    throw new Error(toFriendlyRequestError(exc, REQUEST_TIMEOUT_MS))
  } finally {
    controller.clear()
  }
}

export async function sendImageMessage(
  question: string,
  images: File | File[],
  topK = 5,
  sensorData?: SensorData | null,
  modelConfig?: ModelConfig | null,
): Promise<string> {
  const controller = createTimeoutController(IMAGE_REQUEST_TIMEOUT_MS)
  const formData = new FormData()
  const imageFiles = normalizeImages(images)
  formData.append('question', question)
  formData.append('top_k', String(topK))
  if (sensorData) formData.append('sensor_data', JSON.stringify(sensorData))
  if (modelConfig) formData.append('model_config', JSON.stringify(modelConfigForRequest(modelConfig)))
  imageFiles.forEach((image) => {
    formData.append('images', image)
  })
  if (imageFiles[0]) formData.append('image', imageFiles[0])

  try {
    const response = await fetch(`${API_BASE}/api/chat/upload/`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
    return parseAnswer(response)
  } catch (exc) {
    throw new Error(toFriendlyRequestError(exc, IMAGE_REQUEST_TIMEOUT_MS))
  } finally {
    controller.clear()
  }
}

export async function streamMessage(
  question: string,
  images: File | File[] | null,
  handlers: StreamHandlers,
  topK = 5,
  sensorData?: SensorData | null,
  modelConfig?: ModelConfig | null,
): Promise<void> {
  const controller = createTimeoutController(STREAM_TIMEOUT_MS)
  const init: RequestInit = { method: 'POST', signal: controller.signal }
  const imageFiles = normalizeImages(images)

  if (imageFiles.length) {
    const formData = new FormData()
    formData.append('question', question)
    formData.append('top_k', String(topK))
    if (sensorData) formData.append('sensor_data', JSON.stringify(sensorData))
    if (modelConfig) formData.append('model_config', JSON.stringify(modelConfigForRequest(modelConfig)))
    imageFiles.forEach((image) => {
      formData.append('images', image)
    })
    if (imageFiles[0]) formData.append('image', imageFiles[0])
    init.body = formData
  } else {
    init.headers = { 'Content-Type': 'application/json' }
    init.body = JSON.stringify({
      question,
      top_k: topK,
      sensor_data: sensorData || undefined,
      model_config: modelConfig ? modelConfigForRequest(modelConfig) : undefined,
    })
  }

  try {
    const response = await fetch(`${API_BASE}/api/chat/stream/`, init)
    if (!response.ok || !response.body) {
      throw new Error(friendlyHttpError(response, await response.text()))
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
  } catch (exc) {
    throw new Error(toFriendlyRequestError(exc, STREAM_TIMEOUT_MS))
  } finally {
    controller.clear()
  }
}

export async function testModelConfig(modelConfig: ModelConfig): Promise<ModelTestResult> {
  const controller = createTimeoutController(MODEL_TEST_TIMEOUT_MS)
  try {
    const response = await fetch(`${API_BASE}/api/chat/model/test/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_config: modelConfigForRequest(modelConfig) }),
      signal: controller.signal,
    })
    const rawText = await response.text()
    let payload: ModelTestResult & { error?: string }
    try {
      payload = JSON.parse(rawText) as ModelTestResult & { error?: string }
    } catch {
      throw new Error(friendlyHttpError(response, rawText))
    }
    if (!response.ok) {
      throw new Error(payload.error || friendlyHttpError(response, rawText))
    }
    return payload
  } catch (exc) {
    throw new Error(toFriendlyRequestError(exc, MODEL_TEST_TIMEOUT_MS))
  } finally {
    controller.clear()
  }
}

function normalizeImages(images: File | File[] | null): File[] {
  if (!images) return []
  return Array.isArray(images) ? images.filter(Boolean) : [images]
}

async function parseAnswer(response: Response): Promise<string> {
  const rawText = await response.text()
  let payload: ChatResponse
  try {
    payload = JSON.parse(rawText) as ChatResponse
  } catch {
    if (!response.ok) {
      throw new Error(friendlyHttpError(response, rawText))
    }
    throw new Error('后端返回了无法解析的响应')
  }

  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`)
  }
  return payload.answer || ''
}

function dispatchSseEvent(eventText: string, handlers: StreamHandlers) {
  const lines = eventText.split('\n')
  const eventLine = lines.find((line) => line.startsWith('event:'))
  const dataLines = lines
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.replace('data:', '').trim())
  const event = eventLine?.replace('event:', '').trim() || 'message'
  const rawData = dataLines.join('\n') || '{}'

  let data: Record<string, unknown> = {}
  try {
    data = JSON.parse(rawData) as Record<string, unknown>
  } catch {
    data = { content: rawData }
  }

  if (event === 'token') handlers.onToken(String(data.content || ''))
  if (event === 'status') handlers.onStatus?.(String(data.message || ''))
  if (event === 'step') {
    handlers.onStep?.({
      step: String(data.step || 'step'),
      message: String(data.message || ''),
      data: (data.data as Record<string, unknown> | undefined) || {},
    })
  }
  if (event === 'error') {
    handlers.onError(String(data.message || '流式响应失败'))
    handlers.onDone()
  }
  if (event === 'done') handlers.onDone()
}

function toFriendlyRequestError(exc: unknown, timeoutMs: number) {
  const message = exc instanceof Error ? exc.message : String(exc || '')
  const lowerMessage = message.toLowerCase()
  const isAbort =
    (exc instanceof DOMException && exc.name === 'AbortError') ||
    lowerMessage.includes('aborted') ||
    lowerMessage.includes('bodystreambuffer')

  if (isAbort) {
    const minutes = Math.round(timeoutMs / 60_000)
    return `请求超时或被浏览器中止：已等待约 ${minutes} 分钟。图片识别可能仍在后端处理中，请确认后端服务、DashScope 额度和模型配置后重试。`
  }

  if (lowerMessage.includes('failed to fetch') || lowerMessage.includes('networkerror')) {
    return '无法连接后端服务：请确认 Django 服务正在 http://127.0.0.1:8000 运行。'
  }

  return message || '请求失败'
}

function createTimeoutController(timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  return {
    signal: controller.signal,
    clear: () => window.clearTimeout(timer),
  }
}
