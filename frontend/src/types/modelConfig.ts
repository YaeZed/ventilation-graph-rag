export type ModelProvider = 'dashscope' | 'openai' | 'ollama' | 'custom'

export type ModelConfig = {
  provider: ModelProvider
  textModel: string
  textEndpoint: string
  textApiKey: string
  visionModel: string
  visionEndpoint: string
  visionApiKey: string
}

export type ModelPreset = Omit<ModelConfig, 'textApiKey' | 'visionApiKey'> & {
  name: string
}

export const MODEL_PRESETS: Record<ModelProvider, ModelPreset> = {
  dashscope: {
    provider: 'dashscope',
    name: 'DashScope',
    textModel: 'qwen-plus',
    textEndpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    visionModel: 'qwen3.5-omni-plus',
    visionEndpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  },
  openai: {
    provider: 'openai',
    name: 'OpenAI',
    textModel: 'gpt-4o',
    textEndpoint: 'https://api.openai.com/v1',
    visionModel: 'gpt-4o',
    visionEndpoint: 'https://api.openai.com/v1',
  },
  ollama: {
    provider: 'ollama',
    name: 'Ollama',
    textModel: 'qwen2.5:latest',
    textEndpoint: 'http://localhost:11434/v1',
    visionModel: 'llava:latest',
    visionEndpoint: 'http://localhost:11434/v1',
  },
  custom: {
    provider: 'custom',
    name: '自定义',
    textModel: '',
    textEndpoint: '',
    visionModel: '',
    visionEndpoint: '',
  },
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  provider: 'dashscope',
  textModel: MODEL_PRESETS.dashscope.textModel,
  textEndpoint: MODEL_PRESETS.dashscope.textEndpoint,
  textApiKey: '',
  visionModel: MODEL_PRESETS.dashscope.visionModel,
  visionEndpoint: MODEL_PRESETS.dashscope.visionEndpoint,
  visionApiKey: '',
}

export function normalizeModelConfig(config?: Partial<ModelConfig> | null): ModelConfig {
  const provider = isModelProvider(config?.provider) ? config.provider : DEFAULT_MODEL_CONFIG.provider
  const preset = MODEL_PRESETS[provider] || MODEL_PRESETS.dashscope
  return {
    provider,
    textModel: cleanText(config?.textModel, preset.textModel, 120),
    textEndpoint: cleanText(config?.textEndpoint, preset.textEndpoint, 300),
    textApiKey: cleanText(config?.textApiKey, '', 500),
    visionModel: cleanText(config?.visionModel, preset.visionModel, 120),
    visionEndpoint: cleanText(config?.visionEndpoint, preset.visionEndpoint, 300),
    visionApiKey: cleanText(config?.visionApiKey, '', 500),
  }
}

export function presetToModelConfig(provider: ModelProvider, current?: ModelConfig): ModelConfig {
  const preset = MODEL_PRESETS[provider]
  return normalizeModelConfig({
    provider,
    textModel: preset.textModel,
    textEndpoint: preset.textEndpoint,
    textApiKey: current?.textApiKey || '',
    visionModel: preset.visionModel,
    visionEndpoint: preset.visionEndpoint,
    visionApiKey: current?.visionApiKey || '',
  })
}

export function modelConfigForRequest(config?: ModelConfig | null) {
  const normalized = normalizeModelConfig(config)
  return {
    provider: normalized.provider,
    textModel: normalized.textModel,
    textEndpoint: normalized.textEndpoint,
    textApiKey: normalized.textApiKey || undefined,
    visionModel: normalized.visionModel,
    visionEndpoint: normalized.visionEndpoint,
    visionApiKey: normalized.visionApiKey || undefined,
  }
}

function isModelProvider(value: unknown): value is ModelProvider {
  return typeof value === 'string' && value in MODEL_PRESETS
}

function cleanText(value: unknown, fallback: string, maxLength: number) {
  const text = String(value ?? '').trim()
  return (text || fallback).slice(0, maxLength)
}
