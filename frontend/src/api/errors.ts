const MAX_ERROR_TEXT_LENGTH = 240

export function friendlyHttpError(response: Response, rawText: string) {
  const statusText = response.status ? `HTTP ${response.status}` : '请求失败'
  const text = rawText.trim()
  if (!text) return statusText

  if (looksLikeHtml(response, text)) {
    const plainText = htmlToText(text)
    const lowerText = plainText.toLowerCase()

    if (response.status === 403 && lowerText.includes('origin checking failed')) {
      const origin = extractCsrfOrigin(plainText)
      const originText = origin ? `${origin} ` : '当前前端地址 '
      return `CSRF 校验失败：${originText}未加入后端信任来源。请重启 Django 后端，或在 DJANGO_CSRF_TRUSTED_ORIGINS 中加入当前前端地址。`
    }

    if (response.status === 403 && lowerText.includes('csrf')) {
      return 'CSRF 校验失败：请刷新页面后重试；如果前端端口变化，请把当前地址加入 DJANGO_CSRF_TRUSTED_ORIGINS 并重启后端。'
    }

    const title = extractTagText(text, 'title') || extractTagText(text, 'h1')
    return `${statusText}：${compactErrorText(title || '后端返回了 HTML 错误页，请查看后端日志。')}`
  }

  return compactErrorText(text) || statusText
}

function looksLikeHtml(response: Response, text: string) {
  const contentType = response.headers.get('content-type')?.toLowerCase() || ''
  return contentType.includes('text/html') || /^<!doctype html/i.test(text) || /^<html[\s>]/i.test(text)
}

function htmlToText(text: string) {
  return text
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&amp;/gi, '&')
    .replace(/&quot;/gi, '"')
    .replace(/&#x27;|&#39;/gi, "'")
    .replace(/\s+/g, ' ')
    .trim()
}

function extractTagText(text: string, tagName: string) {
  const match = text.match(new RegExp(`<${tagName}[^>]*>([\\s\\S]*?)<\\/${tagName}>`, 'i'))
  return match ? htmlToText(match[1] || '') : ''
}

function extractCsrfOrigin(text: string) {
  const match = text.match(/Origin checking failed\s*-\s*(\S+)\s+does not match/i)
  return match?.[1] || ''
}

function compactErrorText(text: string) {
  const compact = text.replace(/\s+/g, ' ').trim()
  if (compact.length <= MAX_ERROR_TEXT_LENGTH) return compact
  return `${compact.slice(0, MAX_ERROR_TEXT_LENGTH)}...`
}
