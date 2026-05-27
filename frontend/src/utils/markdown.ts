import MarkdownIt from 'markdown-it'

const escapedBreakTagPattern = /&lt;br\s*\/?&gt;/gi

export function createSafeMarkdownRenderer() {
  const markdown = new MarkdownIt({
    html: false,
    linkify: true,
    breaks: true,
  })
  const defaultTextRule = markdown.renderer.rules.text

  markdown.renderer.rules.text = (...args) => {
    const [tokens, idx] = args
    const tokenContent = tokens[idx]?.content || ''
    const rendered = defaultTextRule
      ? defaultTextRule(...args)
      : markdown.utils.escapeHtml(tokenContent)

    return rendered.replace(escapedBreakTagPattern, '<br />')
  }

  return markdown
}
