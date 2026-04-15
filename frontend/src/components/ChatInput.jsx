import { useState, useRef, useEffect } from 'react'

function SendIcon({ spinning }) {
  if (spinning) {
    return (
      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
      </svg>
    )
  }
  return (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
      <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
    </svg>
  )
}

export default function ChatInput({ onSendMessage, isLoading, selectedDocs }) {
  const [text, setText] = useState('')
  const [compareMode, setCompareMode] = useState(false)
  const textareaRef = useRef(null)

  const canCompare = selectedDocs && selectedDocs.length > 1
  const charCount = text.length
  const showCharCount = charCount > 200
  const isEmpty = text.trim().length === 0
  const canSend = !isLoading && !isEmpty

  // Auto-resize textarea up to 4 rows.
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    const lineHeight = 24 // px — matches leading-6
    const maxHeight = lineHeight * 4 + 24 // 4 rows + padding
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
  }, [text])

  // Reset compare mode when selection drops below 2.
  useEffect(() => {
    if (!canCompare) setCompareMode(false)
  }, [canCompare])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (canSend) submit()
    }
  }

  function submit() {
    const trimmed = text.trim()
    if (!trimmed) return
    onSendMessage(trimmed, compareMode ? selectedDocs : null)
    setText('')
    // Reset height after clearing.
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  return (
    <div className="border-t border-gray-800 bg-gray-950 px-4 pt-3 pb-4">
      {/* Compare mode toggle — only visible when 2+ docs are selected */}
      {canCompare && (
        <div className="mb-2 flex items-center gap-2">
          <button
            type="button"
            onClick={() => setCompareMode((v) => !v)}
            className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none
              ${compareMode ? 'bg-blue-600' : 'bg-gray-700'}`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform
                ${compareMode ? 'translate-x-4' : 'translate-x-0'}`}
            />
          </button>
          <span className="text-xs text-gray-400">
            Compare across{' '}
            <span className="text-white font-medium">{selectedDocs.length} documents</span>
          </span>
        </div>
      )}

      {/* Input row */}
      <div className={`flex items-end gap-2 rounded-xl border bg-gray-900 px-3 py-2 transition-colors
        ${isLoading ? 'border-gray-800' : 'border-gray-700 focus-within:border-blue-600'}`}>

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          placeholder="Ask a question about your documents…"
          rows={1}
          className="flex-1 resize-none bg-transparent text-sm text-white placeholder-gray-500 outline-none leading-6 py-1 disabled:opacity-50"
        />

        <button
          type="button"
          onClick={submit}
          disabled={!canSend}
          className={`flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-lg transition-colors mb-0.5
            ${canSend
              ? 'bg-blue-600 hover:bg-blue-500 text-white'
              : 'bg-gray-800 text-gray-600 cursor-not-allowed'}`}
        >
          <SendIcon spinning={isLoading} />
        </button>
      </div>

      {/* Footer row: char count + hint */}
      <div className="mt-1.5 flex items-center justify-between px-1">
        <p className="text-xs text-gray-600">
          Enter to send · Shift+Enter for new line
        </p>
        {showCharCount && (
          <span className={`text-xs tabular-nums ${charCount > 2000 ? 'text-red-400' : 'text-gray-500'}`}>
            {charCount.toLocaleString()} chars
          </span>
        )}
      </div>
    </div>
  )
}
