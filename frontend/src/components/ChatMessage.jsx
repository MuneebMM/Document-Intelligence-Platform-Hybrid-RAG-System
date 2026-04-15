import ReactMarkdown from 'react-markdown'

function TypingDots() {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full bg-gray-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  )
}

function CitationBadge({ citation }) {
  const label = `${citation.filename}  •  Page ${citation.page_number}`
  return (
    <span
      title={citation.quote || ''}
      className="inline-flex items-center gap-1 rounded-full bg-gray-700 px-2.5 py-0.5 text-xs text-gray-300 cursor-default hover:bg-gray-600 hover:text-white transition-colors"
    >
      <svg className="w-3 h-3 flex-shrink-0 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      {label}
    </span>
  )
}

function ConfidenceBar({ confidence }) {
  if (!confidence) return null

  const score = confidence.confidence_score ?? 0
  const pct = Math.round(score * 100)

  const barColor =
    score >= 0.8 ? 'bg-green-500' :
    score >= 0.5 ? 'bg-yellow-500' :
    'bg-red-500'

  const labelColor =
    score >= 0.8 ? 'text-green-400' :
    score >= 0.5 ? 'text-yellow-400' :
    'text-red-400'

  return (
    <div className="mt-3 pt-3 border-t border-gray-700">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500">Confidence</span>
        <span className={`text-xs font-medium ${labelColor}`}>{pct}%</span>
      </div>
      <div className="w-full h-1.5 rounded-full bg-gray-700 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {confidence.warning && (
        <p className="mt-1.5 text-xs text-yellow-400 flex items-start gap-1">
          <span className="flex-shrink-0">⚠</span>
          <span>{confidence.warning}</span>
        </p>
      )}
    </div>
  )
}

function AssistantMessage({ message }) {
  const hasCitations = message.citations && message.citations.length > 0

  return (
    <div className="flex items-start gap-3 max-w-3xl">
      {/* Avatar */}
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-indigo-600 flex items-center justify-center mt-0.5">
        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
          <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
        </svg>
      </div>

      <div className="flex-1 rounded-2xl rounded-tl-sm bg-gray-800 px-4 py-3 text-sm text-gray-100 shadow-sm">
        {message.isLoading ? (
          <TypingDots />
        ) : (
          <>
            {/* Answer */}
            <div className="prose prose-invert prose-sm max-w-none
              prose-p:my-1 prose-p:leading-relaxed
              prose-headings:text-gray-100 prose-headings:font-semibold
              prose-strong:text-white
              prose-code:bg-gray-700 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs
              prose-ul:my-1 prose-li:my-0.5
              prose-blockquote:border-gray-600 prose-blockquote:text-gray-400">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>

            {/* Citations */}
            {hasCitations && (
              <div className="mt-3 pt-3 border-t border-gray-700">
                <p className="text-xs text-gray-500 mb-2 flex items-center gap-1">
                  <span>📎</span> Sources
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {message.citations.map((c, i) => (
                    <CitationBadge key={c.chunk_id || i} citation={c} />
                  ))}
                </div>
              </div>
            )}

            {/* Confidence */}
            <ConfidenceBar confidence={message.confidence} />
          </>
        )}
      </div>
    </div>
  )
}

function UserMessage({ message }) {
  return (
    <div className="flex justify-end max-w-3xl ml-auto">
      <div className="rounded-2xl rounded-tr-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-sm max-w-lg">
        {message.content}
      </div>
    </div>
  )
}

export default function ChatMessage({ message }) {
  if (message.role === 'user') {
    return <UserMessage message={message} />
  }
  return <AssistantMessage message={message} />
}
