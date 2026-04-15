import { useState, useRef, useEffect, useCallback } from 'react'
import DocumentSidebar from './components/DocumentSidebar'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import { askQuestion, compareDocuments } from './services/api'

function formatComparisonAnswer(data) {
  if (!data.comparisons || data.comparisons.length === 0) {
    return 'No comparison results were returned.'
  }
  return data.comparisons
    .map((c) => `### ${c.filename}\n\n${c.answer}`)
    .join('\n\n---\n\n')
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 text-center px-6">
      <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center">
        <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-3 3v-3z" />
        </svg>
      </div>
      <div>
        <p className="text-sm font-medium text-gray-700">No messages yet</p>
        <p className="text-xs text-gray-400 mt-1">
          Upload a document on the left, then ask a question.
        </p>
      </div>
    </div>
  )
}

let messageIdCounter = 0
function nextId() {
  return ++messageIdCounter
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [uploadedDocs, setUploadedDocs] = useState([])
  const [selectedDocs, setSelectedDocs] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const messagesEndRef = useRef(null)

  // Scroll to bottom whenever messages change.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleUploadSuccess = useCallback((filename) => {
    setUploadedDocs((prev) =>
      prev.includes(filename) ? prev : [...prev, filename]
    )
  }, [])

  const handleDocumentSelect = useCallback((filename) => {
    setSelectedDocs((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename]
    )
  }, [])

  const handleSendMessage = useCallback(async (query, compareDocs) => {
    const userMsg = { id: nextId(), role: 'user', content: query }
    const loadingId = nextId()
    const loadingMsg = {
      id: loadingId,
      role: 'assistant',
      content: '',
      citations: null,
      confidence: null,
      isLoading: true,
    }

    setMessages((prev) => [...prev, userMsg, loadingMsg])
    setIsLoading(true)

    try {
      let assistantMsg

      if (compareDocs && compareDocs.length > 1) {
        // Compare mode — one answer per selected document.
        const data = await compareDocuments(query, compareDocs)
        assistantMsg = {
          id: loadingId,
          role: 'assistant',
          content: formatComparisonAnswer(data),
          citations: null,
          confidence: null,
          isLoading: false,
        }
      } else {
        // Standard Q&A with citations and confidence.
        const data = await askQuestion(query)
        assistantMsg = {
          id: loadingId,
          role: 'assistant',
          content: data.answer,
          citations: data.citations ?? null,
          confidence: data.confidence ?? null,
          isLoading: false,
        }
      }

      // Replace the loading placeholder with the real response.
      setMessages((prev) =>
        prev.map((m) => (m.id === loadingId ? assistantMsg : m))
      )
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                ...m,
                content: `Error: ${err.message}`,
                isLoading: false,
              }
            : m
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [])

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Sidebar */}
      <DocumentSidebar
        uploadedDocs={uploadedDocs}
        onUploadSuccess={handleUploadSuccess}
        onDocumentSelect={handleDocumentSelect}
        selectedDocs={selectedDocs}
      />

      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex-shrink-0 flex items-center gap-3 px-6 py-4 bg-gray-900 border-b border-gray-800">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
            </svg>
          </div>
          <div>
            <h1 className="text-white font-semibold text-sm leading-tight">
              Document Intelligence Platform
            </h1>
            <p className="text-gray-400 text-xs">Legal &amp; Compliance Q&amp;A</p>
          </div>

          {selectedDocs.length > 0 && (
            <div className="ml-auto flex items-center gap-1.5 rounded-full bg-gray-800 px-3 py-1">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
              <span className="text-xs text-gray-300">
                {selectedDocs.length} doc{selectedDocs.length > 1 ? 's' : ''} selected
              </span>
            </div>
          )}
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto bg-gray-50 px-6 py-6">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="flex flex-col gap-4 max-w-4xl mx-auto">
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <ChatInput
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          selectedDocs={selectedDocs}
        />
      </div>
    </div>
  )
}
