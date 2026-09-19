import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'

const MAX_CHARS = 1000
const ORIGIN_LABELS = { ocr_text: 'Document text', extracted_fields: 'Extracted field', calculated: 'Calculated' }
const SUGGESTED_QUESTIONS = [
  'When does the licence expire?',
  'How many days until it expires?',
  'Which vehicles can the holder drive?',
  'Who issued this licence?',
]

// Models sometimes format answers as Markdown. Show them as tidy plain text (never as HTML):
// `code` spans keep their quote as “…”, bold markers are dropped, list markers become bullets.
function tidyAnswer(text) {
  return text
    .replace(/`([^`\n]+)`/g, '“$1”')
    .replace(/\*\*([^*\n]+)\*\*/g, '$1')
    .replace(/^[ \t]*[*-][ \t]+/gm, '• ')
}

// Each source is expandable; expanding one with a bbox highlights it on the document.
function Sources({ messageId, sources, activeSourceId, onSourceSelect }) {
  if (!sources.length) return null
  return (
    <details className="mt-2">
      <summary className="cursor-pointer select-none text-xs font-medium text-blue-700 hover:underline">
        Sources ({sources.length})
      </summary>
      <ol className="mt-2 space-y-2">
        {sources.map((s, i) => {
          const id = `${messageId}-${i}`
          const active = id === activeSourceId
          return (
            <li key={id}>
              <details
                open={active || undefined}
                onToggle={(e) => {
                  if (e.currentTarget.open) onSourceSelect({ id, box: s.bbox })
                  else if (active) onSourceSelect(null)
                }}
                className={`rounded-md bg-white text-xs ring-1 ${active ? 'ring-violet-400' : 'ring-slate-200'}`}
              >
                <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2">
                  <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 font-medium text-slate-600">
                    {ORIGIN_LABELS[s.origin] ?? s.origin}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-slate-700">{s.text.split('\n')[0]}</span>
                </summary>
                <div className="border-t border-slate-100 px-3 py-2">
                  <p className="whitespace-pre-line text-slate-700">{s.text}</p>
                  <p className={`mt-1 ${s.bbox ? 'text-violet-700' : 'text-slate-400'}`}>
                    {s.bbox
                      ? 'Highlighted on the document.'
                      : s.origin === 'calculated'
                        ? 'Worked out from the licence dates and today’s date.'
                        : 'Not located on the image.'}
                  </p>
                </div>
              </details>
            </li>
          )
        })}
      </ol>
    </details>
  )
}

function Message({ message, messageId, activeSourceId, onSourceSelect }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-blue-700 px-4 py-2 text-sm text-white">
          {message.text}
        </div>
      </div>
    )
  }
  if (message.role === 'error') {
    return (
      <div role="alert" className="max-w-[85%] rounded-2xl rounded-bl-sm bg-red-50 px-4 py-2 text-sm text-red-900 ring-1 ring-red-200">
        {message.text}
      </div>
    )
  }
  return (
    <div className="max-w-[85%] rounded-2xl rounded-bl-sm bg-slate-100 px-4 py-2 text-sm text-slate-900">
      <p className="whitespace-pre-wrap">{tidyAnswer(message.text)}</p>
      <Sources
        messageId={messageId}
        sources={message.sources}
        activeSourceId={activeSourceId}
        onSourceSelect={onSourceSelect}
      />
    </div>
  )
}

// onSourceSelect({ id, box } | null) drives the source highlight on the document.
export default function ChatPanel({ docId, activeSourceId, onSourceSelect }) {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [messages, pending])

  // `suggested` is set when a suggested question is clicked; otherwise the typed question is sent.
  async function ask(e, suggested) {
    e?.preventDefault()
    const text = (suggested ?? question).trim()
    if (!text || pending) return
    setMessages((m) => [...m, { role: 'user', text }])
    setQuestion('')
    setPending(true)
    try {
      const res = await api.chat(docId, text)
      setMessages((m) => [...m, { role: 'assistant', text: res.answer, sources: res.sources }])
    } catch (err) {
      setMessages((m) => [...m, { role: 'error', text: `Could not get an answer: ${err.message}` }])
      setQuestion(text) // keep the question so it can be re-sent
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="flex h-[min(70vh,44rem)] flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto p-6" aria-live="polite">
        {messages.length === 0 && (
          <div className="text-sm text-slate-500">
            <p>Ask a question about this licence. Answers come only from the document and cite their sources.</p>
            <p className="mt-3">Try one of these:</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {SUGGESTED_QUESTIONS.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => ask(null, q)}
                  className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1.5 text-sm text-blue-800 hover:bg-blue-100"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={i} messageId={i} message={m} activeSourceId={activeSourceId} onSourceSelect={onSourceSelect} />
        ))}
        {pending && (
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-200 border-t-slate-500" />
            Reading the document…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={ask} className="border-t border-slate-200 p-4">
        <div className="flex items-end gap-2">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) ask(e)
            }}
            disabled={pending}
            maxLength={MAX_CHARS}
            rows={2}
            placeholder={pending ? 'Waiting for the answer…' : 'Ask about this licence…'}
            aria-label="Question"
            className="block w-full resize-none rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200 disabled:bg-slate-50 disabled:text-slate-400"
          />
          <button
            type="submit"
            disabled={pending || !question.trim()}
            className="rounded-md bg-blue-700 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            Ask
          </button>
        </div>
        {question.length > MAX_CHARS - 100 && (
          <p className="mt-1 text-right text-xs text-slate-500">
            {question.length}/{MAX_CHARS}
          </p>
        )}
      </form>
    </div>
  )
}
