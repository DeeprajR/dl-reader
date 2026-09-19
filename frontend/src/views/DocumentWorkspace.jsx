// The page for one document: the image on the left, and on the right two tabs, the review form
// and the chat. This component owns what links the two sides: which field or chat source is
// selected, and therefore which highlights are drawn on the image.

import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import DocumentViewer from '../components/DocumentViewer.jsx'
import ExtractedForm from '../components/ExtractedForm.jsx'
import ChatPanel from '../components/ChatPanel.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'
import { fieldDomId } from '../fields.js'

const TABS = [
  { id: 'form', label: 'Extracted form' },
  { id: 'chat', label: 'Ask the document' },
]

export default function DocumentWorkspace({ docId }) {
  const [info, setInfo] = useState(null) // { meta, label }
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('form')
  const [fieldBoxes, setFieldBoxes] = useState([]) // [{ key, target, label, box, confidence }]
  const [activeField, setActiveField] = useState(null)
  const [activeSource, setActiveSource] = useState(null) // { id, box }

  // Load the image size (needed to scale the highlights) and the file's display name.
  const load = useCallback(() => {
    setError(null)
    Promise.all([api.getMeta(docId), api.listDocuments()])
      .then(([meta, docs]) => {
        const label = docs.find((d) => d.doc_id === docId)?.filename_label ?? 'Document'
        setInfo({ meta, label })
      })
      .catch((e) => setError(e.message))
  }, [docId])

  useEffect(load, [load])

  // Clicking a highlight on the document focuses (and scrolls to) its form field.
  const focusField = useCallback((key) => {
    setActiveField(key)
    const input = document.getElementById(fieldDomId(key))
    // Focus without the browser's instant jump, then scroll smoothly to the centre instead.
    input?.focus({ preventScroll: true })
    input?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [])

  if (error) {
    return (
      <div className="mx-auto max-w-2xl space-y-4">
        <a href="#/" className="text-sm text-slate-600 hover:text-slate-900">
          ← All documents
        </a>
        <ErrorMessage message={error} onRetry={load} />
      </div>
    )
  }
  if (!info) return <p className="text-sm text-slate-500">Loading document…</p>

  // Form tab: one highlight per located field. Chat tab: only the source the user has opened.
  const highlights =
    tab === 'form'
      ? fieldBoxes.map((f) => ({
          id: f.key,
          box: f.box,
          label: f.label,
          tone: f.confidence === 'review' ? 'review' : 'field',
          active: f.target === activeField, // the form field this highlight belongs to
          onClick: () => focusField(f.target),
        }))
      : activeSource?.box
        ? [{ id: activeSource.id, box: activeSource.box, label: 'Chat source', tone: 'source', active: true }]
        : []

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <a href="#/" className="text-sm text-slate-600 hover:text-slate-900">
          ← All documents
        </a>
        <h1 className="min-w-0 truncate text-lg font-semibold">{info.label}</h1>
      </div>

      {/* Two columns on wide screens, stacked on narrow ones. */}
      <div className="grid items-start gap-6 lg:grid-cols-2">
        {/* Left: the document stays visible while reviewing the form or chatting. */}
        <section className="lg:sticky lg:top-6">
          <DocumentViewer docId={docId} meta={info.meta} highlights={highlights} />
          <p className="mt-2 text-center text-xs text-slate-500">
            {tab === 'form'
              ? 'Click a field to see where it is on the document, or click a highlight to edit that field.'
              : 'Open a source under an answer to see where it is on the document.'}
          </p>
        </section>

        <section className="rounded-xl bg-white shadow-sm ring-1 ring-slate-200">
          <div role="tablist" className="flex gap-1 border-b border-slate-200 px-3">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => setTab(t.id)}
                className={`-mb-px border-b-2 px-3 py-3 text-sm font-medium ${
                  tab === t.id
                    ? 'border-blue-700 text-blue-800'
                    : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          {/* Panels stay mounted so unsaved edits and the chat thread survive tab switches. */}
          <div role="tabpanel" hidden={tab !== 'form'}>
            <ExtractedForm
              docId={docId}
              activeField={activeField}
              onFieldFocus={setActiveField}
              onBoxesChange={setFieldBoxes}
            />
          </div>
          <div role="tabpanel" hidden={tab !== 'chat'}>
            <ChatPanel docId={docId} activeSourceId={activeSource?.id} onSourceSelect={setActiveSource} />
          </div>
        </section>
      </div>
    </div>
  )
}
