import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import DocumentViewer from '../components/DocumentViewer.jsx'
import ExtractedForm from '../components/ExtractedForm.jsx'
import ErrorMessage from '../components/ErrorMessage.jsx'

const TABS = [{ id: 'form', label: 'Extracted form' }]

export default function DocumentWorkspace({ docId }) {
  const [info, setInfo] = useState(null) // { meta, label }
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('form')

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

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <a href="#/" className="text-sm text-slate-600 hover:text-slate-900">
          ← All documents
        </a>
        <h1 className="min-w-0 truncate text-lg font-semibold">{info.label}</h1>
      </div>

      <div className="grid items-start gap-6 lg:grid-cols-2">
        {/* Left: the document stays visible while reviewing the form or chatting. */}
        <section className="lg:sticky lg:top-6">
          <DocumentViewer docId={docId} meta={info.meta} />
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
          {/* Panels stay mounted so unsaved edits survive tab switches. */}
          <div role="tabpanel" hidden={tab !== 'form'}>
            <ExtractedForm docId={docId} />
          </div>
        </section>
      </div>
    </div>
  )
}
