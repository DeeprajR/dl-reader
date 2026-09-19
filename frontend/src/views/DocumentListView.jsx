import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import { navigate } from '../router.js'
import ErrorMessage from '../components/ErrorMessage.jsx'

function formatDate(iso) {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

export default function DocumentListView() {
  const [docs, setDocs] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    setError(null)
    api
      .listDocuments()
      .then(setDocs)
      .catch((e) => setError(e.message))
  }, [])

  useEffect(load, [load])

  return (
    <div className="space-y-6">
      <section className="rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200 flex flex-col sm:flex-row sm:items-center gap-4 justify-between">
        <div>
          <h1 className="text-xl font-semibold">Driving licence reader</h1>
          <p className="mt-1 text-sm text-slate-600">
            Upload a licence (JPG, PNG or PDF) to extract its details, review them next to the document,
            and ask questions about it.
          </p>
        </div>
        <a
          href="#/upload"
          className="shrink-0 rounded-lg bg-blue-700 px-5 py-2.5 text-center font-medium text-white shadow-sm hover:bg-blue-800"
        >
          Upload a licence
        </a>
      </section>

      <section className="rounded-xl bg-white shadow-sm ring-1 ring-slate-200">
        <h2 className="px-6 pt-5 pb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Your documents
        </h2>
        {error && (
          <div className="px-6 pb-6">
            <ErrorMessage message={error} onRetry={load} />
          </div>
        )}
        {!error && docs === null && <p className="px-6 pb-6 text-sm text-slate-500">Loading…</p>}
        {!error && docs?.length === 0 && (
          <p className="px-6 pb-8 pt-2 text-center text-slate-600">No documents yet.</p>
        )}
        {!error && docs?.length > 0 && (
          <ul className="divide-y divide-slate-100 border-t border-slate-100">
            {docs.map((d) => (
              <li key={d.doc_id}>
                <button
                  onClick={() => navigate(`/documents/${d.doc_id}`)}
                  className="w-full px-6 py-3.5 flex items-center gap-4 text-left hover:bg-slate-50 focus:bg-slate-50 focus:outline-none"
                >
                  <span className="flex-1 min-w-0">
                    <span className="block truncate font-medium text-slate-900">{d.filename_label}</span>
                    <span className="block text-xs text-slate-500">Uploaded {formatDate(d.uploaded_at)}</span>
                  </span>
                  <span
                    className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                      d.has_extraction ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {d.has_extraction ? 'Extracted' : 'Not extracted'}
                  </span>
                  <span className="text-slate-400" aria-hidden>
                    ›
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}
