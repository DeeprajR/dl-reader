import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { useToast } from './Toast.jsx'
import ErrorMessage from './ErrorMessage.jsx'

const CORE_FIELDS = [
  { name: 'full_name', label: 'Full name' },
  { name: 'licence_number', label: 'Licence number' },
  { name: 'date_of_birth', label: 'Date of birth', kind: 'date' },
  { name: 'date_of_issue', label: 'Date of issue', kind: 'date' },
  { name: 'date_of_expiry', label: 'Date of expiry', kind: 'date' },
  { name: 'address', label: 'Address', kind: 'multiline' },
  { name: 'vehicle_classes', label: 'Vehicle classes' },
  { name: 'issuing_authority', label: 'Issuing authority' },
]

// Progress text while POST /extract runs (a single request, so stages are time-based).
const STAGES = [
  { at: 0, text: 'Reading document…' },
  { at: 2500, text: 'Extracting fields…' },
  { at: 9000, text: 'Verifying…' },
]

function humanize(key) {
  const s = key.replace(/_/g, ' ').trim()
  return s.charAt(0).toUpperCase() + s.slice(1)
}

function Field({ id, label, kind, field, onChange }) {
  const review = field.confidence === 'review'
  const Input = kind === 'multiline' ? 'textarea' : 'input'
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium text-slate-700">
          {label}
        </label>
        {review && (
          <span className="text-xs font-medium text-amber-700" id={`${id}-hint`}>
            ⚠ Please verify
          </span>
        )}
      </div>
      <Input
        id={id}
        value={field.value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        rows={kind === 'multiline' ? 3 : undefined}
        placeholder={kind === 'date' ? 'YYYY-MM-DD' : field.value == null ? 'Not found on the document' : ''}
        aria-describedby={`${review ? `${id}-hint ` : ''}${id}-source`}
        className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 ${
          review
            ? 'border-amber-400 bg-amber-50/50 focus:border-amber-500 focus:ring-amber-200'
            : 'border-slate-300 bg-white focus:border-blue-500 focus:ring-blue-200'
        }`}
      />
      <p id={`${id}-source`} className="mt-1 whitespace-pre-line text-xs text-slate-500">
        Source: {field.source_text ?? <em>not found on the document</em>}
      </p>
    </div>
  )
}

function ExtractionProgress({ stage }) {
  return (
    <ol className="space-y-3" aria-live="polite">
      {STAGES.map((s, i) => (
        <li key={s.text} className="flex items-center gap-3 text-sm">
          {i < stage ? (
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-600 text-xs text-white">✓</span>
          ) : i === stage ? (
            <span className="h-5 w-5 animate-spin rounded-full border-2 border-blue-200 border-t-blue-700" />
          ) : (
            <span className="h-5 w-5 rounded-full border-2 border-slate-200" />
          )}
          <span className={i <= stage ? 'text-slate-900' : 'text-slate-400'}>{s.text}</span>
        </li>
      ))}
    </ol>
  )
}

export default function ExtractedForm({ docId }) {
  const toast = useToast()
  const [status, setStatus] = useState('loading') // loading | extracting | ready | error
  const [stage, setStage] = useState(0)
  const [error, setError] = useState(null) // { message, retry }
  const [warnings, setWarnings] = useState([])
  const [form, setForm] = useState(null) // LicenceData being edited
  const [saved, setSaved] = useState(null) // last persisted LicenceData
  const [saving, setSaving] = useState(false)

  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const apply = useCallback((result) => {
    if (!mounted.current) return
    setWarnings(result.warnings)
    setForm(result.data)
    setSaved(result.data)
    setStatus('ready')
  }, [])

  const runExtraction = useCallback(async () => {
    setStatus('extracting')
    setError(null)
    setStage(0)
    const timers = STAGES.slice(1).map((s, i) => setTimeout(() => setStage(i + 1), s.at))
    try {
      apply(await api.runExtraction(docId))
    } catch (e) {
      if (mounted.current) {
        setError({ message: `Extraction failed: ${e.message}`, retry: runExtraction })
        setStatus('error')
      }
    } finally {
      timers.forEach(clearTimeout)
    }
  }, [docId, apply])

  const load = useCallback(async () => {
    setStatus('loading')
    setError(null)
    try {
      apply(await api.getExtraction(docId)) // persisted result: reopening never re-extracts
    } catch (e) {
      if (e.status === 404) return runExtraction()
      if (mounted.current) {
        setError({ message: e.message, retry: load })
        setStatus('error')
      }
    }
  }, [docId, apply, runExtraction])

  // Run once per mount (StrictMode re-runs effects in dev; that must not trigger a second LLM call).
  const started = useRef(false)
  useEffect(() => {
    if (started.current) return
    started.current = true
    load()
  }, [load])

  function updateCore(name, value) {
    setForm((f) => ({ ...f, [name]: { ...f[name], value } }))
  }

  function updateOther(key, value) {
    setForm((f) => ({ ...f, other_fields: { ...f.other_fields, [key]: { ...f.other_fields[key], value } } }))
  }

  async function save(e) {
    e.preventDefault()
    setSaving(true)
    try {
      const data = await api.saveData(docId, form)
      setForm(data)
      setSaved(data)
      toast('Changes saved.')
    } catch (err) {
      toast(`Could not save: ${err.message}`, 'error')
    } finally {
      setSaving(false)
    }
  }

  if (status === 'loading') return <p className="p-6 text-sm text-slate-500">Loading extracted data…</p>
  if (status === 'extracting') {
    return (
      <div className="p-6 space-y-4">
        <p className="text-sm text-slate-600">Extracting details from the document. This usually takes 10–20 seconds.</p>
        <ExtractionProgress stage={stage} />
      </div>
    )
  }
  if (status === 'error') {
    return (
      <div className="p-6">
        <ErrorMessage message={error.message} onRetry={error.retry} />
      </div>
    )
  }

  const dirty = JSON.stringify(form) !== JSON.stringify(saved)
  const otherKeys = Object.keys(form.other_fields)
  const reviewCount = [...CORE_FIELDS.map((f) => form[f.name]), ...Object.values(form.other_fields)].filter(
    (f) => f.confidence === 'review',
  ).length

  return (
    <form onSubmit={save}>
      <div className="space-y-5 p-6">
        {warnings.length > 0 && (
          <div className="rounded-lg bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">
            {warnings.map((w) => (
              <p key={w}>{w}</p>
            ))}
          </div>
        )}
        <p className="text-sm text-slate-600">
          {reviewCount === 0
            ? 'All fields were confirmed against the document text.'
            : `${reviewCount} field${reviewCount === 1 ? '' : 's'} could not be confirmed against the document text. Please verify the highlighted fields.`}
        </p>

        {CORE_FIELDS.map((f) => (
          <Field
            key={f.name}
            id={`field-${f.name}`}
            label={f.label}
            kind={f.kind}
            field={form[f.name]}
            onChange={(v) => updateCore(f.name, v)}
          />
        ))}

        {otherKeys.length > 0 && (
          <>
            <h3 className="border-t border-slate-200 pt-5 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Other details
            </h3>
            {otherKeys.map((key) => (
              <Field
                key={key}
                id={`field-other-${key}`}
                label={humanize(key)}
                field={form.other_fields[key]}
                onChange={(v) => updateOther(key, v)}
              />
            ))}
          </>
        )}
      </div>

      <div className="sticky bottom-0 flex items-center justify-end gap-3 rounded-b-xl border-t border-slate-200 bg-white/95 px-6 py-3 backdrop-blur">
        {dirty && <span className="text-xs text-slate-500">Unsaved changes</span>}
        <button
          type="submit"
          disabled={!dirty || saving}
          className="rounded-md bg-blue-700 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>
    </form>
  )
}
