import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { useToast } from './Toast.jsx'
import ErrorMessage from './ErrorMessage.jsx'
import { fieldDomId, listFields } from '../fields.js'

// Progress text while POST /extract runs (a single request, so stages are time-based).
const STAGES = [
  { at: 0, text: 'Reading document…' },
  { at: 2500, text: 'Extracting fields…' },
  { at: 9000, text: 'Verifying…' },
]

function Field({ id, label, kind, field, active, onChange, onFocus }) {
  const review = field.confidence === 'review'
  const locatable = field.bbox && field.value != null
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
        onFocus={onFocus}
        onClick={onFocus}
        rows={kind === 'multiline' ? 3 : undefined}
        placeholder={kind === 'date' ? 'YYYY-MM-DD' : field.value == null ? 'Not found on the document' : ''}
        aria-describedby={`${review ? `${id}-hint ` : ''}${id}-source`}
        className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 ${
          active && locatable ? 'ring-2 ring-blue-300 ' : ''
        }${
          review
            ? 'border-amber-400 bg-amber-50/50 focus:border-amber-500 focus:ring-amber-200'
            : 'border-slate-300 bg-white focus:border-blue-500 focus:ring-blue-200'
        }`}
      />
      <p id={`${id}-source`} className="mt-1 whitespace-pre-line text-xs text-slate-500">
        Source: {field.source_text ?? <em>not found on the document</em>}
        {field.value != null && field.source_text && !field.bbox && (
          <span className="text-slate-400"> · not located on the image</span>
        )}
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

// onBoxesChange([{ key, label, box, confidence }]) reports highlightable fields to the viewer;
// onFieldFocus(key) tells it which one to emphasise.
export default function ExtractedForm({ docId, activeField, onFieldFocus, onBoxesChange }) {
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

  // Fields with a bbox (and a value) become clickable highlights on the document.
  useEffect(() => {
    if (!form) return
    onBoxesChange?.(
      listFields(form)
        .filter(({ field }) => field.bbox && field.value != null)
        .map(({ key, label, field }) => ({ key, label, box: field.bbox, confidence: field.confidence })),
    )
  }, [form, onBoxesChange])

  function update(key, value) {
    setForm((f) => {
      if (!key.startsWith('other_fields.')) return { ...f, [key]: { ...f[key], value } }
      const k = key.slice('other_fields.'.length)
      return { ...f, other_fields: { ...f.other_fields, [k]: { ...f.other_fields[k], value } } }
    })
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
  const fields = listFields(form)
  const core = fields.filter((f) => !f.key.startsWith('other_fields.'))
  const other = fields.filter((f) => f.key.startsWith('other_fields.'))
  const reviewCount = fields.filter(({ field }) => field.confidence === 'review').length
  const renderField = (f) => (
    <Field
      key={f.key}
      id={fieldDomId(f.key)}
      label={f.label}
      kind={f.kind}
      field={f.field}
      active={activeField === f.key}
      onChange={(v) => update(f.key, v)}
      onFocus={() => onFieldFocus?.(f.key)}
    />
  )

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

        {core.map(renderField)}

        {other.length > 0 && (
          <>
            <h3 className="border-t border-slate-200 pt-5 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Other details
            </h3>
            {other.map(renderField)}
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
