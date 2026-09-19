// The review form: loads (or runs) the extraction, shows the nine fields with their sources and
// "Please verify" marks, and saves the user's edits.

import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { useToast } from './Toast.jsx'
import ErrorMessage from './ErrorMessage.jsx'
import { CORE_FIELDS, OTHER_FIELD, composeOther, fieldDomId, otherItems, parseOther } from '../fields.js'

// Progress text while POST /extract runs (a single request, so stages are time-based).
const STAGES = [
  { at: 0, text: 'Reading document…' },
  { at: 2500, text: 'Extracting fields…' },
  { at: 9000, text: 'Verifying…' },
]

// One labelled input with its "Please verify" mark and its source line.
// `active` = this field's highlight is selected on the document.
function Field({ id, label, kind, field, active, onChange, onFocus }) {
  // review: OCR could not confirm this value. locatable: it has a highlight on the document.
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

// One editable box for everything outside the eight core fields: one "Label: value" line per
// item. Each item keeps its own source and highlight; the sources are listed underneath.
function OtherField({ id, text, items, active, onChange, onFocus }) {
  const review = items.some(({ field }) => field.confidence === 'review')
  // The box grows with its content: one row per item, plus a spare row.
  const lines = text ? text.split('\n').length : 0
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium text-slate-700">
          {OTHER_FIELD.label}
        </label>
        {review && (
          <span className="text-xs font-medium text-amber-700" id={`${id}-hint`}>
            ⚠ Please verify
          </span>
        )}
      </div>
      <textarea
        id={id}
        value={text}
        onChange={(e) => onChange(e.target.value)}
        onFocus={onFocus}
        onClick={onFocus}
        rows={Math.max(3, lines + 1)}
        placeholder="One item per line, e.g. Blood group: O+"
        aria-describedby={`${review ? `${id}-hint ` : ''}${id}-source`}
        className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 ${
          active && items.some(({ field }) => field.bbox) ? 'ring-2 ring-blue-300 ' : ''
        }${
          review
            ? 'border-amber-400 bg-amber-50/50 focus:border-amber-500 focus:ring-amber-200'
            : 'border-slate-300 bg-white focus:border-blue-500 focus:ring-blue-200'
        }`}
      />
      <div id={`${id}-source`} className="mt-1 text-xs text-slate-500">
        {items.length === 0 ? (
          <p>Source: <em>nothing else found on the document</em></p>
        ) : (
          <>
            <p>Source:</p>
            <ul className="mt-0.5 space-y-0.5">
              {items.map(({ key, label, field }) => (
                <li key={key} className="whitespace-pre-line">
                  {field.confidence === 'review' && <span className="text-amber-700">⚠ </span>}
                  {label}: {field.source_text ?? <em>not found on the document</em>}
                  {field.source_text && !field.bbox && <span className="text-slate-400"> · not located on the image</span>}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}

// The three-step progress list: done steps get a tick, the current one spins, later ones are grey.
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

// onBoxesChange([{ key, target, label, box, confidence }]) reports highlightable items to the
// viewer (target = the form field they belong to); onFieldFocus(target) marks the active field.
export default function ExtractedForm({ docId, activeField, onFieldFocus, onBoxesChange }) {
  const toast = useToast()
  const [status, setStatus] = useState('loading') // loading | extracting | ready | error
  const [stage, setStage] = useState(0)
  const [error, setError] = useState(null) // { message, retry }
  const [warnings, setWarnings] = useState([])
  const [form, setForm] = useState(null) // LicenceData being edited
  const [saved, setSaved] = useState(null) // last persisted LicenceData
  const [otherText, setOtherText] = useState('') // the "Other relevant information" box
  const [saving, setSaving] = useState(false)

  // True while this component is on the page. A request can finish after the user has left the
  // page; its result is then ignored.
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  // Put an extraction result on the screen. `saved` is the copy used to detect unsaved changes.
  const apply = useCallback((result) => {
    if (!mounted.current) return
    setWarnings(result.warnings)
    setForm(result.data)
    setSaved(result.data)
    setOtherText(composeOther(result.data.other_fields))
    setStatus('ready')
  }, [])

  // Ask the backend to read the document (the slow call: OCR plus the AI model).
  const runExtraction = useCallback(async () => {
    setStatus('extracting')
    setError(null)
    setStage(0)
    // Move the progress text on by the clock, and cancel the timers as soon as the request ends.
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

  // Open the document: use the saved extraction when there is one, otherwise run it now.
  const load = useCallback(async () => {
    setStatus('loading')
    setError(null)
    try {
      apply(await api.getExtraction(docId)) // persisted result: reopening never re-extracts
    } catch (e) {
      // 404 means "not extracted yet", which is normal for a document that was just uploaded.
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

  // Fields and other items with a bbox become clickable highlights on the document.
  useEffect(() => {
    if (!form) return
    const core = CORE_FIELDS.map((f) => ({ key: f.name, target: f.name, label: f.label, field: form[f.name] }))
    const other = otherItems(form.other_fields).map(({ key, label, field }) => ({
      key: `other_fields.${key}`,
      target: OTHER_FIELD.name,
      label,
      field,
    }))
    onBoxesChange?.(
      [...core, ...other]
        .filter(({ field }) => field.bbox && field.value != null)
        .map(({ key, target, label, field }) => ({ key, target, label, box: field.bbox, confidence: field.confidence })),
    )
  }, [form, onBoxesChange])

  // Change one field's value, keeping its source, confidence and highlight.
  function update(name, value) {
    setForm((f) => ({ ...f, [name]: { ...f[name], value } }))
  }

  // Save the form. The "other" box is turned back from text lines into separate items first.
  async function save(e) {
    e.preventDefault()
    setSaving(true)
    try {
      const data = await api.saveData(docId, { ...form, other_fields: parseOther(otherText, saved.other_fields) })
      // The server returns the tidied data (trimmed, dates normalised), and that is what is shown.
      setForm(data)
      setSaved(data)
      setOtherText(composeOther(data.other_fields))
      toast('Changes saved.')
    } catch (err) {
      toast(`Could not save: ${err.message}`, 'error')
    } finally {
      setSaving(false)
    }
  }

  // What to show depends on the status: loading, extracting, error, or the form itself.
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

  // dirty: something differs from what was last saved, so the Save button is enabled.
  const dirty = JSON.stringify(form) !== JSON.stringify(saved) || otherText !== composeOther(saved.other_fields)
  const items = otherItems(form.other_fields)
  // How many of the nine fields need a look. The "other" box counts once, however many items it holds.
  const reviewCount =
    CORE_FIELDS.filter((f) => form[f.name].confidence === 'review').length +
    (items.some(({ field }) => field.confidence === 'review') ? 1 : 0)

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
            id={fieldDomId(f.name)}
            label={f.label}
            kind={f.kind}
            field={form[f.name]}
            active={activeField === f.name}
            onChange={(v) => update(f.name, v)}
            onFocus={() => onFieldFocus?.(f.name)}
          />
        ))}
        <OtherField
          id={fieldDomId(OTHER_FIELD.name)}
          text={otherText}
          items={items}
          active={activeField === OTHER_FIELD.name}
          onChange={setOtherText}
          onFocus={() => onFieldFocus?.(OTHER_FIELD.name)}
        />
      </div>

      {/* The save bar stays visible at the bottom while the form scrolls. */}
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
