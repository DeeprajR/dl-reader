import { useRef, useState } from 'react'
import { api } from '../api.js'
import { navigate } from '../router.js'
import ErrorMessage from '../components/ErrorMessage.jsx'

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.pdf']
const MAX_MB = 10 // mirrors the server default MAX_UPLOAD_MB; the server enforces its own limit

// Checked in the browser for a fast, friendly message; the server repeats every check.
function validate(file) {
  const name = file.name.toLowerCase()
  if (!ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return `"${file.name}" is not a supported file. Please choose a JPG, PNG or PDF.`
  }
  if (file.size === 0) return 'The selected file is empty.'
  if (file.size > MAX_MB * 1024 * 1024) {
    return `"${file.name}" is ${(file.size / 1024 / 1024).toFixed(1)} MB. The maximum size is ${MAX_MB} MB.`
  }
  return null
}

// Drag-and-drop or file picker for one document; on success it opens the document's workspace.
export default function UploadView() {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(null) // file name while uploading
  const [error, setError] = useState(null)

  async function handleFiles(files) {
    if (uploading || !files?.length) return
    if (files.length > 1) {
      setError('Please upload one document at a time.')
      return
    }
    const file = files[0]
    const problem = validate(file)
    if (problem) {
      setError(problem)
      return
    }
    setError(null)
    setUploading(file.name)
    try {
      const { doc_id } = await api.uploadDocument(file)
      navigate(`/documents/${doc_id}`)
    } catch (e) {
      setError(e.message)
      setUploading(null)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <a href="#/" className="text-sm text-slate-600 hover:text-slate-900">
        ← All documents
      </a>
      <div className="rounded-xl bg-white p-6 shadow-sm ring-1 ring-slate-200 space-y-4">
        <div>
          <h1 className="text-xl font-semibold">Upload a driving licence</h1>
          <p className="mt-1 text-sm text-slate-600">
            JPG, PNG or PDF up to {MAX_MB} MB. For PDFs, the first two pages (front and back) are used.
          </p>
        </div>

        <div
          role="button"
          tabIndex={0}
          aria-disabled={!!uploading}
          onClick={() => !uploading && inputRef.current?.click()}
          onKeyDown={(e) => {
            if ((e.key === 'Enter' || e.key === ' ') && !uploading) {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
            uploading
              ? 'cursor-wait border-slate-300 bg-slate-50'
              : dragging
                ? 'cursor-copy border-blue-500 bg-blue-50'
                : 'cursor-pointer border-slate-300 hover:border-blue-400 hover:bg-slate-50'
          }`}
        >
          {uploading ? (
            <>
              <span className="h-8 w-8 animate-spin rounded-full border-4 border-blue-200 border-t-blue-700" />
              <p className="mt-4 font-medium text-slate-800">Uploading {uploading}…</p>
            </>
          ) : (
            <>
              <svg viewBox="0 0 24 24" className="h-10 w-10 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
              </svg>
              <p className="mt-3 font-medium text-slate-800">Drag and drop your licence here</p>
              <p className="mt-1 text-sm text-slate-500">
                or <span className="font-medium text-blue-700">browse files</span>
              </p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.pdf,image/jpeg,image/png,application/pdf"
            className="hidden"
            onChange={(e) => {
              handleFiles(e.target.files)
              e.target.value = ''
            }}
          />
        </div>

        {error && <ErrorMessage message={error} />}
      </div>
    </div>
  )
}
