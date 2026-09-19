import { useState } from 'react'
import { api } from '../api.js'
import ErrorMessage from './ErrorMessage.jsx'

export default function DocumentViewer({ docId, meta }) {
  const [failed, setFailed] = useState(false)

  if (failed) {
    return <ErrorMessage message="The document image could not be loaded." onRetry={() => setFailed(false)} />
  }

  return (
    <div className="flex justify-center rounded-xl bg-white p-3 shadow-sm ring-1 ring-slate-200">
      <div className="relative inline-block">
        <img
          src={api.imageUrl(docId)}
          alt="Uploaded driving licence"
          width={meta.width}
          height={meta.height}
          onError={() => setFailed(true)}
          className="block h-auto max-h-[calc(100vh-9rem)] w-auto max-w-full"
        />
      </div>
    </div>
  )
}
