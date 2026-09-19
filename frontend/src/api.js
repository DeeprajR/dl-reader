// All calls use the relative /api base: proxied by Vite in dev, same-origin in the container.
const API = '/api'

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(API + path, options)
  } catch {
    throw new ApiError('Could not reach the server. Check that the backend is running.', 0)
  }
  const text = await res.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  if (!res.ok) {
    throw new ApiError(body?.error || `Request failed (HTTP ${res.status}).`, res.status)
  }
  return body
}

function json(method, body) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

// YYYY-MM-DD in the user's own time zone (toISOString would give the UTC date).
function localDate() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

export const api = {
  listDocuments: () => request('/documents'),
  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },
  getMeta: (docId) => request(`/documents/${docId}/meta`),
  getExtraction: (docId) => request(`/documents/${docId}/extract`),
  runExtraction: (docId) => request(`/documents/${docId}/extract`, { method: 'POST' }),
  saveData: (docId, data) => request(`/documents/${docId}/data`, json('PUT', data)),
  // `today` is the browser's local date: the server may be in another time zone (a container is UTC).
  chat: (docId, question) => request(`/documents/${docId}/chat`, json('POST', { question, today: localDate() })),
  imageUrl: (docId) => `${API}/documents/${docId}/image`,
}
