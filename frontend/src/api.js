// All calls use the relative /api base: proxied by Vite in dev, same-origin in the container.
const API = '/api'

// An error with the HTTP status attached (0 = the server could not be reached at all).
export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

// One place for every call to the backend. It returns the parsed JSON, or throws an ApiError
// whose message can be shown to the user as it is.
async function request(path, options = {}) {
  let res
  try {
    res = await fetch(API + path, options)
  } catch {
    throw new ApiError('Could not reach the server. Check that the backend is running.', 0)
  }
  // Read the body as text first: an error page from a proxy may not be JSON at all.
  const text = await res.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = null
  }
  // 401: the app has a password and this browser is not signed in (or no longer is).
  if (res.status === 401) window.location.assign('/api/login')
  // The backend sends every error as {"error": "message"}.
  if (!res.ok) {
    throw new ApiError(body?.error || `Request failed (HTTP ${res.status}).`, res.status)
  }
  return body
}

// The fetch options for sending `body` as JSON.
function json(method, body) {
  return { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
}

// Today's date in the user's own time zone (toISOString would give the UTC date). It is sent as
// YYYY-MM-DD because that is the form every JSON API reads a date in; it is never shown.
function localDate() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

// One function per backend endpoint.
export const api = {
  listDocuments: () => request('/documents'),
  uploadDocument: (file) => {
    // Files are sent as a form upload. The browser sets the Content-Type header itself.
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
  // Not a request: the address the <img> tag loads the document image from.
  imageUrl: (docId) => `${API}/documents/${docId}/image`,
}
