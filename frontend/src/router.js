import { useEffect, useState } from 'react'

// Hash routes: #/ (list), #/upload, #/documents/<id>
function parseRoute(hash) {
  const path = hash.replace(/^#/, '') || '/'
  const doc = path.match(/^\/documents\/([^/]+)$/)
  if (doc) return { view: 'workspace', docId: decodeURIComponent(doc[1]) }
  if (path === '/upload') return { view: 'upload' }
  return { view: 'list' }
}

export function navigate(path) {
  window.location.hash = path
}

export function useRoute() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
