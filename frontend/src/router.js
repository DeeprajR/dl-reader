// A tiny router. The current view is kept in the address after the "#", for example
// "#/documents/123". The browser never sends that part to the server, so every view works
// without any server configuration, and the back button and reloading work as expected.

import { useEffect, useState } from 'react'

// Hash routes: #/ (list), #/upload, #/documents/<id>
// Turn the address into { view, docId }. Anything unknown shows the document list.
function parseRoute(hash) {
  const path = hash.replace(/^#/, '') || '/'
  const doc = path.match(/^\/documents\/([^/]+)$/)
  if (doc) return { view: 'workspace', docId: decodeURIComponent(doc[1]) }
  if (path === '/upload') return { view: 'upload' }
  return { view: 'list' }
}

// Go to another view, e.g. navigate('/upload'). Changing the hash triggers `useRoute` below.
export function navigate(path) {
  window.location.hash = path
}

// React hook: the current route. The component re-renders whenever the address changes.
export function useRoute() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash))
    window.addEventListener('hashchange', onChange)
    // Stop listening when the component goes away.
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}
