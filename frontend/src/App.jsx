// The page frame: the header bar, and below it one of the three views, chosen by the address.

import DocumentListView from './views/DocumentListView.jsx'
import UploadView from './views/UploadView.jsx'
import DocumentWorkspace from './views/DocumentWorkspace.jsx'
import { ToastProvider } from './components/Toast.jsx'
import { useRoute } from './router.js'

// `useRoute` re-renders this component whenever the address (the part after #) changes.
export default function App() {
  const route = useRoute()

  return (
    // ToastProvider lets any component below it show a pop-up message.
    <ToastProvider>
      <div className="min-h-screen flex flex-col">
        <header className="bg-white border-b border-slate-200">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 h-14 flex items-center justify-between">
            <a href="#/" className="flex items-center gap-2 font-semibold text-slate-900">
              <img src="/favicon.svg" alt="" className="h-6 w-6" />
              Licence Reader
            </a>
            {/* Home has its own prominent upload button; document pages have no other. */}
            {route.view === 'workspace' && (
              <a
                href="#/upload"
                className="rounded-md bg-blue-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-800"
              >
                Upload licence
              </a>
            )}
          </div>
        </header>
        {/* Exactly one view is shown at a time. */}
        <main className="flex-1 mx-auto w-full max-w-7xl px-4 sm:px-6 py-6">
          {route.view === 'list' && <DocumentListView />}
          {route.view === 'upload' && <UploadView />}
          {/* `key` makes React build a fresh workspace for each document, so nothing from the previous */}
          {/* document (form edits, chat messages) carries over. */}
          {route.view === 'workspace' && <DocumentWorkspace key={route.docId} docId={route.docId} />}
        </main>
      </div>
    </ToastProvider>
  )
}
