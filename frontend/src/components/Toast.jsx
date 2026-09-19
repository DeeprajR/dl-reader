// Toasts: small messages that pop up in the bottom-right corner and disappear by themselves
// ("Saved", "Could not save ...").
//
// Usage:  const toast = useToast();  toast('Saved');  toast('Something failed', 'error')

import { createContext, useCallback, useContext, useRef, useState } from 'react'

// The context carries the `show` function. Outside a provider it is a function that does nothing.
const ToastContext = createContext(() => {})

export function useToast() {
  return useContext(ToastContext)
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])
  // A counter for unique ids. A ref, because changing it must not re-render anything.
  const nextId = useRef(1)

  const dismiss = useCallback((id) => setToasts((all) => all.filter((t) => t.id !== id)), [])

  // show(message, 'success' | 'error')
  const show = useCallback(
    (message, kind = 'success') => {
      const id = nextId.current++
      setToasts((all) => [...all, { id, message, kind }])
      // Errors stay up longer than successes, because they take longer to read.
      setTimeout(() => dismiss(id), kind === 'error' ? 6000 : 3500)
    },
    [dismiss],
  )

  return (
    <ToastContext.Provider value={show}>
      {children}
      {/* The stack of visible toasts, drawn on top of the page. */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-80 max-w-[calc(100vw-2rem)]">
        {toasts.map((t) => (
          <div
            key={t.id}
            role={t.kind === 'error' ? 'alert' : 'status'}
            className={`flex items-start gap-3 rounded-lg px-4 py-3 text-sm shadow-lg ring-1 ${
              t.kind === 'error'
                ? 'bg-red-50 text-red-900 ring-red-200'
                : 'bg-emerald-50 text-emerald-900 ring-emerald-200'
            }`}
          >
            <span className="flex-1">{t.message}</span>
            <button
              onClick={() => dismiss(t.id)}
              className="text-current opacity-60 hover:opacity-100"
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
