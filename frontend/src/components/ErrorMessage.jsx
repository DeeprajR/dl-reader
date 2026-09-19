export default function ErrorMessage({ message, onRetry, retryLabel = 'Retry' }) {
  return (
    <div role="alert" className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-900 ring-1 ring-red-200">
      <div className="flex items-start gap-3">
        <span className="flex-1">{message}</span>
        {onRetry && (
          <button
            onClick={onRetry}
            className="shrink-0 rounded-md bg-white px-3 py-1 text-sm font-medium text-red-800 ring-1 ring-red-300 hover:bg-red-100"
          >
            {retryLabel}
          </button>
        )}
      </div>
    </div>
  )
}
