import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import ErrorMessage from './ErrorMessage.jsx'

const PAD = 3 // rendered pixels around each highlight

const TONES = {
  field: 'border-blue-500/50 bg-blue-400/10 hover:border-blue-600 hover:bg-blue-400/25',
  review: 'border-amber-500/70 bg-amber-300/15 hover:border-amber-600 hover:bg-amber-300/30',
  active: 'z-10 border-blue-700 bg-blue-500/25 ring-4 ring-blue-500/30',
  source: 'z-10 border-violet-600 bg-violet-500/20 ring-4 ring-violet-500/25',
}

// highlights: [{ id, box: {x, y, w, h} in natural image pixels, label, tone, active, onClick? }]
// Boxes are scaled by rendered/natural size and recomputed whenever the image is resized.
export default function DocumentViewer({ docId, meta, highlights = [] }) {
  const imgRef = useRef(null)
  const [failed, setFailed] = useState(false)
  const [rendered, setRendered] = useState(null) // { width, height } of the displayed image

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const measure = () => setRendered({ width: img.clientWidth, height: img.clientHeight })
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(img)
    window.addEventListener('resize', measure)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [failed])

  if (failed) {
    return <ErrorMessage message="The document image could not be loaded." onRetry={() => setFailed(false)} />
  }

  const sx = rendered ? rendered.width / meta.width : 0
  const sy = rendered ? rendered.height / meta.height : 0

  return (
    <div className="flex justify-center rounded-xl bg-white p-3 shadow-sm ring-1 ring-slate-200">
      <div className="relative inline-block overflow-hidden">
        <img
          ref={imgRef}
          src={api.imageUrl(docId)}
          alt="Uploaded driving licence"
          width={meta.width}
          height={meta.height}
          onError={() => setFailed(true)}
          className="block h-auto max-h-[calc(100vh-9rem)] w-auto max-w-full"
        />
        {sx > 0 &&
          highlights.map((h) => {
            const style = {
              left: h.box.x * sx - PAD,
              top: h.box.y * sy - PAD,
              width: h.box.w * sx + 2 * PAD,
              height: h.box.h * sy + 2 * PAD,
            }
            const tone = h.active ? (h.tone === 'source' ? TONES.source : TONES.active) : TONES[h.tone] ?? TONES.field
            const className = `absolute rounded border-2 transition-colors ${tone}`
            return h.onClick ? (
              <button
                key={h.id}
                type="button"
                tabIndex={-1}
                title={h.label}
                aria-label={`Edit ${h.label}`}
                data-highlight={h.id}
                onClick={h.onClick}
                style={style}
                className={`${className} cursor-pointer`}
              />
            ) : (
              <div
                key={h.id}
                aria-hidden="true"
                data-highlight={h.id}
                style={style}
                className={`${className} pointer-events-none`}
              />
            )
          })}
      </div>
    </div>
  )
}
