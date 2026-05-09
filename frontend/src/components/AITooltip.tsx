import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

interface AITooltipProps {
  card: string
  modelId: string
  data: Record<string, unknown>
}

export default function AITooltip({ card, modelId, data }: AITooltipProps) {
  const [open, setOpen] = useState(false)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [fetched, setFetched] = useState(false)
  const [popoverPos, setPopoverPos] = useState({ top: 0, left: 0 })
  const buttonRef = useRef<HTMLButtonElement>(null)
  const cursorRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [showCursor, setShowCursor] = useState(true)

  // Blink cursor while streaming
  useEffect(() => {
    if (loading) {
      cursorRef.current = setInterval(() => setShowCursor(p => !p), 500)
    } else {
      if (cursorRef.current) clearInterval(cursorRef.current)
      setShowCursor(false)
    }
    return () => { if (cursorRef.current) clearInterval(cursorRef.current) }
  }, [loading])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (buttonRef.current && !buttonRef.current.contains(e.target as Node)) {
        const popover = document.getElementById('ai-tooltip-popover')
        if (popover && popover.contains(e.target as Node)) return
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  function calcPosition() {
    if (!buttonRef.current) return
    const rect = buttonRef.current.getBoundingClientRect()
    setPopoverPos({
      top: rect.bottom + window.scrollY + 6,
      left: Math.min(rect.right - 280, window.innerWidth - 296) + window.scrollX,
    })
  }

  async function fetchInterpretation() {
    calcPosition()
    if (fetched) { setOpen(true); return }
    setOpen(true)
    setLoading(true)
    setText('')

    try {
      const res = await fetch(`${API_BASE}/ai/interpret`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ card, model_id: modelId, data }),
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let done = false
      while (!done) {
        const { value, done: d } = await reader.read()
        done = d
        if (value) setText(prev => prev + decoder.decode(value, { stream: !d }))
      }
      setFetched(true)
    } catch {
      setText('Could not load interpretation.')
    } finally {
      setLoading(false)
    }
  }

  const popover = open ? createPortal(
    <div
      id="ai-tooltip-popover"
      className="rounded-xl p-3"
      style={{
        position: 'absolute',
        top: popoverPos.top,
        left: popoverPos.left,
        width: 300,
        zIndex: 99999,
        backgroundColor: '#141414',
        border: '1px solid rgba(251,191,36,0.25)',
        boxShadow: '0 12px 40px rgba(0,0,0,0.8)',
      }}
      onClick={e => e.stopPropagation()}
    >
      <div className="flex items-center gap-1.5 mb-2">
        <span style={{ fontSize: 9, color: '#fbbf24', letterSpacing: '0.1em', textTransform: 'uppercase', fontWeight: 600 }}>
          AI Interpretation
        </span>
        {loading && (
          <span className="inline-flex gap-0.5 ml-1">
            {[0, 1, 2].map(i => (
              <span key={i} className="rounded-full animate-bounce" style={{
                width: 3, height: 3, backgroundColor: '#fbbf24',
                animationDelay: `${i * 0.15}s`,
              }} />
            ))}
          </span>
        )}
      </div>
      <p className="text-xs leading-relaxed" style={{ color: '#d1d5db' }}>
        {text || (loading ? '' : '—')}
        {showCursor && loading && (
          <span className="inline-block w-0.5 h-3 ml-0.5 align-middle" style={{ backgroundColor: '#fbbf24' }} />
        )}
      </p>
    </div>,
    document.body,
  ) : null

  return (
    <>
      <button
        ref={buttonRef}
        onClick={e => { e.stopPropagation(); fetchInterpretation() }}
        className="flex items-center justify-center rounded-full transition-all"
        style={{
          width: 20, height: 20,
          backgroundColor: open ? 'rgba(251,191,36,0.2)' : 'rgba(255,255,255,0.06)',
          border: `1px solid ${open ? 'rgba(251,191,36,0.5)' : 'rgba(255,255,255,0.12)'}`,
          color: open ? '#fbbf24' : '#6b7280',
        }}
        title="AI interpretation"
      >
        <span style={{ fontSize: 10, lineHeight: 1, fontWeight: 600 }}>?</span>
      </button>
      {popover}
    </>
  )
}
