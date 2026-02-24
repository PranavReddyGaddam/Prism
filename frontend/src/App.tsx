import React, { useState } from 'react'
import AiPrompt from '@/components/kokonutui/ai-prompt'
import Silk from '@/components/Silk'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const API_BASE = 'https://integrate-commission-surgeons-conceptual.trycloudflare.com'

// ── Types ────────────────────────────────────────────────────────────────────

interface GenerationResult {
  response: string
  thinking: string | null
  final_answer: string | null
  token_count: number
}

interface TokenConfidence {
  token: string
  confidence: number
}

interface AttentionData {
  tokens: string[]
  matrix: number[][]
  layer: number
  head: number
}

interface LogitLensLayer {
  layer: number
  word_position?: number
  predicted_token: string
  probability: number
}

interface GradientAttribution {
  token: string
  score: number
}

interface HiddenStateNorm {
  layer: number
  norm: number
}

interface ExplainData {
  confidence: TokenConfidence[]
  attention: AttentionData | null
  logitLens: LogitLensLayer[]
  attribution: GradientAttribution[]
  hiddenStates: HiddenStateNorm[]
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function smoothScrollTo(targetY: number, duration: number) {
  const startPosition = window.pageYOffset
  const distance = targetY - startPosition
  let start: number | null = null
  const animation = (currentTime: number) => {
    if (start === null) start = currentTime
    const elapsed = currentTime - start
    const progress = Math.min(elapsed / duration, 1)
    const ease = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2
    window.scrollTo(0, startPosition + distance * ease)
    if (elapsed < duration) requestAnimationFrame(animation)
  }
  requestAnimationFrame(animation)
}

function cleanToken(t: string): string {
  return t.replace(/^▁/, ' ').replace(/<0x[0-9A-Fa-f]+>/g, '').trim() || t
}

// ── Visualization Components ─────────────────────────────────────────────────

function confidenceColor(c: number): string {
  if (c >= 0.8) return '#22c55e'
  if (c >= 0.5) return '#eab308'
  return '#ef4444'
}

function TokenConfidenceViz({ data }: { data: TokenConfidence[] }) {
  if (!data.length) return <p className="text-gray-500 text-sm">No token data.</p>
  return (
    <div className="flex flex-wrap gap-1.5">
      {data.map((t, i) => (
        <div key={i} className="flex flex-col items-center gap-0.5">
          <span
            className="px-2 py-1 rounded text-xs font-mono text-black font-semibold"
            style={{ backgroundColor: confidenceColor(t.confidence) }}
          >
            {cleanToken(t.token)}
          </span>
          <span className="text-gray-500 text-[10px]">{(t.confidence * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  )
}

function AttentionViz({ data }: { data: AttentionData }) {
  const MAX_TOKENS = 20
  const tokens = data.tokens.slice(0, MAX_TOKENS).map(cleanToken)
  const matrix = data.matrix.slice(0, MAX_TOKENS).map(r => r.slice(0, MAX_TOKENS))
  // Use a more sensitive scaling for attention weights
  const maxVal = Math.max(...matrix.flat(), 0.1)
  return (
    <div className="overflow-x-auto">
      <p className="text-gray-500 text-xs mb-3">Layer {data.layer} · Head {data.head} · showing first {tokens.length} tokens</p>
      <div style={{ display: 'grid', gridTemplateColumns: `60px repeat(${tokens.length}, 28px)`, gap: '1px' }}>
        <div />
        {tokens.map((t, i) => (
          <div key={i} className="text-gray-500 text-[9px] text-center truncate" style={{ writingMode: 'vertical-rl', height: 60, paddingBottom: 4 }}>
            {t}
          </div>
        ))}
        {matrix.map((row, ri) => (
          <React.Fragment key={`row-${ri}`}>
            <div className="text-gray-400 text-[9px] flex items-center justify-end pr-1 truncate">{tokens[ri]}</div>
            {row.map((val, ci) => (
              <div
                key={`${ri}-${ci}`}
                title={`${tokens[ri]} → ${tokens[ci]}: ${val.toFixed(3)}`}
                style={{
                  width: 28, height: 28,
                  backgroundColor: `rgba(139,92,246,${(val / maxVal).toFixed(3)})`,
                  border: '1px solid #1f1f1f',
                  opacity: val === 0 ? 0.1 : 1,
                }}
              />
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}

function LogitLensViz({ data }: { data: LogitLensLayer[] }) {
  if (!data.length) return <p className="text-gray-500 text-sm">No logit lens data.</p>
  
  // Group data by word position
  const groupedByPosition = data.reduce((acc, item) => {
    const pos = item.word_position || 0
    if (!acc[pos]) acc[pos] = []
    acc[pos].push(item)
    return acc
  }, {} as Record<number, LogitLensLayer[]>)
  
  const positions = Object.keys(groupedByPosition).map(Number).sort()
  const [selectedWordIndex, setSelectedWordIndex] = React.useState(0)
  
  const selectedPos = positions[selectedWordIndex]
  const selectedData = groupedByPosition[selectedPos] || []
  const selectedWord = selectedData[0]?.predicted_token || ''
  
  // Use a better scaling that handles extreme probabilities
  const scaledData = selectedData.map(d => ({
    ...d,
    scaledProb: Math.pow(d.probability, 0.3) * 100 // Power law scaling for better contrast
  }))
  const maxScaled = Math.max(...scaledData.map(d => d.scaledProb), 1)
  
  return (
    <div className="space-y-4">
      <p className="text-gray-500 text-xs mb-3">Select a word to see its layer-by-layer prediction confidence</p>
      
      {/* Word Selector */}
      <div className="flex items-center gap-4">
        <label className="text-gray-400 text-xs w-16">Word:</label>
        <select 
          value={selectedWordIndex}
          onChange={(e) => setSelectedWordIndex(Number(e.target.value))}
          className="bg-gray-800 text-white text-xs px-3 py-1 rounded border border-gray-700 flex-1"
        >
          {positions.map((pos, index) => (
            <option key={pos} value={index}>
              Word {pos + 1}: {cleanToken(groupedByPosition[pos][0]?.predicted_token || '')}
            </option>
          ))}
        </select>
        <div className="bg-gray-800 px-3 py-1 rounded border border-gray-700">
          <span className="text-purple-400 text-xs font-mono">{cleanToken(selectedWord)}</span>
        </div>
      </div>
      
      {/* Layer-by-layer visualization for selected word */}
      <div className="space-y-1">
        <p className="text-gray-500 text-[10px] mb-2">Layer confidence for word "{cleanToken(selectedWord)}"</p>
        {scaledData.map((d) => (
          <div key={d.layer} className="flex items-center gap-3">
            <span className="text-gray-500 text-[10px] w-12 text-right shrink-0">L{d.layer}</span>
            <div className="flex-1 bg-gray-900 rounded-sm h-4 relative overflow-hidden">
              <div className="h-full rounded-sm" style={{ width: `${Math.min(100, (d.scaledProb / maxScaled) * 100)}%`, backgroundColor: '#8b5cf6' }} />
              <span className="absolute inset-0 flex items-center px-2 text-[9px] text-white font-mono truncate">
                {cleanToken(d.predicted_token)}
              </span>
            </div>
            <span className="text-gray-500 text-[9px] w-10 shrink-0">{(d.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function AttributionViz({ data }: { data: GradientAttribution[] }) {
  if (!data.length) return <p className="text-gray-500 text-sm">No attribution data.</p>
  const maxScore = Math.max(...data.map(d => d.score), 0.001)
  return (
    <div className="space-y-1">
      <p className="text-gray-500 text-xs mb-3">Input token importance via gradient norms</p>
      {data.map((d, i) => (
        <div key={i} className="flex items-center gap-3">
          <span className="text-gray-400 text-[10px] font-mono w-24 truncate shrink-0">{cleanToken(d.token)}</span>
          <div className="flex-1 bg-gray-900 rounded-sm h-4 relative overflow-hidden">
            <div className="h-full rounded-sm" style={{ width: `${(d.score / maxScore) * 100}%`, backgroundColor: '#f59e0b' }} />
          </div>
          <span className="text-gray-500 text-[10px] w-14 shrink-0">{d.score.toFixed(4)}</span>
        </div>
      ))}
    </div>
  )
}

function HiddenStatesViz({ data }: { data: HiddenStateNorm[] }) {
  if (!data.length) return <p className="text-gray-500 text-sm">No hidden state data.</p>
  const chartData = data.map(d => ({
    layer: `L${d.layer}`,
    norm: d.norm,
  }))
  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
        <XAxis dataKey="layer" tick={{ fill: '#9ca3af', fontSize: 10 }} />
        <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #374151', borderRadius: '4px' }}
          labelStyle={{ color: '#ffffff', fontSize: 12 }}
          itemStyle={{ color: '#ffffff' }}
          formatter={(value: any) => [(value ?? 0).toFixed(3), 'Norm']}
        />
        <Bar dataKey="norm" radius={[4, 4, 0, 0]} fill="#38bdf8" />
      </BarChart>
    </ResponsiveContainer>
  )
}

function LoadingSection({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-gray-500 text-sm">
      <div className="w-3 h-3 rounded-full border border-gray-600 border-t-white animate-spin" />
      Loading {label}...
    </div>
  )
}

// ── App ──────────────────────────────────────────────────────────────────────

function App() {
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [explainData, setExplainData] = useState<Partial<ExplainData>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [explainLoading, setExplainLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [attentionLayer, setAttentionLayer] = useState(0)
  const [attentionHead, setAttentionHead] = useState(0)
  const [currentPrompt, setCurrentPrompt] = useState('')

  const fetchAttentionForLayer = async (prompt: string, response: string) => {
    const modelId = result?.response ? (result.thinking ? 'deepseek' : 'phi3') : 'phi3'
    const explainBase = { model_id: modelId, prompt, response, max_new_tokens: 64 }
    
    try {
      const res = await fetch(`${API_BASE}/explain/attention`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...explainBase, attn_layer: attentionLayer, attn_head: attentionHead })
      })
      
      if (res.ok) {
        const data = await res.json()
        setExplainData(prev => ({ ...prev, attention: data }))
      }
    } catch (err) {
      console.error('Failed to fetch attention:', err)
    }
  }

  const handleSubmit = async (prompt: string, _model: string) => {
    setIsLoading(true)
    setResult(null)
    setExplainData({})
    setError(null)
    setCurrentPrompt(prompt)

    // Temporarily map both models to deepseek since phi3 is not working
    const modelId = 'deepseek'

    setTimeout(() => {
      const section = document.getElementById('explanation-section')
      if (section) smoothScrollTo(section.offsetTop - 100, 1500)
    }, 100)

    try {
      const res = await fetch(`${API_BASE}/generate/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, prompt, max_new_tokens: 1024 }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }
      const data: GenerationResult = await res.json()
      setResult(data)
      setIsLoading(false)

      setExplainLoading(true)
      const explainBase = { model_id: modelId, prompt, response: data.response, max_new_tokens: 64 }
      const body = JSON.stringify(explainBase)
      const headers = { 'Content-Type': 'application/json' }

      const [confRes, attnRes, logitRes, hiddenRes] = await Promise.allSettled([
        fetch(`${API_BASE}/explain/confidence`, { method: 'POST', headers, body }),
        fetch(`${API_BASE}/explain/attention`, { method: 'POST', headers, body: JSON.stringify({ ...explainBase, attn_layer: attentionLayer, attn_head: attentionHead }) }),
        fetch(`${API_BASE}/explain/logit-lens`, { method: 'POST', headers, body }),
        fetch(`${API_BASE}/explain/hidden-states`, { method: 'POST', headers, body }),
      ])

      const partial: Partial<ExplainData> = {}
      if (confRes.status === 'fulfilled' && confRes.value.ok) {
        const d = await confRes.value.json()
        partial.confidence = d.token_confidence ?? []
      }
      if (attnRes.status === 'fulfilled' && attnRes.value.ok) {
        partial.attention = await attnRes.value.json()
      }
      if (logitRes.status === 'fulfilled' && logitRes.value.ok) {
        const d = await logitRes.value.json()
        partial.logitLens = d.logit_lens ?? []
      }
      if (hiddenRes.status === 'fulfilled' && hiddenRes.value.ok) {
        const d = await hiddenRes.value.json()
        partial.hiddenStates = d.hidden_state_norms ?? []
      }

      setExplainData(partial)
      setExplainLoading(false)

      // Attribution is slow — fire separately
      fetch(`${API_BASE}/explain/attribution`, { method: 'POST', headers, body: JSON.stringify({ ...explainBase }) })
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (d) setExplainData(prev => ({ ...prev, attribution: d.gradient_attribution ?? [] }))
        })
        .catch(() => {})

    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
      setIsLoading(false)
      setExplainLoading(false)
    }
  }

  return (
    <div className="min-h-screen relative">
      {/* Silk full-page fixed background */}
      <div className="fixed inset-0 z-0">
        <Silk speed={2} scale={1.5} color="#39383d" noiseIntensity={0.3} rotation={0} />
      </div>

      {/* All content above background */}
      <div className="relative z-10">
        <nav className="px-8 py-3">
          <div className="text-white text-2xl font-bold">Prism</div>
        </nav>

        {/* Hero section */}
        <div className="flex flex-col items-center justify-center px-20 py-16 h-screen">
          <div className="text-center mb-20">
            <h1 className="text-7xl font-bold text-white leading-tight mb-2 tracking-wide">
              Inside the Black Box
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
              Attention, reasoning, and confidence - all visible for the first time
            </p>
          </div>
          <div className="w-full max-w-3xl">
            <AiPrompt onSubmit={handleSubmit} />
          </div>
        </div>

        {/* Response + Explainability */}
        {(result || isLoading || error) && (
          <div id="explanation-section" className="px-20 py-16">
            <div className="flex border border-gray-800 rounded-lg bg-black">

              {/* Left — Chain of Thought (sticky) */}
              <div className="w-1/2 border-r border-gray-800 sticky top-0 self-start h-screen p-6 overflow-y-auto bg-black">
                <h2 className="text-white text-lg font-semibold mb-4">Chain of Thought</h2>
                {isLoading ? (
                  <div className="text-gray-400 animate-pulse">Thinking...</div>
                ) : error ? (
                  <div className="text-red-400 text-sm">{error}</div>
                ) : result ? (
                  <div className="space-y-4">
                    {result.thinking && (
                      <div>
                        <p className="text-gray-400 text-xs uppercase tracking-widest mb-2">Reasoning</p>
                        <div className="text-gray-300 text-sm whitespace-pre-wrap leading-relaxed">{result.thinking}</div>
                      </div>
                    )}
                    {result.final_answer && (
                      <div>
                        <p className="text-gray-400 text-xs uppercase tracking-widest mb-2">Final Answer</p>
                        <div className="text-white font-medium whitespace-pre-wrap">{result.final_answer}</div>
                      </div>
                    )}
                    {!result.thinking && !result.final_answer && (
                      <div className="text-white whitespace-pre-wrap">{result.response}</div>
                    )}
                  </div>
                ) : null}
              </div>

              {/* Right — Explainability panels */}
              <div className="w-1/2 bg-black">

                {/* Token Analysis */}
                <div className="p-6 min-h-screen border-b border-gray-800">
                  <h2 className="text-white text-lg font-semibold mb-1">Token Analysis</h2>
                  <p className="text-gray-500 text-xs mb-4">Per-token output confidence</p>
                  {explainLoading && !explainData.confidence
                    ? <LoadingSection label="token confidence" />
                    : explainData.confidence
                      ? <TokenConfidenceViz data={explainData.confidence} />
                      : <p className="text-gray-600 text-sm">Submit a query to see token confidence.</p>
                  }
                </div>

                {/* Attention Patterns */}
                <div className="p-6 min-h-screen border-b border-gray-800">
                  <h2 className="text-white text-lg font-semibold mb-1">Attention Patterns</h2>
                  <p className="text-gray-500 text-xs mb-4">Layer {attentionLayer}, Head {attentionHead} attention weights</p>
                  
                  {/* Layer and Head Controls */}
                  {result && (
                    <div className="space-y-3 mb-4">
                      <div className="flex items-center gap-4">
                        <label className="text-gray-400 text-xs w-12">Layer:</label>
                        <input 
                          type="range"
                          min="0"
                          max="31"
                          value={attentionLayer}
                          onChange={(e) => {
                            const newLayer = Number(e.target.value)
                            setAttentionLayer(newLayer)
                            fetchAttentionForLayer(currentPrompt, result.response)
                          }}
                          className="flex-1 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                        />
                        <span className="text-white text-xs w-8 text-right">{attentionLayer}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <label className="text-gray-400 text-xs w-12">Head:</label>
                        <input 
                          type="range"
                          min="0"
                          max="31"
                          value={attentionHead}
                          onChange={(e) => {
                            const newHead = Number(e.target.value)
                            setAttentionHead(newHead)
                            fetchAttentionForLayer(currentPrompt, result.response)
                          }}
                          className="flex-1 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                        />
                        <span className="text-white text-xs w-8 text-right">{attentionHead}</span>
                      </div>
                    </div>
                  )}

                  {explainLoading && !explainData.attention
                    ? <LoadingSection label="attention" />
                    : explainData.attention
                      ? <AttentionViz data={explainData.attention} />
                      : <p className="text-gray-600 text-sm">Submit a query to see attention patterns.</p>
                  }
                </div>

                {/* Layer Activations */}
                <div className="p-6 min-h-screen border-b border-gray-800">
                  <h2 className="text-white text-lg font-semibold mb-1">Layer Activations</h2>
                  <p className="text-gray-500 text-xs mb-4">Logit lens — prediction at each layer</p>
                  {explainLoading && !explainData.logitLens
                    ? <LoadingSection label="logit lens" />
                    : explainData.logitLens
                      ? <LogitLensViz data={explainData.logitLens} />
                      : <p className="text-gray-600 text-sm">Submit a query to see layer activations.</p>
                  }
                </div>

                {/* Gradient Attribution */}
                <div className="p-6 min-h-screen border-b border-gray-800">
                  <h2 className="text-white text-lg font-semibold mb-1">Gradient Attribution</h2>
                  <p className="text-gray-500 text-xs mb-4">Input token importance scores</p>
                  {!explainData.attribution
                    ? <LoadingSection label="gradient attribution" />
                    : <AttributionViz data={explainData.attribution} />
                  }
                </div>

                {/* Hidden States */}
                <div className="p-6 min-h-screen">
                  <h2 className="text-white text-lg font-semibold mb-1">Hidden States</h2>
                  <p className="text-gray-500 text-xs mb-4">L2 norm of hidden states across layers</p>
                  {explainLoading && !explainData.hiddenStates
                    ? <LoadingSection label="hidden states" />
                    : explainData.hiddenStates
                      ? <HiddenStatesViz data={explainData.hiddenStates} />
                      : <p className="text-gray-600 text-sm">Submit a query to see hidden states.</p>
                  }
                </div>

              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
