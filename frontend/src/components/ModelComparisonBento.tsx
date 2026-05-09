import { useState, useEffect, useMemo } from 'react'
import AITooltip from '@/components/AITooltip'
import AttributionGraphVisualization from '@/components/visualizations/AttributionGraph'
import NodeDetailPanel from '@/components/visualizations/NodeDetailPanel'
import TokenFlow from '@/components/visualizations/TokenFlow'
import type { AttributionGraph, AttributionNode } from '@/types/attribution'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList } from 'recharts'
import { buildAttributionGraph, DEFAULT_MAX_LAYERS, DEFAULT_MAX_OUTPUTS } from '@/lib/attributionGraph'

// Strip LaTeX delimiters and normalize math notation for plain-text display
export function cleanMath(text: string): string {
  return text
    .replace(/\\\(|\\\)/g, '')           // remove \( \)
    .replace(/\\\[|\\\]/g, '')           // remove \[ \]
    .replace(/\\boxed\{([^}]*)\}/g, '[$1]')  // \boxed{x} → [x]
    .replace(/\\\\/g, '')                // remove leftover \\
    .replace(/\\_/g, '_')               // \_ → _
    .replace(/\\text\{([^}]*)\}/g, '$1') // \text{x} → x
    .replace(/\$\$([^$]+)\$\$/g, '$1')  // $$x$$ → x
    .replace(/\$([^$]+)\$/g, '$1')      // $x$ → x
    .trim()
}

export type SlotStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface GenerationResult {
  model_id: string
  response: string
  thinking: string | null
  final_answer: string | null
  token_count: number
}

export interface TokenConfidence {
  token: string
  confidence: number
}

export interface LogitLensLayer {
  layer: number
  word_position?: number
  predicted_token: string
  probability: number
}

export interface GradientAttribution {
  token: string
  score: number
}

export interface HiddenStateNorm {
  layer: number
  norm: number
  delta?: number
  token_norms?: number[]
}

export interface ExplainData {
  confidence: TokenConfidence[]
  logitLens: LogitLensLayer[]
  attribution: GradientAttribution[]
  hiddenStates: HiddenStateNorm[]
}

export interface RawAttribution {
  gradientAttribution: { token: string; score: number }[]
  positionPredictions: {
    position: number
    token: string
    top: { token: string; probability: number; rank: number }[]
    bottom: { token: string; probability: number; rank: number }[]
  }[]
  layerAttention: { layer: number; weights: { token: string; weight: number }[] }[]
  prompt: string
}

export interface SlotState {
  status: SlotStatus
  result: GenerationResult | null
  explain: Partial<ExplainData>
  graph: AttributionGraph | null
  rawAttribution: RawAttribution | null
  hiddenTokens?: string[]
}

interface ModelComparisonBentoProps {
  slot: SlotState
  onExpandCard: (card: string) => void
}

export function ModelComparisonBento({ slot, onExpandCard }: ModelComparisonBentoProps) {
  const isLoading = slot.status === 'loading'
  const result = slot.result
  const explainData = slot.explain
  const attributionGraph = slot.graph
  const tokens = explainData.confidence || []

  return (
    <div className="grid grid-cols-12 gap-3 auto-rows-[140px]">
      <div
        onClick={() => !isLoading && result && onExpandCard('response')}
        className="col-span-3 row-span-2 border border-gray-700/50 rounded-xl p-4 hover:border-white/60 transition-all duration-200 cursor-pointer group active:scale-[0.98]"
        style={{ backgroundColor: '#202020' }}
      >
        <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">AI Response</div>
          <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </h3>
        <div className="flex flex-col gap-2 h-[calc(100%-2.5rem)] overflow-hidden">
          {isLoading ? (
            <>
              <div className="p-3 bg-gray-800/50 rounded-lg animate-pulse flex-1">
                <div className="h-2 bg-gray-700 rounded w-20 mb-2"></div>
                <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                <div className="h-2 bg-gray-700 rounded w-3/4"></div>
              </div>
              <div className="p-4 bg-gradient-to-r from-amber-500/10 to-green-500/10 border border-amber-500/30 rounded-lg animate-pulse flex-1">
                <div className="h-2 bg-amber-500/30 rounded w-24 mb-2"></div>
                <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                <div className="h-2 bg-gray-700 rounded w-2/3"></div>
              </div>
            </>
          ) : result ? (
            <div className="p-4 rounded-lg flex-1 overflow-hidden border border-gray-700/40" style={{backgroundColor: '#2a2a2a'}}>
              <p className="text-gray-400 text-[10px] font-semibold uppercase tracking-wider mb-2">Final Answer</p>
              <p className="text-white text-sm line-clamp-6">{cleanMath(result.final_answer || result.response)}</p>
            </div>
          ) : null}
        </div>
      </div>

      <div
        onClick={() => !isLoading && result && onExpandCard('attribution')}
        className="col-span-3 row-span-2 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-white/60 transition-all duration-200 cursor-pointer group active:scale-[0.98]"
        style={{
          backgroundImage:
            'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)',
        }}
      >
        <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">Attribution Graph</div>
          <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </h3>
        <div className="flex flex-col h-[calc(100%-2rem)] gap-2">
          {isLoading ? (
            <div className="flex-1 rounded-lg bg-gray-800/40 animate-pulse" />
          ) : attributionGraph ? (
            <>
              {/* Mini graph preview */}
              <div className="flex-1 relative overflow-hidden rounded-lg bg-black/20">
                {(() => {
                  const W = 300, H = 160
                  const layers = new Map<number, typeof attributionGraph.nodes>()
                  attributionGraph.nodes.forEach(n => {
                    const l = n.layer ?? (n.type === 'input' ? 0 : n.type === 'output' ? 3 : 2)
                    if (!layers.has(l)) layers.set(l, [])
                    layers.get(l)!.push(n)
                  })
                  const layerArr = Array.from(layers.entries()).sort((a, b) => a[0] - b[0])
                  const pos = new Map<string, {x: number, y: number}>()
                  layerArr.forEach(([, nodes], li) => {
                    const x = (li + 1) * (W / (layerArr.length + 1))
                    nodes.forEach((n, ni) => {
                      pos.set(n.id, { x, y: (ni + 1) * (H / (nodes.length + 1)) })
                    })
                  })
                  const maxW = Math.max(...attributionGraph.edges.map(e => Math.abs(e.weight)), 1)
                  const nodeColor = (type: string) =>
                    type === 'input' ? '#e5e5e5' : type === 'output' ? '#ffffff' : type === 'error' ? '#555' : '#aaa'
                  return (
                    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
                      <g opacity="0.6">
                        {attributionGraph.edges.slice(0, 60).map((e, i) => {
                          const s = pos.get(e.source), t = pos.get(e.target)
                          if (!s || !t) return null
                          return <line key={i} x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                            stroke="#666" strokeWidth={Math.max(0.3, Math.abs(e.weight) / maxW * 1.5)}
                            strokeOpacity={0.3 + (Math.abs(e.weight) / maxW) * 0.4} />
                        })}
                      </g>
                      {attributionGraph.nodes.map(n => {
                        const p = pos.get(n.id)
                        if (!p) return null
                        const r = n.type === 'output' ? 5 : n.type === 'input' ? 4 : 3
                        return <circle key={n.id} cx={p.x} cy={p.y} r={r}
                          fill={nodeColor(n.type)} opacity={0.9} />
                      })}
                    </svg>
                  )
                })()}
              </div>
              {/* Stats row */}
              <div className="flex items-center justify-between text-[10px] px-0.5">
                <div className="flex items-center gap-3 text-gray-500">
                  <span>{attributionGraph.nodes.length} nodes</span>
                  <span className="text-gray-700">·</span>
                  <span>{attributionGraph.edges.length} edges</span>
                </div>
                <span className="text-purple-400 font-semibold">
                  {((attributionGraph.explainedBehavior || 0) * 100).toFixed(0)}% explained
                </span>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-gray-500 text-xs">No attribution data</div>
          )}
        </div>
      </div>

      <div
        onClick={() => !isLoading && result && onExpandCard('hidden')}
        className="col-span-6 row-span-4 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-white/60 transition-all duration-200 cursor-pointer group active:scale-[0.98]"
        style={{
          backgroundImage:
            'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)',
        }}
      >
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white font-semibold flex items-center gap-2">
            Hidden State Norms
            <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </h3>
          <div className="text-xs text-gray-400">Layer-wise activation magnitudes</div>
        </div>
        {isLoading ? (
          <div className="overflow-hidden animate-pulse h-64">
            <div className="flex items-end justify-between h-full gap-1">
              {Array.from({ length: 35 }).map((_, i) => (
                <div key={i} className="flex-1 bg-gray-700 rounded-t" style={{ height: `${Math.random() * 100}%` }}></div>
              ))}
            </div>
          </div>
        ) : explainData.hiddenStates && explainData.hiddenStates.length > 0 ? (
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={explainData.hiddenStates} margin={{ top: 5, right: 5, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis dataKey="layer" stroke="#9ca3af" fontSize={9} tickLine={false} interval="preserveStartEnd" />
                <YAxis stroke="#9ca3af" fontSize={9} tickLine={false} width={40} />
                <Tooltip
                  cursor={false}
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    fontSize: '12px',
                  }}
                  labelStyle={{ color: '#c9a96e' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="norm" fill="#c9a96e" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">No hidden state data available</div>
        )}
      </div>

      <div
        onClick={() => !isLoading && result && onExpandCard('flow')}
        className="col-span-6 row-span-2 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-white/60 transition-all duration-200 cursor-pointer group active:scale-[0.98]"
        style={{
          backgroundImage:
            'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)',
        }}
      >
        <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2">Token Flow</div>
          <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </h3>
        {isLoading ? (
          <div className="flex items-end gap-2 h-[160px] animate-pulse px-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="flex-1 bg-gray-700 rounded-t" style={{ height: `${30 + Math.random() * 70}%` }} />
            ))}
          </div>
        ) : tokens.length > 0 ? (() => {
          const W = 600, H = 160, pad = { t: 10, r: 8, b: 28, l: 8 }
          const vals = tokens.map(t => t.confidence * 100)
          const minV = Math.max(0, Math.min(...vals) - 8)
          const maxV = Math.min(100, Math.max(...vals) + 8)
          const xStep = (W - pad.l - pad.r) / (vals.length - 1)
          const toY = (v: number) => pad.t + (1 - (v - minV) / (maxV - minV)) * (H - pad.t - pad.b)
          const avg = vals.reduce((s, v) => s + v, 0) / vals.length
          const pts = vals.map((v, i) => [pad.l + i * xStep, toY(v)] as [number, number])
          const linePath = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')
          const areaPath = `${linePath} L${pts[pts.length-1][0].toFixed(1)},${(H - pad.b).toFixed(1)} L${pts[0][0].toFixed(1)},${(H - pad.b).toFixed(1)} Z`
          const avgY = toY(avg)
          const clean = (tok: string) => tok.replace(/^▁/, '').replace(/<0x[0-9A-Fa-f]+>/g, '').trim() || tok
          return (
            <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
              <defs>
                <linearGradient id="bentoCG" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="hsl(221.2 83.2% 53.3%)" stopOpacity={0.7}/>
                  <stop offset="95%" stopColor="hsl(221.2 83.2% 53.3%)" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              {/* avg reference line */}
              <line x1={pad.l} y1={avgY} x2={W - pad.r} y2={avgY}
                stroke="hsl(45 93.4% 47.5%)" strokeWidth={1} strokeDasharray="4 3" opacity={0.7}/>
              {/* area fill */}
              <path d={areaPath} fill="url(#bentoCG)"/>
              {/* line */}
              <path d={linePath} fill="none" stroke="hsl(221.2 83.2% 53.3%)" strokeWidth={2} strokeLinejoin="round"/>
              {/* dots + token labels */}
              {pts.map((p, i) => (
                <g key={i}>
                  <circle cx={p[0]} cy={p[1]} r={3} fill="hsl(221.2 83.2% 65%)" />
                  <text x={p[0]} y={H - 6} textAnchor="middle" fontSize={9}
                    fill="hsl(0 0% 55%)" fontFamily="monospace">
                    {clean(tokens[i].token).slice(0, 6)}
                  </text>
                </g>
              ))}
            </svg>
          )
        })() : null}
      </div>

    </div>
  )
}

interface ExpandedModalProps {
  expandedCard: string
  slot: SlotState
  onClose: () => void
}

export function ModelComparisonModal({ expandedCard, slot, onClose }: ExpandedModalProps) {
  const result = slot.result
  const explainData = slot.explain
  const tokens = explainData.confidence || []
  const [inspectedNode, setInspectedNode] = useState<AttributionNode | null>(null)
  const [mounted, setMounted] = useState(false)
  const [closing, setClosing] = useState(false)
  const [maxLayers, setMaxLayers] = useState(DEFAULT_MAX_LAYERS)
  const [maxOutputs, setMaxOutputs] = useState(DEFAULT_MAX_OUTPUTS)
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null)

  const totalLayers = slot.rawAttribution?.layerAttention.length ?? 0
  const totalOutputs = slot.rawAttribution
    ? Math.max(0, slot.rawAttribution.positionPredictions.length - Math.floor(slot.rawAttribution.positionPredictions.length / 2))
    : 0

  const attributionGraph = useMemo(() => {
    if (!slot.rawAttribution) return slot.graph
    const r = slot.rawAttribution
    return buildAttributionGraph(
      r.prompt,
      r.gradientAttribution,
      r.positionPredictions,
      r.layerAttention,
      maxLayers,
      maxOutputs,
    )
  }, [slot.rawAttribution, slot.graph, maxLayers, maxOutputs])

  const handleClose = () => {
    setClosing(true)
    setMounted(false)
    setTimeout(onClose, 250)
  }

  useEffect(() => {
    const t = requestAnimationFrame(() => setMounted(true))
    return () => cancelAnimationFrame(t)
  }, [])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') handleClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  if (!result) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-8 backdrop-blur-sm transition-all duration-250 ease-out"
      style={{ backgroundColor: closing ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,0.8)' }}
      onClick={handleClose}
    >
      <div
        className="bg-gradient-to-br from-gray-900/90 via-gray-800/80 to-gray-900/90 border border-gray-700/50 rounded-2xl p-6 max-w-[95vw] w-full max-h-[92vh] overflow-hidden backdrop-blur-xl transition-all duration-250 ease-out"
        style={{
          backgroundImage:
            'linear-gradient(145deg, rgba(67, 67, 67, 0.4) 0%, rgba(29, 29, 29, 0.6) 50%, rgba(67, 67, 67, 0.4) 100%)',
          opacity: mounted ? 1 : 0,
          transform: mounted ? 'scale(1) translateY(0)' : closing ? 'scale(0.96) translateY(8px)' : 'scale(0.95) translateY(12px)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="flex items-center justify-between mb-4 transition-all duration-300 ease-out"
          style={{ opacity: mounted ? 1 : 0, transform: mounted ? 'translateY(0)' : 'translateY(-8px)' }}
        >
          <div className="flex items-center gap-3">
            <h2 className="text-2xl font-bold text-white">
              {expandedCard === 'response' && 'AI Response'}
              {expandedCard === 'attribution' && 'Attribution Graph Analysis'}
              {expandedCard === 'hidden' && 'Hidden State Analysis'}
              {expandedCard === 'flow' && 'Token Generation Flow'}
            </h2>
            {result && (() => {
              const modelId = result.model_id
              if (expandedCard === 'response') return (
                <AITooltip card="response" modelId={modelId} data={{
                  prompt: cleanMath(slot.rawAttribution?.prompt ?? '').slice(0, 200),
                  token_count: result.token_count,
                  final_answer: result.final_answer,
                  response_preview: cleanMath(result.response ?? '').slice(0, 600),
                }} />
              )
              if (expandedCard === 'attribution' && explainData.attribution?.length) {
                const sorted = [...explainData.attribution].sort((a,b) => Math.abs(b.score)-Math.abs(a.score))
                return (
                  <AITooltip card="attribution" modelId={modelId} data={{
                    prompt: cleanMath(slot.rawAttribution?.prompt ?? '').slice(0, 200),
                    top_tokens: sorted.slice(0, 8).map(t => ({ token: t.token.trim(), score: +t.score.toFixed(4) })),
                    bottom_tokens: sorted.slice(-3).map(t => ({ token: t.token.trim(), score: +t.score.toFixed(4) })),
                  }} />
                )
              }
              if (expandedCard === 'hidden' && explainData.hiddenStates?.length) {
                const hs = explainData.hiddenStates
                const maxNorm = Math.max(...hs.map(h => h.norm))
                const peakLayer = hs.find(h => h.norm === maxNorm)?.layer ?? 0
                const maxDrift = Math.max(...hs.map(h => h.delta ?? 0))
                const maxDriftLayer = hs.find(h => (h.delta ?? 0) === maxDrift)?.layer ?? 0
                // Sample every 4th layer for the trend
                const trend = hs.filter((_,i) => i % 4 === 0).map(h => ({ layer: h.layer, norm: +h.norm.toFixed(1) }))
                return <AITooltip card="hidden" modelId={modelId} data={{
                  layer_count: hs.length,
                  avg_norm: +(hs.reduce((s,h)=>s+h.norm,0)/hs.length).toFixed(1),
                  peak_norm: +maxNorm.toFixed(1), peak_layer: peakLayer,
                  max_drift: +maxDrift.toFixed(1), max_drift_layer: maxDriftLayer,
                  norm_trend: trend,
                }} />
              }
              if (expandedCard === 'flow' && tokens.length > 0) {
                // Pass lowest-confidence tokens so GPT can name them specifically
                const sorted = [...tokens].sort((a,b) => a.confidence - b.confidence)
                return (
                  <AITooltip card="confidence" modelId={modelId} data={{
                    token_count: tokens.length,
                    avg_confidence: +(tokens.reduce((s,t)=>s+t.confidence,0)/tokens.length).toFixed(4),
                    min_confidence: +Math.min(...tokens.map(t=>t.confidence)).toFixed(4),
                    max_confidence: +Math.max(...tokens.map(t=>t.confidence)).toFixed(4),
                    lowest_confidence_tokens: sorted.slice(0, 5).map(t => ({ token: t.token, confidence: +t.confidence.toFixed(3) })),
                    highest_confidence_tokens: sorted.slice(-5).reverse().map(t => ({ token: t.token, confidence: +t.confidence.toFixed(3) })),
                  }} />
                )
              }
              return null
            })()}
          </div>
          <button
            type="button"
            onClick={handleClose}
            className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-800 rounded-lg"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <div
          className="bg-gradient-to-br from-gray-800/60 via-gray-900/50 to-gray-800/60 rounded-xl p-4 backdrop-blur-md transition-all duration-500 ease-out"
          style={{
            backgroundImage:
              'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.4) 50%, rgba(67, 67, 67, 0.3) 100%)',
            opacity: mounted ? 1 : 0,
            transform: mounted ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.99)',
            transitionDelay: '80ms',
          }}
        >
          {expandedCard === 'attribution' && attributionGraph && (
            <div className="flex flex-col gap-3" style={{ height: '600px' }}>
              {totalLayers > 0 && (
                <div className="flex items-center gap-6 px-3 py-2 rounded-lg bg-gray-900/40 border border-gray-700/40">
                  <div className="flex items-center gap-3 flex-1">
                    <label className="text-gray-400 text-xs font-medium whitespace-nowrap">
                      Layers shown:
                      <span className="text-white ml-2 tabular-nums">{Math.min(maxLayers, totalLayers)} / {totalLayers}</span>
                    </label>
                    <input
                      type="range"
                      min={1}
                      max={totalLayers}
                      value={Math.min(maxLayers, totalLayers)}
                      onChange={(e) => setMaxLayers(parseInt(e.target.value, 10))}
                      className="flex-1 accent-white"
                    />
                  </div>
                  <div className="flex items-center gap-3 flex-1">
                    <label className="text-gray-400 text-xs font-medium whitespace-nowrap">
                      Output tokens:
                      <span className="text-white ml-2 tabular-nums">{Math.min(maxOutputs, totalOutputs)} / {totalOutputs}</span>
                    </label>
                    <input
                      type="range"
                      min={1}
                      max={Math.max(1, totalOutputs)}
                      value={Math.min(maxOutputs, Math.max(1, totalOutputs))}
                      onChange={(e) => setMaxOutputs(parseInt(e.target.value, 10))}
                      className="flex-1 accent-white"
                    />
                  </div>
                </div>
              )}
              <div className="flex gap-3 flex-1 min-h-0">
              <div className="flex-1 min-w-0 overflow-hidden transition-all duration-300 ease-out">
                <AttributionGraphVisualization
                  data={attributionGraph}
                  width={1200}
                  height={560}
                  onInspectNode={setInspectedNode}
                />
              </div>
              <div
                className="shrink-0 overflow-hidden transition-all duration-300 ease-out"
                style={{ width: inspectedNode ? '18rem' : '0', opacity: inspectedNode ? 1 : 0 }}
              >
                {inspectedNode && (
                  <div style={{ width: '18rem', height: '560px' }}>
                    <NodeDetailPanel node={inspectedNode} onClose={() => setInspectedNode(null)} />
                  </div>
                )}
              </div>
              </div>
            </div>
          )}
          {expandedCard === 'hidden' && explainData.hiddenStates && explainData.hiddenStates.length > 0 && (() => {
            const data = explainData.hiddenStates
            const hiddenTokens = slot.hiddenTokens ?? []
            const maxNorm = Math.max(...data.map((d) => d.norm))
            const avgNorm = data.reduce((sum, d) => sum + d.norm, 0) / data.length
            const maxDelta = Math.max(...data.map(d => d.delta ?? 0), 1e-9)

            const selectedLayerData = selectedLayer !== null ? data.find(d => d.layer === selectedLayer) : null
            const tokenNorms = selectedLayerData?.token_norms ?? []
            const maxTokenNorm = Math.max(...tokenNorms, 1e-9)

            return (
              <div className="space-y-6">
                {/* Summary stats */}
                <div className="grid grid-cols-4 gap-3">
                  {[
                    { label: 'Total Layers', value: data.length, color: '#fff' },
                    { label: 'Avg Norm', value: avgNorm.toFixed(1), color: '#c9a96e' },
                    { label: 'Peak Norm', value: maxNorm.toFixed(1), color: '#86efac' } as const,
                    { label: 'Max Drift', value: maxDelta.toFixed(1), color: '#f9a8d4' },
                  ].map(s => (
                    <div key={s.label} className="p-4 rounded-xl" style={{ backgroundColor: '#2a2a2a', border: '1px solid rgba(255,255,255,0.08)' }}>
                      <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: '#6b7280' }}>{s.label}</div>
                      <div className="text-2xl font-bold font-mono" style={{ color: s.color }}>{s.value}</div>
                    </div>
                  ))}
                </div>

                <div className="text-xs" style={{ color: '#6b7280' }}>
                  Click any bar to drill down into per-token activations at that layer.
                  {selectedLayer !== null && <span className="ml-2" style={{ color: '#c9a96e' }}>Showing Layer {selectedLayer} ↓</span>}
                </div>

                {/* Main bar chart — clickable */}
                <div className="h-72 rounded-xl p-4" style={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                    <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 20 }} onClick={(e: any) => {
                      if (e?.activePayload?.[0]) {
                        const layer = (e.activePayload[0].payload as HiddenStateNorm).layer
                        setSelectedLayer((prev: number | null) => prev === layer ? null : layer)
                      }
                    }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="layer" stroke="#4b5563" fontSize={11} tickLine={false} tick={{ fill: '#6b7280' }}
                        label={{ value: 'Layer', position: 'insideBottom', offset: -12, fill: '#6b7280', fontSize: 12 }} />
                      <YAxis stroke="#4b5563" fontSize={11} tickLine={false} tick={{ fill: '#6b7280' }} width={50} />
                      <Tooltip
                        cursor={{ fill: 'transparent' }}
                        contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                        labelStyle={{ color: '#c9a96e' }}
                        itemStyle={{ color: '#fff' }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={((v: number) => [v.toFixed(2), 'Norm']) as any}
                        labelFormatter={(l) => `Layer ${l}`}
                      />
                      <Bar dataKey="norm" radius={[3, 3, 0, 0]} cursor="pointer">
                        <LabelList
                          dataKey="norm"
                          position="top"
                          fontSize={9}
                          fill="#9ca3af"
                          // eslint-disable-next-line @typescript-eslint/no-explicit-any
                          formatter={((v: number) => v >= 100 ? v.toFixed(0) : v.toFixed(1)) as any}
                        />
                        {data.map((d) => (
                          <Cell
                            key={d.layer}
                            fill={selectedLayer === d.layer ? '#f9a8d4' : (d.delta ?? 0) > maxDelta * 0.6 ? '#86efac' : '#c9a96e'}
                            opacity={selectedLayer !== null && selectedLayer !== d.layer ? 0.4 : 1}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Drift overlay chart */}
                {data.some(d => d.delta !== undefined) && (
                  <div>
                    <div className="text-xs uppercase tracking-wider mb-2" style={{ color: '#6b7280' }}>Layer-to-layer drift (Δ norm)</div>
                    <div className="h-24 rounded-xl px-4 pt-2" style={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(255,255,255,0.06)' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data.slice(1)} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                          <XAxis dataKey="layer" hide />
                          <YAxis hide />
                          <Tooltip
                            cursor={{ fill: 'transparent' }}
                            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 11 }}
                            labelStyle={{ color: '#f9a8d4' }}
                            itemStyle={{ color: '#fff' }}
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            formatter={((v: number) => [v.toFixed(2), 'Drift']) as any}
                            labelFormatter={(l) => `Layer ${l}`}
                          />
                          <Bar dataKey="delta" radius={[2, 2, 0, 0]}>
                            {data.slice(1).map((d) => (
                              <Cell key={d.layer} fill={(d.delta ?? 0) > maxDelta * 0.6 ? '#86efac' : 'rgba(249,168,212,0.5)'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {/* Drill-down: per-token heatmap for selected layer */}
                {selectedLayerData && tokenNorms.length > 0 && (
                  <div className="rounded-xl p-4" style={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(249,168,212,0.2)' }}>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm font-semibold" style={{ color: '#f9a8d4' }}>
                        Layer {selectedLayer} — per-token activations
                      </span>
                      <span className="text-xs font-mono" style={{ color: '#6b7280' }}>
                        mean norm: {selectedLayerData.norm.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {tokenNorms.map((norm, idx) => {
                        const intensity = norm / maxTokenNorm
                        const label = hiddenTokens[idx] ?? `t${idx}`
                        const bg = `rgba(201,169,110,${0.1 + intensity * 0.75})`
                        const textCol = intensity > 0.5 ? '#000' : '#fff'
                        return (
                          <div
                            key={idx}
                            className="rounded px-2 py-1 text-xs font-mono"
                            style={{ backgroundColor: bg, color: textCol }}
                            title={`${label}: ${norm.toFixed(3)}`}
                          >
                            {label.replace(/^▁/, ' ').trim() || '·'}
                            <span className="ml-1 text-[9px] opacity-70">{norm.toFixed(1)}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}

                <div className="text-xs" style={{ color: '#4b5563' }}>
                  Green bars = high-drift layers where the model is doing the most computation.
                  Pink = selected layer. Click again to deselect.
                </div>
              </div>
            )
          })()}
          {expandedCard === 'flow' && tokens.length > 0 && (
            <div className="space-y-4">
              <div className="text-sm text-gray-400">Showing {tokens.length} tokens with confidence scores</div>
              <div className="overflow-auto" style={{ maxHeight: '70vh' }}>
                <TokenFlow tokens={tokens.map((t) => t.token)} confidence={tokens.map((t) => t.confidence)} />
              </div>
            </div>
          )}
          {expandedCard === 'response' && (() => {
            const response = result.response || ''
            const stepMatches = response.match(/Step \d+:.*?(?=Step \d+:|$)/gs) || []
            const hasSteps = stepMatches.length > 0
            const avgConfidence = tokens.length > 0
              ? (tokens.reduce((sum, t) => sum + t.confidence, 0) / tokens.length) * 100
              : 0

            return (
              <div className={`flex gap-6 ${hasSteps ? 'items-start' : ''}`}>
                {/* Left — answer + thinking */}
                <div className={`flex flex-col gap-4 ${hasSteps ? 'w-2/5 shrink-0' : 'w-full'}`}>
                  {result.thinking && (
                    <div className="rounded-xl border border-gray-800 overflow-hidden">
                      <div className="px-4 py-2 border-b border-gray-800" style={{backgroundColor:'#1a1a1a'}}>
                        <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">Thinking</span>
                      </div>
                      <div className="px-4 py-3 max-h-48 overflow-y-auto" style={{backgroundColor:'#161616'}}>
                        <p className="text-gray-400 text-sm leading-relaxed whitespace-pre-wrap">{cleanMath(result.thinking)}</p>
                      </div>
                    </div>
                  )}
                  <div className="rounded-xl border border-gray-700 overflow-hidden">
                    <div className="px-4 py-2 border-b border-gray-700" style={{backgroundColor:'#1e1e1e'}}>
                      <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">Final Answer</span>
                    </div>
                    <div className="px-4 py-4" style={{backgroundColor:'#181818'}}>
                      <p className="text-white text-base leading-relaxed whitespace-pre-wrap">{cleanMath(result.final_answer || result.response)}</p>
                    </div>
                  </div>
                </div>

                {/* Right — reasoning steps */}
                {hasSteps && (
                  <div className="flex-1 flex flex-col gap-3 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">Reasoning Steps</span>
                      {tokens.length > 0 && (
                        <span className="text-[10px] text-gray-600">avg confidence <span className="text-gray-300 font-semibold">{avgConfidence.toFixed(1)}%</span></span>
                      )}
                    </div>
                    <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-1">
                      {stepMatches.map((stepText, idx) => {
                        const stepTokens = tokens.filter((t) => {
                          const tokenPos = tokens.indexOf(t)
                          const stepStart = response.indexOf(stepText)
                          return tokenPos >= stepStart && tokenPos < stepStart + stepText.length
                        })
                        const stepAvgConf = stepTokens.length > 0
                          ? (stepTokens.reduce((sum, t) => sum + t.confidence, 0) / stepTokens.length) * 100
                          : 0
                        const confColor = stepAvgConf >= 80 ? '#4ade80' : stepAvgConf >= 60 ? '#a3a3a3' : stepAvgConf > 0 ? '#facc15' : '#555'
                        return (
                          <div key={idx} className="rounded-xl border border-gray-800 overflow-hidden" style={{backgroundColor:'#181818'}}>
                            <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800" style={{backgroundColor:'#1e1e1e'}}>
                              <span className="text-white text-xs font-semibold">Step {idx + 1}</span>
                              {stepTokens.length > 0 && (
                                <div className="flex items-center gap-3 text-[10px]">
                                  <span className="text-gray-600">{stepTokens.length} tokens</span>
                                  <span className="font-semibold tabular-nums" style={{color: confColor}}>{stepAvgConf.toFixed(0)}%</span>
                                  <div className="w-16 h-1 rounded-full bg-gray-800 overflow-hidden">
                                    <div className="h-full rounded-full" style={{width:`${stepAvgConf}%`, backgroundColor: confColor, opacity: 0.8}}/>
                                  </div>
                                </div>
                              )}
                            </div>
                            <div className="px-4 py-3">
                              <p className="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap">{stepText.trim()}</p>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
