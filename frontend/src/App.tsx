import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import AiPrompt from '@/components/kokonutui/ai-prompt'
import SiteBackground from '@/components/SiteBackground'
import {
  ModelComparisonBento,
  ModelComparisonModal,
  type SlotState,
  type GenerationResult,
  cleanMath,
} from '@/components/ModelComparisonBento'
import { COMPARISON_MODEL_TABS, type ComparisonModelId } from '@/modelSpecs'
import {
  buildAttributionGraph,
  type PositionPrediction,
  type LayerAttention,
} from '@/lib/attributionGraph'
import PostHocBento from '@/components/PostHocBento'
import Presentation from '@/components/Presentation'
import type {
  PostHocSlot,
  LimeResult,
  TokenShapResult,
  CounterfactualResult,
} from '@/types/posthoc'

// ── Config ────────────────────────────────────────────────────────────────────

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

// ── Helpers ───────────────────────────────────────────────────────────────────

function loadingSlot(): SlotState {
  return { status: 'loading', result: null, explain: {}, graph: null, rawAttribution: null }
}

function idleSlots(): Record<ComparisonModelId, SlotState> {
  return Object.fromEntries(
    COMPARISON_MODEL_TABS.map((t) => [
      t.id,
      { status: 'idle', result: null, explain: {}, graph: null, rawAttribution: null },
    ]),
  ) as Record<ComparisonModelId, SlotState>
}


async function post(url: string, body: object) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) return null
  return res.json()
}

async function fetchModelSlot(
  tab: (typeof COMPARISON_MODEL_TABS)[number],
  prompt: string,
): Promise<SlotState> {
  const res = await fetch(`${API_BASE}/generate/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: tab.id, prompt, max_new_tokens: 256 }),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => `HTTP ${res.status}`)
    throw new Error(detail)
  }
  const data = await res.json()
  const result: GenerationResult = {
    model_id: tab.id,
    response: data.response ?? '',
    thinking: data.thinking ?? null,
    final_answer: data.final_answer ?? data.response ?? '',
    token_count: data.token_count ?? 0,
  }
  const response = result.final_answer || result.response

  // Sequential to avoid GPU OOM — pod serialises these via _GEN_LOCK anyway
  const confData   = await post(`${API_BASE}/explain/confidence`,    { model_id: tab.id, prompt, response })
  const hiddenData = await post(`${API_BASE}/explain/hidden-states`,  { model_id: tab.id, prompt })
  const logitData  = await post(`${API_BASE}/explain/logit-lens`,     { model_id: tab.id, prompt, response })
  const attrData   = await post(`${API_BASE}/explain/attribution`,    { model_id: tab.id, prompt, response })

  const attrTokens: { token: string; score: number }[] = attrData?.gradient_attribution ?? []
  const confTokens: { token: string; confidence: number }[] = confData?.token_confidence ?? []
  const positionPredictions: PositionPrediction[] = attrData?.position_predictions ?? []
  const layerAttention: LayerAttention[] = attrData?.layer_attention ?? []

  // Build attribution graph using real attention rollout + per-position predictions
  const graph = attrTokens.length > 0
    ? buildAttributionGraph(prompt, attrTokens, positionPredictions, layerAttention)
    : null

  const explain = {
    confidence: confTokens,
    hiddenStates: hiddenData?.hidden_state_norms ?? [],
    logitLens: logitData?.logit_lens ?? [],
    attribution: attrTokens,
  }

  const rawAttribution = attrTokens.length > 0
    ? {
        gradientAttribution: attrTokens,
        positionPredictions,
        layerAttention,
        prompt,
      }
    : null

  return { status: 'ready', result, explain, graph, rawAttribution, hiddenTokens: hiddenData?.tokens ?? [] }
}

function errorSlot(tab: (typeof COMPARISON_MODEL_TABS)[number], message: string): SlotState {
  return {
    status: 'ready',
    result: {
      model_id: tab.id,
      response: `Error: ${message}`,
      thinking: null,
      final_answer: `Error: ${message}`,
      token_count: 0,
    },
    explain: {},
    graph: null,
    rawAttribution: null,
  }
}

// ── Comparison Table ─────────────────────────────────────────────────────────

type AllTabId = ComparisonModelId | 'comparison'

interface ComparisonTableProps {
  slots: Record<ComparisonModelId, SlotState>
}

function ComparisonTable({ slots }: ComparisonTableProps) {
  const derived = COMPARISON_MODEL_TABS.map((tab) => {
    const slot = slots[tab.id]
    const ex = slot.explain

    const avgConf = ex.confidence && ex.confidence.length
      ? ex.confidence.reduce((s, c) => s + c.confidence, 0) / ex.confidence.length
      : null

    const earlyConf = ex.confidence && ex.confidence.length >= 4
      ? ex.confidence.slice(0, Math.floor(ex.confidence.length / 3)).reduce((s, c) => s + c.confidence, 0) / Math.floor(ex.confidence.length / 3)
      : null
    const lateConf = ex.confidence && ex.confidence.length >= 4
      ? ex.confidence.slice(-Math.floor(ex.confidence.length / 3)).reduce((s, c) => s + c.confidence, 0) / Math.floor(ex.confidence.length / 3)
      : null
    const confDelta = earlyConf !== null && lateConf !== null ? lateConf - earlyConf : null

    const hiddenNorms = ex.hiddenStates?.map(h => h.norm) ?? []
    const avgNorm = hiddenNorms.length ? hiddenNorms.reduce((a, b) => a + b, 0) / hiddenNorms.length : null
    const normStd = hiddenNorms.length > 1
      ? Math.sqrt(hiddenNorms.reduce((s, v) => s + Math.pow(v - (avgNorm ?? 0), 2), 0) / hiddenNorms.length)
      : null

    const peakAttn = ex.hiddenStates && ex.hiddenStates.length
      ? Math.max(...ex.hiddenStates.map(h => h.norm))
      : null

    const topAttrToken = ex.attribution && ex.attribution.length
      ? ex.attribution.reduce((best, a) => Math.abs(a.score) > Math.abs(best.score) ? a : best)
      : null

    return {
      id: tab.id,
      label: tab.label,
      status: slot.status,
      finalAnswer: slot.result?.final_answer ?? null,
      tokenCount: slot.result?.token_count ?? null,
      avgConf,
      confDelta,
      normStd,
      peakAttn,
      topAttrToken,
    }
  })

  // Which model has best value for each numeric metric (higher = better)
  const bestOf = (key: keyof typeof derived[0]) => {
    const vals = derived.map(d => d[key] as number | null).filter(v => v !== null) as number[]
    return vals.length ? Math.max(...vals) : null
  }
  const bestConf = bestOf('avgConf')
  const minTokens = (() => {
    const vals = derived.map(d => d.tokenCount).filter(v => v !== null) as number[]
    return vals.length ? Math.min(...vals) : null
  })()

  const Pill = ({ children, green }: { children: React.ReactNode; green?: boolean }) => (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs font-mono font-semibold"
      style={{ backgroundColor: green ? 'rgba(134,239,172,0.15)' : 'rgba(252,165,165,0.12)', color: green ? '#86efac' : '#fca5a5' }}
    >
      {children}
    </span>
  )

  const NumCell = ({ val, fmt, isBest }: { val: number | null; fmt: (v: number) => string; isBest?: boolean }) => {
    if (val === null) return <span style={{ color: '#4b5563' }}>—</span>
    return (
      <span style={{ color: isBest ? '#fff' : '#9ca3af', fontWeight: isBest ? 600 : 400 }} className="font-mono">
        {fmt(val)}
        {isBest && <span className="ml-1 text-[10px]" style={{ color: '#6b7280' }}>▲</span>}
      </span>
    )
  }

  const sectionHeader = (label: string) => (
    <tr style={{ backgroundColor: '#181818' }}>
      <td colSpan={5} className="px-5 py-2">
        <span className="text-[10px] uppercase tracking-widest" style={{ color: '#6b7280' }}>{label}</span>
      </td>
    </tr>
  )

  return (
    <div className="p-1">
      <div className="mb-6">
        <h2 className="text-white text-lg font-semibold">Model Comparison</h2>
        <p className="text-sm mt-1" style={{ color: '#6b7280' }}>Side-by-side metrics across all four models for the last prompt.</p>
      </div>

      <div className="rounded-xl overflow-hidden" style={{ backgroundColor: '#202020', border: '1px solid rgba(255,255,255,0.08)' }}>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ backgroundColor: '#181818', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <th className="text-left px-5 py-3 font-medium w-52" style={{ color: '#9ca3af' }}>Metric</th>
              {COMPARISON_MODEL_TABS.map(tab => (
                <th key={tab.id} className="text-center px-4 py-3 font-medium" style={{ color: '#d1d5db' }}>
                  <div className="text-xs">{tab.label}</div>
                  <div className={`text-[10px] font-normal mt-0.5 ${
                    slots[tab.id].status === 'ready' ? 'text-emerald-500/80' :
                    slots[tab.id].status === 'loading' ? 'text-gray-500' : 'text-gray-700'
                  }`}>
                    {slots[tab.id].status}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sectionHeader('Answer')}
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>Final Answer</td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   d.finalAnswer ? <Pill green>{d.finalAnswer}</Pill> : <span style={{ color: '#6b7280' }}>—</span>}
                </td>
              ))}
            </tr>

            {sectionHeader('Generation')}
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', backgroundColor: '#1c1c1c' }}>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>Response Tokens</td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   <NumCell val={d.tokenCount} fmt={v => v.toString()} isBest={d.tokenCount !== null && d.tokenCount === minTokens} />}
                </td>
              ))}
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>Avg Token Confidence</td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   <NumCell val={d.avgConf} fmt={v => `${(v * 100).toFixed(1)}%`} isBest={d.avgConf !== null && d.avgConf === bestConf} />}
                </td>
              ))}
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', backgroundColor: '#1c1c1c' }}>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>
                <span>Confidence Trend</span>
                <span className="block text-[10px]" style={{ color: '#6b7280' }}>late − early tokens</span>
              </td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   d.confDelta !== null ? (
                     <span className="font-mono text-xs" style={{ color: d.confDelta >= 0 ? '#86efac' : '#fca5a5' }}>
                       {d.confDelta >= 0 ? '+' : ''}{(d.confDelta * 100).toFixed(1)}%
                     </span>
                   ) : <span style={{ color: '#6b7280' }}>—</span>}
                </td>
              ))}
            </tr>

            {sectionHeader('Mechanistic Signals')}
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>
                <span>Hidden State Drift</span>
                <span className="block text-[10px]" style={{ color: '#6b7280' }}>std dev of layer norms</span>
              </td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   <NumCell val={d.normStd} fmt={v => v.toFixed(2)} />}
                </td>
              ))}
            </tr>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', backgroundColor: '#1c1c1c' }}>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>
                <span>Peak Activation Layer</span>
                <span className="block text-[10px]" style={{ color: '#6b7280' }}>max hidden state norm</span>
              </td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   <NumCell val={d.peakAttn} fmt={v => v.toFixed(1)} />}
                </td>
              ))}
            </tr>
            <tr>
              <td className="px-5 py-3.5 font-medium" style={{ color: '#9ca3af' }}>
                <span>Top Attribution Token</span>
                <span className="block text-[10px]" style={{ color: '#6b7280' }}>most influential input word</span>
              </td>
              {derived.map(d => (
                <td key={d.id} className="px-4 py-3.5 text-center">
                  {d.status === 'loading' ? <span className="inline-block w-10 h-2 rounded animate-pulse" style={{ backgroundColor: '#374151' }} /> :
                   d.status === 'idle' ? <span style={{ color: '#374151' }}>—</span> :
                   d.topAttrToken ? (
                     <span className="font-mono text-xs px-2 py-0.5 rounded" style={{ backgroundColor: 'rgba(255,255,255,0.06)', color: '#e5e7eb' }}>
                       {d.topAttrToken.token}
                       <span className="ml-1.5 text-[10px]" style={{ color: '#6b7280' }}>{d.topAttrToken.score.toFixed(3)}</span>
                     </span>
                   ) : <span style={{ color: '#6b7280' }}>—</span>}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      <div className="mt-6 rounded-xl overflow-hidden" style={{ backgroundColor: '#202020', border: '1px solid rgba(255,255,255,0.08)' }}>
        <div className="px-5 py-3" style={{ backgroundColor: '#181818', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
          <span className="font-medium text-sm" style={{ color: '#d1d5db' }}>Response Preview</span>
        </div>
        <div className="grid grid-cols-4 divide-x" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          {COMPARISON_MODEL_TABS.map(tab => {
            const slot = slots[tab.id]
            return (
              <div key={tab.id} className="px-5 py-4">
                <p className="text-[10px] font-semibold uppercase tracking-wider mb-2" style={{ color: '#6b7280' }}>{tab.label}</p>
                {slot.status === 'loading' && (
                  <div className="space-y-1.5">
                    <div className="h-2 rounded animate-pulse w-full" style={{ backgroundColor: '#1f2937' }} />
                    <div className="h-2 rounded animate-pulse w-4/5" style={{ backgroundColor: '#1f2937' }} />
                    <div className="h-2 rounded animate-pulse w-3/5" style={{ backgroundColor: '#1f2937' }} />
                  </div>
                )}
                {slot.status === 'idle' && <p className="text-xs" style={{ color: '#374151' }}>No data yet.</p>}
                {slot.status === 'ready' && slot.result && (
                  <p className="text-xs leading-relaxed line-clamp-5" style={{ color: '#d1d5db' }}>
                    {cleanMath(slot.result.final_answer || slot.result.response)}
                  </p>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ── Post-hoc helpers ─────────────────────────────────────────────────────────

type ParentTabId = 'mech' | 'posthoc'

function idlePostHocSlots(): Record<ComparisonModelId, PostHocSlot> {
  return Object.fromEntries(
    COMPARISON_MODEL_TABS.map((t) => [
      t.id,
      { status: 'idle', lime: null, tokenshap: null, counterfactual: null } as PostHocSlot,
    ]),
  ) as Record<ComparisonModelId, PostHocSlot>
}

async function fetchPostHocSlot(
  tab: (typeof COMPARISON_MODEL_TABS)[number],
  prompt: string,
  response: string,
  onPartial: (patch: Partial<PostHocSlot>) => void,
): Promise<void> {
  // Fire all three sequentially; emit each piece as it lands so the UI can stream.
  // Counterfactual first (cheapest, most demo-worthy), then LIME, then TokenSHAP.
  try {
    const cf = await post(`${API_BASE}/posthoc/counterfactual`, {
      model_id: tab.id,
      prompt,
      response,
    })
    if (cf) onPartial({ counterfactual: cf as CounterfactualResult })
  } catch (err) {
    console.error(`[posthoc] counterfactual failed for ${tab.id}:`, err)
  }

  try {
    const lime = await post(`${API_BASE}/posthoc/lime`, {
      model_id: tab.id,
      prompt,
      response,
      n_samples: 20,
    })
    if (lime) onPartial({ lime: lime as LimeResult })
  } catch (err) {
    console.error(`[posthoc] LIME failed for ${tab.id}:`, err)
    onPartial({ lime: { method: 'lime', words: [], attributions: [], reference_answer: null, reference_response: '', n_samples: 0, r_squared: 0 } as LimeResult })
  }

  try {
    const ts = await post(`${API_BASE}/posthoc/tokenshap`, {
      model_id: tab.id,
      prompt,
      response,
      n_samples: 20,
    })
    if (ts) onPartial({ tokenshap: ts as TokenShapResult })
  } catch (err) {
    console.error(`[posthoc] TokenSHAP failed for ${tab.id}:`, err)
    onPartial({ tokenshap: { method: 'tokenshap', words: [], attributions: [], reference_answer: null, reference_response: '', n_samples: 0 } as TokenShapResult })
  }
}

// ── App ──────────────────────────────────────────────────────────────────────

function MainApp() {
  const [slots, setSlots] = useState<Record<ComparisonModelId, SlotState>>(idleSlots)
  const [postHocSlots, setPostHocSlots] = useState<Record<ComparisonModelId, PostHocSlot>>(idlePostHocSlots)
  const [activeTab, setActiveTab] = useState<AllTabId>(COMPARISON_MODEL_TABS[0].id)
  const [parentTab, setParentTab] = useState<ParentTabId>('mech')
  const [lastPrompt, setLastPrompt] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [showComparison, setShowComparison] = useState(false)
  const [infoBanner, setInfoBanner] = useState<string | null>(null)
  const [expandedCard, setExpandedCard] = useState<string | null>(null)

  const activeModelTab: ComparisonModelId = activeTab === 'comparison' ? COMPARISON_MODEL_TABS[0].id : activeTab

  const handleSubmit = (prompt: string) => {
    setIsLoading(true)
    setInfoBanner(null)
    setShowComparison(true)
    setExpandedCard(null)
    setLastPrompt(prompt)
    setSlots(
      Object.fromEntries(COMPARISON_MODEL_TABS.map((t) => [t.id, loadingSlot()])) as Record<
        ComparisonModelId,
        SlotState
      >,
    )
    // Reset post-hoc — user must switch to the post-hoc tab to trigger fetches
    setPostHocSlots(idlePostHocSlots())
    setActiveTab(COMPARISON_MODEL_TABS[0].id)
    setParentTab('mech')

    setTimeout(() => {
      window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })
    }, 100)

    let completed = 0
    COMPARISON_MODEL_TABS.forEach((tab) => {
      fetchModelSlot(tab, prompt)
        .then((slot) => {
          setSlots((prev) => ({ ...prev, [tab.id]: slot }))
        })
        .catch((err: Error) => {
          setSlots((prev) => ({ ...prev, [tab.id]: errorSlot(tab, err.message) }))
          const is524 = err.message.includes('524')
          setInfoBanner(is524
            ? 'Request timed out — models may still be loading on the pod. Try again in 30s.'
            : 'One or more models failed — is the backend running and MODEL_BASE_URL set?'
          )
        })
        .finally(() => {
          completed += 1
          if (completed === COMPARISON_MODEL_TABS.length) setIsLoading(false)
        })
    })
  }

  const triggerPostHoc = (modelId: ComparisonModelId) => {
    const current = postHocSlots[modelId]
    if (current.status !== 'idle') return  // already fetching or done
    const slot = slots[modelId]
    if (!slot.result) return  // mech-interp must have completed first
    if (!lastPrompt) return

    const tab = COMPARISON_MODEL_TABS.find((t) => t.id === modelId)
    if (!tab) return

    setPostHocSlots((prev) => ({
      ...prev,
      [modelId]: { ...prev[modelId], status: 'loading' },
    }))

    const referenceResponse = slot.result.final_answer || slot.result.response

    fetchPostHocSlot(tab, lastPrompt, referenceResponse, (patch) => {
      setPostHocSlots((prev) => ({
        ...prev,
        [modelId]: { ...prev[modelId], ...patch, status: 'loading' },
      }))
    })
      .then(() => {
        setPostHocSlots((prev) => ({ ...prev, [modelId]: { ...prev[modelId], status: 'ready' } }))
      })
      .catch((err: Error) => {
        setPostHocSlots((prev) => ({
          ...prev,
          [modelId]: { ...prev[modelId], status: 'error', error: err.message },
        }))
      })
  }

  const handleParentTabChange = (tab: ParentTabId) => {
    setParentTab(tab)
    if (tab === 'posthoc') {
      // Fire post-hoc for the active model tab if it hasn't been computed
      const modelTab: ComparisonModelId =
        activeTab === 'comparison' ? COMPARISON_MODEL_TABS[0].id : activeTab
      triggerPostHoc(modelTab)
    }
  }

  const handleModelTabChange = (tab: AllTabId) => {
    setActiveTab(tab)
    if (parentTab === 'posthoc' && tab !== 'comparison') {
      triggerPostHoc(tab)
    }
  }

  const activeSlot = slots[activeModelTab]

  return (
    <div className="min-h-screen relative">
      <SiteBackground />

      <div className="relative z-10">
        <nav className="px-8 py-4 flex items-center justify-between border-b border-gray-800/50">
          <div className="flex items-center gap-3">
            <div className="text-white text-2xl font-bold">Prism</div>
            {infoBanner && (
              <span className="text-amber-400/90 text-xs max-w-xl truncate border-l border-gray-700 pl-3">{infoBanner}</span>
            )}
          </div>
          <div className="flex items-center gap-4" />
        </nav>

        <div className="flex flex-col items-center justify-center px-20 py-16 h-screen">
          <div className="text-center mb-20">
            <h1 className="text-7xl font-bold text-white leading-tight mb-2 tracking-wide">Inside the Black Box</h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
              Attention, reasoning, and confidence - all visible for the first time
            </p>
          </div>
          <div className="w-full max-w-3xl">
            <AiPrompt onSubmit={handleSubmit} />
          </div>

          {isLoading && (
            <div className="mt-12 text-center">
              <div className="w-12 h-12 rounded-full border-4 border-gray-700 border-t-white animate-spin mx-auto" />
            </div>
          )}
        </div>

        {(showComparison || isLoading) && (
          <div className="px-8 pt-0 pb-8 max-w-[1920px] mx-auto -mt-16">
            {/* Parent tabs: Mechanistic Interpretability vs Post-hoc Analysis */}
            <div
              className="mb-3 flex w-full max-w-3xl mx-auto rounded-lg border border-gray-700/40 bg-black/30 p-1"
              role="tablist"
              aria-label="Analysis mode"
            >
              {([
                { id: 'mech' as ParentTabId, label: 'Mechanistic Interpretability', subtitle: 'During the forward pass' },
                { id: 'posthoc' as ParentTabId, label: 'Post-hoc Analysis', subtitle: 'After generation completes' },
              ]).map((p) => {
                const active = parentTab === p.id
                return (
                  <button
                    key={p.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => handleParentTabChange(p.id)}
                    className={`flex-1 rounded-md px-4 py-2.5 text-center transition-colors ${
                      active
                        ? 'bg-white/10 text-white'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-white/5'
                    }`}
                  >
                    <div className="text-sm font-semibold">{p.label}</div>
                    <div className="text-[10px] text-gray-500 mt-0.5">{p.subtitle}</div>
                  </button>
                )
              })}
            </div>

            {/* Model sub-tabs */}
            <div
              className="mb-4 flex w-full border border-gray-600/50 bg-black/20"
              role="tablist"
              aria-label="Model comparison"
            >
              {COMPARISON_MODEL_TABS.map((tab) => {
                const st = parentTab === 'posthoc' ? postHocSlots[tab.id] : slots[tab.id]
                const active = activeTab === tab.id
                return (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => handleModelTabChange(tab.id)}
                    className={`relative flex min-w-0 flex-1 flex-col items-center justify-center gap-1 px-3 py-3 text-center text-sm font-medium transition-colors box-border border-b-2 border-r border-gray-600/50 outline-none focus-visible:z-10 focus-visible:ring-1 focus-visible:ring-white/30 focus-visible:ring-inset ${
                      active
                        ? 'border-b-white bg-gray-950 text-white'
                        : 'border-b-transparent text-gray-500 hover:bg-gray-950/70 hover:text-gray-300'
                    }`}
                  >
                    <span className="truncate w-full px-1">{tab.label}</span>
                    <span className="flex items-center gap-1.5 text-[10px] font-normal tabular-nums">
                      {st.status === 'loading' && (
                        <>
                          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-white/60" title="Loading" />
                          <span className="text-gray-500">loading</span>
                        </>
                      )}
                      {st.status === 'ready' && <span className="text-emerald-500/90">ready</span>}
                      {st.status === 'idle' && <span className="text-gray-600">idle</span>}
                      {st.status === 'error' && <span className="text-red-400/90">error</span>}
                    </span>
                  </button>
                )
              })}
              {parentTab === 'mech' && (
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'comparison'}
                  onClick={() => handleModelTabChange('comparison')}
                  className={`relative flex min-w-0 flex-1 flex-col items-center justify-center gap-1 px-3 py-3 text-center text-sm font-medium transition-colors box-border border-b-2 outline-none focus-visible:z-10 focus-visible:ring-1 focus-visible:ring-white/30 focus-visible:ring-inset ${
                    activeTab === 'comparison'
                      ? 'border-b-white bg-gray-950 text-white'
                      : 'border-b-transparent text-gray-500 hover:bg-gray-950/70 hover:text-gray-300'
                  }`}
                >
                  <span className="truncate w-full px-1">Comparison</span>
                  <span className="text-[10px] font-normal text-gray-600">overview</span>
                </button>
              )}
            </div>

            {parentTab === 'mech' && activeTab !== 'comparison' && (
              <ModelComparisonBento slot={activeSlot} onExpandCard={(c) => setExpandedCard(c)} />
            )}
            {parentTab === 'mech' && activeTab === 'comparison' && (
              <ComparisonTable slots={slots} />
            )}
            {parentTab === 'posthoc' && (
              <PostHocBento slot={postHocSlots[activeModelTab]} />
            )}
          </div>
        )}

        {expandedCard && activeSlot.result && (
          <ModelComparisonModal
            expandedCard={expandedCard}
            slot={activeSlot}
            onClose={() => setExpandedCard(null)}
          />
        )}
      </div>
    </div>
  )
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainApp />} />
        <Route path="/presentation" element={<Presentation />} />
      </Routes>
    </Router>
  )
}

export default App
