import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import AiPrompt from '@/components/kokonutui/ai-prompt'
import SiteBackground from '@/components/SiteBackground'
import AttentionMatrix from '@/components/visualizations/AttentionMatrix'
import AttributionGraphVisualization from '@/components/visualizations/AttributionGraph'
import TokenFlow from '@/components/visualizations/TokenFlow'
import { generateMockAttributionGraph } from '@/utils/mockAttributionData'
import type { AttributionGraph } from '@/types/attribution'

const API_BASE = 'http://localhost:8000'

// ── Types ────────────────────────────────────────────────────────────────────

interface GenerationResult {
  model_id: string
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

function generateFillerData(prompt: string) {
  // Generate mock tokens from prompt
  const words = prompt.split(' ').filter(w => w.length > 0)
  const tokens = words.slice(0, 15)
  
  // Generate mock confidence values
  const confidence = tokens.map(() => 
    Math.max(0.1, Math.min(0.95, 0.5 + (Math.random() - 0.5) * 0.6))
  )
  
  // Generate mock attention matrix
  const attentionMatrix = Array(tokens.length).fill(0).map(() =>
    Array(tokens.length).fill(0).map(() => Math.random() * 0.8)
  )
  
  // Generate mock logit lens data
  const logitLens = []
  for (let layer = 0; layer < 12; layer++) {
    for (let pos = 0; pos < Math.min(5, tokens.length); pos++) {
      logitLens.push({
        layer,
        word_position: pos,
        predicted_token: tokens[pos] || 'the',
        probability: Math.max(0.01, Math.random() * 0.9)
      })
    }
  }
  
  // Generate mock attribution data
  const attribution = tokens.map(token => ({
    token,
    score: Math.random() * 0.05
  }))
  
  // Generate mock hidden states
  const hiddenStates = Array(12).fill(0).map((_, i) => ({
    layer: i,
    norm: 10 + Math.random() * 20
  }))
  
  // Generate mock response
  const response = `This is a simulated response for: "${prompt}". In a real scenario, the model would provide a thoughtful answer with reasoning and analysis.`
  
  return {
    response,
    thinking: "This is simulated thinking that would show the model's reasoning process step by step.",
    final_answer: "This is the final simulated answer.",
    token_count: tokens.length,
    confidence: confidence.map((conf, i) => ({ token: tokens[i], confidence: conf })),
    attention: {
      tokens,
      matrix: attentionMatrix,
      layer: 0,
      head: 0
    },
    logitLens,
    attribution,
    hiddenStates
  }
}

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


// ── Visualization Components ─────────────────────────────────────────────────

// ── App ──────────────────────────────────────────────────────────────────────

function MainApp() {
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [explainData, setExplainData] = useState<Partial<ExplainData>>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [attentionLayer, setAttentionLayer] = useState(0)
  const [attentionHead, setAttentionHead] = useState(0)
  const [currentPrompt, setCurrentPrompt] = useState('')
  const [expandedCard, setExpandedCard] = useState<string | null>(null)
  const [attributionGraph, setAttributionGraph] = useState<AttributionGraph | null>(null)

  const fetchAttentionForLayer = async (prompt: string, response: string) => {
    // Use the current model ID from result or default to gemma-finetuned
    const modelId = result?.model_id || 'gemma-finetuned'
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

  const handleSubmit = async (prompt: string, model: string) => {
    setCurrentPrompt(prompt)
    setIsLoading(true)
    setError(null)
    setResult(null)
    setExplainData({})
    
    // Auto-scroll to bento boxes after a brief delay
    setTimeout(() => {
      window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })
    }, 100)

    // Map frontend model names to backend model IDs
    const modelMap: Record<string, string> = {
      'Gemma Base': 'gemma-base',
      'Gemma Fine-tuned': 'gemma-finetuned',
    }
    const modelId = modelMap[model] || 'gemma-finetuned'

    try {
      const res = await fetch(`${API_BASE}/generate/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId, prompt, max_new_tokens: 1024 }),
      })
      
      if (!res.ok) {
        throw new Error('Backend unavailable')
      }
      
      const data: GenerationResult = await res.json()
      setResult(data)
      setIsLoading(false)

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
        const d = await attnRes.value.json()
        partial.attention = d
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

      // Attribution is slow — fire separately
      fetch(`${API_BASE}/explain/attribution`, { method: 'POST', headers, body: JSON.stringify({ ...explainBase }) })
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (d && d.gradient_attribution) {
            setExplainData(prev => ({ ...prev, attribution: d.gradient_attribution }))
            
            // Generate attribution graph from the data
            const graph = generateMockAttributionGraph(currentPrompt, result?.response || '')
            setAttributionGraph(graph)
          }
        })
        .catch(() => {})

    } catch (e: unknown) {
      // Backend unavailable - use fallback data
      console.log('Backend unavailable, using fallback data')
      const fallbackData = generateFillerData(prompt)
      
      setResult({
        model_id: modelId,
        response: fallbackData.response,
        thinking: fallbackData.thinking,
        final_answer: fallbackData.final_answer,
        token_count: fallbackData.token_count,
      })
      
      setExplainData({
        confidence: fallbackData.confidence,
        attention: fallbackData.attention,
        logitLens: fallbackData.logitLens,
        attribution: fallbackData.attribution,
        hiddenStates: fallbackData.hiddenStates
      })
      
      // Generate attribution graph
      const graph = generateMockAttributionGraph(prompt, fallbackData.response)
      setAttributionGraph(graph)
      
      setError('Using demo data - backend is offline')
      setIsLoading(false)
    }
  }

  const confidenceColor = (conf: number) => {
    if (conf >= 0.8) return '#10b981'
    if (conf >= 0.5) return '#fbbf24'
    return '#ef4444'
  }

  const tokens = explainData.confidence || []

  return (
    <div className="min-h-screen relative">
      <SiteBackground />

      {/* All content above background */}
      <div className="relative z-10">
        <nav className="px-8 py-4 flex items-center justify-between border-b border-gray-800/50">
          <div className="flex items-center gap-3">
            <div className="text-white text-2xl font-bold">Prism</div>
            {error && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                <span className="text-red-400 text-xs">Offline</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-4">
          </div>
        </nav>

        {/* Hero section - Always visible */}
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
          
          {/* Loading indicator in hero section */}
          {isLoading && (
            <div className="mt-12 text-center">
              <div className="w-12 h-12 rounded-full border-4 border-gray-700 border-t-purple-500 animate-spin mx-auto"></div>
            </div>
          )}
        </div>

        {/* Bento Box Dashboard - One viewport below */}
        {(result || isLoading) && (
          <div className="p-8 max-w-[1920px] mx-auto">

            {/* Bento Grid Layout */}
            <div className="grid grid-cols-12 gap-3 auto-rows-[140px]">
              
              {/* AI Response - Square Card (Top Left) */}
              <div 
                onClick={() => !isLoading && setExpandedCard('response')}
                className="col-span-3 row-span-2 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-amber-500 transition-all cursor-pointer group"
                style={{
                  backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)'
                }}>
                <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    AI Response
                  </div>
                  <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                  </svg>
                </h3>
                <div className="space-y-2 max-h-[220px] overflow-hidden">
                  {isLoading ? (
                    <>
                      <div className="p-2 bg-gray-800/50 rounded-lg animate-pulse">
                        <div className="h-2 bg-gray-700 rounded w-20 mb-2"></div>
                        <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                        <div className="h-2 bg-gray-700 rounded w-3/4"></div>
                      </div>
                      <div className="p-3 bg-gradient-to-r from-amber-500/10 to-green-500/10 border border-amber-500/30 rounded-lg animate-pulse">
                        <div className="h-2 bg-amber-500/30 rounded w-24 mb-2"></div>
                        <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                        <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                        <div className="h-2 bg-gray-700 rounded w-2/3"></div>
                      </div>
                    </>
                  ) : result ? (
                    <>
                      {result.thinking && (
                        <div className="p-2 bg-gray-800/50 rounded-lg">
                          <p className="text-gray-400 text-[10px] uppercase tracking-wider mb-1">Reasoning</p>
                          <p className="text-gray-300 text-xs line-clamp-2">{result.thinking}</p>
                        </div>
                      )}
                      <div className="p-3 bg-gradient-to-r from-amber-500/10 to-green-500/10 border border-amber-500/30 rounded-lg">
                        <p className="text-amber-400 text-[10px] font-semibold mb-1">FINAL ANSWER</p>
                        <p className="text-white text-xs line-clamp-3">{result.final_answer || result.response}</p>
                      </div>
                    </>
                  ) : null}
                </div>
              </div>

              {/* Attribution Graph - Card */}
              <div 
                onClick={() => !isLoading && setExpandedCard('attribution')}
                className="col-span-3 row-span-2 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-purple-500 transition-all cursor-pointer group"
                style={{
                  backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)'
                }}>
                <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    Attribution Graph
                  </div>
                  <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                  </svg>
                </h3>
                <div className="flex flex-col items-center justify-center h-[180px]">
                  {isLoading ? (
                    <div className="text-center space-y-3 animate-pulse">
                      <div className="w-16 h-16 bg-gray-700 rounded-full mx-auto"></div>
                      <div className="h-3 bg-gray-700 rounded w-32 mx-auto"></div>
                      <div className="h-2 bg-gray-700 rounded w-24 mx-auto"></div>
                    </div>
                  ) : attributionGraph ? (
                    <div className="text-center space-y-4 w-full">
                      <div className="relative w-20 h-20 mx-auto">
                        <svg className="w-full h-full transform -rotate-90">
                          <circle cx="40" cy="40" r="35" fill="none" stroke="hsl(0 0% 20%)" strokeWidth="6"/>
                          <circle 
                            cx="40" 
                            cy="40" 
                            r="35" 
                            fill="none" 
                            stroke="hsl(262.1 83.3% 57.8%)" 
                            strokeWidth="6"
                            strokeDasharray={`${(attributionGraph.explainedBehavior || 0) * 220} 220`}
                            className="transition-all duration-1000"
                          />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center">
                          <span className="text-white text-lg font-bold">
                            {((attributionGraph.explainedBehavior || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <p className="text-gray-400 text-xs">Network Analysis</p>
                        <div className="flex items-center justify-center gap-4 text-[10px]">
                          <span className="text-gray-500">{attributionGraph.nodes.length} nodes</span>
                          <span className="text-gray-600">•</span>
                          <span className="text-gray-500">{attributionGraph.edges.length} edges</span>
                        </div>
                      </div>
                      <p className="text-purple-400 text-xs font-semibold group-hover:text-purple-300 transition-colors">
                        Click to explore graph →
                      </p>
                    </div>
                  ) : (
                    <div className="text-center text-gray-500 text-xs">
                      No attribution data
                    </div>
                  )}
                </div>
              </div>

              {/* Attention Matrix - Large Card */}
              <div 
                onClick={() => !isLoading && setExpandedCard('attention')}
                className="col-span-6 row-span-4 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-cyan-500 transition-all cursor-pointer group"
                style={{
                  backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)'
                }}>
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-white font-semibold flex items-center gap-2">
                    Attention Patterns
                    <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                    </svg>
                  </h3>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <label className="text-gray-400 text-xs">Layer:</label>
                      <input 
                        type="range"
                        min="0"
                        max="31"
                        value={attentionLayer}
                        onChange={(e) => {
                          const newLayer = Number(e.target.value)
                          setAttentionLayer(newLayer)
                          if (result) fetchAttentionForLayer(currentPrompt, result.response)
                        }}
                        className="w-24 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                      />
                      <span className="text-white text-xs w-6">{attentionLayer}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="text-gray-400 text-xs">Head:</label>
                      <input 
                        type="range"
                        min="0"
                        max="31"
                        value={attentionHead}
                        onChange={(e) => {
                          const newHead = Number(e.target.value)
                          setAttentionHead(newHead)
                          if (result) fetchAttentionForLayer(currentPrompt, result.response)
                        }}
                        className="w-24 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                      />
                      <span className="text-white text-xs w-6">{attentionHead}</span>
                    </div>
                  </div>
                </div>
                {isLoading ? (
                  <div className="overflow-hidden animate-pulse">
                    <div className="grid grid-cols-8 gap-1 max-w-full">
                      {Array.from({ length: 48 }).map((_, i) => (
                        <div key={i} className="w-full aspect-square bg-gray-700 rounded"></div>
                      ))}
                    </div>
                  </div>
                ) : explainData.attention ? (
                  <div className="overflow-hidden">
                    <AttentionMatrix
                      tokens={explainData.attention.tokens}
                      attentionWeights={explainData.attention.matrix}
                      layer={attentionLayer}
                      head={attentionHead}
                      maxTokens={12}
                    />
                  </div>
                ) : null}
              </div>

              {/* Token Flow - Card */}
              <div 
                onClick={() => !isLoading && setExpandedCard('flow')}
                className="col-span-3 row-span-2 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-green-500 transition-all cursor-pointer group"
                style={{
                  backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)'
                }}>
                <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    Token Flow
                  </div>
                  <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                  </svg>
                </h3>
                {isLoading ? (
                  <div className="overflow-hidden max-h-[220px] animate-pulse space-y-2">
                    {Array.from({ length: 4 }).map((_, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <div className="w-16 h-6 bg-gray-700 rounded"></div>
                        <div className="flex-1 h-2 bg-gray-700 rounded"></div>
                        <div className="w-8 h-4 bg-gray-700 rounded"></div>
                      </div>
                    ))}
                  </div>
                ) : tokens.length > 0 ? (
                  <div className="overflow-hidden max-h-[220px]">
                    <TokenFlow 
                      tokens={tokens.slice(0, 6).map((t: any) => t.token)}
                      confidence={tokens.slice(0, 6).map((t: any) => t.confidence)}
                    />
                  </div>
                ) : null}
              </div>

              {/* Reasoning Steps Card */}
              <div 
                onClick={() => !isLoading && setExpandedCard('reasoning')}
                className="col-span-3 row-span-2 bg-gradient-to-br from-gray-900/80 via-gray-800/60 to-gray-900/80 border border-gray-700/50 rounded-xl p-4 backdrop-blur-md hover:border-pink-500 transition-all cursor-pointer group"
                style={{
                  backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.5) 50%, rgba(67, 67, 67, 0.3) 100%)'
                }}>
                <h3 className="text-white font-semibold mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    Reasoning Steps
                  </div>
                  <svg className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                  </svg>
                </h3>
                <div className="space-y-2 max-h-[200px] overflow-y-auto">
                  {isLoading ? (
                    <>
                      {[1, 2, 3].map(i => (
                        <div key={i} className="p-2 bg-gray-800/50 rounded-lg animate-pulse">
                          <div className="h-3 bg-gray-700 rounded w-20 mb-2"></div>
                          <div className="h-2 bg-gray-700 rounded w-full mb-1"></div>
                          <div className="h-2 bg-gray-700 rounded w-16"></div>
                        </div>
                      ))}
                    </>
                  ) : result ? (() => {
                    const response = result.response || ''
                    const stepMatches = response.match(/Step \d+:.*?(?=Step \d+:|$)/gs) || []
                    
                    if (stepMatches.length === 0) {
                      return (
                        <div className="text-center p-4 text-gray-500 text-xs">
                          No reasoning steps detected
                        </div>
                      )
                    }

                    return stepMatches.slice(0, 3).map((stepText, idx) => {
                      const stepNum = idx + 1
                      const stepTokens = tokens.filter((t: any) => {
                        const tokenPos = tokens.indexOf(t)
                        const stepStart = response.indexOf(stepText)
                        const stepEnd = stepStart + stepText.length
                        return tokenPos >= stepStart && tokenPos < stepEnd
                      })
                      
                      const avgConf = stepTokens.length > 0 
                        ? stepTokens.reduce((sum: number, t: any) => sum + t.confidence, 0) / stepTokens.length * 100
                        : 0
                      
                      const status = avgConf >= 80 ? '✓' : avgConf >= 60 ? '⚠️' : '✗'
                      const statusColor = avgConf >= 80 ? 'text-green-400' : avgConf >= 60 ? 'text-yellow-400' : 'text-red-400'
                      const preview = stepText.substring(0, 40).trim() + (stepText.length > 40 ? '...' : '')
                      
                      return (
                        <div key={idx} className="p-2 bg-gray-800/50 rounded-lg border border-gray-700/30">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-white text-xs font-semibold">Step {stepNum}</span>
                            <span className={`text-sm ${statusColor}`}>{status}</span>
                          </div>
                          <p className="text-gray-400 text-[10px] mb-1 line-clamp-1">{preview}</p>
                          <div className="flex items-center justify-between text-[9px]">
                            <span className="text-gray-500">{stepTokens.length} tokens</span>
                            <span className={`font-semibold ${statusColor}`}>{avgConf.toFixed(0)}%</span>
                          </div>
                        </div>
                      )
                    })
                  })() : null}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Expanded Card Modal */}
        {expandedCard && result && (
          <div 
            className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200"
            onClick={() => setExpandedCard(null)}
          >
            <div 
              className="bg-gradient-to-br from-gray-900/90 via-gray-800/80 to-gray-900/90 border border-gray-700/50 rounded-2xl p-8 max-w-6xl w-full max-h-[90vh] overflow-y-auto backdrop-blur-xl animate-in zoom-in-95 duration-300 slide-in-from-bottom-4"
              style={{
                backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.4) 0%, rgba(29, 29, 29, 0.6) 50%, rgba(67, 67, 67, 0.4) 100%)'
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">
                  {expandedCard === 'response' && 'AI Response'}
                  {expandedCard === 'attribution' && 'Attribution Graph Analysis'}
                  {expandedCard === 'attention' && 'Attention Patterns'}
                  {expandedCard === 'flow' && 'Token Generation Flow'}
                  {expandedCard === 'stats' && 'Statistics'}
                </h2>
                <button
                  onClick={() => setExpandedCard(null)}
                  className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-800 rounded-lg"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="bg-gradient-to-br from-gray-800/60 via-gray-900/50 to-gray-800/60 rounded-xl p-6 backdrop-blur-md"
                style={{
                  backgroundImage: 'linear-gradient(145deg, rgba(67, 67, 67, 0.3) 0%, rgba(29, 29, 29, 0.4) 50%, rgba(67, 67, 67, 0.3) 100%)'
                }}>
                {expandedCard === 'attribution' && attributionGraph && (
                  <AttributionGraphVisualization 
                    data={attributionGraph}
                    width={1000}
                    height={600}
                  />
                )}
                {expandedCard === 'flow' && tokens.length > 0 && (
                  <div className="space-y-4">
                    <div className="text-sm text-gray-400">
                      Showing {tokens.length} tokens with confidence scores
                    </div>
                    <div className="overflow-auto" style={{ maxHeight: '70vh' }}>
                      <TokenFlow 
                        tokens={tokens.map((t: any) => t.token)}
                        confidence={tokens.map((t: any) => t.confidence)}
                      />
                    </div>
                  </div>
                )}
                {expandedCard === 'reasoning' && (() => {
                  const response = result.response || ''
                  const stepMatches = response.match(/Step \d+:.*?(?=Step \d+:|$)/gs) || []
                  
                  if (stepMatches.length === 0) {
                    return (
                      <div className="text-center p-8 text-gray-500">
                        No reasoning steps detected in the response
                      </div>
                    )
                  }

                  const avgConfidence = tokens.length > 0 
                    ? tokens.reduce((sum: number, t: any) => sum + t.confidence, 0) / tokens.length * 100
                    : 0

                  return (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between mb-4">
                        <div className="text-sm text-gray-400">
                          Chain of Thought Analysis - {stepMatches.length} steps detected
                        </div>
                        <div className="text-sm text-gray-400">
                          Overall Confidence: <span className="text-white font-semibold">{avgConfidence.toFixed(1)}%</span>
                        </div>
                      </div>
                      <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-2">
                        {stepMatches.map((stepText, idx) => {
                          const stepNum = idx + 1
                          const stepTokens = tokens.filter((t: any) => {
                            const tokenPos = tokens.indexOf(t)
                            const stepStart = response.indexOf(stepText)
                            const stepEnd = stepStart + stepText.length
                            return tokenPos >= stepStart && tokenPos < stepEnd
                          })
                          
                          const stepAvgConf = stepTokens.length > 0 
                            ? stepTokens.reduce((sum: number, t: any) => sum + t.confidence, 0) / stepTokens.length * 100
                            : 0
                          
                          const status = stepAvgConf >= 80 ? '✓ High Confidence' : stepAvgConf >= 60 ? '⚠️ Medium Confidence' : '✗ Low Confidence'
                          const statusColor = stepAvgConf >= 80 ? 'text-green-400' : stepAvgConf >= 60 ? 'text-yellow-400' : 'text-red-400'
                          const bgColor = stepAvgConf >= 80 ? 'bg-green-500/10 border-green-500/30' : stepAvgConf >= 60 ? 'bg-yellow-500/10 border-yellow-500/30' : 'bg-red-500/10 border-red-500/30'
                          
                          return (
                            <div key={idx} className={`p-4 rounded-lg border ${bgColor}`}>
                              <div className="flex items-start justify-between mb-3">
                                <div className="flex-1">
                                  <div className="flex items-center gap-3 mb-2">
                                    <h4 className="text-white font-semibold text-lg">Step {stepNum}</h4>
                                    <span className={`text-sm font-semibold ${statusColor}`}>{status}</span>
                                  </div>
                                  <p className="text-gray-300 text-sm whitespace-pre-wrap leading-relaxed">{stepText.trim()}</p>
                                </div>
                              </div>
                              <div className="flex items-center gap-6 text-xs mt-3 pt-3 border-t border-gray-700/50">
                                <div className="flex items-center gap-2">
                                  <span className="text-gray-500">Tokens:</span>
                                  <span className="text-white font-semibold">{stepTokens.length}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <span className="text-gray-500">Avg Confidence:</span>
                                  <span className={`font-semibold ${statusColor}`}>{stepAvgConf.toFixed(1)}%</span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <span className="text-gray-500">vs Overall:</span>
                                  <span className={stepAvgConf > avgConfidence ? 'text-green-400' : 'text-red-400'}>
                                    {stepAvgConf > avgConfidence ? '↑' : '↓'} {Math.abs(stepAvgConf - avgConfidence).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })()}
                {expandedCard === 'response' && (
                  <div className="space-y-4">
                    {result.thinking && (
                      <div className="p-4 bg-gray-800/50 rounded-lg">
                        <p className="text-gray-400 text-sm uppercase tracking-wider mb-2">Reasoning</p>
                        <p className="text-gray-300 whitespace-pre-wrap">{result.thinking}</p>
                      </div>
                    )}
                    <div className="p-6 bg-gradient-to-r from-amber-500/10 to-green-500/10 border border-amber-500/30 rounded-lg">
                      <p className="text-amber-400 text-sm font-semibold mb-3">FINAL ANSWER</p>
                      <p className="text-white text-lg whitespace-pre-wrap">{result.final_answer || result.response}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
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
      </Routes>
    </Router>
  )
}

export default App
