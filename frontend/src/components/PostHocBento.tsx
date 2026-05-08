import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { PostHocSlot, LimeResult, TokenShapResult, CounterfactualResult } from '@/types/posthoc'

const POS_COLOR = '#86efac'
const NEG_COLOR = '#fca5a5'
const NEUTRAL_BG = 'rgba(255,255,255,0.04)'

const CARD_BG = '#202020'
const INNER_BG = '#2a2a2a'
const BORDER = '1px solid rgba(255,255,255,0.08)'

function colorForAttr(value: number, max: number): string {
  if (max <= 0) return NEUTRAL_BG
  const intensity = Math.min(1, Math.abs(value) / max)
  if (value >= 0) return `rgba(134, 239, 172, ${0.15 + intensity * 0.55})`
  return `rgba(252, 165, 165, ${0.15 + intensity * 0.55})`
}

function TokenHighlights({ words, attributions }: { words: string[]; attributions: number[] }) {
  const max = Math.max(...attributions.map((a) => Math.abs(a)), 1e-9)
  return (
    <div className="flex flex-wrap gap-1 leading-relaxed">
      {words.map((w, i) => (
        <span
          key={i}
          className="rounded px-1.5 py-0.5 text-sm text-white/90"
          style={{ backgroundColor: colorForAttr(attributions[i] ?? 0, max) }}
          title={`${w} → ${(attributions[i] ?? 0).toFixed(4)}`}
        >
          {w}
        </span>
      ))}
    </div>
  )
}

function TopKBars({ words, attributions, k = 12 }: { words: string[]; attributions: number[]; k?: number }) {
  const data = useMemo(() => {
    const pairs = words.map((w, i) => ({ word: w, score: attributions[i] ?? 0 }))
    return pairs
      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
      .slice(0, k)
      .reverse()
  }, [words, attributions, k])

  return (
    <div style={{ outline: 'none' }}>
      <ResponsiveContainer width="100%" height={Math.max(220, data.length * 26)}>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 30, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis type="number" stroke="rgba(255,255,255,0.4)" fontSize={10} tickLine={false} tick={{ fill: 'rgba(255,255,255,0.6)' }} />
          <YAxis type="category" dataKey="word" stroke="rgba(255,255,255,0.6)" fontSize={11} width={100} tickLine={false} axisLine={false} tick={{ fill: 'rgba(255,255,255,0.85)' }} />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12, color: '#fff' }}
            labelStyle={{ color: '#fff' }}
            itemStyle={{ color: '#fff' }}
            formatter={(v: number | undefined) => [(v ?? 0).toFixed(4), 'score']}
          />
          <Bar dataKey="score" radius={[0, 3, 3, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.score >= 0 ? POS_COLOR : NEG_COLOR} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── LIME card ────────────────────────────────────────────────────────────────

function LimeCard({ data }: { data: LimeResult }) {
  if (data.n_samples === 0) {
    return <div className="text-sm text-red-400 p-3">LIME computation failed — check pod logs.</div>
  }
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs">
        <span className="text-gray-500">
          Reference answer: <span style={{ color: '#fff' }} className="font-mono">{data.reference_answer ?? '—'}</span>
        </span>
        <span className="text-gray-500">
          R<sup style={{ color: '#9ca3af' }}>2</sup>: <span style={{ color: '#fff' }} className="font-mono">{data.r_squared.toFixed(3)}</span>
        </span>
        <span className="text-gray-500">
          Samples: <span style={{ color: '#fff' }} className="font-mono">{data.n_samples}</span>
        </span>
      </div>
      <div className="rounded-lg p-3" style={{ backgroundColor: INNER_BG, border: BORDER }}>
        <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Word-level attributions</p>
        <TokenHighlights words={data.words} attributions={data.attributions} />
      </div>
      <div className="rounded-lg p-3" style={{ backgroundColor: INNER_BG, border: BORDER }}>
        <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Top influential words</p>
        <TopKBars words={data.words} attributions={data.attributions} />
      </div>
    </div>
  )
}

// ── TokenSHAP card ───────────────────────────────────────────────────────────

function TokenShapCard({ data }: { data: TokenShapResult }) {
  if (data.n_samples === 0) {
    return <div className="text-sm text-red-400 p-3">TokenSHAP computation failed — check pod logs.</div>
  }
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs">
        <span className="text-gray-500">
          Reference answer: <span style={{ color: '#fff' }} className="font-mono">{data.reference_answer ?? '—'}</span>
        </span>
        <span className="text-gray-500">
          Coalitions evaluated: <span style={{ color: '#fff' }} className="font-mono">{data.n_samples}</span>
        </span>
      </div>
      <div className="rounded-lg p-3" style={{ backgroundColor: INNER_BG, border: BORDER }}>
        <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Shapley values per word</p>
        <TokenHighlights words={data.words} attributions={data.attributions} />
      </div>
      <div className="rounded-lg p-3" style={{ backgroundColor: INNER_BG, border: BORDER }}>
        <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Top contributors</p>
        <TopKBars words={data.words} attributions={data.attributions} />
      </div>
    </div>
  )
}

// ── Counterfactual card ──────────────────────────────────────────────────────

function CounterfactualCard({ data }: { data: CounterfactualResult }) {
  if (data.note && data.edits.length === 0) {
    return (
      <div className="text-sm text-gray-400 p-4 rounded-lg" style={{ backgroundColor: INNER_BG, border: BORDER }}>
        {data.note}
      </div>
    )
  }
  const flipped = data.edits.filter((e) => e.flipped)
  const stable = data.edits.filter((e) => !e.flipped)

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-xs">
        <span className="text-gray-500">
          Reference answer: <span style={{ color: '#fff' }} className="font-mono">{data.reference_answer ?? '—'}</span>
        </span>
        <span className="text-gray-500">
          Edits tried: <span style={{ color: '#fff' }} className="font-mono">{data.n_edits}</span>
        </span>
        <span className="text-gray-500">
          Flipped: <span style={{ color: '#34d399' }} className="font-mono">{data.n_flipped}</span>
        </span>
      </div>

      {flipped.length > 0 && (
        <div className="rounded-lg p-3" style={{ backgroundColor: INNER_BG, border: BORDER }}>
          <p className="text-[10px] uppercase tracking-wider text-emerald-400/90 mb-2">
            Edits that flipped the answer
          </p>
          <div className="space-y-2">
            {flipped.map((e, i) => (
              <div key={i} className="rounded p-2 text-xs" style={{ backgroundColor: '#1a1a1a', border: BORDER }}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-gray-500">change</span>
                  <span className="rounded bg-red-900/30 text-red-200 px-1.5 py-0.5 font-mono">{e.original_token}</span>
                  <span className="text-gray-500">→</span>
                  <span className="rounded bg-emerald-900/30 text-emerald-200 px-1.5 py-0.5 font-mono">{e.replacement}</span>
                  <span className="ml-auto text-gray-500">
                    answer: <span className="text-red-300 font-mono">{data.reference_answer}</span>
                    <span className="mx-1">→</span>
                    <span className="text-emerald-300 font-mono">{e.new_answer ?? '?'}</span>
                  </span>
                </div>
                <div className="text-gray-500 line-clamp-2">{e.new_response}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {stable.length > 0 && (
        <details className="rounded-lg p-3" style={{ backgroundColor: INNER_BG, border: BORDER }}>
          <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-gray-500">
            Edits that did not flip ({stable.length})
          </summary>
          <div className="space-y-1.5 mt-2">
            {stable.slice(0, 12).map((e, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
                <span className="rounded bg-gray-800/40 px-1.5 py-0.5 font-mono">{e.original_token}</span>
                <span className="text-gray-600">→</span>
                <span className="rounded bg-gray-800/40 px-1.5 py-0.5 font-mono">{e.replacement}</span>
                <span className="text-gray-600 ml-auto">
                  still <span className="text-gray-300 font-mono">{e.new_answer ?? '—'}</span>
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

// ── Container ────────────────────────────────────────────────────────────────

interface PostHocBentoProps {
  slot: PostHocSlot
}

function Pane({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl p-4" style={{ backgroundColor: CARD_BG, border: BORDER }}>
      <div className="flex items-baseline gap-3 mb-3">
        <h3 className="text-white text-base font-semibold">{title}</h3>
        <span className="text-gray-500 text-xs">{subtitle}</span>
      </div>
      {children}
    </div>
  )
}

function LoadingPane({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <Pane title={title} subtitle={subtitle}>
      <div className="space-y-2">
        <div className="h-3 bg-gray-800 rounded animate-pulse w-3/4" />
        <div className="h-3 bg-gray-800 rounded animate-pulse w-2/3" />
        <div className="h-3 bg-gray-800 rounded animate-pulse w-5/6" />
        <div className="h-32 rounded animate-pulse mt-3" style={{ backgroundColor: INNER_BG }} />
      </div>
    </Pane>
  )
}

export default function PostHocBento({ slot }: PostHocBentoProps) {
  if (slot.status === 'idle') {
    return (
      <div className="rounded-xl p-10 text-center" style={{ backgroundColor: CARD_BG, border: BORDER }}>
        <p className="text-gray-400 text-sm">
          Submit a prompt to compute post-hoc explanations (LIME, TokenSHAP, counterfactuals).
        </p>
      </div>
    )
  }

  if (slot.status === 'error') {
    return (
      <div className="rounded-xl p-6 text-sm text-red-300" style={{ backgroundColor: '#1a0a0a', border: '1px solid rgba(239,68,68,0.3)' }}>
        Post-hoc analysis failed: {slot.error ?? 'unknown error'}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <div className="lg:col-span-2">
        {slot.counterfactual ? (
          <Pane
            title="Counterfactual Edits"
            subtitle="Smallest input changes that flip the model's answer"
          >
            <CounterfactualCard data={slot.counterfactual} />
          </Pane>
        ) : (
          <LoadingPane
            title="Counterfactual Edits"
            subtitle="Smallest input changes that flip the model's answer"
          />
        )}
      </div>

      {slot.lime ? (
        <Pane title="LIME" subtitle="Word-level attribution from local linear surrogate">
          <LimeCard data={slot.lime} />
        </Pane>
      ) : (
        <LoadingPane title="LIME" subtitle="Word-level attribution from local linear surrogate" />
      )}

      {slot.tokenshap ? (
        <Pane title="TokenSHAP" subtitle="Shapley-value attribution from token coalitions">
          <TokenShapCard data={slot.tokenshap} />
        </Pane>
      ) : (
        <LoadingPane title="TokenSHAP" subtitle="Shapley-value attribution from token coalitions" />
      )}
    </div>
  )
}
