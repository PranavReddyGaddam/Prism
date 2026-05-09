import { useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { PostHocSlot, LimeResult, TokenShapResult, CounterfactualResult, CounterfactualEdit } from '@/types/posthoc'
import AITooltip from '@/components/AITooltip'

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

// ── Counterfactual diff modal ────────────────────────────────────────────────

function lcsTable(a: string[], b: string[]): number[][] {
  const dp = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i--)
    for (let j = b.length - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1])
  return dp
}

function diffTokens(orig: string[], mod: string[]): { text: string; type: 'same' | 'removed' | 'added' }[] {
  const dp = lcsTable(orig, mod)
  const result: { text: string; type: 'same' | 'removed' | 'added' }[] = []
  let i = 0, j = 0
  while (i < orig.length || j < mod.length) {
    if (i < orig.length && j < mod.length && orig[i] === mod[j]) {
      result.push({ text: orig[i], type: 'same' }); i++; j++
    } else if (j < mod.length && (i >= orig.length || dp[i][j + 1] >= dp[i + 1][j])) {
      result.push({ text: mod[j], type: 'added' }); j++
    } else {
      result.push({ text: orig[i], type: 'removed' }); i++
    }
  }
  return result
}

// Split into sentences, diff sentence-by-sentence at word level.
// Unchanged sentences render as plain text; changed sentences show word-level highlights.
function diffSentences(original: string, modified: string): { text: string; type: 'same' | 'removed' | 'added' }[] {
  const sentRe = /[^.!?\n]+[.!?\n]*/g
  const origSents = original.match(sentRe) ?? [original]
  const modSents = modified.match(sentRe) ?? [modified]

  const sentDp = lcsTable(origSents, modSents)
  const result: { text: string; type: 'same' | 'removed' | 'added' }[] = []

  let i = 0, j = 0
  while (i < origSents.length || j < modSents.length) {
    if (i < origSents.length && j < modSents.length && origSents[i] === modSents[j]) {
      result.push({ text: origSents[i], type: 'same' }); i++; j++
    } else if (j < modSents.length && (i >= origSents.length || sentDp[i][j + 1] >= sentDp[i + 1][j])) {
      // New sentence in modified — diff at word level against removed counterpart if present
      if (i < origSents.length && sentDp[i + 1][j] < sentDp[i][j + 1]) {
        // paired change: orig[i] ↔ mod[j]
        const wordDiff = diffTokens(
          origSents[i].split(/(\s+)/),
          modSents[j].split(/(\s+)/),
        )
        result.push(...wordDiff); i++; j++
      } else {
        result.push({ text: modSents[j], type: 'added' }); j++
      }
    } else {
      result.push({ text: origSents[i], type: 'removed' }); i++
    }
  }
  return result
}

function CounterfactualModal({
  edit,
  referenceResponse,
  referenceAnswer,
  modelId,
  prompt,
  onClose,
}: {
  edit: CounterfactualEdit
  referenceResponse: string
  referenceAnswer: string | null
  modelId: string
  prompt: string
  onClose: () => void
}) {
  const diff = useMemo(() => diffSentences(referenceResponse, edit.new_response), [referenceResponse, edit.new_response])

  return createPortal(
    <div
      className="fixed inset-0 flex items-center justify-center"
      style={{ zIndex: 99999, backgroundColor: 'rgba(0,0,0,0.75)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="rounded-2xl flex flex-col"
        style={{
          width: 'min(860px, 92vw)',
          maxHeight: '85vh',
          backgroundColor: '#181818',
          border: '1px solid rgba(255,255,255,0.1)',
          boxShadow: '0 24px 80px rgba(0,0,0,0.9)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4" style={{ borderBottom: BORDER }}>
          <div className="flex items-center gap-3">
            <h3 className="text-white font-semibold text-base">Counterfactual Response</h3>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="rounded bg-red-900/30 text-red-200 px-2 py-0.5 font-mono">{edit.original_token}</span>
              <span className="text-gray-500">→</span>
              <span className="rounded bg-emerald-900/30 text-emerald-200 px-2 py-0.5 font-mono">{edit.replacement}</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs text-gray-500">
              answer: <span className="text-red-300 font-mono">{referenceAnswer}</span>
              <span className="mx-1 text-gray-600">→</span>
              <span className="text-emerald-300 font-mono">{edit.new_answer ?? '?'}</span>
            </span>
            <AITooltip card="counterfactual" modelId={modelId} data={{
              prompt: prompt.slice(0, 200),
              original_token: edit.original_token,
              replacement: edit.replacement,
              reference_answer: referenceAnswer,
              new_answer: edit.new_answer,
              original_response: referenceResponse.slice(0, 400),
              new_response: edit.new_response.slice(0, 400),
            }} />
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white transition-colors p-1.5 hover:bg-gray-800 rounded-lg"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 p-5 space-y-4">
          {/* Diff view */}
          <div>
            <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Response diff</p>
            <div
              className="rounded-lg p-4 text-sm leading-relaxed"
              style={{ backgroundColor: INNER_BG, border: BORDER }}
            >
              {diff.map((seg, i) => {
                if (seg.type === 'same') return <span key={i} className="text-gray-300">{seg.text}</span>
                if (seg.type === 'removed') return (
                  <span key={i} className="rounded px-0.5" style={{ backgroundColor: 'rgba(239,68,68,0.2)', color: '#fca5a5', textDecoration: 'line-through' }}>
                    {seg.text}
                  </span>
                )
                return (
                  <span key={i} className="rounded px-0.5" style={{ backgroundColor: 'rgba(52,211,153,0.2)', color: '#6ee7b7' }}>
                    {seg.text}
                  </span>
                )
              })}
            </div>
          </div>

          {/* Side by side */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-wider text-gray-500 mb-2">Original response</p>
              <div className="rounded-lg p-3 text-xs text-gray-400 leading-relaxed" style={{ backgroundColor: INNER_BG, border: BORDER }}>
                {referenceResponse}
              </div>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-wider text-emerald-400/80 mb-2">Counterfactual response</p>
              <div className="rounded-lg p-3 text-xs text-gray-300 leading-relaxed" style={{ backgroundColor: INNER_BG, border: '1px solid rgba(52,211,153,0.15)' }}>
                {edit.new_response}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

// ── Counterfactual card ──────────────────────────────────────────────────────

function CounterfactualCard({ data, modelId, prompt }: { data: CounterfactualResult; modelId: string; prompt: string }) {
  const [selectedEdit, setSelectedEdit] = useState<CounterfactualEdit | null>(null)

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
    <>
      {selectedEdit && (
        <CounterfactualModal
          edit={selectedEdit}
          referenceResponse={data.reference_response}
          referenceAnswer={data.reference_answer}
          modelId={modelId}
          prompt={prompt}
          onClose={() => setSelectedEdit(null)}
        />
      )}

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
              Edits that flipped the answer <span className="text-gray-600 normal-case tracking-normal">— click to expand</span>
            </p>
            <div className="space-y-2">
              {flipped.map((e, i) => (
                <div
                  key={i}
                  className="rounded p-2 text-xs cursor-pointer transition-colors"
                  style={{ backgroundColor: '#1a1a1a', border: BORDER }}
                  onClick={() => setSelectedEdit(e)}
                  onMouseEnter={(el) => (el.currentTarget.style.borderColor = 'rgba(52,211,153,0.35)')}
                  onMouseLeave={(el) => (el.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)')}
                >
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
                    <svg className="w-3 h-3 text-gray-600 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
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
              Edits that did not flip ({stable.length}) <span className="normal-case tracking-normal">— click to expand</span>
            </summary>
            <div className="space-y-1.5 mt-2">
              {stable.slice(0, 12).map((e, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 text-xs text-gray-400 rounded px-2 py-1 cursor-pointer transition-colors hover:bg-gray-800/40"
                  onClick={() => setSelectedEdit(e)}
                >
                  <span className="rounded bg-gray-800/40 px-1.5 py-0.5 font-mono">{e.original_token}</span>
                  <span className="text-gray-600">→</span>
                  <span className="rounded bg-gray-800/40 px-1.5 py-0.5 font-mono">{e.replacement}</span>
                  <span className="text-gray-600 ml-auto">
                    still <span className="text-gray-300 font-mono">{e.new_answer ?? '—'}</span>
                  </span>
                  <svg className="w-3 h-3 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </>
  )
}

// ── Container ────────────────────────────────────────────────────────────────

interface PostHocBentoProps {
  slot: PostHocSlot
}

function Pane({ title, subtitle, tooltip, children }: { title: string; subtitle: string; tooltip?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="rounded-xl p-4" style={{ backgroundColor: CARD_BG, border: BORDER }}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-baseline gap-3">
          <h3 className="text-white text-base font-semibold">{title}</h3>
          <span className="text-gray-500 text-xs">{subtitle}</span>
        </div>
        {tooltip}
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
            <CounterfactualCard data={slot.counterfactual} modelId={slot.model_id ?? ''} prompt={slot.prompt ?? ''} />
          </Pane>
        ) : (
          <LoadingPane
            title="Counterfactual Edits"
            subtitle="Smallest input changes that flip the model's answer"
          />
        )}
      </div>

      {slot.lime ? (
        <Pane
          title="LIME"
          subtitle="Word-level attribution from local linear surrogate"
          tooltip={(() => {
            const d = slot.lime!
            const sorted = d.words.map((w, i) => ({ word: w, score: d.attributions[i] ?? 0 }))
              .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
            return (
              <AITooltip card="lime" modelId={slot.model_id ?? ''} data={{
                reference_answer: d.reference_answer,
                r_squared: +d.r_squared.toFixed(3),
                n_samples: d.n_samples,
                top_words: sorted.slice(0, 6).map(x => ({ word: x.word, score: +x.score.toFixed(4) })),
                bottom_words: sorted.slice(-3).map(x => ({ word: x.word, score: +x.score.toFixed(4) })),
              }} />
            )
          })()}
        >
          <LimeCard data={slot.lime} />
        </Pane>
      ) : (
        <LoadingPane title="LIME" subtitle="Word-level attribution from local linear surrogate" />
      )}

      {slot.tokenshap ? (
        <Pane
          title="TokenSHAP"
          subtitle="Shapley-value attribution from token coalitions"
          tooltip={(() => {
            const d = slot.tokenshap!
            const sorted = d.words.map((w, i) => ({ word: w, score: d.attributions[i] ?? 0 }))
              .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
            return (
              <AITooltip card="tokenshap" modelId={slot.model_id ?? ''} data={{
                reference_answer: d.reference_answer,
                n_samples: d.n_samples,
                top_words: sorted.slice(0, 6).map(x => ({ word: x.word, score: +x.score.toFixed(4) })),
                bottom_words: sorted.slice(-3).map(x => ({ word: x.word, score: +x.score.toFixed(4) })),
              }} />
            )
          })()}
        >
          <TokenShapCard data={slot.tokenshap} />
        </Pane>
      ) : (
        <LoadingPane title="TokenSHAP" subtitle="Shapley-value attribution from token coalitions" />
      )}
    </div>
  )
}
