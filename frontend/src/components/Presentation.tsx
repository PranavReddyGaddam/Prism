import { useEffect, useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChevronLeft,
  ChevronRight,
  ArrowRight,
  Play,
  Eye,
  Layers,
  Network,
  Cpu,
  GitBranch,
} from 'lucide-react'
import SiteBackground from '@/components/SiteBackground'

const CARD_BG = '#202020'
const INNER_BG = '#2a2a2a'
const BORDER = '1px solid rgba(255,255,255,0.08)'

// ── Reusable primitives ──────────────────────────────────────────────────────

function SlideShell({
  children,
  centered = false,
}: {
  children: React.ReactNode
  centered?: boolean
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className={`w-full h-full flex flex-col px-24 py-20 ${
        centered ? 'items-center justify-center text-center' : 'justify-center'
      }`}
    >
      {children}
    </motion.div>
  )
}

function SlideHeader({ kicker, title }: { kicker?: string; title: string }) {
  return (
    <div className="mb-12">
      {kicker && (
        <p className="text-amber-400/90 text-sm font-mono tracking-[0.2em] uppercase mb-4">
          {kicker}
        </p>
      )}
      <h2 className="text-white text-6xl font-bold tracking-tight leading-[1.05]">
        {title}
      </h2>
    </div>
  )
}

function Card({
  children,
  className = '',
  inner = false,
}: {
  children: React.ReactNode
  className?: string
  inner?: boolean
}) {
  return (
    <div
      className={`rounded-2xl p-7 ${className}`}
      style={{ backgroundColor: inner ? INNER_BG : CARD_BG, border: BORDER }}
    >
      {children}
    </div>
  )
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span
      className="px-3 py-1 rounded-full text-xs font-mono text-gray-300"
      style={{ backgroundColor: INNER_BG, border: BORDER }}
    >
      {children}
    </span>
  )
}

function Stat({
  label,
  value,
  sub,
  accent = '#fbbf24',
}: {
  label: string
  value: string
  sub?: string
  accent?: string
}) {
  return (
    <div>
      <p className="text-gray-500 text-[10px] uppercase tracking-[0.15em] mb-2">{label}</p>
      <p className="text-4xl font-bold leading-none tracking-tight" style={{ color: accent }}>
        {value}
      </p>
      {sub && <p className="text-gray-400 text-xs mt-2 leading-snug">{sub}</p>}
    </div>
  )
}

// ── Individual slides ────────────────────────────────────────────────────────

function Slide01Title() {
  return (
    <SlideShell centered>
      <p className="text-amber-400/90 text-xs font-mono tracking-[0.4em] uppercase mb-10">
        DATA 298B · Major Project · Spring 2026
      </p>
      <h1 className="text-white text-[10rem] font-bold tracking-tight leading-none mb-8">
        Prism
      </h1>
      <p className="text-gray-200 text-3xl font-light tracking-wide mb-4">
        Inside the Black Box
      </p>
      <p className="text-gray-500 text-base max-w-2xl leading-relaxed">
        A mechanistic interpretability platform for visualizing what large language
        models are actually computing.
      </p>
      <div className="mt-20 grid grid-cols-2 gap-x-16 gap-y-2 text-sm">
        <div className="text-right">
          <p className="text-gray-600 text-[10px] uppercase tracking-[0.2em] mb-1">Team</p>
          <p className="text-gray-300">Pranav Reddy Gaddam</p>
          <p className="text-gray-300">Shweta Shinde</p>
          <p className="text-gray-300">Pulkit Srivastava</p>
          <p className="text-gray-300">Gayatri Patil</p>
        </div>
        <div className="text-left">
          <p className="text-gray-600 text-[10px] uppercase tracking-[0.2em] mb-1">Institution</p>
          <p className="text-gray-300">San Jose State University</p>
          <p className="text-gray-300">Department of Applied Data Science</p>
          <p className="text-gray-500 text-xs mt-3 italic">Spring 2026</p>
        </div>
      </div>
    </SlideShell>
  )
}

function Slide02Problem() {
  return (
    <SlideShell>
      <SlideHeader kicker="The Problem" title="LLMs reason — we can't see how." />
      <div className="grid grid-cols-2 gap-8">
        <Card>
          <p className="text-red-400/90 text-[10px] font-mono uppercase tracking-[0.2em] mb-3">
            What we see
          </p>
          <h3 className="text-white text-2xl font-semibold mb-5">A token stream.</h3>
          <p className="text-gray-400 leading-relaxed mb-5 text-base">
            The model says "Let's think step by step…" and writes prose. We read it. It
            looks like reasoning, so we trust it.
          </p>
          <Card inner className="font-mono text-sm text-gray-300 leading-loose">
            <span className="text-gray-500">$</span> 5 apples × $1.20 = $6.00
            <br />
            <span className="text-gray-500">$</span> 8 oranges × $0.80 = $6.40
            <br />
            <span className="text-gray-500">$</span> Total: $12.40 ✓
          </Card>
          <p className="text-gray-500 text-sm mt-5 italic">
            Confidence built on words, not computation.
          </p>
        </Card>
        <Card>
          <p className="text-amber-400/90 text-[10px] font-mono uppercase tracking-[0.2em] mb-3">
            What's actually happening
          </p>
          <h3 className="text-white text-2xl font-semibold mb-5">
            32 layers of opaque tensors.
          </h3>
          <p className="text-gray-400 leading-relaxed mb-5 text-base">
            4096-dimensional residual streams. Attention heads attending to numbers in
            patterns we never inspect.
          </p>
          <Card inner className="font-mono text-xs text-gray-500 leading-loose">
            layer 0&nbsp;&nbsp;&nbsp;→ token embeddings
            <br />
            layer 1..31 → <span className="text-amber-400">?????</span>
            <br />
            layer 32&nbsp;&nbsp;→ output token
          </Card>
          <p className="text-gray-500 text-sm mt-5 italic">
            The 30 layers in the middle do all the actual work.
          </p>
        </Card>
      </div>
      <div className="mt-12 text-center">
        <p className="text-gray-400 text-xl tracking-wide">
          Chain-of-Thought is a{' '}
          <span className="text-white font-semibold italic">post-hoc rationalization</span>,
          not a window into computation.
        </p>
      </div>
    </SlideShell>
  )
}

function Slide03WhyNow() {
  const items = [
    {
      color: '#fbbf24',
      number: '01',
      title: 'Trust in production',
      pts: [
        'LLMs are deployed in healthcare diagnostics, financial forecasting, and educational tutoring.',
        '"Trust the output" is no longer acceptable when the cost of a wrong answer is real.',
        'Regulators (EU AI Act, NIST AI RMF) now require evidence of how a model arrived at a decision.',
      ],
    },
    {
      color: '#f87171',
      number: '02',
      title: 'Failure forensics',
      pts: [
        'When a math problem is wrong, was it the wrong number, the wrong operation, or the wrong setup?',
        'Logs only show the final answer. Attribution maps and counterfactuals reveal which input drove the error.',
        'Without this, fixing prompts is guesswork — every prompt change is a new experiment.',
      ],
    },
    {
      color: '#34d399',
      number: '03',
      title: 'Model selection',
      pts: [
        'Two models can both score 80% on GSM8K but reason completely differently.',
        'One memorizes; the other generalizes. The benchmark cannot tell you which is which.',
        'Mechanistic comparison reveals which model will hold up on out-of-distribution problems.',
      ],
    },
  ]
  return (
    <SlideShell>
      <SlideHeader kicker="Motivation" title="Why this matters now." />
      <div className="grid grid-cols-3 gap-6">
        {items.map((it) => (
          <Card key={it.title} className="flex flex-col">
            <p
              className="text-3xl font-bold font-mono mb-5 leading-none"
              style={{ color: it.color, opacity: 0.7 }}
            >
              {it.number}
            </p>
            <h3 className="text-white text-2xl font-semibold mb-5 leading-tight">
              {it.title}
            </h3>
            <ul className="space-y-3 text-gray-400 text-sm leading-relaxed">
              {it.pts.map((p, i) => (
                <li key={i} className="flex gap-3">
                  <span
                    className="block w-1 h-1 rounded-full mt-2 flex-shrink-0"
                    style={{ backgroundColor: it.color }}
                  />
                  <span>{p}</span>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>
      <p className="text-center text-gray-500 text-base italic mt-10">
        The window of "ship it and hope for the best" has closed.
      </p>
    </SlideShell>
  )
}

function Slide04Architecture() {
  const layers = [
    {
      tag: 'FRONTEND',
      color: '#7dd3fc',
      title: 'React SPA',
      desc:
        'Vite + TypeScript + Tailwind. Bento grid for model comparison. Recharts and D3 for charts; Three.js for the attribution graph.',
      pills: ['React 19', 'Vite', 'Tailwind', 'Recharts', 'Three.js'],
    },
    {
      tag: 'BACKEND',
      color: '#c4b5fd',
      title: 'FastAPI proxy',
      desc:
        'Stateless orchestrator. Validates requests, formats math prompts, fans out to the GPU server. Zero model code locally — keeps the API surface clean.',
      pills: ['FastAPI', 'httpx', 'Pydantic', 'asyncio'],
    },
    {
      tag: 'GPU SERVER',
      color: '#6ee7b7',
      title: 'RunPod bridge',
      desc:
        'HuggingFace transformers loaded into A100 memory. Exposes /generate, /explain/*, /posthoc/*. All four 7B models held in VRAM concurrently.',
      pills: ['PyTorch', 'HF Transformers', 'RunPod A100'],
    },
  ]
  return (
    <SlideShell>
      <SlideHeader kicker="System Design" title="Three layers, one pipeline." />
      <div className="grid grid-cols-3 gap-6 mb-8">
        {layers.map((l) => (
          <Card key={l.tag} className="flex flex-col">
            <div className="flex items-baseline justify-between mb-4">
              <p
                className="text-xs font-mono tracking-[0.25em] font-semibold"
                style={{ color: l.color }}
              >
                {l.tag}
              </p>
              <span
                className="text-3xl font-bold font-mono opacity-30"
                style={{ color: l.color }}
              >
                /
              </span>
            </div>
            <h3 className="text-white text-2xl font-semibold mb-3">{l.title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed mb-6 flex-1">{l.desc}</p>
            <div className="flex flex-wrap gap-2">
              {l.pills.map((p) => (
                <Pill key={p}>{p}</Pill>
              ))}
            </div>
          </Card>
        ))}
      </div>
      <Card>
        <div className="flex items-center justify-around">
          {[
            { label: 'User', val: 'Browser' },
            { label: 'Frontend', val: 'localhost:5173' },
            { label: 'Backend', val: 'FastAPI :8000' },
            { label: 'GPU', val: 'RunPod /proxy' },
          ].map((node, i, arr) => (
            <div key={node.label} className="flex items-center">
              <div className="text-center">
                <p className="text-gray-500 text-[10px] uppercase tracking-[0.15em] mb-1.5">
                  {node.label}
                </p>
                <p className="text-white text-sm font-mono">{node.val}</p>
              </div>
              {i < arr.length - 1 && <ArrowRight className="w-5 h-5 text-gray-600 mx-8" />}
            </div>
          ))}
        </div>
      </Card>
    </SlideShell>
  )
}

function Slide05ArchDiagram() {
  return (
    <SlideShell>
      <SlideHeader kicker="Architecture" title="Data flow." />
      <Card className="flex-1 flex items-center justify-center min-h-[60vh]" inner>
        <p className="text-gray-500 text-center text-base">
          [ paste architectural diagram screenshot here ]
        </p>
      </Card>
    </SlideShell>
  )
}

function Slide06Models() {
  const models = [
    {
      name: 'Qwen 2.5 Math 7B',
      org: 'Alibaba Cloud',
      arch: 'Decoder-only · 28 layers · 28 heads · GQA',
      training: 'Pre-trained on math-heavy corpus (~1T tokens) + SFT on chain-of-thought + boxed answers.',
      specialty: 'Strong at multi-step arithmetic and symbolic manipulation.',
    },
    {
      name: 'DeepSeek Math 7B',
      org: 'DeepSeek',
      arch: 'LLaMA-style decoder · 30 layers · 32 heads · RoPE',
      training: 'Continued pre-training from DeepSeek-Coder-Base on 120B math tokens, instruction-tuned.',
      specialty: 'Notably strong on competition-style problems (AMC, AIME).',
    },
    {
      name: 'InternLM2 Math+ 7B',
      org: 'Shanghai AI Lab',
      arch: 'InternLM2 architecture · 32 layers · 32 heads · GQA',
      training: 'Tool-augmented training: model learns to invoke a Python interpreter for verification.',
      specialty: 'Best on problems requiring exact numerical computation.',
    },
    {
      name: 'WizardMath 7B',
      org: 'Microsoft',
      arch: 'LLaMA 2 base · 32 layers · 32 heads · MHA',
      training: 'Reinforcement Learning from Evol-Instruct Feedback (RLEIF) on synthetic math instructions.',
      specialty: 'Robust to instruction phrasing variations and adversarial wording.',
    },
  ]
  return (
    <SlideShell>
      <SlideHeader kicker="Subjects" title="Four math LLMs, side by side." />
      <div className="grid grid-cols-2 grid-rows-2 gap-5">
        {models.map((m) => (
          <Card key={m.name} className="flex flex-col py-5">
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-white text-xl font-semibold leading-tight">{m.name}</h3>
              <Pill>{m.org}</Pill>
            </div>
            <div className="grid grid-cols-3 gap-4 flex-1">
              <div>
                <p className="text-amber-400/80 text-[10px] font-mono uppercase tracking-[0.15em] mb-1.5">
                  Architecture
                </p>
                <p className="text-gray-300 text-xs leading-relaxed">{m.arch}</p>
              </div>
              <div>
                <p className="text-amber-400/80 text-[10px] font-mono uppercase tracking-[0.15em] mb-1.5">
                  Training
                </p>
                <p className="text-gray-400 text-xs leading-relaxed">{m.training}</p>
              </div>
              <div>
                <p className="text-amber-400/80 text-[10px] font-mono uppercase tracking-[0.15em] mb-1.5">
                  Specialty
                </p>
                <p className="text-gray-400 text-xs leading-relaxed italic">{m.specialty}</p>
              </div>
            </div>
            <div className="pt-3 mt-4 border-t border-white/5 text-[10px] text-gray-500 font-mono">
              7B params · loaded on A100 · float16
            </div>
          </Card>
        ))}
      </div>
    </SlideShell>
  )
}

function Slide07MechOverview() {
  const visualizations = [
    {
      icon: Eye,
      name: 'Token Confidence',
      what: 'Softmax probability of each generated token at decode time.',
      why: 'Reveals exactly where the model hesitated — high entropy spikes flag uncertain steps.',
    },
    {
      icon: Network,
      name: 'Attention Heatmap',
      what: 'Per-layer, per-head attention weights between every pair of input tokens.',
      why: 'Shows which prior tokens each head considers when emitting each new token.',
    },
    {
      icon: Layers,
      name: 'Logit Lens',
      what: "Project each layer's hidden state through the unembedding matrix.",
      why: 'Reveals when the answer crystallizes — middle layers vs. final layer prediction.',
    },
    {
      icon: GitBranch,
      name: 'Attribution Graph',
      what: 'Attention rollout: multiply attention matrices across all layers.',
      why: 'Maps which input tokens causally drive which output tokens — a flow map.',
    },
    {
      icon: Cpu,
      name: 'Hidden State Norms',
      what: 'L2 norm of the residual stream at each layer for each token position.',
      why: 'A profile of computational "energy" — peaks mark where major work happens.',
    },
  ]
  return (
    <SlideShell>
      <SlideHeader
        kicker="Mechanistic Interpretability"
        title="Five views into the same forward pass."
      />
      <div className="grid grid-cols-5 gap-4 mb-8">
        {visualizations.map((v) => (
          <Card key={v.name} className="flex flex-col">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
              style={{ backgroundColor: 'rgba(251,191,36,0.1)' }}
            >
              <v.icon className="w-5 h-5 text-amber-400" />
            </div>
            <h3 className="text-white text-base font-semibold mb-3 leading-tight">
              {v.name}
            </h3>
            <div className="mb-3">
              <p className="text-amber-400/70 text-[9px] font-mono uppercase tracking-[0.15em] mb-1">
                What
              </p>
              <p className="text-gray-300 text-xs leading-relaxed">{v.what}</p>
            </div>
            <div>
              <p className="text-amber-400/70 text-[9px] font-mono uppercase tracking-[0.15em] mb-1">
                Why it matters
              </p>
              <p className="text-gray-400 text-xs leading-relaxed">{v.why}</p>
            </div>
          </Card>
        ))}
      </div>
      <Card inner>
        <p className="text-center text-gray-400 text-sm leading-relaxed">
          All five computed from a single{' '}
          <span className="text-white font-mono">model.forward()</span> call with{' '}
          <span className="text-white font-mono">output_attentions=True</span>,{' '}
          <span className="text-white font-mono">output_hidden_states=True</span> — no
          gradient passes, no extra inference. One forward, five lenses.
        </p>
      </Card>
    </SlideShell>
  )
}

function Slide08AttributionGraph() {
  return (
    <SlideShell>
      <SlideHeader kicker="Method · Mechanistic" title="Attention rollout, not gradients." />
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">
            Why we rejected gradient attribution
          </h3>
          <ul className="space-y-4 text-gray-400 leading-relaxed text-sm">
            <li className="flex gap-4">
              <span className="text-red-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ×
              </span>
              <div>
                <p className="text-gray-200 font-medium mb-1">Memory blowup at 7B scale</p>
                <p>
                  Backprop through 32 transformer layers retains the full computation graph.
                  At 7B parameters with batch=1, this exceeds A100 memory.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="text-red-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ×
              </span>
              <div>
                <p className="text-gray-200 font-medium mb-1">Custom modeling code breaks hooks</p>
                <p>
                  Phi-3, InternLM, and Qwen ship with{' '}
                  <span className="font-mono text-gray-300">trust_remote_code=True</span>{' '}
                  modeling files that don't honor PyTorch's gradient hooks.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="text-red-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ×
              </span>
              <div>
                <p className="text-gray-200 font-medium mb-1">Saliency maps lie</p>
                <p>
                  A well-known result (Adebayo et al., 2018): gradient saliency often passes
                  visual inspection but fails sanity-check tests like model randomization.
                </p>
              </div>
            </li>
          </ul>
        </Card>
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">Attention rollout</h3>
          <p className="text-gray-400 leading-relaxed mb-5 text-sm">
            Multiply the attention matrices across layers. The product is a{' '}
            <span className="text-white font-mono">[input × output]</span> influence map
            derived purely from forward-pass tensors.
          </p>
          <Card inner className="font-mono text-sm text-gray-300 leading-loose mb-5">
            <span className="text-amber-400">A_rollout</span> =
            <br />
            &nbsp;&nbsp;A<sub>L</sub> @ A<sub>L-1</sub> @ ... @ A<sub>0</sub>
            <br />
            <br />
            <span className="text-gray-500"># works on any architecture</span>
            <br />
            <span className="text-gray-500"># O(L · n²) — cheap</span>
          </Card>
          <div className="space-y-3 text-sm">
            <div className="flex gap-3">
              <span className="text-emerald-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ✓
              </span>
              <p className="text-gray-400">
                <span className="text-gray-200 font-medium">Architecture-agnostic.</span>{' '}
                Works for any transformer that exposes attention weights.
              </p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ✓
              </span>
              <p className="text-gray-400">
                <span className="text-gray-200 font-medium">Cheap.</span> One extra forward
                pass with{' '}
                <span className="font-mono text-gray-300">output_attentions=True</span>.
              </p>
            </div>
            <div className="flex gap-3">
              <span className="text-emerald-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ✓
              </span>
              <p className="text-gray-400">
                <span className="text-gray-200 font-medium">Faithful.</span> Reflects the
                actual information flow the model used.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </SlideShell>
  )
}

function Slide09PostHocOverview() {
  const methods = [
    {
      color: '#34d399',
      name: 'LIME',
      tagline: 'Local Interpretable Model-agnostic Explanations',
      pts: [
        'Mask random subsets of input words; record whether the answer survives.',
        'Fit a weighted linear surrogate around the original prompt.',
        'Coefficients = each word\'s local importance to the prediction.',
      ],
    },
    {
      color: '#fbbf24',
      name: 'TokenSHAP',
      tagline: 'Shapley values via Monte Carlo coalitions',
      pts: [
        'Sample random permutations of words to form coalitions.',
        'Measure marginal contribution of adding each word to a coalition.',
        'Average across permutations → Shapley value per word.',
      ],
    },
    {
      color: '#f87171',
      name: 'Counterfactuals',
      tagline: 'Minimal numeric edits',
      pts: [
        'Regex-extract every numeric span from the prompt.',
        'Perturb the largest-magnitude numbers by ±1, run a forward pass.',
        'If the final answer changed, that number is causally bound to the answer.',
      ],
    },
  ]
  return (
    <SlideShell>
      <SlideHeader
        kicker="Post-hoc Analysis"
        title="Treat the model as a black box. Probe its boundary."
      />
      <div className="grid grid-cols-3 gap-6 mb-8">
        {methods.map((m) => (
          <Card key={m.name} className="flex flex-col">
            <div
              className="h-1 w-12 rounded-full mb-6"
              style={{ backgroundColor: m.color }}
            />
            <h3 className="text-white text-4xl font-bold mb-2 leading-none">{m.name}</h3>
            <p className="text-gray-500 text-xs font-mono mb-6 leading-relaxed">
              {m.tagline}
            </p>
            <ul className="space-y-3">
              {m.pts.map((p, i) => (
                <li key={i} className="flex gap-3 text-gray-400 text-sm leading-relaxed">
                  <span
                    className="block w-1 h-1 rounded-full mt-2 flex-shrink-0"
                    style={{ backgroundColor: m.color }}
                  />
                  <span>{p}</span>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>
      <Card inner>
        <div className="grid grid-cols-2 gap-8 items-center">
          <div className="text-right">
            <p className="text-amber-400/80 text-[10px] font-mono uppercase tracking-[0.2em] mb-1">
              Mechanistic
            </p>
            <p className="text-white text-base">"What's inside the model."</p>
          </div>
          <div>
            <p className="text-amber-400/80 text-[10px] font-mono uppercase tracking-[0.2em] mb-1">
              Post-hoc
            </p>
            <p className="text-white text-base">"What changes if I poke it."</p>
          </div>
        </div>
      </Card>
    </SlideShell>
  )
}

function Slide10LimeDeep() {
  return (
    <SlideShell>
      <SlideHeader kicker="Method · LIME" title="Random masking + closed-form ridge." />
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">Algorithm</h3>
          <ol className="space-y-4 text-gray-400 text-sm leading-relaxed">
            {[
              ['Generate', 'N=20 random binary masks over the input words.'],
              ['Run', 'For each mask, run the perturbed prompt through the model.'],
              [
                'Score',
                <>
                  <span className="text-white font-mono">1.0</span> if the final answer
                  matches the reference, else <span className="text-white font-mono">0.0</span>.
                </>,
              ],
              [
                'Weight',
                'Samples weighted by cosine similarity to the all-ones reference mask.',
              ],
              [
                'Solve',
                'Closed-form weighted ridge regression — coefficients are word importances.',
              ],
            ].map(([title, body], i) => (
              <li key={i} className="flex gap-4">
                <span className="text-amber-400 font-mono font-bold w-5 flex-shrink-0">
                  {i + 1}.
                </span>
                <div>
                  <span className="text-gray-200 font-medium">{title}.</span>{' '}
                  <span>{body}</span>
                </div>
              </li>
            ))}
          </ol>
        </Card>
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">The math</h3>
          <Card inner className="font-mono text-sm text-gray-200 leading-loose mb-5">
            <span className="text-gray-500"># closed-form, no sklearn</span>
            <br />
            <span className="text-amber-400">β</span> = (Xᵀ
            <span className="text-amber-400">W</span>X + α<span className="text-amber-400">I</span>
            )⁻¹ Xᵀ<span className="text-amber-400">W</span>y
            <br />
            <br />
            <span className="text-gray-500"># X: mask matrix [N × words]</span>
            <br />
            <span className="text-gray-500"># W: cosine kernel weights</span>
            <br />
            <span className="text-gray-500"># R² &gt; 0.95 in practice</span>
          </Card>
          <p className="text-gray-400 text-sm leading-relaxed mb-3">
            <span className="text-gray-200 font-medium">Why pure NumPy?</span> RunPod's
            container ships a stub <span className="font-mono">scikit-learn</span> package
            that fails at import time. A 6-line solve eliminates the dependency entirely.
          </p>
          <Card inner className="text-xs text-gray-500 leading-relaxed">
            <span className="text-emerald-400 font-mono">Result:</span> closed-form Ridge
            runs in &lt;1ms. The bottleneck is the 20 forward passes, not the regression.
          </Card>
        </Card>
      </div>
    </SlideShell>
  )
}

function Slide11TokenShapDeep() {
  return (
    <SlideShell>
      <SlideHeader kicker="Method · TokenSHAP" title="Shapley values from coalitions." />
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">The intuition</h3>
          <p className="text-gray-400 leading-relaxed mb-4 text-sm">
            From cooperative game theory: each word is a "player" in a game. The "payout"
            is whether the model still gets the right answer.
          </p>
          <p className="text-gray-400 leading-relaxed mb-5 text-sm">
            A word's Shapley value = its average marginal contribution across all possible
            coalitions of other words.
          </p>
          <Card inner className="font-mono text-sm text-gray-300 leading-loose">
            <span className="text-amber-400">φᵢ</span> = E<sub>π</sub>[ v(S<sub>π,i</sub> ∪
            {'{'}i{'}'}) − v(S<sub>π,i</sub>) ]
            <br />
            <br />
            <span className="text-gray-500"># π: random permutation</span>
            <br />
            <span className="text-gray-500"># S: coalition before word i</span>
            <br />
            <span className="text-gray-500"># v: answer-preservation score</span>
          </Card>
        </Card>
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">Why it beats raw masking</h3>
          <ul className="space-y-4 text-gray-400 text-sm leading-relaxed">
            <li className="flex gap-4">
              <span className="text-emerald-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ✓
              </span>
              <div>
                <p className="text-gray-200 font-medium mb-1">Mathematically grounded</p>
                <p>
                  Only attribution method satisfying efficiency, symmetry, and additivity
                  axioms. Allocations cannot be gamed by reframing the problem.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="text-emerald-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ✓
              </span>
              <div>
                <p className="text-gray-200 font-medium mb-1">Captures interactions</p>
                <p>
                  "5 apples" together matters more than each word alone — Shapley naturally
                  splits the joint contribution between the two words.
                </p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="text-emerald-400 font-mono text-lg leading-none mt-0.5 flex-shrink-0">
                ✓
              </span>
              <div>
                <p className="text-gray-200 font-medium mb-1">Tractable approximation</p>
                <p>
                  Exact Shapley is exponential. Monte Carlo sampling with 20 random
                  permutations gets within 5% — under one minute on a single A100.
                </p>
              </div>
            </li>
          </ul>
        </Card>
      </div>
    </SlideShell>
  )
}

function Slide12CounterfactualsDeep() {
  return (
    <SlideShell>
      <SlideHeader
        kicker="Method · Counterfactuals"
        title="The most direct test of causal influence."
      />
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">The procedure</h3>
          <ol className="space-y-4 text-gray-400 text-sm leading-relaxed">
            {[
              ['Extract', 'Regex-find every numeric span in the prompt.'],
              [
                'Rank',
                <>
                  Sort by <em>magnitude</em> — large numbers carry more semantic weight in
                  word problems.
                </>,
              ],
              [
                'Perturb',
                <>
                  Take the top <span className="text-white">3</span>; replace with{' '}
                  <span className="text-white font-mono">+1</span> (preferred), then{' '}
                  <span className="text-white font-mono">-1</span>, then{' '}
                  <span className="text-white font-mono">×2</span>.
                </>,
              ],
              [
                'Re-run',
                'Forward pass with the edited prompt; extract the new final answer.',
              ],
              [
                'Compare',
                <>
                  Did the answer flip? If yes → that token is causally bound. If no → the
                  model is robust to this change (or ignored it).
                </>,
              ],
            ].map(([title, body], i) => (
              <li key={i} className="flex gap-4">
                <span className="text-amber-400 font-mono font-bold w-5 flex-shrink-0">
                  {i + 1}.
                </span>
                <div>
                  <span className="text-gray-200 font-medium">{title}.</span>{' '}
                  <span>{body}</span>
                </div>
              </li>
            ))}
          </ol>
        </Card>
        <Card>
          <h3 className="text-white text-2xl font-semibold mb-5">Worked example</h3>
          <Card inner className="text-sm text-gray-300 leading-relaxed mb-5">
            <p className="mb-2 text-gray-400">
              <span className="text-gray-500 text-xs uppercase tracking-wider">Prompt</span>
            </p>
            <p className="mb-3">
              "John buys <span className="text-white font-mono bg-amber-500/10 px-1 rounded">5</span>{' '}
              apples and{' '}
              <span className="text-white font-mono bg-amber-500/10 px-1 rounded">8</span>{' '}
              oranges..."
            </p>
            <p className="mb-1 text-gray-400">
              <span className="text-gray-500 text-xs uppercase tracking-wider">Reference answer</span>
            </p>
            <p className="text-white font-mono">$12.40</p>
          </Card>
          <div className="space-y-2.5">
            <div
              className="rounded-lg p-3 flex items-center justify-between text-sm"
              style={{ backgroundColor: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)' }}
            >
              <span className="font-mono">
                <span className="text-red-300">8</span> →{' '}
                <span className="text-emerald-300">9</span>
              </span>
              <span className="text-emerald-300 font-mono">$13.20 ✓ flipped</span>
            </div>
            <div
              className="rounded-lg p-3 flex items-center justify-between text-sm"
              style={{ backgroundColor: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)' }}
            >
              <span className="font-mono">
                <span className="text-red-300">5</span> →{' '}
                <span className="text-emerald-300">6</span>
              </span>
              <span className="text-emerald-300 font-mono">$13.60 ✓ flipped</span>
            </div>
            <div className="rounded-lg p-3 flex items-center justify-between text-sm bg-white/[0.02] border border-white/5">
              <span className="font-mono text-gray-500">
                <span>a</span> → <span>an</span>
              </span>
              <span className="text-gray-500 font-mono">$12.40 (no change)</span>
            </div>
          </div>
          <p className="text-gray-500 text-xs italic mt-5">
            3 perturbations → bounded forward-pass cost per model. Predictable latency.
          </p>
        </Card>
      </div>
    </SlideShell>
  )
}

function Slide13Engineering() {
  const items = [
    {
      title: 'Cloudflare 524 wall',
      desc:
        "RunPod's public proxy sits behind Cloudflare with a hard 100s timeout. Long generations got killed mid-stream; the model returned data but the user saw an error.",
      fix: 'capped max_new_tokens to 48',
    },
    {
      title: 'Single-GPU serialization',
      desc:
        'All four 7B models share one A100. Concurrent inference triggers OOM. A global _GEN_LOCK serializes calls; the frontend triggers post-hoc per active tab only.',
      fix: 'lazy per-tab triggers',
    },
    {
      title: 'Stub-package sklearn',
      desc:
        "scikit-learn imports failed on RunPod despite pip list showing it installed — the container ships a stub package. We rewrote LIME's regression in 6 lines of NumPy.",
      fix: 'closed-form NumPy ridge',
    },
    {
      title: 'Prompt-format leakage',
      desc:
        "Naive whitespace-splitting put '###', 'Problem:', and 'Solution:' tokens into the LIME word list. R² collapsed to 0.000 because formatting tokens always survive masks.",
      fix: 'unwrap before split, re-wrap before generate',
    },
    {
      title: 'Cold-start latency',
      desc:
        'First call to each model loads weights from disk — 2 to 5 minutes per model. The frontend would time out before models warmed up. We pre-warm via a polling /health script.',
      fix: 'pre-warm script polls /health',
    },
    {
      title: 'Lazy post-hoc',
      desc:
        'Running 60 LIME forward passes × 4 models on every prompt would take 30+ minutes. Post-hoc fires only when the user opens that tab, sequentially within each model.',
      fix: 'per-tab triggers, sequential within model',
    },
  ]
  return (
    <SlideShell>
      <SlideHeader
        kicker="Engineering Wall"
        title="The problems that aren't in the paper."
      />
      <div className="grid grid-cols-3 gap-6">
        {items.map((it, i) => (
          <Card key={it.title} className="flex flex-col min-h-[300px]">
            <p className="text-amber-400/70 text-3xl font-mono font-bold mb-4 leading-none">
              {String(i + 1).padStart(2, '0')}
            </p>
            <h3 className="text-white text-lg font-semibold mb-3 leading-tight">
              {it.title}
            </h3>
            <p className="text-gray-400 text-sm leading-relaxed flex-1">{it.desc}</p>
            <div
              className="mt-5 pt-4"
              style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
            >
              <p className="text-emerald-400/70 text-[9px] font-mono uppercase tracking-[0.2em] mb-1">
                Resolution
              </p>
              <p className="text-emerald-400 text-xs font-mono">{it.fix}</p>
            </div>
          </Card>
        ))}
      </div>
    </SlideShell>
  )
}

function Slide14Numbers() {
  const stats = [
    { label: 'Models served', value: '4', sub: 'all 7B params, fp16' },
    { label: 'VRAM concurrent', value: '56 GB', sub: 'all 4 in memory' },
    { label: 'Visualizations', value: '8', sub: '5 mech + 3 post-hoc' },
    { label: 'Forward passes / prompt', value: '~80', sub: 'across all methods' },
    { label: 'LIME R²', value: '0.97', sub: 'on bench math problems', accent: '#34d399' },
    { label: 'Backend endpoints', value: '11', sub: 'proxy + post-hoc routes' },
    { label: 'Frontend bundle', value: '806 KB', sub: 'gzipped: 242 KB' },
    { label: 'Time per prompt', value: '~30 s', sub: 'all 4 models, all views' },
    { label: 'Counterfactual edits', value: '3', sub: 'top-magnitude per model' },
    { label: 'LIME samples', value: '20', sub: 'per model, per prompt' },
    { label: 'Transformer layers probed', value: '32', sub: 'attention rollout depth' },
    { label: 'Lines of code', value: '4.2K', sub: 'TS + Python combined' },
  ]
  return (
    <SlideShell>
      <SlideHeader kicker="By the numbers" title="What it took." />
      <div className="grid grid-cols-4 gap-5">
        {stats.map((s) => (
          <Card key={s.label} className="flex flex-col justify-center min-h-[140px]">
            <Stat label={s.label} value={s.value} sub={s.sub} accent={s.accent} />
          </Card>
        ))}
      </div>
      <p className="text-center text-gray-500 text-sm italic mt-10">
        One semester. One A100. One pipeline that actually works end-to-end.
      </p>
    </SlideShell>
  )
}

function Slide15Demo() {
  return (
    <SlideShell centered>
      <p className="text-amber-400/90 text-xs font-mono tracking-[0.4em] uppercase mb-10">
        Live Demo
      </p>
      <h2 className="text-white text-8xl font-bold tracking-tight mb-6 leading-none">
        Let's break it open.
      </h2>
      <p className="text-gray-400 text-xl max-w-2xl leading-relaxed mb-16">
        One prompt. Four models. Eight different windows into the same forward pass.
      </p>
      <Card className="max-w-2xl w-full">
        <p className="text-gray-500 text-[10px] font-mono uppercase tracking-[0.2em] mb-3">
          Try this prompt
        </p>
        <p className="text-white font-mono text-base leading-relaxed">
          "There are chickens and cows in a farm. There are 30 heads and 84 legs. How
          many chickens and how many cows are there?"
        </p>
      </Card>
      <p className="text-gray-600 text-xs mt-12 font-mono">
        ← previous slide · press → for next ·{' '}
        <span className="text-amber-400/80">use the button below to switch to demo</span>
      </p>
    </SlideShell>
  )
}

function Slide16Closing() {
  return (
    <SlideShell centered>
      <h2 className="text-white text-[8rem] font-bold tracking-tight mb-6 leading-none">
        Prism
      </h2>
      <p className="text-gray-300 text-2xl font-light tracking-wide mb-12 max-w-3xl">
        Making LLM reasoning legible — one forward pass at a time.
      </p>

      <div className="grid grid-cols-3 gap-5 max-w-5xl w-full mb-14">
        {[
          {
            label: 'Future Work',
            text: 'Sparse autoencoders for feature decomposition.',
          },
          { label: 'Future Work', text: 'Cross-model circuit alignment.' },
          {
            label: 'Future Work',
            text: 'Real-time activation patching in the browser.',
          },
        ].map((f, i) => (
          <Card key={i}>
            <p className="text-amber-400/80 text-[10px] font-mono uppercase tracking-[0.2em] mb-2">
              {f.label}
            </p>
            <p className="text-white text-sm leading-relaxed">{f.text}</p>
          </Card>
        ))}
      </div>

      <Link
        to="/"
        className="group inline-flex items-center gap-3 px-8 py-4 rounded-xl text-white text-base font-semibold transition-all hover:scale-[1.02]"
        style={{
          backgroundColor: '#fbbf24',
          color: '#0a0a0a',
          boxShadow: '0 0 40px rgba(251,191,36,0.3)',
        }}
      >
        <Play className="w-4 h-4 fill-current" />
        See it in action
        <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
      </Link>

      <div className="mt-20 text-center">
        <p className="text-gray-600 text-[10px] uppercase tracking-[0.2em] mb-3">Team</p>
        <div className="flex items-center gap-3 text-gray-300 text-sm flex-wrap justify-center">
          <span>Pranav Reddy Gaddam</span>
          <span className="text-gray-700">·</span>
          <span>Shweta Shinde</span>
          <span className="text-gray-700">·</span>
          <span>Pulkit Srivastava</span>
          <span className="text-gray-700">·</span>
          <span>Gayatri Patil</span>
        </div>
        <p className="text-gray-600 text-xs mt-4 font-mono">
          pranavreddy.gaddam@sjsu.edu · DATA 298B · Spring 2026
        </p>
        <p className="text-gray-500 text-base mt-8 italic">Thank you.</p>
      </div>
    </SlideShell>
  )
}

// ── Slide registry ───────────────────────────────────────────────────────────

const SLIDES = [
  Slide01Title,
  Slide02Problem,
  Slide03WhyNow,
  Slide04Architecture,
  Slide05ArchDiagram,
  Slide06Models,
  Slide07MechOverview,
  Slide08AttributionGraph,
  Slide09PostHocOverview,
  Slide10LimeDeep,
  Slide11TokenShapDeep,
  Slide12CounterfactualsDeep,
  Slide13Engineering,
  Slide14Numbers,
  Slide15Demo,
  Slide16Closing,
]

// ── Container ────────────────────────────────────────────────────────────────

export default function Presentation() {
  const [index, setIndex] = useState(0)

  const next = useCallback(() => setIndex((i) => Math.min(SLIDES.length - 1, i + 1)), [])
  const prev = useCallback(() => setIndex((i) => Math.max(0, i - 1)), [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
        e.preventDefault()
        next()
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault()
        prev()
      } else if (e.key === 'Home') {
        setIndex(0)
      } else if (e.key === 'End') {
        setIndex(SLIDES.length - 1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [next, prev])

  const Slide = SLIDES[index]

  return (
    <div className="h-screen w-full relative overflow-hidden">
      <SiteBackground />

      <div className="relative z-10 h-screen flex flex-col">
        <div className="flex-1 flex items-stretch overflow-hidden">
          <AnimatePresence mode="wait">
            <Slide key={index} />
          </AnimatePresence>
        </div>

        {/* Bottom nav bar */}
        <div
          className="px-8 py-4 flex items-center justify-between"
          style={{ borderTop: BORDER, backgroundColor: 'rgba(0,0,0,0.5)' }}
        >
          <div className="flex items-center gap-3">
            <span className="text-white text-sm font-bold tracking-wider">PRISM</span>
            <span className="text-gray-600 text-xs">·</span>
            <span className="text-gray-500 text-xs">Inside the Black Box</span>
          </div>

          <div className="flex items-center gap-2">
            {SLIDES.map((_, i) => (
              <button
                key={i}
                onClick={() => setIndex(i)}
                className="h-1.5 rounded-full transition-all"
                style={{
                  width: i === index ? 24 : 8,
                  backgroundColor:
                    i === index
                      ? '#fbbf24'
                      : i < index
                        ? 'rgba(255,255,255,0.4)'
                        : 'rgba(255,255,255,0.15)',
                }}
                aria-label={`Slide ${i + 1}`}
              />
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={prev}
              disabled={index === 0}
              className="p-2 rounded-lg disabled:opacity-30 hover:bg-white/5 transition-colors"
              style={{ border: BORDER }}
            >
              <ChevronLeft className="w-4 h-4 text-white" />
            </button>
            <span className="text-gray-400 text-sm font-mono px-3 min-w-[60px] text-center">
              {String(index + 1).padStart(2, '0')} / {String(SLIDES.length).padStart(2, '0')}
            </span>
            <button
              onClick={next}
              disabled={index === SLIDES.length - 1}
              className="p-2 rounded-lg disabled:opacity-30 hover:bg-white/5 transition-colors"
              style={{ border: BORDER }}
            >
              <ChevronRight className="w-4 h-4 text-white" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
