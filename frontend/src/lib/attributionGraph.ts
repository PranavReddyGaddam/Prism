import type { AttributionGraph, AttributionNode, AttributionEdge } from '@/types/attribution'

export interface PositionPrediction {
  position: number
  token: string
  top: { token: string; probability: number; rank: number }[]
  bottom: { token: string; probability: number; rank: number }[]
}

export interface LayerAttention {
  layer: number
  weights: { token: string; weight: number }[]
}

export const DEFAULT_MAX_LAYERS = 12
export const DEFAULT_MAX_OUTPUTS = 10

export function buildAttributionGraph(
  prompt: string,
  attrTokens: { token: string; score: number }[],
  positionPredictions: PositionPrediction[],
  layerAttention: LayerAttention[],
  maxLayers: number = DEFAULT_MAX_LAYERS,
  maxOutputs: number = DEFAULT_MAX_OUTPUTS,
): AttributionGraph {
  const nodes: AttributionNode[] = []
  const edges: AttributionEdge[] = []
  const promptTokens = prompt.split(/\s+/).filter((w) => w.length > 0)

  // Input nodes: top 8 tokens by attention-rollout score
  const topInputs = [...attrTokens].sort((a, b) => b.score - a.score).slice(0, 8)
  topInputs.forEach((t, i) => {
    nodes.push({
      id: `input-${i}`,
      type: 'input',
      label: t.token.trim() || `t${i}`,
      layer: 0,
      position: i,
      activation: t.score,
    })
  })

  // Intermediate nodes: evenly-spaced sample of attention layers
  const numLayers = layerAttention.length
  const targetLayers = Math.min(maxLayers, Math.max(1, numLayers))
  const layerStep = Math.max(1, (numLayers - 1) / Math.max(1, targetLayers - 1))
  const sampledLayers = Array.from({ length: targetLayers }, (_, k) =>
    Math.min(Math.round(k * layerStep), numLayers - 1),
  ).filter((v, i, arr) => arr.indexOf(v) === i)

  sampledLayers.forEach((layerIdx, f) => {
    const la = layerAttention[layerIdx]
    const topWeights = [...(la?.weights ?? [])].sort((a, b) => b.weight - a.weight).slice(0, 5)
    const activation = topWeights.length ? topWeights[0].weight : 0.5
    nodes.push({
      id: `feat-${f}`,
      type: 'intermediate',
      label: `Layer ${la?.layer ?? layerIdx}`,
      layer: 1,
      activation,
      inputFeatures: topWeights.map((w) => ({ feature: w.token, weight: w.weight })),
    })
  })

  // Output nodes: evenly-spaced positions from response half of positionPredictions
  const halfIdx = Math.floor(positionPredictions.length / 2)
  const responsePreds = positionPredictions.slice(halfIdx)
  const targetOutputs = Math.min(maxOutputs, Math.max(1, responsePreds.length))
  const outputStep = Math.max(1, (responsePreds.length - 1) / Math.max(1, targetOutputs - 1))
  const outputPreds = Array.from({ length: targetOutputs }, (_, k) =>
    responsePreds[Math.min(Math.round(k * outputStep), responsePreds.length - 1)],
  ).filter(Boolean)

  outputPreds.forEach((pred, i) => {
    const topToken = pred.top[0]
    nodes.push({
      id: `output-${i}`,
      type: 'output',
      label: topToken?.token?.trim() || pred.token.trim() || `o${i}`,
      layer: 2,
      position: pred.position,
      probability: topToken?.probability ?? 0,
      tokenPredictions: [
        ...pred.top.map((p) => ({ token: p.token, logit: 0, probability: p.probability, rank: p.rank })),
        ...pred.bottom.map((p) => ({ token: p.token, logit: 0, probability: p.probability, rank: p.rank + 100 })),
      ],
    })
  })

  // Edges: inputs → intermediate, weighted by attention at that layer
  sampledLayers.forEach((layerIdx, f) => {
    const la = layerAttention[layerIdx]
    topInputs.forEach((inp, i) => {
      const layerWeight = la?.weights.find((w) => w.token === inp.token)?.weight ?? 0
      const w = (inp.score + layerWeight) / 2
      if (w > 0.05) {
        edges.push({ source: `input-${i}`, target: `feat-${f}`, weight: w })
      }
    })
  })

  // Edges: intermediate → outputs, weighted by layer activation
  outputPreds.forEach((_, o) => {
    sampledLayers.forEach((layerIdx, f) => {
      const la = layerAttention[layerIdx]
      const activation = la?.weights.length ? la.weights[0].weight : 0.3
      edges.push({ source: `feat-${f}`, target: `output-${o}`, weight: activation })
    })
  })

  const explainedBehavior = topInputs.reduce((s, t) => s + t.score, 0) / Math.max(attrTokens.length, 1)

  return {
    nodes,
    edges,
    prompt,
    promptTokens,
    totalNodes: nodes.length,
    totalEdges: edges.length,
    explainedBehavior: Math.min(explainedBehavior, 1),
  }
}
