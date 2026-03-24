import type { AttributionGraph, AttributionNode, AttributionEdge } from '@/types/attribution'

export function generateMockAttributionGraph(prompt: string, response: string): AttributionGraph {
  const promptTokens = prompt.split(' ').filter(w => w.length > 0).slice(0, 10)
  const responseTokens = response.split(' ').filter(w => w.length > 0).slice(0, 5)
  
  const nodes: AttributionNode[] = []
  const edges: AttributionEdge[] = []
  
  // Create input nodes (prompt token embeddings)
  promptTokens.forEach((token, i) => {
    nodes.push({
      id: `input-${i}`,
      type: 'input',
      label: token,
      layer: 0,
      position: i,
    })
  })
  
  // Create intermediate nodes (features at different layers)
  const numLayers = 3
  const featuresPerLayer = 4
  
  for (let layer = 1; layer <= numLayers; layer++) {
    for (let feat = 0; feat < featuresPerLayer; feat++) {
      const activation = Math.random() * 0.8 + 0.2 // 0.2 to 1.0
      
      // Generate input features (from previous layer)
      const inputFeatures = layer > 1 ? Array.from({ length: 3 }, (_, i) => ({
        feature: `L${layer - 1}F${i}`,
        weight: (Math.random() - 0.5) * 2,
        description: `Feature from layer ${layer - 1}`
      })) : promptTokens.slice(0, 3).map(token => ({
        feature: token,
        weight: (Math.random() - 0.5) * 2,
        description: 'Input token embedding'
      }))
      
      // Generate output features (to next layer)
      const outputFeatures = layer < numLayers ? Array.from({ length: 3 }, (_, i) => ({
        feature: `L${layer + 1}F${i}`,
        weight: (Math.random() - 0.5) * 2,
        description: `Feature to layer ${layer + 1}`
      })) : responseTokens.slice(0, 3).map(token => ({
        feature: token,
        weight: (Math.random() - 0.5) * 2,
        description: 'Output token prediction'
      }))
      
      // Generate top activations
      const topActivations = [
        `"${promptTokens[Math.floor(Math.random() * promptTokens.length)]}" context`,
        `Pattern: ${['arithmetic', 'semantic', 'syntactic'][Math.floor(Math.random() * 3)]}`,
        `Activation strength: ${(activation * 100).toFixed(1)}%`
      ]
      
      nodes.push({
        id: `feat-L${layer}-F${feat}`,
        type: 'intermediate',
        label: `L${layer}F${feat}`,
        layer,
        position: feat,
        activation,
        featureId: feat,
        inputFeatures,
        outputFeatures,
        topActivations,
        subgraphInterval: `Subgraph Interval ${feat}`
      })
    }
  }
  
  // Create output nodes (candidate tokens)
  responseTokens.forEach((token, i) => {
    const probability = Math.random() * 0.3 + (i === 0 ? 0.4 : 0.1) // First token has higher prob
    
    // Generate full token prediction list
    const allTokens = [...responseTokens, 'the', 'a', 'is', 'was', 'in', 'on', 'at', 'to', 'for', 'with']
    const tokenPredictions = allTokens.map((t, idx) => ({
      token: t,
      logit: 10 - idx * 0.5 + (Math.random() - 0.5),
      probability: idx === i ? probability : Math.random() * 0.1,
      rank: idx + 1
    })).sort((a, b) => b.probability - a.probability).map((p, idx) => ({ ...p, rank: idx + 1 }))
    
    nodes.push({
      id: `output-${i}`,
      type: 'output',
      label: token,
      layer: numLayers + 1,
      position: i,
      probability,
      tokenPredictions: tokenPredictions.slice(0, 20), // Top 20 predictions
    })
  })
  
  // Create error nodes (unexplained portions)
  for (let layer = 1; layer <= numLayers; layer++) {
    nodes.push({
      id: `error-L${layer}`,
      type: 'error',
      label: `Error L${layer}`,
      layer,
      position: featuresPerLayer,
    })
  }
  
  // Create edges from inputs to first layer features
  promptTokens.forEach((_, i) => {
    const inputId = `input-${i}`
    for (let feat = 0; feat < featuresPerLayer; feat++) {
      const targetId = `feat-L1-F${feat}`
      const weight = (Math.random() - 0.5) * 2 // -1 to 1
      if (Math.abs(weight) > 0.3) { // Only keep significant edges
        edges.push({
          source: inputId,
          target: targetId,
          weight,
          contribution: Math.abs(weight),
        })
      }
    }
  })
  
  // Create edges between intermediate layers
  for (let layer = 1; layer < numLayers; layer++) {
    for (let srcFeat = 0; srcFeat < featuresPerLayer; srcFeat++) {
      const sourceId = `feat-L${layer}-F${srcFeat}`
      const sourceNode = nodes.find(n => n.id === sourceId)
      const activation = sourceNode?.activation ?? 0.5
      
      for (let tgtFeat = 0; tgtFeat < featuresPerLayer; tgtFeat++) {
        const targetId = `feat-L${layer + 1}-F${tgtFeat}`
        const baseWeight = (Math.random() - 0.5) * 1.5
        const weight = activation * baseWeight // A_s→t = a_s * w_s→t
        
        if (Math.abs(weight) > 0.2) {
          edges.push({
            source: sourceId,
            target: targetId,
            weight,
            contribution: Math.abs(weight),
          })
        }
      }
    }
  }
  
  // Create edges from last layer features to outputs
  for (let feat = 0; feat < featuresPerLayer; feat++) {
    const sourceId = `feat-L${numLayers}-F${feat}`
    const sourceNode = nodes.find(n => n.id === sourceId)
    const activation = sourceNode?.activation ?? 0.5
    
    responseTokens.forEach((_, i) => {
      const targetId = `output-${i}`
      const baseWeight = (Math.random() - 0.3) * 2
      const weight = activation * baseWeight
      
      if (Math.abs(weight) > 0.3) {
        edges.push({
          source: sourceId,
          target: targetId,
          weight,
          contribution: Math.abs(weight),
        })
      }
    })
  }
  
  // Create edges from error nodes to features
  for (let layer = 1; layer <= numLayers; layer++) {
    const errorId = `error-L${layer}`
    for (let feat = 0; feat < featuresPerLayer; feat++) {
      const targetId = `feat-L${layer}-F${feat}`
      const weight = (Math.random() - 0.5) * 0.5 // Smaller weights for error
      
      if (Math.abs(weight) > 0.1) {
        edges.push({
          source: errorId,
          target: targetId,
          weight,
          contribution: Math.abs(weight),
        })
      }
    }
  }
  
  // Simulate pruning - we've already filtered edges, so these are "after pruning" stats
  const totalNodesBefore = nodes.length * 3 // Simulate 3x more nodes before pruning
  const totalEdgesBefore = edges.length * 5 // Simulate 5x more edges before pruning
  
  return {
    nodes,
    edges,
    prompt,
    promptTokens,
    totalNodes: totalNodesBefore,
    totalEdges: totalEdgesBefore,
    pruningThreshold: 0.2,
    explainedBehavior: 0.82, // 82% of behavior explained after pruning
  }
}
