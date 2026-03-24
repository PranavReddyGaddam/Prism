// Attribution Graph Types based on Anthropic's paper

export type NodeType = 'output' | 'intermediate' | 'input' | 'error';

export interface FeatureWeight {
  feature: string;
  weight: number;
  description?: string;
}

export interface TokenPrediction {
  token: string;
  logit: number;
  probability: number;
  rank: number;
}

export interface AttributionNode {
  id: string;
  type: NodeType;
  label: string;
  layer?: number;
  position?: number; // context position
  activation?: number; // a_s for source features
  probability?: number; // for output nodes
  featureId?: number; // for intermediate nodes
  
  // Detailed feature information
  inputFeatures?: FeatureWeight[]; // Features feeding into this node
  outputFeatures?: FeatureWeight[]; // Features this node feeds into
  topActivations?: string[]; // Top activating contexts
  tokenPredictions?: TokenPrediction[]; // Full token prediction list
  subgraphInterval?: string; // e.g., "Subgraph Interval 0"
}

export interface AttributionEdge {
  source: string; // node id
  target: string; // node id
  weight: number; // A_s→t = a_s * w_s→t
  contribution?: number; // normalized contribution to target
}

export interface AttributionGraph {
  nodes: AttributionNode[];
  edges: AttributionEdge[];
  prompt: string;
  promptTokens: string[];
  totalNodes: number; // before pruning
  totalEdges: number; // before pruning
  pruningThreshold?: number;
  explainedBehavior?: number; // percentage of behavior explained after pruning
  subgraphs?: AttributionSubgraph[]; // Available subgraphs for detailed inspection
}

export interface AttributionSubgraph {
  id: string;
  parentNodeId: string;
  nodes: AttributionNode[];
  edges: AttributionEdge[];
  description: string;
  interval?: string;
}

export interface AttributionPath {
  nodes: AttributionNode[];
  edges: AttributionEdge[];
  totalWeight: number;
  description: string;
}

export interface GraphPruningConfig {
  topK: number; // number of top output nodes to keep
  minEdgeWeight: number; // minimum edge weight threshold
  minNodeInfluence: number; // minimum cumulative influence threshold
  maxNodes?: number; // maximum nodes to display
  preserveProbabilityMass: number; // e.g., 0.95 for 95%
}
