// Post-hoc analysis result types — mirror the pod /posthoc/* responses.

export interface LimeResult {
  method: 'lime'
  words: string[]
  attributions: number[]
  reference_answer: string | null
  reference_response: string
  n_samples: number
  r_squared: number
}

export interface TokenShapResult {
  method: 'tokenshap'
  words: string[]
  attributions: number[]
  reference_answer: string | null
  reference_response: string
  n_samples: number
}

export interface CounterfactualEdit {
  original_token: string
  replacement: string
  position: number
  new_response: string
  new_answer: string | null
  flipped: boolean
}

export interface CounterfactualResult {
  method: 'counterfactual'
  reference_answer: string | null
  reference_response: string
  edits: CounterfactualEdit[]
  n_edits: number
  n_flipped: number
  note?: string
}

export type PostHocStatus = 'idle' | 'loading' | 'ready' | 'error'

export interface PostHocSlot {
  status: PostHocStatus
  lime: LimeResult | null
  tokenshap: TokenShapResult | null
  counterfactual: CounterfactualResult | null
  error?: string
  model_id?: string
  prompt?: string
}
