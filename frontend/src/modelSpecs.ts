/** Four-model comparison tabs — labels match planned RunPod checkpoints. */
export const COMPARISON_MODEL_TABS = [
  { id: 'qwen-2.5-math-7b', label: 'Qwen 2.5 Math 7B' },
  { id: 'deepseek-math-7b', label: 'DeepSeek Math 7B' },
  { id: 'internlm2-math-7b', label: 'InternLM2 Math+ 7B' },
  { id: 'wizardmath-7b', label: 'WizardMath 7B' },
] as const

export type ComparisonModelId = (typeof COMPARISON_MODEL_TABS)[number]['id']
