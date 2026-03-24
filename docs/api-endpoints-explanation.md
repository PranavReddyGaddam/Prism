# Prism API Endpoints - Demo Presentation Guide

## Overview: Mechanistic Interpretability vs Chain-of-Thought

**Key Message:** Unlike Chain-of-Thought (CoT) which shows what the model *says* it's thinking, our explainability endpoints reveal what the model is *actually* computing internally. These are direct measurements of neural activations, attention flows, and probability distributions—not narratives the model constructs.

---

## 1. `/generate/` - Text Generation Endpoint

### What It Does
Generates a response to the user's prompt using either Phi-3 or DeepSeek models.

### Demo Story
"This is where the model produces its answer. For DeepSeek, we extract the `<think>` tags to show its reasoning process. But here's the key: **this CoT reasoning is a learned behavior**—the model generates text that sounds like reasoning because it was trained to do so. It could be confabulating. That's why we need the explainability endpoints below to see what's **really** happening inside."

### Key Points
- Returns: `response`, `thinking` (for DeepSeek), `final_answer`, `token_count`
- This is the baseline—what the model outputs
- CoT is valuable but not ground truth

---

## 2. `/explain/confidence` - Token Confidence Analysis

### What It Does
Computes the probability distribution for each generated token, showing how confident the model was when producing each word.

### The Real Story
"This shows the model's **actual certainty** during generation. Each token has a probability score from the softmax layer—this is direct measurement, not explanation. When you see green tokens (high confidence), the model had strong internal agreement across all possible next words. Red tokens (low confidence) show where the model was genuinely uncertain—not just saying 'I'm uncertain,' but **mathematically uncertain** in its probability distribution."

### Demo Script
1. Point to green tokens: "The model was 95%+ confident here—this is solid ground truth."
2. Point to yellow/red tokens: "Here the model was struggling—only 40% confident. This is where errors are likely."
3. **Contrast with CoT**: "CoT might say 'I'm confident' but show low probability scores here—that's confabulation."

### Key Technical Point
- Direct measurement from `softmax(logits)`
- Cannot be faked or confabulated
- Shows real-time decision certainty

---

## 3. `/explain/attention` - Attention Pattern Heatmap

### What It Does
Visualizes which input tokens the model attends to when processing each word. Shows the actual information flow through the transformer architecture.

### The Real Story
"This is **not** what the model says it's looking at—this is a direct measurement of the attention mechanism. When you see strong attention from 'son' to 'father' and '4', that's the model **mechanistically retrieving relevant context** from earlier in the prompt. This is the actual computation happening in the transformer layers."

### Demo Script
1. **Select Layer 0-5**: "Early layers focus on syntax—see how it attends to nearby words?"
2. **Select Layer 15-20**: "Middle layers show semantic relationships—'father' attends to 'son' and '4 times'."
3. **Select Layer 25-31**: "Deep layers show abstract reasoning—the model is connecting the entire problem structure."
4. **Use the sliders**: "We can explore any of the 32 layers and 32 attention heads to see different aspects of processing."

### Key Technical Point
- Direct architectural mechanism (scaled dot-product attention)
- Layer 0 = syntax, Layer 15 = semantics, Layer 30 = reasoning
- Different heads specialize in different patterns
- This is **computation**, not narrative

---

## 4. `/explain/logit-lens` - Layer Activations (Word Evolution)

### What It Does
Probes the model's hidden states at each of the 33 transformer layers to see what token it would predict at each stage. Shows how the answer "crystallizes" layer by layer.

### The Real Story
"This reveals the model's **internal reasoning trajectory**. We're literally decoding the model's hidden representations at each layer to see what it's 'thinking' at that computational step. Early layers might predict random tokens, but watch how the correct answer emerges in deeper layers. This isn't the model explaining itself—it's us **directly probing the model's brain** at each computation step."

### Demo Script
1. **Select Word 1 (e.g., 'Let')**: 
   - "Layer 0-10: The model is uncertain—predicting random tokens with <1% confidence."
   - "Layer 15-20: It starts converging—'Let' appears with 30-50% confidence."
   - "Layer 25-33: Full confidence—99-100% certain the answer starts with 'Let'."

2. **Select Word 6 (e.g., 'calculate')**:
   - "Notice how this word follows a different trajectory—it becomes confident faster because it's building on the previous context."

3. **Select Word 11 (e.g., 'answer')**:
   - "By the time we reach the conclusion, the model is highly confident from early layers—it knows where it's going."

### Key Technical Point
- We apply the language model head to intermediate hidden states
- Shows computational evolution, not post-hoc explanation
- Each word has its own confidence trajectory
- CoT shows the final narrative; logit lens shows the **actual computational process**

---

## 5. `/explain/attribution` - Gradient Attribution

### What It Does
Uses calculus (backpropagation) to measure how much each input token influenced the final prediction. Shows causal importance.

### The Real Story
"Gradient attribution uses **mathematics to trace causality**. We compute gradients—the derivative of the output with respect to each input token. High gradient scores mean that token was **mathematically critical** to the output. This isn't the model's opinion—it's a **causal measurement** using calculus. If 'father' and '4' have high attribution scores, we have mathematical proof they were essential to generating the answer."

### Demo Script
1. Point to high-scoring tokens: "These tokens have the strongest causal influence—change these and the answer changes dramatically."
2. Point to low-scoring tokens: "These are filler words—they don't affect the computation much."
3. **Contrast with CoT**: "CoT might mention certain words, but attribution shows which words **actually mattered** in the neural computation."

### Key Technical Point
- Gradient norm: `||∂output/∂input||`
- Mathematical proof of influence
- Cannot be confabulated—it's calculus
- Shows **causal importance**, not narrative importance

---

## 6. `/explain/hidden-states` - Hidden State Norms

### What It Does
Measures the L2 norm (magnitude) of neural activations at each layer. Shows computational effort.

### The Real Story
"This shows the **magnitude of neural activations** across layers. High norms mean the model is processing complex information; low norms suggest simpler computation. This reveals the model's **computational effort** at each layer—something CoT never shows. You can see which layers are 'working hardest' on the problem."

### Demo Script
1. Point to peaks: "Layers 15-20 show high activation—this is where the model is doing heavy semantic processing."
2. Point to valleys: "Early layers have lower norms—they're just processing syntax."
3. **Key insight**: "The model doesn't just 'think' uniformly—different layers do different amounts of work."

### Key Technical Point
- L2 norm: `||hidden_state||₂`
- Direct neural measurement
- Shows computational intensity
- Reveals which layers are most active

---

## The Overarching Demo Narrative

### Opening (30 seconds)
"Today we're showing **Prism**—a tool that reveals what LLMs are actually computing, not just what they say they're thinking. Chain-of-Thought is valuable, but it's fundamentally a learned behavior. What we're showing here is **mechanistic interpretability**: direct measurements of neural computations."

### For Each Endpoint (1-2 minutes each)
1. **Show the visualization**
2. **Explain what it measures** (probability, attention, gradients, etc.)
3. **Contrast with CoT**: "CoT is narrative; this is measurement"
4. **Interactive demo**: Use sliders/dropdowns to explore different layers/words/heads

### Closing (30 seconds)
"These aren't stories the model constructs—they're **ground truth measurements** of what's happening inside the neural network. CoT can hallucinate; these metrics cannot. We're not asking the model 'why did you do that?'—we're **directly observing the computational mechanisms** that produced the answer. This is the difference between asking someone to explain their thought process versus using an fMRI to see which brain regions activated. One is a story; the other is measurement."

---

## Key Talking Points for Each Teammate

### Token Confidence
- "Direct probability measurements from softmax"
- "Shows real-time certainty, not post-hoc explanation"
- "Green = confident, red = uncertain—mathematically"

### Attention Patterns
- "Information flow architecture—what the model actually looks at"
- "Early layers = syntax, deep layers = reasoning"
- "32 layers × 32 heads = 1,024 different attention patterns to explore"

### Layer Activations (Logit Lens)
- "Computational trajectory through 33 layers"
- "Watch the answer crystallize layer by layer"
- "Every 5th word—see how different words evolve differently"

### Gradient Attribution
- "Mathematical causality using calculus"
- "Gradients prove which tokens mattered"
- "Cannot be faked—it's derivatives"

### Hidden States
- "Neural activation magnitude"
- "Shows computational effort per layer"
- "Reveals which layers work hardest"

---

## Technical Backup (If Asked)

### "How is this different from CoT?"
"CoT is generated text—a learned behavior. Our metrics are direct measurements: probabilities from softmax, attention weights from the transformer architecture, gradients from backpropagation, and hidden state norms from layer activations. These are mathematical operations, not narratives."

### "Can the model fake these metrics?"
"No. These are computed by us, not generated by the model. The model can't fake its attention weights or probability distributions—those are architectural outputs we measure directly."

### "Why does this matter?"
"Because CoT can confabulate. A model might say 'I'm confident' while having low probability scores. Or it might claim to focus on certain words while its attention is elsewhere. Our metrics reveal the truth—what's actually happening in the neural network."

---

## Demo Flow Recommendation

1. **Start with generation** (30s): Show the model's answer
2. **Token confidence** (1m): "Here's how confident it really was"
3. **Attention patterns** (2m): "Here's what it was actually looking at" (use sliders!)
4. **Layer activations** (2m): "Here's how the answer emerged layer by layer" (use dropdown!)
5. **Gradient attribution** (1m): "Here's what mathematically mattered"
6. **Hidden states** (1m): "Here's the computational effort"
7. **Closing** (30s): "Measurement vs narrative"

**Total: ~8 minutes**

---

## Questions to Anticipate

**Q: "Why not just use CoT?"**
A: "CoT is valuable but can confabulate. Our metrics are ground truth measurements that can't be faked."

**Q: "Is this real-time?"**
A: "Yes—we compute these metrics during or immediately after generation. The attention patterns, probabilities, and gradients are from the actual forward/backward pass."

**Q: "Can this help debug model failures?"**
A: "Absolutely. Low confidence scores, scattered attention, or weak gradients can reveal where and why a model failed—not just that it failed."

**Q: "What's the computational cost?"**
A: "Minimal. Attention and probabilities are already computed during generation. Gradients require one backward pass. Hidden states are free. Total overhead: ~2-3x generation time."

---

## Final Notes

- **Be confident**: You're showing real measurements, not speculation
- **Use the interactive features**: Sliders and dropdowns make it engaging
- **Contrast with CoT repeatedly**: That's the key differentiator
- **Focus on 'measurement vs narrative'**: That's the core message
- **Have fun**: This is genuinely cool technology!

Good luck with the demo! 🚀
