"""
AI interpretation endpoint — streams a GPT-4o explanation for a given card's data.
"""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Any

router = APIRouter()

CARD_SYSTEM = """You are an expert AI researcher helping users understand mechanistic interpretability visualizations of transformer language models.
Be concise (2-4 sentences), precise, and use plain English. Focus on what the data MEANS mechanistically — not just what it shows numerically.
When interpreting math reasoning models, consider whether the model attended to the right quantities, whether its confidence aligns with reasoning difficulty, and what the activations suggest about where computation is happening."""

CARD_PROMPTS = {
    "response": lambda d: f"""Model '{d.get('model_id')}' was given this math problem: "{d.get('prompt', 'unknown')}"
It generated a {d.get('token_count')}-token response. Final answer extracted: {d.get('final_answer') or 'none found'}.
Full response: "{d.get('response_preview', '')}"
In 2-3 sentences, interpret what this response reveals — did the model reason step-by-step or jump to an answer, is the extracted answer correct given the problem, and are there any signs of uncertainty, repetition, or hallucination?""",

    "confidence": lambda d: f"""Token-level softmax confidence scores for model '{d.get('model_id')}' on a math problem.
Each score is the probability the model assigned to the token it actually generated (higher = more certain).
- Total tokens generated: {d.get('token_count', 0)}
- Average confidence: {d.get('avg_confidence', 0):.1%}
- Range: {d.get('min_confidence', 0):.1%} – {d.get('max_confidence', 0):.1%}
- Least confident tokens (where the model hesitated): {d.get('lowest_confidence_tokens', [])}
- Most confident tokens (where the model was certain): {d.get('highest_confidence_tokens', [])}
In 2-3 sentences, interpret what this confidence pattern tells us — where did the model hesitate, do the low-confidence tokens correspond to reasoning steps or arithmetic operations, and what does the overall confidence level suggest about the model's certainty on this problem?""",

    "hidden": lambda d: f"""Hidden state L2 norms across {d.get('layer_count', 0)} transformer layers for model '{d.get('model_id')}'.
The L2 norm of a hidden state vector measures the magnitude of the representation at that layer — higher norms indicate layers where the model is doing more transformative computation. A sudden spike (drift) between adjacent layers often signals where a key reasoning step or concept resolution occurs.
- Average norm: {d.get('avg_norm', 0):.1f}
- Peak norm: {d.get('peak_norm', 0):.1f} at layer {d.get('peak_layer', '?')}
- Largest layer-to-layer drift: {d.get('max_drift', 0):.1f} at layer {d.get('max_drift_layer', '?')}
- Norm trend (sampled across layers): {d.get('norm_trend', [])}
In 2-3 sentences, interpret what this activation pattern reveals — which layers are doing the heaviest computation, what does the drift spike location suggest about where the model resolves the math, and does the norm profile suggest early pattern matching or late-stage reasoning?""",

    "attribution": lambda d: f"""Gradient-based input attribution scores for model '{d.get('model_id')}' on this math problem: "{d.get('prompt', 'unknown')}"
Attribution scores measure how much each input token influenced the model's output — computed via gradients of the output with respect to input embeddings. Higher score = the model's answer would change more if this token were different.
- Most influential tokens (token → score): {d.get('top_tokens', [])}
- Least influential tokens: {d.get('bottom_tokens', [])}
In 2-3 sentences, interpret what these scores reveal — did the model correctly focus on the numerically important words and quantities in the problem, or did it attribute high importance to irrelevant tokens like punctuation or filler words? Does this suggest the model understood the problem structure?""",

    "attention": lambda d: f"""Attention flow graph for model '{d.get('model_id')}' on this math problem: "{d.get('prompt', 'unknown')}"
Each edge represents attention weight from one token to another — thicker edges mean stronger attention. Attention heads in different layers capture different linguistic and mathematical relationships.
- Input tokens: {d.get('input_tokens', [])}
- Layers visualized: {d.get('layer_count', 0)}
- Strongest attention connections: {d.get('top_edges', [])}
In 2-3 sentences, explain what the attention flow reveals — which tokens are receiving the most attention, does the model appear to be linking quantities and operators correctly, and what does this suggest about how it parsed the mathematical structure of the problem?""",

    "lime": lambda d: f"""LIME (Local Interpretable Model-agnostic Explanations) results for model '{d.get('model_id')}' on this math problem.
LIME works by randomly masking words and measuring how the model's answer changes — words that cause the answer to change when removed get high attribution scores.
- Reference answer: {d.get('reference_answer', '?')}
- R² (surrogate fit quality): {d.get('r_squared', 0):.3f}
- Samples used: {d.get('n_samples', 0)}
- Top influential words (word, score): {d.get('top_words', [])}
- Least influential words: {d.get('bottom_words', [])}
In 2-3 sentences, interpret what these attributions reveal — did the model focus on the numerically important words in the problem, does the R² suggest the surrogate model was a good fit, and what does this tell us about the model's reasoning?""",

    "tokenshap": lambda d: f"""TokenSHAP (Shapley value attribution) results for model '{d.get('model_id')}' on this math problem.
Shapley values measure each word's average marginal contribution to the model's answer across all possible word subsets.
- Reference answer: {d.get('reference_answer', '?')}
- Coalitions evaluated: {d.get('n_samples', 0)}
- Highest Shapley value words (word, score): {d.get('top_words', [])}
- Lowest Shapley value words: {d.get('bottom_words', [])}
In 2-3 sentences, interpret what these Shapley values reveal — which words were truly necessary for the model to reach its answer, and how does this compare to what a human would consider the key information in the problem?""",

    "counterfactual": lambda d: f"""Counterfactual analysis for model '{d.get('model_id')}' on this math problem: "{d.get('prompt', 'unknown')}"
A single token was changed in the input — '{d.get('original_token')}' was replaced with '{d.get('replacement')}' — and the model's answer flipped from '{d.get('reference_answer')}' to '{d.get('new_answer')}'.
Original response: "{d.get('original_response', '')}"
Counterfactual response: "{d.get('new_response', '')}"
In 2-3 sentences, explain what this flip reveals about the model's reasoning: was the model's original answer fragile or robust, does the changed token represent a genuinely important quantity in the problem, and what does this tell us about how sensitive the model is to this type of input perturbation?""",
}


class InterpretRequest(BaseModel):
    card: str
    model_id: str
    data: dict[str, Any]


@router.post("/interpret")
async def interpret(request: InterpretRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    prompt_fn = CARD_PROMPTS.get(request.card)
    if not prompt_fn:
        raise HTTPException(status_code=400, detail=f"Unknown card type: {request.card!r}")

    payload = {"model_id": request.model_id, **request.data}
    user_prompt = prompt_fn(payload)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    async def stream():
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": CARD_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.4,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[Error: {e}]"

    return StreamingResponse(stream(), media_type="text/plain")
