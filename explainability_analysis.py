import os
import math
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "./deepseek-model"
QUESTION = "If a train travels 120km in 2 hours, what is its average speed?"
ATTN_LAYER = 0
ATTN_HEAD = 0
MAX_TOKENS = 128


def load_model():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map="cpu",
        output_attentions=True,
        output_hidden_states=True
    )
    model.eval()
    print("Model ready.\n")
    return tokenizer, model


def run_generation(tokenizer, model, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True
        )
    decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    return output, decoded, inputs


def extract_chain_of_thought(text: str) -> Tuple[str, str]:
    if "<think>" in text and "</think>" in text:
        thinking = text.split("<think>")[1].split("</think>")[0].strip()
        final = text.split("</think>")[-1].strip()
        return thinking, final
    return "(No <think> block found)", text


def visualize_attention(model, tokenizer, inputs):
    print("Generating attention heatmap...")
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        attn_outputs = model(**inputs, output_attentions=True)
    attentions = attn_outputs.attentions
    layer_attn = attentions[ATTN_LAYER][0, ATTN_HEAD].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        layer_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=len(tokens) <= 16,
        fmt=".2f"
    )
    plt.title(f"Attention Pattern - Layer {ATTN_LAYER}, Head {ATTN_HEAD}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.close()
    print("Attention heatmap saved as attention_heatmap.png\n")


def analyze_token_confidence(tokenizer, generation_output):
    print("Token confidence analysis:")
    probs_per_token: List[Tuple[str, float]] = []
    scores = generation_output.scores
    sequence = generation_output.sequences[0]
    generated_ids = sequence[generation_output.sequences.shape[-1] - len(scores):]

    for idx, score in enumerate(scores):
        prob = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[idx]
        confidence = prob[token_id].item()
        token_str = tokenizer.decode([token_id])
        probs_per_token.append((token_str, confidence))

    for token, conf in probs_per_token:
        bar = "█" * int(conf * 20)
        print(f"Token: {token:<10} | Confidence: {conf:>6.2%} {bar}")
    print()


def logit_lens_analysis(model, tokenizer, inputs):
    print("Logit lens analysis (first 5 layers):")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple length num_layers + 1
    lm_head = model.lm_head.weight

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    last_token_id = inputs["input_ids"][0, -1]

    for layer_idx in range(min(5, len(hidden_states))):
        state = hidden_states[layer_idx][0, -1]  # last token
        logits = torch.matmul(state, lm_head.T)
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_id = torch.topk(probs, 1)
        top_token = tokenizer.decode([top_id.item()])
        print(f"Layer {layer_idx:02d}: predicts '{top_token}' (prob={top_prob.item():.2%}) vs true token '{tokenizer.decode([last_token_id])}'")
    print()


def gradient_token_attribution(model, tokenizer, inputs, answer_token_id: int):
    print("Computing gradient-based input attribution...")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    embeddings = model.get_input_embeddings()(input_ids).detach()
    embeddings.requires_grad_()

    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :]
    target_logit = logits[answer_token_id]

    model.zero_grad()
    target_logit.backward()

    gradients = embeddings.grad[0]
    token_scores = (gradients * embeddings[0]).sum(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("Token attribution scores:")
    for token, score in zip(tokens, token_scores):
        print(f"  {token:<10} -> {score.item(): .4f}")
    print()


def hidden_state_summary(model, tokenizer, inputs):
    print("Hidden state norms per layer:")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    for idx, state in enumerate(hidden_states):
        norm = state.norm(dim=-1).mean().item()
        print(f"  Layer {idx:02d}: average hidden state norm = {norm:.4f}")
    print()


def main():
    tokenizer, model = load_model()
    gen_output, decoded_text, inputs = run_generation(tokenizer, model, QUESTION)

    print("=== Model Response ===")
    print(decoded_text)
    thinking, final_answer = extract_chain_of_thought(decoded_text)
    print("\n--- Chain of Thought ---")
    print(thinking)
    print("\n--- Final Answer ---")
    print(final_answer)
    print()

    visualize_attention(model, tokenizer, inputs)
    analyze_token_confidence(tokenizer, gen_output)
    logit_lens_analysis(model, tokenizer, inputs)

    last_generated_token = gen_output.sequences[0, -1].item()
    gradient_token_attribution(model, tokenizer, inputs, last_generated_token)
    hidden_state_summary(model, tokenizer, inputs)

    print("Explainability analysis complete.")


if __name__ == "__main__":
    main()
