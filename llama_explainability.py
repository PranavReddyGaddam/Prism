import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

MODEL_PATH = "./llama-3.2-3b-instruct"
QUESTION = "If a train travels 120km in 2 hours, what is its average speed?"
ATTN_LAYER = 0
ATTN_HEAD = 0
MAX_TOKENS = 50


def load_model():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu"
    )
    model.eval()
    print("Model ready.\n")
    return tokenizer, model


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def run_generation(tokenizer, model, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    return output, decoded, inputs


def visualize_attention(model, tokenizer, inputs):
    print("Generating attention heatmap...")
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        attn_outputs = model(**inputs, output_attentions=True)
    attentions = attn_outputs.attentions
    layer_attn = attentions[ATTN_LAYER][0, ATTN_HEAD].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Truncate to last 16 tokens for readability
    N = min(16, len(tokens))
    layer_attn = layer_attn[-N:, -N:]
    tokens = tokens[-N:]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        layer_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.title(f"Llama-3.2-3B-Instruct Attention - Layer {ATTN_LAYER}, Head {ATTN_HEAD}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("llama_attention_heatmap.png", dpi=150)
    plt.close()
    print("Attention heatmap saved as llama_attention_heatmap.png\n")


def analyze_token_confidence(tokenizer, generation_output):
    print("Token confidence analysis:")
    scores = generation_output.scores
    sequence = generation_output.sequences[0]
    generated_ids = sequence[-len(scores):]

    for idx, score in enumerate(scores):
        prob = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[idx]
        confidence = prob[token_id].item()
        token_str = tokenizer.decode([token_id])
        bar = "█" * int(confidence * 20)
        print(f"Token: {token_str:<12} | Confidence: {confidence:>6.2%} {bar}")
    print()


def logit_lens_analysis(model, tokenizer, inputs):
    print("Logit lens analysis (first 5 layers):")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    lm_head = model.lm_head.weight
    last_token_id = inputs["input_ids"][0, -1]

    for layer_idx in range(min(5, len(hidden_states))):
        state = hidden_states[layer_idx][0, -1].float()
        logits = torch.matmul(state, lm_head.float().T)
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_id = torch.topk(probs, 1)
        top_token = tokenizer.decode([top_id.item()])
        true_token = tokenizer.decode([last_token_id])
        print(f"Layer {layer_idx:02d}: predicts '{top_token}' (prob={top_prob.item():.2%}) vs true token '{true_token}'")
    print()


def gradient_token_attribution(model, tokenizer, inputs, answer_token_id: int):
    print("Computing gradient-based input attribution...")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    embeddings = model.get_input_embeddings()(input_ids).to(torch.float32).detach()
    embeddings.requires_grad_()

    outputs = model(inputs_embeds=embeddings.to(next(model.parameters()).dtype), attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :].float()
    target_logit = logits[answer_token_id]

    model.zero_grad()
    target_logit.backward()

    gradients = embeddings.grad[0]
    token_scores = (gradients * embeddings[0]).sum(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Show last 20 tokens for readability
    tokens = tokens[-20:]
    token_scores = token_scores[-20:]

    print("Token attribution scores (last 20 tokens):")
    for token, score in zip(tokens, token_scores):
        print(f"  {token:<15} -> {score.item(): .4f}")
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


def weight_distribution_plot(model):
    print("Generating weight distribution plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    attn_layer = model.model.layers[0].self_attn

    weight_data = [
        (attn_layer.q_proj.weight.float().detach().numpy().flatten(), "Q Projection", axes[0, 0]),
        (attn_layer.k_proj.weight.float().detach().numpy().flatten(), "K Projection", axes[0, 1]),
        (attn_layer.v_proj.weight.float().detach().numpy().flatten(), "V Projection", axes[1, 0]),
        (attn_layer.o_proj.weight.float().detach().numpy().flatten(), "O Projection", axes[1, 1]),
    ]

    for weights, title, ax in weight_data:
        ax.hist(weights, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.set_title(f"{title} - Layer 0")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        ax.axvline(weights.mean(), color="red", linestyle="--", label=f"Mean: {weights.mean():.4f}")
        ax.legend()

    plt.suptitle("Llama-3.2-3B-Instruct Weight Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("llama_weight_distributions.png", dpi=150)
    plt.close()
    print("Weight distribution saved as llama_weight_distributions.png\n")


def main():
    tokenizer, model = load_model()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Attention heads: {model.config.num_attention_heads}")
    print(f"Hidden size: {model.config.hidden_size}\n")

    prompt = build_prompt(tokenizer, QUESTION)
    gen_output, decoded_text, inputs = run_generation(tokenizer, model, prompt)

    print("=== Model Response ===")
    print(decoded_text)
    print()

    visualize_attention(model, tokenizer, inputs)
    analyze_token_confidence(tokenizer, gen_output)
    logit_lens_analysis(model, tokenizer, inputs)

    last_generated_token = gen_output.sequences[0, -1].item()
    gradient_token_attribution(model, tokenizer, inputs, last_generated_token)
    hidden_state_summary(model, tokenizer, inputs)
    weight_distribution_plot(model)

    print("Llama explainability analysis complete.")
    print("Generated files: llama_attention_heatmap.png, llama_weight_distributions.png")


if __name__ == "__main__":
    main()
