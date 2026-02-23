from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Point directly to your downloaded folder
MODEL_PATH = "./deepseek-model"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="cpu"
)

print(f"✅ Model loaded!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Attention heads: {model.config.num_attention_heads}")
print(f"Hidden size: {model.config.hidden_size}")

print("\n" + "="*60)
print("TESTING CHAIN OF THOUGHT")
print("="*60)

question = "If a train travels 120km in 2 hours, what is its average speed?"

# DeepSeek R1 produces CoT automatically in <think> tags
inputs = tokenizer(question, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Extract the <think> section (the CoT)
if "<think>" in response and "</think>" in response:
    thinking = response.split("<think>")[1].split("</think>")[0].strip()
    final_answer = response.split("</think>")[-1].strip()
    
    print("\n--- CHAIN OF THOUGHT ---")
    print(thinking)
    print("\n--- FINAL ANSWER ---")
    print(final_answer)

print("\n" + "="*60)
print("ACCESSING MODEL WEIGHTS")
print("="*60)

# 1. Print full architecture
print("\n--- MODEL ARCHITECTURE ---")
print(model)

# 2. Access specific weight matrices
print("\n--- WEIGHT MATRICES (first 5) ---")
for i, (name, param) in enumerate(model.named_parameters()):
    if i < 5:
        print(f"\n{name}")
        print(f"  Shape: {param.shape}")
        print(f"  Mean:  {param.float().mean():.6f}")
        print(f"  Std:   {param.float().std():.6f}")

# 3. Access embedding weights specifically
embeddings = model.get_input_embeddings().weight
print(f"\n--- EMBEDDING WEIGHTS ---")
print(f"Shape: {embeddings.shape}")
print(f"Vocab size: {embeddings.shape[0]}")
print(f"Embedding dim: {embeddings.shape[1]}")

print("\n" + "="*60)
print("EXTRACTING ATTENTION PATTERNS")
print("="*60)

short_text = "The train travels fast"
inputs = tokenizer(short_text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    # Use eager attention to get attention patterns
    model.config._attn_implementation = "eager"
    outputs = model(
        **inputs,
        output_attentions=True
    )

# Get attention from first layer, first head
attentions = outputs.attentions
print(f"Number of attention layers: {len(attentions)}")
print(f"Shape per layer: {attentions[0].shape}")
# [batch, num_heads, seq_len, seq_len]

# Plot attention heatmap
attn_matrix = attentions[0][0, 0].cpu().numpy()  # layer 0, head 0

plt.figure(figsize=(8, 6))
sns.heatmap(
    attn_matrix,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    annot=True,
    fmt=".2f",
    cbar=True
)
plt.title("Attention Pattern - Layer 0, Head 0")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
plt.show()

print("✅ Saved attention_heatmap.png")

print("\n" + "="*60)
print("WEIGHT DISTRIBUTION VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Get attention weights from layer 0
attn_layer = model.model.layers[0].self_attn

weight_data = [
    (attn_layer.q_proj.weight.float().detach().numpy().flatten(), "Q Projection", axes[0,0]),
    (attn_layer.k_proj.weight.float().detach().numpy().flatten(), "K Projection", axes[0,1]),
    (attn_layer.v_proj.weight.float().detach().numpy().flatten(), "V Projection", axes[1,0]),
    (attn_layer.o_proj.weight.float().detach().numpy().flatten(), "O Projection", axes[1,1]),
]

for weights, title, ax in weight_data:
    ax.hist(weights, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
    ax.set_title(f"{title} - Layer 0")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.axvline(weights.mean(), color="red", linestyle="--", label=f"Mean: {weights.mean():.4f}")
    ax.legend()

plt.suptitle("DeepSeek-R1 Weight Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("weight_distributions.png", dpi=150)
plt.show()

print("✅ Saved weight_distributions.png")
