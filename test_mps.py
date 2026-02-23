import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_PATH = "./llama-3.2-3b-instruct"

print("Testing Apple Silicon GPU (MPS) performance...")

# Test 1: Load model on MPS
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="mps"  # Use MPS instead of CPU
)
load_time = time.time() - start
print(f"Model loaded on MPS in {load_time:.2f} seconds")

# Test 2: Simple generation
prompt = "What is 2+2?"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

start = time.time()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )
gen_time = time.time() - start

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated in {gen_time:.2f} seconds")
print(f"Result: {result}")

# Test 3: Check memory usage
if hasattr(torch, 'mps'):
    print(f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
    print(f"MPS memory reserved: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GB")
