"""
Run this script on the cloud pod to download all required models.
Usage:
    pip install huggingface_hub
    python download_models.py

Set HF_TOKEN env variable if models require authentication:
    export HF_TOKEN=hf_your_token_here
"""
import os
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def download(repo_id: str, local_dir: str):
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"         To: {local_dir}")
    print('='*60)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"✅ Done: {repo_id}")


if __name__ == "__main__":
    # Phi-3 base model
    download(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        local_dir=os.path.join(MODELS_DIR, "phi-3-mini-4k-instruct"),
    )

    # DeepSeek R1 Distill
    download(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        local_dir=os.path.join(MODELS_DIR, "deepseek-model"),
    )

    print("\n✅ All models downloaded.")
    print("Update MODEL_CONFIGS paths in backend/model_manager.py to:")
    print(f"  phi-3:     {os.path.abspath(os.path.join(MODELS_DIR, 'phi-3-mini-4k-instruct'))}")
    print(f"  deepseek:  {os.path.abspath(os.path.join(MODELS_DIR, 'deepseek-model'))}")
    print("\nFor the LoRA adapter, upload phi3_mmlu_fever_hotpot_lora_adapter/ manually via scp:")
    print("  scp -r phi3_mmlu_fever_hotpot_lora_adapter/ root@<pod-ip>:/root/models/")
