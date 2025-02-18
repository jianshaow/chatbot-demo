import os

from dotenv import load_dotenv

load_dotenv()

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
hf_embed_zh_model = os.getenv("HF_EMBED_ZH_MODEL", "BAAI/bge-base-zh-v1.5")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
hf_mm_model = os.getenv("HF_MM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
hf_sd1_model = os.getenv("HF_SD1_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
hf_sd3_model = os.getenv("HF_SD3_MODEL", "stabilityai/stable-diffusion-3.5-medium")

bnb_enabled = os.getenv("BNB_ENABLED", "false") == "true"
