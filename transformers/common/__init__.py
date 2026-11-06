import os

from dotenv import load_dotenv

load_dotenv()

bnb_enabled = os.getenv("BNB_ENABLED", "false") == "true"

hf_embed_model = os.getenv("HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
hf_embed_zh_model = os.getenv("HF_EMBED_ZH_MODEL", hf_embed_model)
hf_chat_model = os.getenv("HF_CHAT_MODEL", "google/gemma-4-E2B-it")
hf_mm_model = os.getenv("HF_MM_MODEL", hf_chat_model)
