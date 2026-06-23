import os

from dotenv import load_dotenv

load_dotenv()

bnb_enabled = os.getenv("BNB_ENABLED", "false") == "true"

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
hf_embed_zh_model = os.getenv("HF_EMBED_ZH_MODEL", "BAAI/bge-base-zh-v1.5")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "google/gemma-4-E2B-it")
hf_mm_model = os.getenv("HF_MM_MODEL", hf_chat_model)
