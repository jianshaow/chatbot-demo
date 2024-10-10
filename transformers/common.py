import os
from dotenv import load_dotenv

load_dotenv()

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

bnb_enabled = os.getenv("BNB_ENABLED", "false") == "true"
