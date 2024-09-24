import os
from dotenv import load_dotenv

load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "localhost")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "vicuna:7b")
ollama_mm_model = os.getenv("OLLAMA_MM_MODEL", "llava:7b")
ollama_fc_model = os.getenv("OLLAMA_FC_MODEL", "llama3.1:8b")
