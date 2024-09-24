import os
from dotenv import load_dotenv

load_dotenv()

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
ollama_embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
ollama_chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "vicuna:7b")
ollama_mm_model = os.environ.get("OLLAMA_MM_MODEL", "llava:7b")
ollama_fc_model = os.environ.get("OLLAMA_FC_MODEL", "llama3.1:8b")
