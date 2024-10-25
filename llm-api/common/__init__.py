import os
from dotenv import load_dotenv

load_dotenv()

ollama_host = os.environ.get("OLLAMA_HOST", "localhost")
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
ollama_embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
ollama_chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2:3b")
ollama_mm_model = os.environ.get("OLLAMA_MM_MODEL", "llava:7b")
ollama_fc_model = os.environ.get("OLLAMA_FC_MODEL", "llama3.1:8b")

openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
openai_mm_model = os.getenv("OPENAI_MM_MODEL", "gpt-4o-mini")
openai_fc_model = os.getenv("OPENAI_FC_MODEL", "gpt-4o-mini")

google_embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/embedding-001")
google_chat_model = os.getenv("GOOGLE_CHAT_MODEL", "models/gemini-1.5-flash")
google_mm_model = os.getenv("GOOGLE_MM_MODEL", "models/gemini-1.5-flash")
google_fc_model = os.getenv("GOOGLE_FC_MODEL", "models/gemini-1.5-flash")
