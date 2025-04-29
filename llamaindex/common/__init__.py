import os
import sys

from dotenv import load_dotenv

load_dotenv()

demo_image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

db_base_dir = os.environ.get("CHROMA_BASE_DIR", "chroma")
data_base_dir = os.environ.get("DATA_BASE_DIR", "data")

ollama_host = os.getenv("OLLAMA_HOST", "localhost")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "paraphrase-multilingual:278m")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
ollama_mm_model = os.getenv("OLLAMA_MM_MODEL", "llama3.2-vision:11b")
ollama_fc_model = os.getenv("OLLAMA_FC_MODEL", "qwen2.5:7b")

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
hf_mm_model = os.getenv("HF_MM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")

openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
openai_llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", openai_llm_model)
openai_mm_model = os.getenv("OPENAI_MM_MODEL", openai_llm_model)
openai_fc_model = os.getenv("OPENAI_FC_MODEL", openai_llm_model)

google_embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
google_llm_model = os.getenv("GOOGLE_LLM_MODEL", "models/gemini-2.0-flash")
google_chat_model = os.getenv("GOOGLE_CHAT_MODEL", google_llm_model)
google_mm_model = os.getenv("GOOGLE_MM_MODEL", google_llm_model)
google_fc_model = os.getenv("GOOGLE_FC_MODEL", google_llm_model)
google_few_shoted = os.getenv("GOOGLE_FEW_SHOTED", "false") == "true"

sse_url = os.getenv("SSE_URL", "http://localhost:8000/sse")

def get_env_bool(key: str, default: str = "false", target: str = "true"):
    return os.getenv(key, default) == target


def get_args(order: int, default: str):
    return len(sys.argv) > order and sys.argv[order] or default
