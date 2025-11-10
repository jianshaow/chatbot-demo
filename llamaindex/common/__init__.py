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
ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL", "deepseek-v3.1:671b-cloud")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", ollama_llm_model)
ollama_fc_model = os.getenv("OLLAMA_FC_MODEL", ollama_llm_model)
ollama_mm_model = os.getenv("OLLAMA_MM_MODEL", "qwen3-vl:235b-cloud")

hf_embed_model = os.getenv(
    "HF_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
hf_mm_model = os.getenv("HF_MM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")

openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
openai_llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", openai_llm_model)
openai_fc_model = os.getenv("OPENAI_FC_MODEL", openai_llm_model)
openai_mm_model = os.getenv("OPENAI_MM_MODEL", openai_llm_model)

openai_like_api_base = os.getenv("OPENAI_LIKE_API_BASE", "https://api.openai.com/v1")
openai_like_api_key = os.getenv("OPENAI_LIKE_API_KEY", "empty")
openai_like_embed_model = os.getenv("OPENAI_LIKE_EMBED_MODEL", "text-embedding-3-small")
openai_like_llm_model = os.getenv("OPENAI_LIKE_LLM_MODEL", openai_llm_model)
openai_like_chat_model = os.getenv("OPENAI_LIKE_CHAT_MODEL", openai_like_llm_model)
openai_like_fc_model = os.getenv("OPENAI_LIKE_FC_MODEL", openai_like_llm_model)
openai_like_mm_model = os.getenv("OPENAI_LIKE_MM_MODEL", openai_like_llm_model)
openai_like_is_chat_model = os.getenv("OPENAI_LIKE_IS_CHAT_MODEL", "true") == "true"
openai_like_embed_batch_size = int(os.getenv("OPENAI_LIKE_EMBED_BATCH_SIZE", "10"))

google_embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
google_llm_model = os.getenv("GOOGLE_LLM_MODEL", "models/gemini-2.5-flash")
google_chat_model = os.getenv("GOOGLE_CHAT_MODEL", google_llm_model)
google_fc_model = os.getenv("GOOGLE_FC_MODEL", google_llm_model)
google_mm_model = os.getenv("GOOGLE_MM_MODEL", google_llm_model)
google_few_shoted = os.getenv("GOOGLE_FEW_SHOTED", "false") == "true"

sse_url = os.getenv("SSE_URL", "http://localhost:8000/sse")

ssl_verify = os.getenv("SSL_VERIFY", "true") == "true"

thinking = os.getenv("THINK", "false") == "true"


def get_env_bool(key: str, default: str = "false", target: str = "true"):
    return os.getenv(key, default) == target


def get_args(order: int, default: str | None = None):
    return len(sys.argv) > order and sys.argv[order] or default
