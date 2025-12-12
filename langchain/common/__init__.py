import base64
import os
import sys
import types

from dotenv import load_dotenv

load_dotenv()

db_base_dir = os.environ.get("CHROMA_BASE_DIR", "chroma")
data_base_dir = os.environ.get("DATA_BASE_DIR", "data")

ollama_host = os.getenv("OLLAMA_HOST", "localhost")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "paraphrase-multilingual:278m")
ollama_llm_model = os.getenv("OLLAMA_LLM_MODEL", "deepseek-v3.2:cloud")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", ollama_llm_model)
ollama_fc_model = os.getenv("OLLAMA_FC_MODEL", ollama_llm_model)
ollama_mm_model = os.getenv("OLLAMA_MM_MODEL", "qwen3-vl:235b-instruct-cloud")

hf_embed_model = os.getenv(
    "HF_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
openai_llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", openai_llm_model)
openai_fc_model = os.getenv("OPENAI_FC_MODEL", openai_llm_model)
openai_mm_model = os.getenv("OPENAI_MM_MODEL", openai_llm_model)

google_embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
google_llm_model = os.getenv("GOOGLE_LLM_MODEL", "models/gemini-2.5-flash")
google_chat_model = os.getenv("GOOGLE_CHAT_MODEL", google_llm_model)
google_fc_model = os.getenv("GOOGLE_FC_MODEL", google_llm_model)
google_mm_model = os.getenv("GOOGLE_MM_MODEL", google_llm_model)
google_few_shoted = os.getenv("GOOGLE_FEW_SHOTED", "false") == "true"

searxng_host = os.getenv("SEARXNG_HOST", "localhost")
searxng_username = os.getenv("SEARXNG_USERNAME", "test")
searxng_password = os.getenv("SEARXNG_PASSWORD", "test")

sse_url = os.getenv("SSE_URL", "http://localhost:8000/sse")

ssl_verify = os.getenv("SSL_VERIFY", "true") == "true"

thinking = os.getenv("THINK", "false") == "true"


def get_env_bool(key: str, default: str = "false", target: str = "true"):
    return os.getenv(key, default) == target


def get_args(order: int, default: str):
    return len(sys.argv) > order and sys.argv[order] or default


def add_method_kwargs(obj: object, method_name: str, **extra_kwargs):
    if hasattr(obj, method_name) and callable(getattr(obj, method_name)):
        method = getattr(obj, method_name)
        wrapper = add_func_kwargs(method, **extra_kwargs)
        setattr(obj, method_name, wrapper)
    else:
        raise ValueError(f"objet '{obj}' do not has such a method '{method_name}'")


def add_func_kwargs(func: types.FunctionType, **extra_args) -> types.FunctionType:
    def wrapper(self, **kwargs):
        return func(self, **{**kwargs, **extra_args})

    return wrapper


def get_basic_auth_headers(username: str, password: str):
    auth_str = f"{username}:{password}"
    auth_bytes = base64.b64encode(auth_str.encode("utf-8"))
    auth_header = f"Basic {auth_bytes.decode('utf-8')}"
    return {
        "Authorization": auth_header,
    }
