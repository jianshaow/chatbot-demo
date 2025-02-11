import os
import sys
import types

from dotenv import load_dotenv

load_dotenv()

demo_image_url = "https://storage.googleapis.com/generativeai-downloads/data/scene.jpg"

db_base_dir = os.environ.get("CHROMA_BASE_DIR", "chroma")
data_base_dir = os.environ.get("DATA_BASE_DIR", "data")

ollama_host = os.getenv("OLLAMA_HOST", "localhost")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "gemma2:9b")
ollama_mm_model = os.getenv("OLLAMA_MM_MODEL", "llama3.2-vision:11b")
ollama_fc_model = os.getenv("OLLAMA_FC_MODEL", "qwen2.5:7b")

hf_embed_model = os.getenv("HF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
hf_chat_model = os.getenv("HF_CHAT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

openai_embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
openai_llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", openai_llm_model)
openai_mm_model = os.getenv("OPENAI_MM_MODEL", openai_llm_model)
openai_fc_model = os.getenv("OPENAI_FC_MODEL", openai_llm_model)

gemini_embed_model = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
gemini_llm_model = os.getenv("GEMINI_LLM_MODEL", "models/gemini-2.0-flash")
gemini_chat_model = os.getenv("GEMINI_CHAT_MODEL", gemini_llm_model)
gemini_mm_model = os.getenv("GEMINI_MM_MODEL", gemini_llm_model)
gemini_fc_model = os.getenv("GEMINI_FC_MODEL", gemini_llm_model)
gemini_few_shoted = os.getenv("GEMINI_FEW_SHOTED", "false") == "true"


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
