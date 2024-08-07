import os, sys, torch, prompts
from typing import Type
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

DATA_PATH = "data"
DATA_PATH_EN = "data_en"
DATA_PATH_ZH = "data_zh"

DEFAULT_QUESTION = "What did the author do growing up?"
DEFAULT_QUESTION_EN = "Why the old man go fishing?"
DEFAULT_QUESTION_ZH = "地球发动机都安装在哪里？"


class RagChatConfig:
    def __init__(
        self,
        name: str,
        embed_model: Type[BaseEmbedding],
        embed_model_name: str,
        chat_model: Type[LLM],
        chat_model_name: str,
        bnb_quantized: bool = True,
        data_path: str = DATA_PATH,
        vector_db_collection: str = "hface",
        defalut_question: str = DEFAULT_QUESTION,
    ):
        self.name = name
        self.__embed_model = embed_model
        self.embed_model_name = embed_model_name
        self.__chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.bnb_quantized = bnb_quantized
        self.data_path = data_path
        self.vector_db_path = os.environ.get("CHROMA_DB_DIR", "chroma")
        self.vector_db_collection = vector_db_collection
        self.defalut_question = defalut_question

    def embed_model(self):
        if self.__embed_model == OllamaEmbedding:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return self.__embed_model(
                base_url=base_url, model_name=self.embed_model_name
            )
        if self.__embed_model == OpenAIEmbedding:
            return self.__embed_model(model=self.embed_model_name)

        return self.__embed_model(model_name=self.embed_model_name)

    def chat_model(self):
        if self.__chat_model == HuggingFaceLLM:
            return self.__hf_chat_model()
        if self.__chat_model == Ollama:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return self.__chat_model(base_url=base_url, model=self.chat_model_name)
        if self.__chat_model == OpenAI:
            return self.__chat_model(model=self.chat_model_name)

        return self.__chat_model(model_name=self.chat_model_name)

    def get_question(self):
        if len(sys.argv) >= 3:
            return sys.argv[2]
        else:
            return self.defalut_question

    def __hf_chat_model(self):
        model_kwargs = {}
        if self.bnb_quantized:
            from transformers import BitsAndBytesConfig

            model_kwargs.update(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                }
            )
        return HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=2048,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            query_wrapper_prompt=PromptTemplate(prompts.rag_template()),
            tokenizer_name=self.chat_model_name,
            model_name=self.chat_model_name,
            model_kwargs=model_kwargs,
        )


def __openai_config(
    embed_model_name=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-ada-002"),
    chat_model_name=os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
    data_path=DATA_PATH,
    vector_db_collection="openai",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "openai",
        OpenAIEmbedding,
        embed_model_name,
        OpenAI,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __gemini_config(
    embed_model_name=os.environ.get("GEMINI_EMBED_MODEL", "models/embedding-001"),
    chat_model_name=os.environ.get("GEMINI_CHAT_MODEL", "models/gemini-1.5-flash"),
    data_path=DATA_PATH,
    vector_db_collection="gemini",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "gemini",
        GeminiEmbedding,
        embed_model_name,
        Gemini,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __ollama_config(
    embed_model_name=os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5"),
    chat_model_name=os.environ.get("OLLAMA_CHAT_MODEL", "vicuna:7b"),
    data_path=DATA_PATH,
    vector_db_collection="ollama",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "ollama",
        OllamaEmbedding,
        embed_model_name,
        Ollama,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __hf_config(
    embed_model_name="BAAI/bge-small-en",
    chat_model_name="lmsys/vicuna-7b-v1.5",
    bnb_quantized=True,
    data_path=DATA_PATH,
    vector_db_collection="hface",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "hface",
        HuggingFaceEmbedding,
        embed_model_name,
        HuggingFaceLLM,
        chat_model_name,
        bnb_quantized=bnb_quantized,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


HYBRID = RagChatConfig(
    "hybrid",
    HuggingFaceEmbedding,
    "BAAI/bge-small-en",
    OpenAI,
    "gpt-3.5-turbo",
)

HYBRID_ZH = RagChatConfig(
    "hybrid_zh",
    HuggingFaceEmbedding,
    "BAAI/bge-small-en",
    OpenAI,
    "gpt-3.5-turbo",
    data_path=DATA_PATH_ZH,
    vector_db_collection="hface_zh",
    defalut_question=DEFAULT_QUESTION_ZH,
)

HYBRID_LARGE = RagChatConfig(
    "hybrid_large",
    HuggingFaceEmbedding,
    "BAAI/bge-large-en-v1.5",
    OpenAI,
    "gpt-3.5-turbo",
    bnb_quantized=False,
    vector_db_collection="hface_large",
)

HYBRID_LARGE_ZH = RagChatConfig(
    "hybrid_large_zh",
    HuggingFaceEmbedding,
    "BAAI/bge-large-zh-v1.5",
    OpenAI,
    "gpt-3.5-turbo",
    bnb_quantized=False,
    data_path=DATA_PATH_ZH,
    vector_db_collection="hface_large_zh",
    defalut_question=DEFAULT_QUESTION_ZH,
)

__config_dict = {
    "openai": __openai_config(),
    "openai_en": __openai_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="openai_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "openai_zh": __openai_config(
        data_path=DATA_PATH_ZH,
        vector_db_collection="openai_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "gemini": __gemini_config(),
    "gemini_en": __gemini_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="gemini_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "gemini_zh": __gemini_config(
        data_path=DATA_PATH_ZH,
        vector_db_collection="gemini_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "ollama": __ollama_config(),
    "ollama_en": __ollama_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="ollama_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "ollama_zh": __ollama_config(
        data_path=DATA_PATH_ZH,
        embed_model_name="nomic-embed-text:v1.5",
        vector_db_collection="ollama_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf": __hf_config(),
    "hf_en": __hf_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="hface_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_zh": __hf_config(
        data_path=DATA_PATH_ZH,
        embed_model_name="BAAI/bge-small-zh",
        vector_db_collection="hface_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf_large": __hf_config(
        embed_model_name="BAAI/bge-large-en-v1.5",
        chat_model_name="TheBloke/vicuna-13B-v1.5-AWQ",
        bnb_quantized=False,
        vector_db_collection="hface_large",
    ),
    "hf_large_en": __hf_config(
        embed_model_name="BAAI/bge-large-en-v1.5",
        chat_model_name="TheBloke/vicuna-13B-v1.5-AWQ",
        bnb_quantized=False,
        vector_db_collection="hface_large_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_large_zh": __hf_config(
        embed_model_name="BAAI/bge-large-zh-v1.5",
        chat_model_name="TheBloke/vicuna-13B-v1.5-AWQ",
        bnb_quantized=False,
        data_path=DATA_PATH_ZH,
        vector_db_collection="hface_large_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hybrid": HYBRID,
    "hybrid_zh": HYBRID_ZH,
    "hybrid_large": HYBRID_LARGE,
    "hybrid_large_zh": HYBRID_LARGE_ZH,
}


def get_config(name="ollama"):
    if len(sys.argv) >= 2:
        return __config_dict[sys.argv[1]]
    else:
        return __config_dict[name]


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print(vars(__config_dict[sys.argv[1]]))
    else:
        for config in __config_dict:
            print(config)
