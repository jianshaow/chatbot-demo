import os, sys, torch
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
DEFAULT_QUESTION_ZH = "杨志是个怎样的人?"


class RagChatConfig:
    def __init__(
        self,
        embedding_model: Type[BaseEmbedding],
        embedding_model_name: str,
        chat_model: Type[LLM],
        chat_model_name: str,
        bnb_quantized: bool = True,
        data_path: str = DATA_PATH,
        vector_db_collection: str = "hf",
        defalut_question: str = DEFAULT_QUESTION,
    ):
        self.__embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.__chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.bnb_quantized = bnb_quantized
        self.data_path = data_path
        self.vector_db_path = os.environ.get("CHROMA_DB_DIR", "chroma_db")
        self.vector_db_collection = vector_db_collection
        self.defalut_question = defalut_question

    def embedding_model(self):
        return self.__embedding_model(model_name=self.embedding_model_name)

    def chat_model(self):
        if self.__chat_model == HuggingFaceLLM:
            return self.__hf_chat_model()
        else:
            return self.__chat_model(self.chat_model_name)

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
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.chat_model_name,
            model_name=self.chat_model_name,
            model_kwargs=model_kwargs,
        )


SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
)


def __openai_config(
    embeddding_model_name="text-embedding-ada-002",
    chat_model_name="gpt-3.5-turbo",
    data_path=DATA_PATH,
    vector_db_collection="openai",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        OpenAIEmbedding,
        embeddding_model_name,
        OpenAI,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __gemini_config(
    embeddding_model_name="models/embedding-001",
    chat_model_name="models/gemini-pro",
    data_path=DATA_PATH,
    vector_db_collection="gemini",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        GeminiEmbedding,
        embeddding_model_name,
        Gemini,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __ollama_config(
    embeddding_model_name="znbang/bge:large-en-v1.5-f16",
    chat_model_name="vicuna:13b-v1.5-q4_0",
    data_path=DATA_PATH,
    vector_db_collection="ollama",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        OllamaEmbedding,
        embeddding_model_name,
        Ollama,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __hf_config(
    embeddding_model_name=None,
    chat_model_name="lmsys/vicuna-7b-v1.5",
    bnb_quantized=True,
    data_path=DATA_PATH,
    vector_db_collection="hf",
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        HuggingFaceEmbedding,
        embeddding_model_name,
        HuggingFaceLLM,
        chat_model_name,
        bnb_quantized=bnb_quantized,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


HYBRID = RagChatConfig(
    HuggingFaceEmbedding,
    None,
    OpenAI,
    "gpt-3.5-turbo",
)

HYBRID_ZH = RagChatConfig(
    HuggingFaceEmbedding,
    None,
    OpenAI,
    "gpt-3.5-turbo",
    data_path=DATA_PATH_ZH,
    vector_db_collection="hf_zh",
    defalut_question=DEFAULT_QUESTION_ZH,
)

HYBRID_LARGE = RagChatConfig(
    HuggingFaceEmbedding,
    "BAAI/bge-large-en-v1.5",
    OpenAI,
    "gpt-3.5-turbo",
    bnb_quantized=False,
    vector_db_collection="hf_large",
)

HYBRID_LARGE_ZH = RagChatConfig(
    HuggingFaceEmbedding,
    "BAAI/bge-large-zh-v1.5",
    OpenAI,
    "gpt-3.5-turbo",
    bnb_quantized=False,
    data_path=DATA_PATH_ZH,
    vector_db_collection="hf_large_zh",
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
        embeddding_model_name="znbang/bge:large-zh-v1.5-f16",
        vector_db_collection="ollama_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf": __hf_config(),
    "hf_en": __hf_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="hf_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_zh": __hf_config(
        data_path=DATA_PATH_ZH,
        embeddding_model_name="BAAI/bge-small-zh",
        vector_db_collection="hf_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf_large": __hf_config(
        embeddding_model_name="BAAI/bge-large-en-v1.5",
        chat_model_name="TheBloke/vicuna-13B-v1.5-AWQ",
        bnb_quantized=False,
        vector_db_collection="hf_large",
    ),
    "hf_large_en": __hf_config(
        embeddding_model_name="BAAI/bge-large-en-v1.5",
        chat_model_name="TheBloke/vicuna-13B-v1.5-AWQ",
        bnb_quantized=False,
        vector_db_collection="hf_large_en",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_large_zh": __hf_config(
        embeddding_model_name="BAAI/bge-large-zh-v1.5",
        chat_model_name="TheBloke/vicuna-13B-v1.5-AWQ",
        bnb_quantized=False,
        data_path=DATA_PATH_ZH,
        vector_db_collection="hf_large_zh",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hybrid": HYBRID,
    "hybrid_zh": HYBRID_ZH,
    "hybrid_large": HYBRID_LARGE,
    "hybrid_large_zh": HYBRID_LARGE_ZH,
}


def get_config(name="openai"):
    if len(sys.argv) >= 2:
        return __config_dict[sys.argv[1]]
    else:
        return __config_dict[name]


if __name__ == "__main__":
    for config in __config_dict:
        print(config)
