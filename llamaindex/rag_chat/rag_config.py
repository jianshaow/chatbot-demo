import os, sys, prompts
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

from common import (
    db_base_dir,
    ollama_base_url,
    ollama_embed_model,
    ollama_chat_model,
    hf_embed_model,
    hf_chat_model,
    openai_embed_model,
    openai_chat_model,
    google_embed_model,
    google_chat_model,
)
from common.models import default_model_kwargs

DATA_PATH = "data"
DATA_PATH_EN = "data_en"
DATA_PATH_ZH = "data_zh"

DEFAULT_COLLECTION = "default"

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
        vector_db_collection: str,
        data_path: str = DATA_PATH,
        defalut_question: str = DEFAULT_QUESTION,
    ):
        self.name = name
        self.__embed_model = embed_model
        self.embed_model_name = embed_model_name
        self.__chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.data_path = data_path
        self.db_base_dir = db_base_dir
        self.vector_db_collection = vector_db_collection
        self.defalut_question = defalut_question

    @property
    def vector_db_path(self):
        return os.path.join(self.db_base_dir, self.name)

    def embed_model(self):
        if self.__embed_model == OllamaEmbedding:
            return self.__embed_model(
                base_url=ollama_base_url, model_name=self.embed_model_name
            )
        if self.__embed_model == HuggingFaceEmbedding:
            return self.__embed_model(
                model_name=self.embed_model_name, trust_remote_code=True
            )
        if self.__embed_model == OpenAIEmbedding:
            return self.__embed_model(model=self.embed_model_name)

        return self.__embed_model(model_name=self.embed_model_name)

    def chat_model(self):
        if self.__chat_model == Ollama:
            return self.__chat_model(
                base_url=ollama_base_url, model=self.chat_model_name
            )
        if self.__chat_model == HuggingFaceLLM:
            return self.__hf_chat_model()
        if self.__chat_model == OpenAI:
            return self.__chat_model(model=self.chat_model_name)

        return self.__chat_model(model_name=self.chat_model_name)

    def get_question(self):
        if len(sys.argv) >= 3:
            return sys.argv[2]
        else:
            return self.defalut_question

    def __hf_chat_model(self):
        model_kwargs = default_model_kwargs()
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
    data_path=DATA_PATH,
    vector_db_collection=DEFAULT_COLLECTION,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "openai",
        OpenAIEmbedding,
        openai_embed_model,
        OpenAI,
        openai_chat_model,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __gemini_config(
    data_path=DATA_PATH,
    vector_db_collection=DEFAULT_COLLECTION,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "gemini",
        GeminiEmbedding,
        google_embed_model,
        Gemini,
        google_chat_model,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


def __ollama_config(
    embed_model_name=ollama_embed_model,
    chat_model_name=ollama_chat_model,
    data_path=DATA_PATH,
    vector_db_collection=DEFAULT_COLLECTION,
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
    embed_model_name=hf_embed_model,
    chat_model_name=hf_chat_model,
    data_path=DATA_PATH,
    vector_db_collection=DEFAULT_COLLECTION,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "hface",
        HuggingFaceEmbedding,
        embed_model_name,
        HuggingFaceLLM,
        chat_model_name,
        data_path=data_path,
        vector_db_collection=vector_db_collection,
        defalut_question=defalut_question,
    )


__config_dict = {
    "openai": __openai_config(),
    "openai_en": __openai_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="en_text",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "openai_zh": __openai_config(
        data_path=DATA_PATH_ZH,
        vector_db_collection="zh_text",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "gemini": __gemini_config(),
    "gemini_en": __gemini_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="en_text",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "gemini_zh": __gemini_config(
        data_path=DATA_PATH_ZH,
        vector_db_collection="zh_text",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "ollama": __ollama_config(),
    "ollama_en": __ollama_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="en_text",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "ollama_zh": __ollama_config(
        data_path=DATA_PATH_ZH,
        embed_model_name="nomic-embed-text:v1.5",
        vector_db_collection="zh_text",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf": __hf_config(),
    "hf_en": __hf_config(
        data_path=DATA_PATH_EN,
        vector_db_collection="en_text",
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_zh": __hf_config(
        data_path=DATA_PATH_ZH,
        embed_model_name="BAAI/bge-small-zh",
        vector_db_collection="zh_text",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
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
