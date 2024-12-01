import os, sys
from typing import Type
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from common import (
    db_base_dir,
    ollama_base_url,
    ollama_embed_model,
    ollama_chat_model,
    openai_embed_model,
    openai_chat_model,
    gemini_embed_model,
    gemini_chat_model,
    hf_embed_model,
    hf_chat_model,
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
        embed_model: Type[Embeddings],
        embed_model_name: str,
        chat_model: Type[BaseChatModel],
        chat_model_name: str,
        data_path: str = DATA_PATH,
        vector_db_collection: str = "hface",
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
        if self.__embed_model == OllamaEmbeddings:
            return self.__embed_model(
                base_url=ollama_base_url, model=self.embed_model_name
            )
        if self.__embed_model == GoogleGenerativeAIEmbeddings:
            return self.__embed_model(model=self.embed_model_name, transport="rest")
        if self.__embed_model == HuggingFaceEmbeddings:
            return self.__embed_model(
                model_name=self.embed_model_name,
                model_kwargs={"trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self.__embed_model(model=self.embed_model_name)

    def chat_model(self):
        if self.__chat_model == ChatOllama:
            return self.__chat_model(
                base_url=ollama_base_url, model=self.chat_model_name
            )
        if self.__chat_model == ChatGoogleGenerativeAI:
            return self.__chat_model(model=self.chat_model_name, transport="rest")
        if self.__chat_model == ChatHuggingFace:
            return self.__hf_chat_model()
        return self.__chat_model(model=self.chat_model_name)

    def get_question(self):
        if len(sys.argv) >= 3:
            return sys.argv[2]
        else:
            return self.defalut_question

    def __hf_chat_model(self):
        model_kwargs = default_model_kwargs()
        llm = HuggingFacePipeline.from_model_id(
            model_id=self.chat_model_name,
            task="text-generation",
            model_kwargs=model_kwargs,
            pipeline_kwargs={"max_new_tokens": 512},
        )
        return ChatHuggingFace(llm=llm)


def __openai_config(
    data_path=DATA_PATH,
    vector_db_collection=DEFAULT_COLLECTION,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "openai",
        OpenAIEmbeddings,
        openai_embed_model,
        ChatOpenAI,
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
        GoogleGenerativeAIEmbeddings,
        gemini_embed_model,
        ChatGoogleGenerativeAI,
        gemini_chat_model,
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
        OllamaEmbeddings,
        embed_model_name,
        ChatOllama,
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
        HuggingFaceEmbeddings,
        embed_model_name,
        ChatHuggingFace,
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
