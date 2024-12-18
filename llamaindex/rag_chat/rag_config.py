import os, sys, prompts
from typing import Type
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

from common.ollama import NormOllamaEmbedding

from common import (
    db_base_dir,
    data_base_dir,
    ollama_base_url,
    ollama_embed_model,
    ollama_chat_model,
    hf_embed_model,
    hf_chat_model,
    openai_embed_model,
    openai_chat_model,
    gemini_embed_model,
    gemini_chat_model,
)
from common.models import default_model_kwargs

DEFAULT_DATA = "default"
DATA_EN = "en-text"
DATA_ZH = "zh-text"

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
        data_dir: str = DEFAULT_DATA,
        defalut_question: str = DEFAULT_QUESTION,
    ):
        self.name = name
        self.__embed_model = embed_model
        self.embed_model_name = embed_model_name
        self.__chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.__data_dir = data_dir
        self.db_base_dir = db_base_dir
        self.data_base_dir = data_base_dir
        self.defalut_question = defalut_question

    @property
    def vector_db_path(self):
        return os.path.join(self.db_base_dir, self.name)

    @property
    def data_dir(self):
        return os.path.join(self.data_base_dir, self.__data_dir)

    @property
    def vector_db_collection(self):
        escaped = self.embed_model_name.replace(":", "_").replace("/", "_")
        return self.__data_dir + "__" + escaped

    def embed_model(self):
        if self.__embed_model == NormOllamaEmbedding:
            return self.__embed_model(
                base_url=ollama_base_url, model_name=self.embed_model_name
            )
        if self.__embed_model == HuggingFaceEmbedding:
            return self.__embed_model(
                model_name=self.embed_model_name, trust_remote_code=True
            )
        if self.__embed_model == OpenAIEmbedding:
            return self.__embed_model(model=self.embed_model_name)
        if self.__embed_model == GeminiEmbedding:
            return self.__embed_model(
                model_name=self.embed_model_name, transport="rest"
            )

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
        if self.__chat_model == Gemini:
            return self.__chat_model(model_name=self.chat_model_name, transport="rest")

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
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "openai",
        OpenAIEmbedding,
        openai_embed_model,
        OpenAI,
        openai_chat_model,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


def __gemini_config(
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "gemini",
        GeminiEmbedding,
        gemini_embed_model,
        Gemini,
        gemini_chat_model,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


def __ollama_config(
    embed_model_name=ollama_embed_model,
    chat_model_name=ollama_chat_model,
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "ollama",
        NormOllamaEmbedding,
        embed_model_name,
        Ollama,
        chat_model_name,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


def __hf_config(
    embed_model_name=hf_embed_model,
    chat_model_name=hf_chat_model,
    data_dir=DEFAULT_DATA,
    defalut_question=DEFAULT_QUESTION,
):
    return RagChatConfig(
        "hface",
        HuggingFaceEmbedding,
        embed_model_name,
        HuggingFaceLLM,
        chat_model_name,
        data_dir=data_dir,
        defalut_question=defalut_question,
    )


__config_dict = {
    "openai": __openai_config(),
    "openai_en": __openai_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "openai_zh": __openai_config(
        data_dir=DATA_ZH,
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "gemini": __gemini_config(),
    "gemini_en": __gemini_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "gemini_zh": __gemini_config(
        data_dir=DATA_ZH,
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "ollama": __ollama_config(),
    "ollama_en": __ollama_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "ollama_zh": __ollama_config(
        data_dir=DATA_ZH,
        embed_model_name="paraphrase-multilingual:278m",
        defalut_question=DEFAULT_QUESTION_ZH,
    ),
    "hf": __hf_config(),
    "hf_en": __hf_config(
        data_dir=DATA_EN,
        defalut_question=DEFAULT_QUESTION_EN,
    ),
    "hf_zh": __hf_config(
        data_dir=DATA_ZH,
        embed_model_name="BAAI/bge-base-zh-v1.5",
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
