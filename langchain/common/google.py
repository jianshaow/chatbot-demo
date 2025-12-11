from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def get_chat_model(model: str, **kwargs) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, **kwargs)


def get_embed_model(model: str, **kwargs) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=model, **kwargs)
