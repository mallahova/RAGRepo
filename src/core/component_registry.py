# Embedding models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from src.core.custom_wrappers.gemini_wrapper import GeminiEmbeddings

# Text splitters
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

# Rerankers
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereRerank

# Generative LLMs
from langchain_openai import ChatOpenAI


# from langchain_community.llms import OpenAI, HuggingFaceHub
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.llms import LlamaCpp


EMBEDDINGS = {
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
    "GeminiEmbeddings": GeminiEmbeddings,
}


def build_embedding_model(model_config: dict):
    """
    Dynamically build an embedding model instance based on the class name and model_config.

    Args:
        class_name (str): The key from the EMBEDDINGS registry.
        model_config (dict): Dictionary with model model_configuration.
                       Expected keys: "name", optionally "model_kwargs", etc.

    Returns:
        An instantiated embedding model.
    """
    class_name = model_config["class"]
    embedding_cls = EMBEDDINGS[class_name]
    name = model_config.get("name", None)

    if class_name == "OpenAIEmbeddings":
        return embedding_cls(model=name)

    if class_name == "HuggingFaceEmbeddings":
        return embedding_cls(model_name=name, model_kwargs={"trust_remote_code": True})

    if class_name == "GeminiEmbeddings":
        return embedding_cls(model_name=name)

    raise ValueError(f"Unknown or unsupported embedding class: {class_name}")


SPLITTERS = {
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    "CharacterTextSplitter": CharacterTextSplitter,
    "TokenTextSplitter": TokenTextSplitter,
}

RERANKERS = {
    "CohereRerank": CohereRerank,
    "HuggingFaceCrossEncoder": HuggingFaceCrossEncoder,
}

GENERATORS = {
    "ChatOpenAI": ChatOpenAI,
}

INDEX_DIR = "src/data/index"
