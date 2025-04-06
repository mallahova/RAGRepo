from langchain_community.embeddings import (
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.embeddings import (
    #  InstructorEmbeddings,
    HuggingFaceInstructEmbeddings,
)

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.llms import OpenAI, HuggingFaceHub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp

from src.core.custom_wrappers.gemini_wrapper import GeminiEmbeddings

EMBEDDINGS = {
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "GeminiEmbeddings": GeminiEmbeddings,
    # "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
    #  "InstructorEmbeddings": InstructorEmbeddings,
    # "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
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
    kwargs = model_config.get("model_kwargs", {})  # Optional additional args

    if class_name == "OpenAIEmbeddings":
        return embedding_cls(model=name, **kwargs)

    if class_name == "HuggingFaceEmbeddings":
        return embedding_cls(model_name=name, trust_remote_code=True, **kwargs)

    if class_name == "GeminiEmbeddings":
        return embedding_cls(model=name, **kwargs)

    raise ValueError(f"Unknown or unsupported embedding class: {class_name}")


SPLITTERS = {
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    "CharacterTextSplitter": CharacterTextSplitter,
    "TokenTextSplitter": TokenTextSplitter,
}

RERANKERS = {
    # "CohereRerank": CohereRerank,
    "HuggingFaceCrossEncoder": HuggingFaceCrossEncoder,
}

GENERATORS = {
    "OpenAI": OpenAI,
    "ChatOpenAI": ChatOpenAI,
    # "GeminiChat": GeminiChat,
    "HuggingFaceHub": HuggingFaceHub,
    "LlamaCpp": LlamaCpp,
}

INDEX_DIR = "src/data/index"
