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

from langchain_google_genai import GoogleGenerativeAIEmbeddings


EMBEDDINGS = {
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "GoogleGenerativeAIEmbeddings": GoogleGenerativeAIEmbeddings,
    # "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
    #  "InstructorEmbeddings": InstructorEmbeddings,
    # "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
}

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
