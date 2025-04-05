from langchain_community.embeddings import (
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings
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
# from langchain_community_community.rerankers import (
#     CohereRerank,
#     SentenceTransformersRerank
# )
from langchain_community.llms import OpenAI, HuggingFaceHub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import LlamaCpp

EMBEDDINGS = {
    "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
    "SentenceTransformerEmbeddings": SentenceTransformerEmbeddings,
   #  "InstructorEmbeddings": InstructorEmbeddings,
    "HuggingFaceInstructEmbeddings": HuggingFaceInstructEmbeddings,
    "OpenAIEmbeddings": OpenAIEmbeddings,
}

SPLITTERS = {
    "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    "CharacterTextSplitter": CharacterTextSplitter,
    "TokenTextSplitter": TokenTextSplitter,
}

# RERANKERS = {
#     "CohereRerank": CohereRerank,
#     "SentenceTransformersRerank": SentenceTransformersRerank,
# }

GENERATORS = {
    "OpenAI": OpenAI,
    "ChatOpenAI": ChatOpenAI,
    "HuggingFaceHub": HuggingFaceHub,
    "LlamaCpp": LlamaCpp,
}
