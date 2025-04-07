from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import logging
from pathlib import Path
from urllib.parse import urlparse
import logging
import pickle
import re

from src.core.component_registry import build_embedding_model, SPLITTERS, INDEX_DIR
from src.core.loaders.config_loader import load_config
from src.indexing.code_chunker import detect_language_from_path
from src.core.loaders.rag_loaders import get_index_subdir


logging.basicConfig(level=logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_repo_dir(github_url: str) -> str:
    path = urlparse(github_url).path
    repo_name = Path(path).stem
    return f".repos/{repo_name}"


def strip_links(text):
    # Removes http/https URLs from text
    return re.sub(r"https?://\S+", "", text)


def make_file_filter(config):
    include_extensions = set(config.get("include_extensions", []))
    exclude_extensions = set(config.get("exclude_extensions", []))
    exclude_dirs = set(config.get("exclude_dirs", []))

    def file_filter(file_path: str) -> bool:
        """
        This filter allows specifying which file extensions to include or exclude.
        If include_extensions is not provided, all extensions are allowed by default.
        """

        file_path = os.path.join(*file_path.split(os.sep)[2:])
        file_extension = "." + file_path.split(".")[-1] if "." in file_path else ""

        if any(file_path.startswith(d) for d in exclude_dirs):
            return False
        if any(file_path.endswith(ext) for ext in exclude_extensions):
            return False
        if include_extensions:
            return file_extension in include_extensions
        return True

    return file_filter


def strip_links(text):
    # Removes http/https URLs from text
    return re.sub(r"https?://\S+", "", text)


def build_indexing_chain(config):
    """
    Builds a runnable indexing pipeline for loading, filtering, chunking, embedding,
    and saving documents from a GitHub repository.

    Args:
        config (dict): Configuration dictionary with keys:
            - "filter": File filtering rules.
            - "chunking": Chunk size, overlap, and splitter class.
            - "embedding": Embedding model class and name.

    Returns:
        Runnable: A LangChain-style pipeline that expects a GitHub repository URL (str)
                  as input and returns a FAISS index after saving it locally.
    """
    file_filter = make_file_filter(config.get("filter", {}))
    splitter_cfg = config["chunking"]
    # splitter_cls = SPLITTERS[splitter_cfg["splitter"]["class"]]
    # splitter = splitter_cls(
    #     chunk_size=splitter_cfg["chunk_size"],
    #     chunk_overlap=splitter_cfg["chunk_overlap"],
    #     length_function=len,
    # )
    embedding_cfg = config["embedding"]
    embedding_model = build_embedding_model(embedding_cfg)

    def fetch_and_clone_repo_docs(github_url):
        repo_path = get_repo_dir(github_url)
        loader = GitLoader(
            clone_url=github_url,
            repo_path=repo_path,
            branch="main",
            file_filter=file_filter,
        )
        return loader.load()

    def split_docs_into_chunks(docs):
        all_chunks = []
        for doc in docs:
            cleaned_text = strip_links(doc.page_content)
            doc.page_content = cleaned_text
            source_path = doc.metadata.get("source", "")
            lang = detect_language_from_path(source_path)

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=splitter_cfg["chunk_size"],
                chunk_overlap=splitter_cfg["chunk_overlap"],
            )

            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} code-aware chunks")
        return all_chunks

    def embed_chunks_and_create_index(chunks):
        faiss_index = FAISS.from_documents(chunks, embedding_model)
        # faiss_index = FAISS.from_documents([chunks[0]], embedding_model)

        # # Step 2: Loop through the rest and add them to the index
        # for doc in chunks[1:]:
        #     while True:
        #         try:
        #             faiss_index.add_documents([doc])
        #             break
        #         except Exception as e:
        #             print(f"Error embedding document: {e}")
        #             print("Waiting 60 seconds before retrying...")
        #             time.sleep(61)

        bm25_retriever = BM25Retriever.from_documents(chunks)
        return (faiss_index, bm25_retriever)

    def save_indexes(indexes):
        faiss_index, bm25_retriever = indexes
        embed_config_name = get_index_subdir(config)
        faiss_index.save_local(os.path.join(INDEX_DIR, embed_config_name, "faiss"))
        bm25_path = os.path.join(INDEX_DIR, embed_config_name, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        logger.info("Saved FAISS and BM25 indexes.")
        return indexes

    chain = (
        RunnableLambda(fetch_and_clone_repo_docs)
        | RunnableLambda(split_docs_into_chunks)
        | RunnableLambda(embed_chunks_and_create_index)
        | RunnableLambda(save_indexes)
    )

    return chain


# Example usage
if __name__ == "__main__":
    config = load_config("config/base.yaml")
    github_url = "https://github.com/viarotel-org/escrcpy.git"
    indexing_chain = build_indexing_chain(config)
    graph = indexing_chain.get_graph()
    print(graph.print_ascii())
    index = indexing_chain.invoke(github_url)
