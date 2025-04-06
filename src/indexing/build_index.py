import os
import logging
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from src.core.component_registry import EMBEDDINGS, SPLITTERS, INDEX_DIR
from src.core.config_loader import load_config
from langchain_core.runnables import RunnableLambda
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_repo_dir(github_url: str) -> str:
    path = urlparse(github_url).path
    repo_name = Path(path).stem
    return f".repos/{repo_name}"


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
    splitter_cls = SPLITTERS[splitter_cfg["splitter"]["class"]]
    splitter = splitter_cls(
        chunk_size=splitter_cfg["chunk_size"],
        chunk_overlap=splitter_cfg["chunk_overlap"],
        length_function=len,
    )
    embedding_cfg = config["embedding"]
    embedding_cls = EMBEDDINGS[embedding_cfg["class"]]
    embedding_model = embedding_cls(
        model_name=embedding_cfg["name"],
        model_kwargs={"trust_remote_code": True},
    )

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
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def embed_chunks_and_create_index(chunks):
        faiss_index = FAISS.from_documents(chunks, embedding_model)
        return faiss_index

    def save_faiss_index(faiss_index):
        faiss_index.save_local(INDEX_DIR)
        return faiss_index

    chain = (
        RunnableLambda(fetch_and_clone_repo_docs)
        | RunnableLambda(split_docs_into_chunks)
        | RunnableLambda(embed_chunks_and_create_index)
        | RunnableLambda(save_faiss_index)
    )

    return chain


if __name__ == "__main__":
    config = load_config("config/base.yaml")
    github_url = "https://github.com/viarotel-org/escrcpy.git"
    indexing_chain = build_indexing_chain(config)
    graph = indexing_chain.get_graph()
    print(graph.print_ascii())
    faiss_index = indexing_chain.invoke(github_url)
