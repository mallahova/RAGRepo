import os
import logging
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from src.core.component_registry import EMBEDDINGS, SPLITTERS
from src.core.config_loader import load_config


for noisy_logger in ["datasets", "sentence_transformers", "faiss"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def make_file_filter(config):
    include_extensions = set(config.get("include_extensions", []))
    exclude_extensions = set(config.get("exclude_extensions", []))
    exclude_dirs = set(config.get("exclude_dirs", []))

    def file_filter(file_path: str) -> bool:
        """
        This filter allows specifying which file extensions to include or exclude.
        If include_extensions is not provided, all extensions are allowed by default.
        """
        file_path = os.path.join(*file_path.split(os.sep)[1:])
        file_extension = "." + file_path.split(".")[-1] if "." in file_path else ""

        if any(file_path.startswith(d) for d in exclude_dirs):
            return False
        if any(file_path.endswith(ext) for ext in exclude_extensions):
            return False
        if include_extensions:
            return file_extension in include_extensions
        return True

    return file_filter


def load_github_repo_and_create_faiss_index(github_url, config: dict):
    """
    Load code from a GitHub repository, split into chunks, and create a FAISS index.
    """
    logger.info(f"Cloning repository from {github_url}...")
    filter_config = config.get("filter", {})
    file_filter = make_file_filter(filter_config)
    loader = GitLoader(
        clone_url=github_url,
        repo_path=".temp_repo",
        branch="main",
        file_filter=file_filter,
    )
    logger.info("Loading documents...")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    logger.info("Splitting documents into chunks...")
    splitter_cfg = config["chunking"]
    splitter_cls = SPLITTERS[splitter_cfg["splitter"]["class"]]
    splitter = splitter_cls(
        chunk_size=splitter_cfg["chunk_size"],
        chunk_overlap=splitter_cfg["chunk_overlap"],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    logger.info("Creating embeddings and FAISS index...")
    embedding_cfg = config["embedding"]
    embedding_cls = EMBEDDINGS[embedding_cfg["class"]]
    embedding_model = embedding_cls(model_name=embedding_cfg["name"])
    faiss_index = FAISS.from_documents(chunks, embedding_model)

    logger.info("FAISS index created successfully")

    index_dir = "src/data/index"
    faiss_index.save_local(index_dir)
    logger.info(f"FAISS index saved to {index_dir} directory")
    return faiss_index


# Example usage
if __name__ == "__main__":
    config = load_config("config/base.yaml")
    github_url = "https://github.com/viarotel-org/escrcpy.git"
    index_dir = "src/data/index"
    faiss_index = load_github_repo_and_create_faiss_index(github_url, config)
    print(f"FAISS index saved to {index_dir} directory")
