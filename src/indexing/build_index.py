import os
from pathlib import Path
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from src.core.component_registry import EMBEDDINGS, SPLITTERS
from src.core.config_loader import load_config


def make_file_filter(config):
    include_extensions = set(config.get("include_extensions", []))
    exclude_extensions = set(config.get("exclude_extensions", []))
    exclude_dirs = set(config.get("exclude_dirs", []))

    def file_filter(file_path: str) -> bool:
        file_path = os.path.join(*file_path.split(os.sep)[1:])
        file_extension = "." + file_path.split(".")[-1] if "." in file_path else ""

        if any(file_path.startswith(d) for d in exclude_dirs):
            return False
        if any(file_path.endswith(ext) for ext in exclude_extensions):
            return False
        return file_extension in include_extensions

    return file_filter


def load_github_repo_and_create_faiss_index(github_url, config: dict):
    """
    Load code from a GitHub repository, split into chunks, and create a FAISS index.
    """
    print(f"Cloning repository from {github_url}...")
    filter_config = config.get("filter", {})
    file_filter = make_file_filter(filter_config)
    loader = GitLoader(
        clone_url=github_url,
        repo_path=".temp_repo",
        branch="main",
        file_filter=file_filter,
    )
    print("Loading documents...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents into chunks...")

    splitter_cfg = config["chunking"]
    splitter_cls = SPLITTERS[splitter_cfg["splitter"]["class"]]
    splitter = splitter_cls(
        chunk_size=splitter_cfg["chunk_size"],
        chunk_overlap=splitter_cfg["chunk_overlap"],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings and FAISS index...")
    embedding_cfg = config["embedding"]
    embedding_cls = EMBEDDINGS[embedding_cfg["class"]]
    embedding_model = embedding_cls(model_name=embedding_cfg["name"])
    faiss_index = FAISS.from_documents(chunks, embedding_model)

    print("FAISS index created successfully")
    return faiss_index


def search_repository(faiss_index, query, k=5):
    """
    Search the FAISS index for similar chunks.
    """
    results = faiss_index.similarity_search_with_score(query, k=k)
    return results


if __name__ == "__main__":
    github_url = "https://github.com/viarotel-org/escrcpy.git"
    config = load_config("config/base.yaml")
    faiss_index = load_github_repo_and_create_faiss_index(github_url, config)

    query = "How does the repository handle IPv6 addresses in ADB commands?"
    results = search_repository(faiss_index, query)

    print(f"\nSearch results for query: '{query}'")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)

    index_dir = "src/data/index"
    faiss_index.save_local(index_dir)
    print(f"FAISS index saved to {index_dir} directory")
