import os
import re
import pickle
import logging
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.component_registry import build_embedding_model, INDEX_DIR
from src.indexing.code_chunker import detect_language_from_path
from src.core.loaders.rag_loaders import get_index_subdir
from src.indexing.file_filter import make_file_filter

logging.basicConfig(level=logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builds FAISS and BM25 indexes from a GitHub repo using code-aware chunking.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Contains settings for filtering, chunking, and embedding.
        """
        self.config = config
        self.file_filter = make_file_filter(config.get("filter", {}))
        self.splitter_cfg = config["chunking"]
        self.embedding_model = build_embedding_model(config["embedding"])
        self.index_subdir = get_index_subdir(config)

    def build_indexing_chain(self):
        return (
            RunnableLambda(self._fetch_and_clone_repo_docs)
            | RunnableLambda(self._split_docs_into_chunks)
            | RunnableLambda(self._embed_chunks_and_create_index)
            | RunnableLambda(self._save_indexes)
        )

    def _get_repo_dir(self, github_url: str) -> str:
        path = urlparse(github_url).path
        repo_name = Path(path).stem
        return f".repos/{repo_name}"

    def _strip_links(self, text: str) -> str:
        return re.sub(r"https?://\S+", "", text)

    def _fetch_and_clone_repo_docs(self, github_url: str):
        repo_path = self._get_repo_dir(github_url)
        loader = GitLoader(
            clone_url=github_url,
            repo_path=repo_path,
            branch="main",
            file_filter=self.file_filter,
        )
        return loader.load()

    def _split_docs_into_chunks(self, docs):
        all_chunks = []
        for doc in docs:
            cleaned_text = self._strip_links(doc.page_content)
            doc.page_content = cleaned_text
            source_path = doc.metadata.get("source", "")
            lang = detect_language_from_path(source_path)

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=self.splitter_cfg["chunk_size"],
                chunk_overlap=self.splitter_cfg["chunk_overlap"],
            )

            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} code-aware chunks")
        return all_chunks

    def _embed_chunks_and_create_index(self, chunks):
        faiss_index = FAISS.from_documents(chunks, self.embedding_model)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        return faiss_index, bm25_retriever

    def _save_indexes(self, indexes):
        faiss_index, bm25_retriever = indexes
        index_path = os.path.join(INDEX_DIR, self.index_subdir)

        faiss_index.save_local(os.path.join(index_path, "faiss"))

        bm25_path = os.path.join(index_path, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

        logger.info("Saved FAISS and BM25 indexes.")
        return indexes
