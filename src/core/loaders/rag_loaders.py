from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from src.core.component_registry import (
    build_embedding_model,
    RERANKERS,
    INDEX_DIR,
    GENERATORS,
)
import pickle
import os


def get_index_subdir(config):
    """
    Returns the directory where the index is stored.
    """

    cs = config["chunking"]["chunk_size"]
    co = config["chunking"]["chunk_overlap"]
    emb = config["embedding"]["name"].split("/")[-1]
    parts = [
        f"cs{cs}",
        f"co{co}",
        emb,
    ]
    embed_config_name = "_".join(parts)
    return embed_config_name


def load_index(config: dict):
    """
    Loads the FAISS and BM25 retrievers and returns an EnsembleRetriever.
    """
    embedding_model = build_embedding_model(config["embedding"])

    # Load FAISS
    embed_config_name = get_index_subdir(config)
    faiss_index = FAISS.load_local(
        os.path.join(INDEX_DIR, embed_config_name, "faiss"),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = faiss_index.as_retriever(
        search_kwargs={"k": config["retriever"]["top_k"]}
    )

    # Load BM25
    bm25_path = os.path.join(INDEX_DIR, embed_config_name, "bm25.pkl")
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 3
    # Build EnsembleRetriever
    dense_weight = config["retriever"]["dense_weight"]
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[dense_weight, 1 - dense_weight],  # [dense, sparse]
    )

    return hybrid_retriever


def load_generator(config):
    """
    Loads the generator model based on the configuration.
    """
    generator_cfg = config["generator"]
    generator_class = GENERATORS[generator_cfg["class"]]
    generator = generator_class(model_name=generator_cfg["name"])
    return generator


def load_reranker(config):
    """
    Loads the reranker model based on the configuration.
    """
    reranker_cfg = config["reranker"]
    reranker_class = RERANKERS[reranker_cfg["class"]]
    reranker_model = reranker_class(model_name=reranker_cfg["name"])
    return reranker_model


def load_reranker(config):
    reranker_cfg = config["reranker"]
    reranker_class = reranker_cfg["class"]

    if reranker_class == "CohereRerank":
        reranker_class = RERANKERS[reranker_cfg["class"]]
        reranker_model = reranker_class(model=reranker_cfg["name"])

    else:
        reranker_class = RERANKERS[reranker_cfg["class"]]
        reranker_model = reranker_class(model_name=reranker_cfg["name"])
    return reranker_model
