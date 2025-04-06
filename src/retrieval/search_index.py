from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
from src.core.config_loader import load_config
from src.core.component_registry import build_embedding_model, RERANKERS, INDEX_DIR
from langchain.retrievers.document_compressors import CrossEncoderReranker
import pickle
import os


def load_index(config: dict):
    """
    Loads the FAISS and BM25 retrievers and returns an EnsembleRetriever.
    """
    embedding_model = build_embedding_model(config["embedding"])

    # Load FAISS
    faiss_index = FAISS.load_local(
        os.path.join(INDEX_DIR, "faiss"),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = faiss_index.as_retriever(
        search_kwargs={"k": config["retriever"]["top_k"]}
    )

    # Load BM25
    bm25_path = os.path.join(INDEX_DIR, "bm25.pkl")
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = config["retriever"]["top_k"]
    # Build EnsembleRetriever
    dense_weight = config["retriever"]["dense_weight"]
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[dense_weight, 1 - dense_weight],  # [dense, sparse]
    )

    return hybrid_retriever


def build_retrieval_chain(config):
    """
    Returns a LangChain Runnable that retrieves documents using a hybrid retriever.
    Optionally applies reranking if specified in the config.
    """

    def retrieve_hybrid_top_k(inputs):
        query = inputs["query"]
        hybrid_retriever = inputs.get("retriever", None)

        if hybrid_retriever is None:
            hybrid_retriever = load_index(config)

        top_k_val = config["retriever"]["top_k"]
        docs = hybrid_retriever.invoke(query)[:top_k_val]

        return {"docs": docs, "query": query}

    def rerank_docs(inputs):
        docs = inputs["docs"]
        query = inputs["query"]
        reranker_cfg = config["reranker"]
        reranker_class = RERANKERS[reranker_cfg["class"]]
        reranker_model = reranker_class(model_name=reranker_cfg["name"])
        compressor = CrossEncoderReranker(
            model=reranker_model, top_n=reranker_cfg["top_n"]
        )
        reranked_docs = compressor.compress_documents(docs, query)

        return {"docs": reranked_docs, "query": query}

    if "reranker" in config:
        return RunnableLambda(retrieve_hybrid_top_k) | RunnableLambda(rerank_docs)

    else:
        return RunnableLambda(retrieve_hybrid_top_k)


if __name__ == "__main__":
    config = load_config("config/base.yaml")
    query = "How does the repository handle IPv6 addresses in ADB commands?"
    retrieval_chain = build_retrieval_chain(config)
    graph = retrieval_chain.get_graph()
    print(graph.print_ascii())
    results = retrieval_chain.invoke({"query": query, "retriever": None})["docs"]

    print(f"\nSearch results for query: '{query}'")
    for doc in results:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)
