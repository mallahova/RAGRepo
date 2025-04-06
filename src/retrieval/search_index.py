from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from src.core.config_loader import load_config
from src.core.component_registry import EMBEDDINGS, RERANKERS, INDEX_DIR
from langchain.retrievers.document_compressors import CrossEncoderReranker


def load_index(config: dict):
    """
    Loads a FAISS index from the specified directory.
    """
    embedding_cfg = config["embedding"]
    embedding_cls = EMBEDDINGS[embedding_cfg["class"]]
    embedding_model = embedding_cls(
        model_name=embedding_cfg["name"], model_kwargs={"trust_remote_code": True}
    )

    return (
        FAISS.load_local(
            INDEX_DIR, embeddings=embedding_model, allow_dangerous_deserialization=True
        ),
        embedding_model,
    )


def build_retrieval_chain(config):
    """
    Returns a LangChain Runnable that retrieves documents from a FAISS index.
    Optionally applies reranking if specified in the config.

    """

    def retrieve_faiss_top_k(inputs):
        query = inputs["query"]
        faiss_index = inputs.get("faiss_index", None)
        if faiss_index is None:
            faiss_index, embedding_model = load_index(config)
        top_k_val = config["retriever"]["top_k"]
        docs = faiss_index.similarity_search(query, k=top_k_val)
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
        return RunnableLambda(retrieve_faiss_top_k) | RunnableLambda(rerank_docs)

    else:
        return RunnableLambda(retrieve_faiss_top_k)


if __name__ == "__main__":
    config = load_config("config/base.yaml")
    query = "How does the repository handle IPv6 addresses in ADB commands?"
    retrieval_chain = build_retrieval_chain(config)
    graph = retrieval_chain.get_graph()
    print(graph.print_ascii())
    results = retrieval_chain.invoke({"query": query, "faiss_index": None})["docs"]

    print(f"\nSearch results for query: '{query}'")
    for doc in results:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)
