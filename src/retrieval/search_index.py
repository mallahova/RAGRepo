from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableLambda
from src.core.config_loader import load_config
from src.core.component_registry import EMBEDDINGS, RERANKERS, INDEX_DIR


def load_index(config: dict):
    """
    Loads a FAISS index from the specified directory.
    """
    embedding_cfg = config["embedding"]
    embedding_cls = EMBEDDINGS[embedding_cfg["class"]]
    embedding_model = embedding_cls(
        model_name=embedding_cfg["name"], model_kwargs={"trust_remote_code": True}
    )

    return FAISS.load_local(
        INDEX_DIR, embeddings=embedding_model, allow_dangerous_deserialization=True
    )


def build_retrieval_chain(faiss_index, config, top_k=None):
    """
    Returns a LangChain Runnable that retrieves documents from a FAISS index.
    Optionally applies reranking if specified in the config.

    """
    top_k_val = config["retriever"]["top_k"] if top_k is None else top_k

    def retrieve_faiss_top_k(query):
        return faiss_index.similarity_search_with_score(query, k=top_k_val)

    return RunnableLambda(retrieve_faiss_top_k)


if __name__ == "__main__":
    config = load_config("config/base.yaml")
    query = "How does the repository handle IPv6 addresses in ADB commands?"

    faiss_index = load_index(config)
    retrieval_chain = build_retrieval_chain(faiss_index, config)
    graph = retrieval_chain.get_graph()
    print(graph.print_ascii())
    results = retrieval_chain.invoke(query)

    print(f"\nSearch results for query: '{query}'")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)
