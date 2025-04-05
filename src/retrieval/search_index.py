from langchain_community.vectorstores import FAISS
from src.core.config_loader import load_config
from src.core.component_registry import (
    EMBEDDINGS,
)


def search_index(faiss_index, query, config, top_k=None):
    """
    Search the FAISS index for similar chunks.
    """
    top_k = config["retriever"]["top_k"] if top_k is None else top_k
    results = faiss_index.similarity_search_with_score(query, k=top_k)
    return results


if __name__ == "__main__":
    config = load_config("config/base.yaml")
    embedding_cfg = config["embedding"]
    embedding_cls = EMBEDDINGS[embedding_cfg["class"]]
    embedding_model = embedding_cls(model_name=embedding_cfg["name"])

    github_url = "https://github.com/viarotel-org/escrcpy.git"
    index_dir = "src/data/index"
    faiss_index = FAISS.load_local(
        index_dir, embeddings=embedding_model, allow_dangerous_deserialization=True
    )

    query = "How does the repository handle IPv6 addresses in ADB commands?"
    results = search_index(faiss_index, query, config)

    print(f"\nSearch results for query: '{query}'")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)
