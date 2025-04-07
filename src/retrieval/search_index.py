import argparse
from src.core.loaders.config_loader import load_config
from src.retrieval.retrieval_chain_builder import RetrievalChainBuilder


def main():
    parser = argparse.ArgumentParser(description="Run retrieval on a codebase.")
    parser.add_argument("--query", required=True, help="Query to search the index with")
    parser.add_argument(
        "--config_path", default="config/base.yaml", help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    builder = RetrievalChainBuilder(config)
    retrieval_chain = builder.build_chain()

    graph = retrieval_chain.get_graph()
    print(graph.print_ascii())

    result = retrieval_chain.invoke({"query": args.query, "retriever": None})["docs"]

    print(f"\nSearch results for query: '{args.query}'")
    for doc in result[:10]:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("=" * 80)


if __name__ == "__main__":
    main()
