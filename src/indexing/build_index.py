from src.indexing.index_builder import IndexBuilder
from src.core.loaders.config_loader import load_config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 index from GitHub repo"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/base.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--repo_url",
        type=str,
        default="https://github.com/viarotel-org/escrcpy.git",
        help="GitHub repository URL",
    )

    args = parser.parse_args()
    config = load_config(args.config_path)
    builder = IndexBuilder(config)
    indexing_chain = builder.build_indexing_chain()

    graph = indexing_chain.get_graph()
    print(graph.print_ascii())

    indexing_chain.invoke(args.repo_url)
    print(f"Indexing completed for {args.repo_url}")
