import json
import wandb
from datetime import datetime
from itertools import product
from typing import Dict, List

# Assuming these are your existing imports and functions
from src.indexing.build_index import build_indexing_chain
from src.retrieval.search_index import build_retrieval_chain
from src.core.loaders.config_loader import load_config
from src.eval.evaluate_retrieval import evaluate_recall_and_latency
from src.core.loaders.rag_loaders import get_index_subdir
from src.core.component_registry import INDEX_DIR
import os
import gc
import torch

EVAL_DATASET = "src/data/eval/escrcpy-commits-generated.json"
GITHUB_URL = "https://github.com/viarotel-org/escrcpy.git"  # Phase 1
PROJECT = "ragrepo-retrieval"
ENTITY = "mallahova"


def update_config(base_config: Dict, params: Dict) -> Dict:
    """Create a new config dictionary with updated parameters"""
    config = base_config.copy()
    for key, value in params.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    return config


def generate_run_name(config: Dict) -> str:
    cs = config["chunking"]["chunk_size"]
    co = config["chunking"]["chunk_overlap"]
    emb = config["embedding"]["name"].split("/")[-1]
    top_k = config["retriever"].get("top_k", "NA")
    d_w = config["retriever"]["dense_weight"]
    qe = config["retriever"].get("query_expansion", False)

    parts = [
        f"cs{cs}",
        f"co{co}",
        emb,
    ]
    if config["retriever"].get("use_reranker", False):
        reranker_name = config["retriever"]["reranker"]["name"]
        parts.append(f"rr_{reranker_name.split('/')[-1]}")
    parts.append(f"tk{top_k}")
    parts.append(f"dw{d_w}")
    parts.append(f"qe{str(qe).lower()}")
    return "_".join(parts)


def run_experiments(
    base_config: Dict,
    chunk_sizes: List[int],
    chunk_overlaps: List[int],
    embedding_models: List[Dict],
    dense_weights: List[float],
    use_reranker_values: List[bool] = [False],
    reranker_models: List[Dict] = [None],
    top_k_values: List[int] = [50],
    query_expansion_values: List[bool] = [False],
) -> List[Dict]:
    """Default configuration: Test chunking params, indexing weights and embedding models - no reranker, query expansion."""
    results = []

    for chunk_size, chunk_overlap, embed_model in product(
        chunk_sizes, chunk_overlaps, embedding_models
    ):
        embedding_params = {
            "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            "embedding": embed_model,
        }
        run_embed_config = update_config(base_config, embedding_params)
        index_dir = os.path.join(INDEX_DIR, get_index_subdir(run_embed_config))
        # Check if the index already exists
        if not os.path.exists(index_dir):
            indexing_chain = build_indexing_chain(run_embed_config)
            index = indexing_chain.invoke(GITHUB_URL)  # Assuming GITHUB_URL is defined
            del index

        for use_reranker, top_k, dense_weight, query_expansion, reranker in product(
            use_reranker_values,
            top_k_values,
            dense_weights,
            query_expansion_values,
            reranker_models,
        ):
            gc.collect()
            torch.cuda.empty_cache()
            # Update config with current parameters
            params = {
                "retriever": {
                    "use_reranker": use_reranker,
                    "top_k": top_k,
                    "dense_weight": dense_weight,
                    "query_expansion": query_expansion,
                },
                "reranker": reranker,
            }

            run_config = update_config(run_embed_config, params)
            run_name = generate_run_name(run_config)
            # CHeck if the run already exists in wandb
            api = wandb.Api()
            runs = api.runs(f"{ENTITY}/{PROJECT}")
            existing_run = None
            for run in runs:
                if run.name == run_name and run.state == "finished":
                    print(f"Run {run_name} already exists. Skipping...")
                    existing_run = run
                    break
            if not existing_run:
                wandb.init(
                    project=PROJECT,
                    name=run_name,
                    config=run_config,
                )
                retrieval_chain = build_retrieval_chain(run_config)

                if not use_reranker or reranker["class"] == "CohereRerank":
                    low_vram = False

                with open(EVAL_DATASET, "r") as f:
                    eval_data = json.load(f)
                try:
                    final_recall, average_latency = evaluate_recall_and_latency(
                        eval_data, retrieval_chain, run_config, low_vram=low_vram
                    )
                    wandb.log(
                        {"recall@10": final_recall, "avg_latency": average_latency}
                    )
                    wandb.finish()

                    results.append(
                        {
                            "run_name": run_name,
                            "recall@10": final_recall,
                            "avg_latency": average_latency,
                        }
                    )

                except Exception as e:
                    print(f"Error during evaluation: {e}")

    return results


# Example usage


def main():
    # Your base config
    base_config = {
        "filter": {
            "exclude_extensions": [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".ico",
                ".svg",
                ".bin",
                ".exe",
                ".dll",
                ".so",
                ".dylib",
                ".apk",
                ".ipk",
                ".zip",
                ".tar",
                ".gz",
                ".o",
                ".obj",
                ".class",
                ".pyc",
            ],
            "exclude_dirs": [
                "build",
                "dist",
                "node_modules",
                "vendor",
                "bin",
                "__pycache__",
                "docs/zhHans",
            ],
        },
        "chunking": {
            "chunk_size": 800,
            "chunk_overlap": 50,
            "splitter": {"class": "RecursiveCharacterTextSplitter"},
        },
        "embedding": {"class": "OpenAIEmbeddings", "name": "text-embedding-3-small"},
        "retriever": {
            "top_k": 50,
            "dense_weight": 1,
            "query_expansion": False,
            "use_reranker": False,
        },
        "reranker": {"class": "CohereRerank", "name": "rerank-v3.5", "top_n": 15},
        "generator": {"class": "ChatOpenAI", "name": "gpt-4o-mini", "temperature": 0.3},
    }

    # Define experiment parameters
    chunk_sizes = [500, 800, 1000]
    chunk_overlaps = [0, 100]
    dense_weight = [1, 0.9]
    embedding_models = [
        {
            "class": "HuggingFaceEmbeddings",
            "name": "Snowflake/snowflake-arctic-embed-l-v2.0",
        },
        {"class": "HuggingFaceEmbeddings", "name": "Alibaba-NLP/gte-multilingual-base"},
        {
            "class": "HuggingFaceEmbeddings",
            "name": "Lajavaness/bilingual-embedding-small",
        },
        {"class": "HuggingFaceEmbeddings", "name": "Alibaba-NLP/gte-modernbert-base"},
        {"class": "OpenAIEmbeddings", "name": "text-embedding-3-small"},
        {"class": "OpenAIEmbeddings", "name": "text-embedding-3-large"},
    ]

    # Search over best chunking params, embedding models and dense weights
    phase1_results = run_experiments(
        base_config, chunk_sizes, chunk_overlaps, embedding_models, dense_weight
    )

    reranker_models = [
        {"class": "HuggingFaceCrossEncoder", "name": "BAAI/bge-reranker-large"},
        {"class": "HuggingFaceCrossEncoder", "name": "BAAI/bge-reranker-v2-m3"},
        {
            "class": "HuggingFaceCrossEncoder",
            "name": "mixedbread-ai/mxbai-rerank-base-v1",
        },
        {"class": "CohereRerank", "name": "rerank-v3.5"},
    ]

    # Search over best advanced RAG techniques
    phase1_results = run_experiments(
        base_config, chunk_sizes, chunk_overlaps, embedding_models, dense_weight
    )
    top_k_values = [50]
    query_expansion_values = [False]

    # Sort by some metric (e.g., recall) and take top N
    phase1_results.sort(key=lambda x: x["results"].get("recall", 0), reverse=True)
    top_configs = phase1_results[:5]  # Take top 3 configs
    print("Top configs from Phase 1:")
    for result in top_configs:
        print(result["run_name"], result["results"])
    # Save top configs to a file for Phase 2
    with open("top_configs_phase1.json", "w") as f:
        json.dump(top_configs, f)

    # # Phase 2
    # phase2_results = phase_2_fine_tuning(
    #     base_config, top_configs, reranker_models, top_k_values, query_expansion_values
    # )


if __name__ == "__main__":
    main()


# chunk_size = [500, 800, 1000]
# chunk_overlap = [0, 100]
# embedding_model = [
#     {
#         "class": "HuggingFaceEmbeddings",
#         "name": "Snowflake/snowflake-arctic-embed-l-v2.0",
#     },
#     {"class": "HuggingFaceEmbeddings", "name": "Alibaba-NLP/gte-multilingual-base"},
#     {"class": "HuggingFaceEmbeddings", "name": "Lajavaness/bilingual-embedding-small"},
#     {"class": "HuggingFaceEmbeddings", "name": "Alibaba-NLP/gte-modernbert-base"},
#     {"class": "OpenAIEmbeddings", "name": "text-embedding-3-small"},
#     {"class": "OpenAIEmbeddings", "name": "text-embedding-3-large"},
# ]

# use_reranker = [True, False]
# # top_k=[50,100]
# top_k = [50]
# # query_expansion=[True, False]
# query_expansion = [False]

# reranker_model = [
#     {"class": "HuggingFaceCrossEncoder", "name": "BAAI/bge-reranker-large"},
#     {"class": "HuggingFaceCrossEncoder", "name": "BAAI/bge-reranker-v2-m3"},
#     {"class": "HuggingFaceCrossEncoder", "name": "mixedbread-ai/mxbai-rerank-base-v1"},
#     {"class": "CohereRerank", "name": "rerank-v3.5"},
# ]

# generator_model = [
#     {"class": "ChatOpenAI", "name": "gpt-4o-mini"},
# ]
# generator_model_temperature = [0.3]
