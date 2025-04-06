import json
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from src.core.config_loader import load_config
from src.core.component_registry import (
    EMBEDDINGS,
)
from src.retrieval.search_index import load_index, build_retrieval_chain
from src.indexing.build_index import build_indexing_chain

import json
import gc
import torch


def evaluate_recall(eval_data, retrieval_chain, low_vram=False):
    """
    Evaluates average Recall@10 over filenames for a given retrieval chain and evaluation data.
    """
    total_recall = 0.0
    num_queries = 0
    if not low_vram:
        faiss_index, _ = load_index(config)
    for item in tqdm(eval_data, desc=f"Evaluating Recall@10"):
        query = item["question"]
        gold_files = set(item["files"])
        if not gold_files:
            continue  # skip queries with no gold files
        results = retrieval_chain.invoke({"query": query, "faiss_index": faiss_index})[
            "docs"
        ]
        results = results[:10]  # Get top 10 results
        retrieved_files = set(
            doc.metadata.get("source") for doc in results if "source" in doc.metadata
        )

        matched = gold_files & retrieved_files
        recall = len(matched) / len(gold_files)
        total_recall += recall
        num_queries += 1
        if low_vram:
            gc.collect()
            torch.cuda.empty_cache()

    if num_queries == 0:
        return 0.0
    final_recall = total_recall / num_queries
    print(f"Recall@10: {final_recall:.3f}")
    return final_recall


# Example usage
if __name__ == "__main__":
    config = load_config("config/base.yaml")
    retrieval_chain = build_retrieval_chain(config)
    with open("src/data/eval/escrcpy-commits-generated.json") as f:
        eval_data = json.load(f)
    evaluate_recall(eval_data, retrieval_chain, low_vram=True)
