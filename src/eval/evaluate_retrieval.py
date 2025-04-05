import json
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from src.core.config_loader import load_config
from src.core.component_registry import (
    EMBEDDINGS,
)
from src.retrieval.search_index import search_index

import json


def evaluate_recall(eval_data, faiss_index, config, k=10):
    """
    Evaluates average Recall@k over filenames using direct FAISS search function.

    Args:
        eval_data: List of {"question": ..., "files": [...]} examples.
        faiss_index: Loaded FAISS index.
        config: Configuration dictionary with retriever settings.
        k: Number of top results to retrieve (default: 10)

    Returns:
        Recall@k as a float.
    """
    total_recall = 0.0
    num_queries = 0

    for item in tqdm(eval_data, desc=f"Evaluating Recall@{k}"):
        question = item["question"]
        gold_files = set(item["files"])
        if not gold_files:
            continue  # skip queries with no gold files

        results = search_index(faiss_index, question, config, top_k=k)
        retrieved_files = set(
            doc.metadata.get("source") for doc, _ in results if "source" in doc.metadata
        )

        matched = gold_files & retrieved_files
        recall = len(matched) / len(gold_files)
        total_recall += recall
        num_queries += 1

    if num_queries == 0:
        return 0.0
    final_recall = total_recall / num_queries
    print(f"Recall@{k}: {final_recall:.3f}")
    return final_recall


# Example usage
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

    with open("src/data/eval/escrcpy-commits-generated.json") as f:
        eval_data = json.load(f)
    evaluate_recall(eval_data, faiss_index, config, k=10)
