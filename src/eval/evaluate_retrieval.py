from src.core.loaders.config_loader import load_config
from src.retrieval.search_index import build_retrieval_chain
from src.core.loaders.rag_loaders import load_index
import json
from tqdm import tqdm
import gc
import torch
import argparse
import time


def get_top_k_unique_filenames(results, k=10):
    seen = set()
    top_k_files = []

    for doc in results:
        filename = doc.metadata.get("source")
        if filename and filename not in seen:
            seen.add(filename)
            top_k_files.append(filename)
        if len(top_k_files) == k:
            break

    return top_k_files


def evaluate_recall_and_latency(eval_data, retrieval_chain, config, low_vram=False):
    """
    Evaluates average Recall@10 and latency (in seconds) over filenames for a given retrieval chain and evaluation data.
    low_vram: If True, uses a low VRAM mode for evaluation. (Only one model loaded at a time)
    """
    total_recall = 0.0
    total_latency = 0.0
    num_queries = 0

    if not low_vram:
        retriever = load_index(config)
    else:
        retriever = None

    for item in tqdm(eval_data, desc="Evaluating Recall@10"):
        query = item["question"]
        gold_files = set(item["files"])
        if not gold_files:
            continue

        while True:
            try:
                start_time = time.time()
                results = retrieval_chain.invoke(
                    {"query": query, "retriever": retriever}
                )["docs"]
                break
            except Exception as e:  # rate limit error
                print(f"Rate limit error: {e}. Retrying...")
                time.sleep(60)
        end_time = time.time()

        latency = end_time - start_time
        total_latency += latency

        retrieved_files = get_top_k_unique_filenames(results, k=10)
        # print(f"{len(retrieved_files)} retrieved files")
        retrieved_files = set(retrieved_files)
        matched = gold_files & retrieved_files
        recall = len(matched) / len(gold_files)
        total_recall += recall
        num_queries += 1

        if low_vram:
            gc.collect()
            torch.cuda.empty_cache()

    if num_queries == 0:
        return 0.0, 0.0

    final_recall = total_recall / num_queries
    average_latency = total_latency / num_queries
    print(f"Recall@10: {final_recall:.3f} | Avg Latency: {average_latency:.3f}s")
    return final_recall, average_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Recall@10 for a RAG system.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="src/data/eval/escrcpy-commits-generated.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/base.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--low_vram", default=False, action="store_true", help="Enable low VRAM mode."
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    retrieval_chain = build_retrieval_chain(config)

    with open(args.dataset_path, "r") as f:
        eval_data = json.load(f)

    evaluate_recall_and_latency(
        eval_data, retrieval_chain, config=config, low_vram=args.low_vram
    )
