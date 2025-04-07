import json
import time
import torch
import gc
import argparse
from tqdm import tqdm

from src.retrieval.retrieval_chain_builder import RetrievalChainBuilder
from src.core.loaders.rag_loaders import load_index


class RetrievalEvaluator:
    def __init__(self, config: dict, low_vram: bool = False):
        self.config = config
        self.low_vram = low_vram
        self.retriever = None if low_vram else load_index(self.config)
        self.retrieval_chain = RetrievalChainBuilder(self.config).build_chain()

    def evaluate(self, eval_data: list, k: int = 10):
        total_recall = 0.0
        total_latency = 0.0
        num_queries = 0

        for item in tqdm(eval_data, desc="Evaluating Recall@10"):
            query = item["question"]
            gold_files = set(item["files"])
            if not gold_files:
                continue

            while True:
                try:
                    start_time = time.time()
                    result = self.retrieval_chain.invoke(
                        {"query": query, "retriever": self.retriever}
                    )
                    break
                except Exception as e:
                    print(f"Rate limit error: {e}. Retrying...")
                    time.sleep(60)

            latency = time.time() - start_time
            total_latency += latency

            retrieved_files = self._get_top_k_unique_filenames(result["docs"], k)
            recall = len(gold_files & set(retrieved_files)) / len(gold_files)

            total_recall += recall
            num_queries += 1

            if self.low_vram:
                gc.collect()
                torch.cuda.empty_cache()

        if num_queries == 0:
            return 0.0, 0.0

        final_recall = total_recall / num_queries
        average_latency = total_latency / num_queries

        print(f"Recall@10: {final_recall:.3f} | Avg Latency: {average_latency:.3f}s")
        return final_recall, average_latency

    def _get_top_k_unique_filenames(self, docs, k: int):
        seen = set()
        top_k_files = []

        for doc in docs:
            filename = doc.metadata.get("source")
            if filename and filename not in seen:
                seen.add(filename)
                top_k_files.append(filename)
            if len(top_k_files) == k:
                break

        return top_k_files


def main():
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
        "--low_vram",
        default=False,
        action="store_true",
        help="Enable low VRAM mode.",
    )
    args = parser.parse_args()

    with open(args.dataset_path, "r") as f:
        eval_data = json.load(f)

    evaluator = RetrievalEvaluator(args.config_path, low_vram=args.low_vram)
    evaluator.evaluate(eval_data)


if __name__ == "__main__":
    main()
