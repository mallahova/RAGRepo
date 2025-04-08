import json
import time
import torch
import gc
import argparse
from tqdm import tqdm

from src.retrieval.retrieval_chain_builder import RetrievalChainBuilder
from src.core.loaders.rag_loaders import load_index


class RetrievalEvaluator:
    """
    Evaluates the performance of a retrieval pipeline using Recall@k and latency.
    """

    def __init__(self, config: dict, low_vram: bool = False):
        """
        Args:
            config (dict): Retrieval and model configuration.
            low_vram (bool): If True, only one model is loaded to VRAM at a time.
        """
        self.config = config
        self.low_vram = low_vram
        self.retriever = None if low_vram else load_index(self.config)
        self.retrieval_chain = RetrievalChainBuilder(self.config).build_chain()

    def evaluate(self, eval_data: list, k: int = 10):
        """
        Evaluates recall@k and average retrieval latency.

        Args:
            eval_data (list): List of dicts with keys "question" and "files".
            k (int): Top-k documents to consider for recall.

        Returns:
            Tuple[float, float]: (Recall@k, Average Latency)
        """
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
