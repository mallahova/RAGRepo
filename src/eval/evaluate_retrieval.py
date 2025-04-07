import json
import argparse

from src.core.loaders.config_loader import load_config
from src.eval.retrieval_evaluator import RetrievalEvaluator


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
    config = load_config(args.config_path)
    evaluator = RetrievalEvaluator(config, low_vram=args.low_vram)
    evaluator.evaluate(eval_data)


if __name__ == "__main__":
    main()
