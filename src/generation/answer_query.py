import argparse
from src.generation.answer_generator import AnswerGenerator
from src.core.loaders.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate an answer from retrieved documents."
    )
    parser.add_argument(
        "--config_path",
        default="config/base.yaml",
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query to answer based on retrieved documents.",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    generator = AnswerGenerator(config)
    answer = generator.run(args.query)

    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
