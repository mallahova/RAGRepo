import argparse
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.loaders.rag_loaders import load_generator
from src.retrieval.retrieval_chain_builder import RetrievalChainBuilder


class AnswerGenerator:
    def __init__(self, config: dict):
        self.config = config
        retrieval_builder = RetrievalChainBuilder(self.config)
        self.retrieval_chain = retrieval_builder.build_chain()
        self.llm = load_generator(self.config)

    def run(self, query: str) -> str:
        docs = self._retrieve_documents(query)
        self._print_top_docs(query, docs)
        return self._generate_answer(query, docs[:10])

    def _retrieve_documents(self, query: str) -> list:
        result = self.retrieval_chain.invoke({"query": query, "retriever": None})
        return result["docs"]

    def _generate_answer(self, query: str, documents: list) -> str:
        formatted_docs = [
            f"(Source: {doc.metadata.get('source', 'Unknown')})\n{doc.page_content}"
            for doc in documents
        ]
        documents_str = "\n\n".join(formatted_docs)

        prompt = """You are part of an information system that summarises related code parts.
You answer a query using the textual content from the documents retrieved for the
following query.
You build the summary answer based only on quoting information from the documents.
You should reference the documents you used to support your answer.
###
Original Query: {query}
Retrieved Documents: {documents}
Summary Answer:"""

        prompt_template = PromptTemplate.from_template(prompt)
        runnable = prompt_template | self.llm | StrOutputParser()
        return runnable.invoke({"query": query, "documents": documents_str})

    def _print_top_docs(self, query: str, docs: list):
        print("=" * 80)
        print(f"\nSearch results for query: '{query}'")
        for doc in docs[:3]:
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print("=" * 80)


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

    generator = AnswerGenerator(args.config_path)
    answer = generator.run(args.query)

    print("\nGenerated Answer:")
    print(answer)


if __name__ == "__main__":
    main()
