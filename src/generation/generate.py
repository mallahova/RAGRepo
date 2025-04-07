import argparse
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.loaders.config_loader import load_config
from src.core.loaders.rag_loaders import load_index, load_generator, load_reranker
from src.retrieval.search_index import build_retrieval_chain


def generate(query: str, documents: list, config) -> str:
    """
    Generates a summary answer based on the provided query and retrieved documents.
    """
    llm = load_generator(config)
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
    runnable = prompt_template | llm | StrOutputParser()

    return runnable.invoke({"query": query, "documents": documents_str})


def main():
    parser = argparse.ArgumentParser(
        description="Generate an answer from retrieved documents."
    )
    parser.add_argument(
        "--config_path",
        required=False,
        default="config/base.yaml",
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--query", required=True, help="Query to answer based on retrieved documents."
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    query = args.query

    retrieval_chain = build_retrieval_chain(config)
    results = retrieval_chain.invoke({"query": query, "retriever": None})["docs"]

    print("=" * 80)
    print(f"\nSearch results for query: '{query}'")
    for doc in results[:3]:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)

    generated_answer = generate(query, results[:10], config)
    print("\nGenerated Answer:")
    print(generated_answer)


if __name__ == "__main__":
    main()
