from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.loaders.config_loader import load_config
from src.core.loaders.rag_loaders import load_index, load_generator, load_reranker
from src.retrieval.search_index import build_retrieval_chain
import gc
import torch


def generate(query: str, documents: list) -> str:
    """
    Generates a summary answer based on the provided query and retrieved documents.
    """
    llm = load_generator(config)
    prompt = """You are part of an information system that summarises related code parts.
You answer a query using the textual content from the documents retrieved for the
following query.
You build the summary answer based only on quoting information from the documents.
You should reference the documents you used to support your answer.
###
Original Query: "{{query}}"
Retrieved Documents: {{documents}}
Summary Answer:"""
    prompt_template = PromptTemplate.from_template(prompt)
    runnable = RunnableSequence(
        [
            RunnableLambda(lambda x: x["query"]),
            prompt_template | llm | StrOutputParser(),
        ]
    )
    return runnable.invoke({"query": query, "documents": documents})["output"]


if __name__ == "main":
    config = load_config("config/base.yaml")
    query = "How does the repository handle IPv6 addresses in ADB commands?"
    retrieval_chain = build_retrieval_chain(config)
    results = retrieval_chain.invoke({"query": query, "retriever": None})["docs"]
    generated_answer = generate(query, results)
    print("\nGenerated Answer:")
    print(generated_answer)
    print("=" * 80)
    print(f"\nSearch results for query: '{query}'")
    for doc in results[:3]:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)
