import argparse

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.loaders.config_loader import load_config
from src.core.loaders.rag_loaders import load_index, load_generator, load_reranker


def build_retrieval_chain(config):
    """
    Returns a LangChain Runnable that retrieves documents using a hybrid retriever.
    Optionally applies query expanding and reranking if specified in the config.
    """

    def expand_query(inputs):
        query = inputs["query"]
        hybrid_retriever = inputs.get("retriever", None)
        llm = load_generator(config)
        prompt = PromptTemplate.from_template(
            "You are a helpful coding expert. Provide an example short answer to the given question, that might be found in a code repository. Question: {query}"
        )
        expanded_query_chain = prompt | llm | StrOutputParser()
        expanded_query = expanded_query_chain.invoke({"query": query})
        expanded_query = query + " " + expanded_query
        return {"retriever": hybrid_retriever, "query": expanded_query}

    def retrieve_hybrid_top_k(inputs):
        query = inputs["query"]
        hybrid_retriever = inputs.get("retriever", None)

        if hybrid_retriever is None:
            hybrid_retriever = load_index(config)

        top_k_val = config["retriever"]["top_k"]
        docs = hybrid_retriever.invoke(query)[:top_k_val]
        return {"docs": docs, "query": query}

    def rerank_docs(inputs):
        docs = inputs["docs"]
        query = inputs["query"]
        reranker_model = load_reranker(config)
        reranker_cfg = config["reranker"]
        reranker_class = reranker_cfg["class"]

        if reranker_class == "CohereRerank":
            compressor = reranker_model
        else:
            compressor = CrossEncoderReranker(
                model=reranker_model, top_n=reranker_cfg["top_n"]
            )
        reranked_docs = compressor.compress_documents(docs, query)
        return {"docs": reranked_docs, "query": query}

    chain = []
    if config["retriever"]["query_expansion"]:
        chain.append(RunnableLambda(expand_query))
    chain.append(RunnableLambda(retrieve_hybrid_top_k))
    if config["retriever"]["use_reranker"]:
        chain.append(RunnableLambda(rerank_docs))

    if len(chain) == 1:
        return chain[0]
    retrieval_chain = RunnableSequence(*chain)
    return retrieval_chain


def main():
    parser = argparse.ArgumentParser(description="Run retrieval on a codebase.")
    parser.add_argument("--query", required=True, help="Query to search the index with")
    parser.add_argument(
        "--config_path",
        required=False,
        default="config/base.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    retrieval_chain = build_retrieval_chain(config)

    graph = retrieval_chain.get_graph()
    print(graph.print_ascii())

    results = retrieval_chain.invoke({"query": args.query, "retriever": None})["docs"]

    print(f"\nSearch results for query: '{args.query}'")
    for doc in results[:10]:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("=" * 80)


if __name__ == "__main__":
    main()
