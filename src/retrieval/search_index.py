from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.loaders.config_loader import load_config
from src.core.loaders.rag_loaders import load_index, load_generator, load_reranker


def build_retrieval_chain(config):
    """
    Returns a LangChain Runnable that retrieves documents using a hybrid retriever.
    Optionally applies reranking if specified in the config.
    """

    def expand_query(inputs):
        query = inputs["query"]
        hybrid_retriever = inputs.get("retriever", None)
        llm = load_generator(config)
        # prompt = PromptTemplate.from_template(
        #     "Expand this code-related search query using synonyms and technical terms. Return only the expanded query.\nQuery: {query}\nExpanded:"
        # )
        prompt = PromptTemplate.from_template(
            """You are a code search query expansion expert. Your task is to expand and improve the given query
        to make it more detailed and comprehensive. Include relevant synonyms and related terms, class and function names to improve retrieval.
        Return only the expanded query without any explanations or additional text.

        Original query: {query}

        Expanded query:"""
        )
        expanded_query = prompt | llm | StrOutputParser()
        expanded_query = expanded_query.invoke({"query": query})
        print(f"query: { query }")
        print(f"Expanded query: {expanded_query}")
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


if __name__ == "__main__":
    config = load_config("config/base.yaml")
    query = "How does the repository handle IPv6 addresses in ADB commands?"
    retrieval_chain = build_retrieval_chain(config)
    graph = retrieval_chain.get_graph()
    print(graph.print_ascii())
    results = retrieval_chain.invoke({"query": query, "retriever": None})["docs"]

    print(f"\nSearch results for query: '{query}'")
    for doc in results[:3]:
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("=" * 80)
