import torch
import gc

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.loaders.rag_loaders import load_index, load_generator, load_reranker


class RetrievalChainBuilder:
    def __init__(self, config: dict):
        self.config = config

    def build_chain(self):
        steps = []

        if self.config["retriever"]["query_expansion"]:
            steps.append(RunnableLambda(self._expand_query))
        steps.append(RunnableLambda(self._retrieve_hybrid_top_k))
        if self.config["retriever"]["use_reranker"]:
            steps.append(RunnableLambda(self._rerank_docs))

        if len(steps) == 1:
            return steps[0]
        return RunnableSequence(*steps)

    def _expand_query(self, inputs):
        query = inputs["query"]
        retriever = inputs.get("retriever")

        llm = load_generator(self.config)
        prompt = PromptTemplate.from_template(
            "You are a helpful coding expert. Provide an example short answer to the given question, that might be found in a code repository. Question: {query}"
        )
        chain = prompt | llm | StrOutputParser()
        expansion = chain.invoke({"query": query})
        return {"retriever": retriever, "query": f"{query} {expansion}"}

    def _retrieve_hybrid_top_k(self, inputs):
        query = inputs["query"]
        retriever = inputs.get("retriever") or load_index(self.config)

        top_k = self.config["retriever"]["top_k"]
        docs = retriever.invoke(query)[:top_k]

        del retriever
        gc.collect()
        torch.cuda.empty_cache()

        return {"docs": docs, "query": query}

    def _rerank_docs(self, inputs):
        docs = inputs["docs"]
        query = inputs["query"]
        reranker_cfg = self.config["reranker"]
        reranker_model = load_reranker(self.config)

        if reranker_cfg["class"] == "CohereRerank":
            compressor = reranker_model
        else:
            compressor = CrossEncoderReranker(
                model=reranker_model, top_n=reranker_cfg["top_n"]
            )

        reranked = compressor.compress_documents(docs, query)
        gc.collect()
        torch.cuda.empty_cache()

        return {"docs": reranked, "query": query}
