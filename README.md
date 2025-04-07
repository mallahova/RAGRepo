
# RAGRepo 🔍

**A flexible Retrieval-Augmented Generation (RAG) system for exploring and answering questions about code repositories.**  
Designed for real-world trade-offs between speed, accuracy, and openness.

---

## 🚀 Quickstart

```bash
conda create -n ragrepo python=3.10.16 -y
conda activate ragrepo
python -m pip install -r requirements.txt
```

If you're using APIs (OpenAI, Cohere, Google, etc.), add your keys in a `.env` file at the project root:

```
OPENAI_API_KEY=...
COHERE_API_KEY=...
```

---

## 🧠 Overview

RAGRepo offers a full Retrieval-Augmented Generation pipeline for question answering over GitHub codebases. Key features include:

- **Direct GitHub indexing** – build indexes straight from repo URLs  
- **Hybrid retrieval** – combines FAISS and BM25 with adjustable weighting  
- **Query expansion** and **reranking** options  
- **Modular config system** – define LLMs, embeddings, and retrievers via YAML (e.g. [config/base.yaml](https://github.com/mallahova/RAGRepo/blob/main/config/base.yaml)  
)  
- **Custom wrapper support** – e.g. [src/core/custom_wrappers/gemini_wrapper.py](https://github.com/mallahova/RAGRepo/blob/main/src/core/custom_wrappers/gemini_wrapper.py)  
` for non-standard providers  
- **Evaluation tools** – measure retrieval with Recall@10 and latency metrics


---

## 📊 Evaluation Dataset

This dataset benchmarks retrieval on the `https://github.com/viarotel-org/escrcpy` repo: [src/data/eval/escrcpy-commits-generated.json](https://github.com/mallahova/RAGRepo/blob/main/src/data/eval/escrcpy-commits-generated.json)  
 
Includes natural language queries and file-level answers.  
Note: It was auto-generated and may contain minor inconsistencies.

---

## 📈 Reports & Trade-offs

Performance analysis and experiment results are available in:[src/eval/notebooks/rag_setups_evaluation_report.md](https://github.com/mallahova/RAGRepo/blob/main/src/eval/notebooks/rag_setups_evaluation_report.md)  
.


Key aspects explored:

- Open-source vs. closed-source model trade-offs  
- Latency vs. retrieval quality  
- Impact of rerankers  
- Effectiveness of query expansion

> ⚠️ One file was missing from the repo and excluded from the evaluation set (checkout  [src/eval/notebooks/explore_eval.ipynb](https://github.com/mallahova/RAGRepo/blob/main/src/eval/notebooks/explore_eval.ipynb)  
)

---

## 🏆 Results

**Recommended setups:**

- **Closed Source**:
  - *Best overall (but costly), Recall@10 = 0.74*: [OpenAI small embedding + Cohere reranker](https://github.com/mallahova/RAGRepo/blob/main/config/closed_source/best_performance_with_reranker.yaml)  
  - *Strong performance, lower cost, Recall@10 = 0.72*: [OpenAI large embedding, no reranker](https://github.com/mallahova/RAGRepo/blob/main/config/closed_source/high_performance_no_reranker.yaml)

- **Open Source**:
  - *Best open option, Recall@10 = 0.67*: [gte_multilingual_base + bge-reranker-v2-m3](https://github.com/mallahova/RAGRepo/blob/main/config/open_source/decent_performance_with_reranker.yaml)  
  - *Fastest option, Recall@10 = 0.66*: [gte_multilingual_base](https://github.com/mallahova/RAGRepo/blob/main/config/open_source/lightweight_no_reranker.yaml)


> Prebuilt indexes for the `escrcpy` repository are included for all recommended configs.

---

## 🔄 Pipeline

### 1. Index the Repository

Build a hybrid FAISS + BM25 index from a GitHub repo:

```bash
python src/indexing/build_index.py --repo_url <URL.git> --config_path <config_path>
```

---

### 2. Retrieve

To retrieve relevant documents or file paths:

```bash
python src/retrieval/search_index.py --query "<your question>" --config_path <config_path>
```

---

### 3. Retrieve + Generate

LLM-generated answer summaries for retrieved code:

```bash
python src/generation/generate.py --query "<your question>" --config_path <config_path>
```

---

### 4. Evaluate Retrieval Performance

To evaluate Recall@10 with the built-in QA dataset:

```bash
python src/eval/evaluate_retrieval.py \
  --dataset_path <dataset_path>
  --config_path <config_path>
```

---
## 🌱 Future Enhancements

Planned enhancements:

- Streamlit-based UI for interactive exploration  
- Expanded model support (embeddings, rerankers, LLMs)  
- More advanced RAG techniques and strategies
