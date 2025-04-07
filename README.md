# RAGRepo

**Retrieval-Augmented Generation (RAG) system for code repositories.**  
Flexible, configurable, and optimized for practical trade-offs between speed, accuracy, and openness.

---

## ⚡ Quickstart

```bash
conda create -n ragrepo python=3.10.16 -y
conda activate ragrepo
python -m pip install -r requirements.txt
```

If using APIs (OpenAI, Cohere, Google, etc.), create a `.env` file in the root directory:

```
OPENAI_API_KEY=...
COHERE_API_KEY=...
```

---

## 📚 Overview

This repository provides a complete RAG pipeline for question answering over codebases. Key features include:

- **Repo-based indexing**: Build indexes directly from GitHub URLs  
- **Hybrid retrieval**: FAISS + BM25 with tunable weighting  
- **Query expansion** and **reranking** support  
- **Config-driven**: Swap models (LLMs, embeddings, retrievers) via YAML (e.g. ```config/base.yaml```)
- **Custom model wrappers** for unsupported providers (checkout `src/core/custom_wrappers/gemini_wrapper.py`)  
- **Evaluation tools** for retrieval quality using Recall@10 and latency

---

## 🔩 Pipeline

### 1. 🏗️ Build the Index

Create the hybrid FAISS + BM25 index from a GitHub repo:

```bash
python src/indexing/build_index.py --repo_url <URL>
```

---

### 2. 🔍 Retrieve Only *(No LLM Generation)*

For systems that only need relevant documents or file locations:

```bash
python src/retrieval/search_index.py --query "<your question>" --config config/base.yaml
```

---

### 3. 💬 Retrieve and Generate

If you want full RAG output including LLM-generated responses:

```bash
python src/generation/generate.py --config config/base.yaml
```

---

### 4. 📈 Evaluate Retrieval

Evaluate Recall@10 using the provided QA dataset:

```bash
python src/eval/evaluate_retrieval.py \
  --dataset_path src/data/eval/escrcpy-commits-generated.json \
  --config config/base.yaml
```

---

## 📊 Reports & Trade-offs

Performance analysis and insights available in:



Explored dimensions include:

- Latency vs. retrieval quality  
- Open- vs. closed-source models  
- With vs. without rerankers  
- Effects of query expansion

> Note: One missing file in the repo was removed from the evaluation set. See `src/eval/notebooks/explore_eval.ipynb` for details.

---

## 📌 Evaluation Dataset

Evaluates retrieval for `https://github.com/viarotel-org/escrcpy` repository.
Path: `src/data/eval/escrcpy-commits-generated.json`  
Includes natural language queries and file-level answers.  
Automatically generated — may contain minor inconsistencies.


---

## 🛠️ Future Enhancements

- Streamlit application* for interactive use
- Support and evaluation for more embedding, reranker, and LLM models  
- Advanced RAG strategies  