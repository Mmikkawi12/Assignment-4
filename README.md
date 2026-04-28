# CSAI 422 - Assignment 4  
## Retrieval Augmented Generation with Advanced Techniques

This project implements an end-to-end text-only Retrieval-Augmented Generation (RAG) pipeline using the PopQA benchmark.  
The system includes dataset preparation, dense retrieval, query expansion, hybrid search, reranking, citation-grounded answer generation, error analysis, and a self-reflective RAG stage.

---

## Project Overview

The goal of this assignment is to build and evaluate an advanced RAG system for factual question answering.

The implemented pipeline includes:

1. PopQA dataset loading and inspection
2. Retrieval corpus construction with passage IDs and metadata
3. Baseline dense retrieval using TF-IDF + TruncatedSVD dense vectors
4. FAISS indexing and retrieval
5. Retrieval evaluation using Recall@k, Precision@k, and MRR
6. Query expansion using PopQA metadata
7. Hybrid search using BM25 + dense retrieval
8. Reranking using subject/property matching
9. Citation-grounded answer generation
10. Prompt design for grounded QA
11. Error analysis
12. Self-reflective RAG
13. Final comparative evaluation

---

## Dataset

The project uses the PopQA dataset from Hugging Face:

```text
akariasai/PopQA
