RAG Text Splitting Strategy Analysis

Comparative evaluation of different LangChain text splitting strategies for Retrieval-Augmented Generation (RAG) pipelines.

This project experimentally analyzes how different chunking techniques impact:

Retrieval accuracy

Query latency

Vector search efficiency

The experiment compares three commonly used LangChain splitters:

CharacterTextSplitter

RecursiveCharacterTextSplitter

TokenTextSplitter

Project Overview

Retrieval-Augmented Generation (RAG) improves Large Language Models by retrieving relevant context from external documents before generating answers.

A critical preprocessing step in RAG pipelines is document chunking, where large documents are split into smaller segments before embedding and indexing.

Poor chunking can cause:

irrelevant retrieval

higher latency

context fragmentation

This project investigates how different splitting strategies affect system performance.

Architecture

RAG pipeline used in this experiment:

Document
   ↓
Text Splitter
   ↓
Embeddings (OpenAI)
   ↓
Vector Store (FAISS)
   ↓
Similarity Search
   ↓
Retrieved Context
   ↓
LLM Response
Tech Stack

Python

LangChain

OpenAI Embeddings (text-embedding-3-small)

GPT-4o-mini

FAISS Vector Database

python-dotenv

Dataset

A structured DOCX business performance report containing sections such as:

Executive Summary

Financial Performance

Market Analysis

Operational Metrics

Customer Satisfaction

Key metrics present in the document:

Metric	Value
Q3 2024 Revenue	$125.3M
Q3 2023 Revenue	$108.8M
Customer Satisfaction Score	4.6 / 5
North America Market Share	45%
Production Capacity Utilization	87%
Evaluation Queries

The following queries were used to evaluate retrieval performance:

What was the revenue in Q3 2024

What was the revenue in Q3 2023

What was the customer satisfaction score

What market share does North America have

What was the production capacity utilization

Results
Splitter	Accuracy	Avg Latency	Chunks
CharacterTextSplitter	1.0	1.03 s	11
RecursiveCharacterTextSplitter	1.0	0.39 s	11
TokenTextSplitter	1.0	0.70 s	11
Key Findings

All splitting strategies achieved perfect retrieval accuracy on the evaluation queries.

RecursiveCharacterTextSplitter achieved the lowest latency.

Chunking strategy plays a significant role in retrieval efficiency and system performance.

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/rag-text-splitting-analysis.git
cd rag-text-splitting-analysis

Create virtual environment:

python -m venv venv

Activate environment:

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Environment Variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key
Running the Experiment

Run the experiment script:

python rag_experiment.py

Provide the path to the document when prompted.

Example Output
Character splitter accuracy: 1.0
Character splitter average latency: 1.03 seconds

Recursive splitter accuracy: 1.0
Recursive splitter average latency: 0.39 seconds

Token splitter accuracy: 1.0
Token splitter average latency: 0.70 seconds
Repository Structure
rag-text-splitting-analysis
│
├── rag_experiment.py
├── research-paper.docx
├── sample_report.docx
├── README.md
├── .gitignore
└── requirements.txt
Future Improvements

Possible extensions to this experiment:

Semantic chunking strategies

Hybrid retrieval (BM25 + vector search)

Large-scale document benchmarks

Embedding model comparison

Retrieval evaluation metrics (MRR, Recall@K)
