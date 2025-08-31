# ⚡ Enhanced Retrieval-Augmented Generation (RAG) System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20Backend-009688?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-RAG-yellow?logo=python)
![Cohere](https://img.shields.io/badge/Cohere-Reranker-purple?logo=cohere)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-red?logo=google)
![Client](https://img.shields.io/badge/Client-Request%20Script-lightgrey?logo=python)

This repository contains an **enhanced RAG system** for **insurance policy analysis** using:
- Hybrid retrieval (semantic + keyword search)  
- Advanced chunking & metadata  
- Cohere re-ranking  
- Google Gemini (LLM) response generation  
- FastAPI backend for queries  
- Python client script for requests  

---

## 📑 Table of Contents
1. [Backend: fecthv5.py](#1️⃣-backend-fecthv5py)
2. [Client: postv2.py](#2️⃣-client-postv2py)
3. [Workflows](#3️⃣-workflows)
   - [Backend Workflow](#backend-workflow)
   - [Client Workflow](#client-workflow)
4. [Getting Started](#🚀-getting-started)
5. [Project Structure](#📂-project-structure)

---

## 1️⃣ Backend: `fecthv5.py`
📖 **Description:**  
This is the **FastAPI backend** implementing the enhanced Retrieval-Augmented Generation (RAG) system. It loads insurance policy documents, splits them into semantic chunks, retrieves relevant content, reranks with Cohere, and generates structured answers using Google Gemini.  

**Key Features:**
- Hybrid retrieval (semantic + keyword-based)  
- Query preprocessing (expansion, entity extraction)  
- Cohere re-ranking for precision  
- Google Gemini LLM for responses  
- Logging & monitoring with relevance scores and confidence levels  

**Workflow:**
```plaintext
          ┌───────────────────────┐
          │   Environment Setup   │
          │ (API keys, Logging)   │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Document Processing   │
          │ - Load PDF from URL   │
          │ - Split into chunks   │
          │ - Add metadata        │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Vector Store Creation │
          │ (Chroma + Embeddings) │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │   Query Handling API  │
          │  (FastAPI /query)     │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Query Preprocessing   │
          │ - Expand query        │
          │ - Extract entities    │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Hybrid Retrieval      │
          │ (Semantic + Keyword)  │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Reranking             │
          │ (Cohere API)          │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Prompt Engineering    │
          │ (Insurance Analyst    │
          │   Instructions)       │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │  LLM Response (Gemini)│
          │ - Generate answer      │
          │ - Justify with clauses │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Monitoring & Logging  │
          │ - Scores, Metrics     │
          │ - Confidence levels   │
          └──────────┴────────────┘

```

## 2️⃣ Client: `postv2.py`
📖 **Description:**  
This is a **Python client script** that sends a PDF and related questions to the FastAPI backend. It handles the request/response cycle and prints clean structured answers.  

**Key Features:**
- Configurable API endpoint  
- Dynamic PDF + question input  
- JSON payload construction  
- Handles API errors gracefully  
- Pretty-prints answers  

**Workflow:**
```plaintext
          ┌───────────────────────┐
          │   API Configuration   │
          │ (API_URL, headers)    │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Input Preparation     │
          │ - PDF URL             │
          │ - List of Questions   │
          │ - JSON Payload        │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Send POST Request     │
          │ (requests.post)       │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ API Response Handling │
          │ - Check status        │
          │ - Parse JSON          │
          └──────────┬────────────┘
                     │
          ┌──────────▼────────────┐
          │ Display Results       │
          │ - Print structured    │
          │   answers             │
          └──────────┴────────────┘

```

## 🚀 Getting Started  

### 🔧 Requirements  
Make sure you have the following installed:
- Python 3.9+  
- FastAPI & Uvicorn (backend)  
- LangChain, Cohere, Google GenAI SDK  
- Requests (for client)  

Install dependencies:  
```bash
pip install fastapi uvicorn langchain cohere google-generativeai requests chromadb
📦 Enhanced-RAG-System
 ┣ 📜 fecthv5.py     # Backend FastAPI server with RAG pipeline
 ┣ 📜 postv2.py      # Client script for sending queries
 ┣ 📜 README.md      # Documentation
 ┗ 📜 requirements.txt (optional)
