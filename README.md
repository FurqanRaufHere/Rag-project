# RAG Application

This project implements a **Retrieval-Augmented Generation (RAG)** system that can intelligently answer questions from PDF documents.  
The system extracts text from a document, creates embeddings, stores them in a FAISS vector database, retrieves relevant context for each user query, and generates an accurate answer using **Groq’s Llama 3.3-70B Versatile model**.

---

## 1. Features

- **PDF Text Extraction** using PyPDF2 and pdfplumber (with comparison)
- **Text Chunking** into manageable segments for embedding
- **Semantic Embeddings** using SentenceTransformers
- **Vector Database** using FAISS (cosine similarity)
- **Retrieval-Augmented Generation (RAG)** pipeline
- **Groq LLM Integration** (Llama 3.3 model)
- **Baseline vs RAG Comparison**
- **Automated Evaluation Report (CSV)**

---

## 2. Tech Stack

| Component | Technology Used |
|------------|-----------------|
| Language | Python 3.10+ |
| Embedding Model | all-MiniLM-L6-v2 (Sentence Transformers) |
| Vector DB | FAISS |
| LLM | Groq - Llama 3.3-70B Versatile |
| PDF Parsing | PyPDF2, pdfplumber |
| Environment | VS Code |
| Data Format | Pickle (.pkl), CSV |

---

## 3. Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/mini-rag-groq.git
cd mini-rag-groq


