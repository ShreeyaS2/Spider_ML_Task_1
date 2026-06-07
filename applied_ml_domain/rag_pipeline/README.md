# RAG Pipeline Application for NLP Research Papers

A Retrieval-Augmented Generation (RAG) pipeline built with Streamlit that lets you ingest, chunk, and query a given collection of NLP research papers. It retrieves relevant chunks and images from the papers and generates answers using the Gemini API.

## Features
- Ingests PDF research papers and splits them into chunks
- PDF chunk embeddings formed using BAAI/bge and stored using FAISS
- Image embeddings and query embedding formed using CLIP and stored using FAISS
- Query text search done using max marginal relevance search
- Query image search done using IndexFlatIP on normalised embedding
- Reranking of retrieved chunks using a CrossEncoder
- Answer generation using Gemini 2.5 Flash
- Streamlit UI showing the final answer, most relevant image and retrieved chunks ranked by relevance

## Setup

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

## Models Used

| Model | Purpose |
|---|---|---|
| BAAI/bge-small-en-v1.5 | Text embeddings |
| openai/clip-vit-base-patch32 | Image embeddings |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranking |
| gemini-2.5-flash | Answer generation | 

---

## Key Dependencies

| Package | Version |
|---|---|
| torch | 2.6.0 |
| sentence-transformers | 3.3.1 |
| transformers | 4.47.0 |
| faiss-cpu | 1.9.0 |
| langchain | latest |
| streamlit | latest |
| pillow | latest |
| numpy | latest |
| dotenv | latest |
| google-genai | latest |
