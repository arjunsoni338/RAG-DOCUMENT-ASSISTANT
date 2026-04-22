# RAG Project — Alice in Wonderland

A Retrieval-Augmented Generation (RAG) system built with LangChain and Google Gemini. Ask questions about *Alice in Wonderland* and get answers grounded in the actual text.

---

## How it works

1. The book is split into small chunks and stored in a Chroma vector database
2. When you ask a question, the most relevant chunks are retrieved
3. Gemini reads those chunks and answers your question

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install langchain-google-genai google-generativeai
```

### 3. Add your Gemini API key

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_key_here
```
---

## Usage

### Step 1 — Build the database (run once, or when data changes)

```bash
python create_database.py
```

Expected output:
```
Split 1 documents into 814 chunks.
Saved 814 chunks to chroma/ using gemini embeddings.
```

### Step 2 — Query

```bash
python query_data.py "Your question here"
```

Examples:

```bash
python query_data.py "Who is Alice?"
python query_data.py "What happens at the tea party?"
python query_data.py "Who is the Queen of Hearts?"
```

### Show retrieved context alongside the answer

```bash
python query_data.py "Who is Alice?" --show-context
```

---

## Compare embeddings

Test how semantically similar two words are:

```bash
python compare_embeddings.py
```

---

## Project structure

```
RAG project/
├── data/
│   └── alice_in_wonderland.md   # Source document
├── chroma/                      # Vector database (auto-generated)
│   ├── documents.json
│   └── embedding_config.json
├── create_database.py           # Chunks and embeds the document
├── query_data.py                # Handles question answering
├── compare_embeddings.py        # Compares word similarity via embeddings
├── rag_utils.py                 # Shared utilities (embeddings, LLM, config)
├── requirements.txt
└── .env                         # Your API key (never commit this)
```
---

## Embedding models

The project uses `models/gemini-embedding-001` by default.
Available models on your key:
- `models/gemini-embedding-001` — stable, recommended
- `models/gemini-embedding-2-preview` — newer preview version
To switch models, update `_build_gemini_embeddings()` in `rag_utils.py`.

---

## Fallback behaviour

If `GOOGLE_API_KEY` is not set, the project falls back to local fake embeddings and keyword-based retrieval (BM25-style). Answers will be lower quality without a real LLM.

---

## Requirements
- Python 3.10+
- Google Gemini API key