# RAG Project — Alice in Wonderland

A Retrieval-Augmented Generation (RAG) system built with LangChain, HuggingFace embeddings, and OpenRouter. Ask questions about *Alice in Wonderland* (and *The Lighthouse Keeper*) and get answers grounded in the actual text.

---

## How it works

1. Documents are split into small chunks and stored in a Chroma vector database
2. When you ask a question, the most relevant chunks are retrieved via semantic search
3. An LLM (via OpenRouter) reads those chunks and answers your question

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
```

### 3. Add your OpenRouter API key

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
HF_TOKEN=your_token_here        # optional, silences HuggingFace rate limit warning
```

Optionally set a different model (defaults to `openai/gpt-4o-mini`):
```
OPENROUTER_MODEL=openai/gpt-4o-mini
```

---

## Usage

### Step 1 — Build the database (run once, or when data changes)

```bash
python create_database.py
```

Expected output:
```
Split 2 documents into 333 chunks.
Saved 333 chunks to chroma/ using huggingface embeddings.
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
│   ├── alice_in_wonderland.md       # Source document
│   └── the_lighthouse_keeper.md     # Source document
├── chroma/                          # Vector database (auto-generated)
│   └── embedding_config.json
├── create_database.py               # Chunks and embeds the documents
├── query_data.py                    # Handles question answering
├── compare_embeddings.py            # Compares word similarity via embeddings
├── rag_utils.py                     # Shared utilities (embeddings, LLM, config)
├── requirements.txt
└── .env                             # Your API key (never commit this)
```

---

## Embedding model

Uses `all-MiniLM-L6-v2` (HuggingFace) — runs locally, no API key required.

---

## Requirements
- Python 3.10+
- OpenRouter API key (for LLM responses)
