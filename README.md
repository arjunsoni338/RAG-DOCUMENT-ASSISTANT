# GitLab Knowledge Base — RAG Assistant

A Retrieval-Augmented Generation (RAG) system built with LangChain, HuggingFace embeddings, and OpenRouter. Ask questions about GitLab's company handbook — values, mission, communication, leadership, HR policies, compensation, and more — and get answers grounded in the actual documentation.

---

## Knowledge Base

The system is loaded with the following publicly available GitLab handbook documents:

| Document | Content |
|----------|---------|
| `gitlab_values.md` | CREDIT values — Collaboration, Results, Efficiency, DIB, Iteration, Transparency |
| `gitlab_mission.md` | Company mission, strategic beliefs, and what GitLab is building |
| `gitlab_communication.md` | Async-first communication guidelines and channel usage |
| `gitlab_leadership.md` | Leadership principles, decision-making, and development programs |
| `gitlab_anti_harassment.md` | Anti-harassment policy, reporting mechanisms, and disciplinary framework |
| `gitlab_hiring.md` | Talent acquisition process and hiring philosophy |
| `gitlab_compensation.md` | Salary, equity, RSUs, review cycles, and pay transparency |
| `gitlab_benefits.md` | Medical, parental leave, pension, and total rewards |
| `gitlab_people_group.md` | HR teams, contacts, response times, and employee lifecycle |

To add more documents, drop any `.md` file into `data/` and rebuild the database.

---

## How it works

1. Documents in `data/` are split into chunks and stored in a Chroma vector database
2. When you ask a question, the most relevant chunks are retrieved via semantic search
3. An LLM (via OpenRouter) reads those chunks and answers your question grounded in the source

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

### 3. Add your API keys

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
HF_TOKEN=your_token_here        # optional, silences HuggingFace rate limit warning
```

Optionally set a different LLM model (defaults to `openai/gpt-4o-mini`):
```
OPENROUTER_MODEL=openai/gpt-4o-mini
```

---

## Usage

### Step 1 — Build the database (run once, or when documents change)

```bash
python create_database.py
```

Expected output:
```
Split 9 documents into 54 chunks.
Saved 54 chunks to chroma/ using huggingface embeddings.
```

### Step 2 — Query

```bash
python query_data.py "Your question here"
```

Examples:

```bash
python query_data.py "What are GitLab's core values?"
python query_data.py "What is GitLab's anti-harassment policy?"
python query_data.py "How does GitLab handle compensation reviews?"
python query_data.py "What is GitLab's mission?"
python query_data.py "How does GitLab approach communication?"
```

### Show retrieved source chunks alongside the answer

```bash
python query_data.py "What are GitLab's core values?" --show-context
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
├── data/                            # Source documents (drop new .md files here)
│   ├── gitlab_values.md
│   ├── gitlab_mission.md
│   ├── gitlab_communication.md
│   ├── gitlab_leadership.md
│   ├── gitlab_anti_harassment.md
│   ├── gitlab_hiring.md
│   ├── gitlab_compensation.md
│   ├── gitlab_benefits.md
│   └── gitlab_people_group.md
├── chroma/                          # Vector database (auto-generated)
│   └── embedding_config.json
├── create_database.py               # Chunks and embeds all documents in data/
├── query_data.py                    # Handles question answering
├── compare_embeddings.py            # Compares word similarity via embeddings
├── rag_utils.py                     # Shared utilities (embeddings, LLM, config)
├── pyrightconfig.json               # Pylance/Pyright venv configuration
├── requirements.txt
└── .env                             # API keys (never commit this)
```

---

## Embedding model

Uses `all-MiniLM-L6-v2` (HuggingFace) — runs locally, no API key required.

---

## Requirements

- Python 3.10+
- OpenRouter API key (for LLM responses)
