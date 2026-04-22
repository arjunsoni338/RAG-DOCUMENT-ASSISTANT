import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

CHROMA_PATH = Path("chroma")
EMBEDDING_CONFIG_PATH = CHROMA_PATH / "embedding_config.json"
LOCAL_STORE_PATH = CHROMA_PATH / "documents.json"
LOCAL_EMBEDDING_SIZE = 256

def load_environment() -> None:
    load_dotenv()

def _build_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def _build_local_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=LOCAL_EMBEDDING_SIZE)

def get_embeddings(provider: str = "auto"):
    load_environment()
    requested_provider = os.getenv("RAG_EMBEDDINGS", provider)
    normalized_provider = requested_provider.lower()
    if normalized_provider not in {"auto", "gemini", "local"}:
        raise ValueError(
            "provider must be one of: auto, gemini, local"
        )
    if normalized_provider == "local":
        return _build_local_embeddings(), "local"
    if normalized_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        return _build_gemini_embeddings(), "gemini"

    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return _build_gemini_embeddings(), "gemini"
    return _build_local_embeddings(), "local"

def get_chat_model():
    load_environment()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash"),
        google_api_key=api_key,
        temperature=0,
    )

def save_embedding_config(provider: str) -> None:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CONFIG_PATH.write_text(
        json.dumps({"provider": provider}, indent=2),
        encoding="utf-8",
    )

def load_embedding_provider(default: str = "auto") -> str:
    if not EMBEDDING_CONFIG_PATH.exists():
        return default
    data = json.loads(EMBEDDING_CONFIG_PATH.read_text(encoding="utf-8"))
    return data.get("provider", default)