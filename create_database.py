from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_community.vectorstores import Chroma 
import json
import os
import shutil #shutil: Used here to delete directories (to reset the DB).
from rag_utils import (
    LOCAL_STORE_PATH,
    get_embeddings,
    load_environment,
    save_embedding_config,
)
load_environment()
DATA_PATH = "data" # this is our input books/texts given or retrived 
CHROMA_PATH = "chroma" # this is ordered/ embedded/ chunked data/books
def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = TextLoader("data/alice_in_wonderland.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100, #we do this so that important text at the boundary is not lost and the model keeps context.
        length_function=len,
        add_start_index=True, #It keeps track of where in the original text the chunk starts.
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_local_documents(chunks: list[Document]) -> None:
    LOCAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_STORE_PATH.write_text(
        json.dumps(
            [
                {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings, provider = get_embeddings("auto")
    save_local_documents(chunks)
    save_embedding_config(provider)
    if provider == "local":
        print(
            f"Saved {len(chunks)} chunks to {LOCAL_STORE_PATH} using local retrieval."
        )
        return
    try:
        Chroma.from_documents(
            chunks, embeddings, persist_directory=CHROMA_PATH
        )
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH} using {provider} embeddings.")
    except Exception as exc:
        save_embedding_config("local")
        print(
            "OpenAI vector storage failed, but the local document store was created.\n"
            f"Reason: {exc}"
        )
if __name__ == "__main__":
    main()