# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader #DirectoryLoader: Loads all files from a folder (like your books).
#from langchain.text_splitter import RecursiveCharacterTextSplitter #Splits big text into smaller overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain.schema import Document #langchain.schema = rules and structures that define how data looks in LangChain.
from langchain_core.documents import Document 
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma 
#langchain_community.vectorstores = collection of vector database wrappers that let you store & search embeddings.
#chroma is a vector database where you store embeddings and later search them.
import openai 
from dotenv import load_dotenv
import os
import shutil #shutil: Used here to delete directories (to reset the DB).

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

DATA_PATH = "data/books" # this is our input books/texts given or retrived 
CHROMA_PATH = "chroma" # this is ordered/ embedded/ chunked data/books


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
        #Go into data/books and load all .md files (Markdown files - A Markdown file is a simple text file that uses easy formatting symbols to style text.).
    documents = loader.load()
        #each file is converted into a Document object with:
        #page_content: text
        #metadata: e.g., filename, path
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
    # Takes all docs and creates smaller document chunks.
    # and then you check how many chunks we got


## it is a debugging step 
    document = chunks[10]
    print(document.page_content) # actual text
    print(document.metadata)    # file, position, etc.

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first. If there is already a chroma folder, DELETE it.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        ##Delete the entire folder at CHROMA_PATH along with all its files and subfolders.
        #Breakdown:
        #shutil → Python's file operations module
        #rmtree → “remove tree”
        #CHROMA_PATH → the folder where your Chroma database is stored
#      Good for dev:
# ✔ Easy
# ✔ Clean resets
# ✔ No duplicates

# Dangerous for prod:
#  Deletes all data
#  Slow rebuilds
# Breaks your application

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist() #Saves the vector DB permanently
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
