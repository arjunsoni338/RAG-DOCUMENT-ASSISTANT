import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
# argparse: Lets you pass text from the command line:
# e.g., python query.py "What is machine learning?"
# Chroma: To load the vector DB you created.
from langchain_openai import OpenAIEmbeddings #Needed to search the DB (same embeddings as before).
from langchain_openai import ChatOpenAI #LangChain wrapper for OpenAI chat models (gpt-4o-mini, etc.).
from langchain_core.prompts import ChatPromptTemplate

 #For building LLM prompts in a structured way.

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
# PROMPT_TEMPLATE: This is your RAG prompt:
# You tell the model:
# “Use only this context” → avoids hallucinations.
# Then provide context and the question.

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
# This creates a parser(Parser = the part of your program that reads and understands user input from the command line.) that understands command-line arguments.
# What it does:
# Reads what the user typed in the terminal
# Splits it into parts
# Converts them into Python variables
    parser.add_argument("query_text", type=str, help="The query text.")
    #The user must type one piece of text after the script name
    args = parser.parse_args()
    # reads the text user typed in. Argparse grabs whatever the user typed and stores it.
    query_text = args.query_text
    #Put that text into a variable
    #You can send this to your LLM or RAG system.


    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    #You recreate the same embedding model.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #Loads the existing vector database from the folder at CHROMA_PATH
    # Connects it with the embedding function, so new queries can also be converted to vectors
    # Open the database and prepare it for searching.

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        # Take the page_content of each document result.
        # Join them into one big string.
        # Separate them with --- for cleanliness.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#You wrote a message with empty blanks {context} and {question}
# LangChain wraps it so it knows how to inject values into it cleanly later
# So prompt_template now becomes an object that knows:
# “I have placeholders”
# “I need variables called context and question”
# “When someone calls .format(), I must replace them”
# It’s like a smart prompt-generator machine
    prompt = prompt_template.format(context=context_text, question=query_text)
# This is the moment your machine says:
# Replace {context} with the actual document text (your retrieved chunks)
# Replace {question} with the user's real question
    print(prompt) # to see what context did we retrive

#These TWO lines are the moment your RAG app becomes ALIVE.
    model = ChatOpenAI()
    #Hey LangChain, give me an OpenAI chatbot brain.
# Creates a chat model object
# That knows how to talk to OpenAI
# With your API key (loaded from .env)
# Using some default OpenAI model like:
# gpt-4o
    response_text = model.predict(prompt)
#     model.predict(prompt) This means:
# “Hey model, here is my beautifully prepared prompt.
# Please think about it, and give me your best answer.”
#That string is stored in response_text
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
#     Collect source from each document’s metadata (like filename).
# Then print:
# Final answer
# List of sources consulted


if __name__ == "__main__":
    main()
