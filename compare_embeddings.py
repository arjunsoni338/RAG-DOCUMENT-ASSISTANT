from langchain_openai import OpenAIEmbeddings 
#LangChain is a Python/JavaScript framework that helps you to build applications using LLMs
#Embeddings in OpenAI are numeric vector representations of text.
#They convert words, sentences, or documents into lists of numbers so that a computer can understand their meaning,
#similarity, and relationships.
#In simple terms:
#Embeddings = Text → Numbers that capture meaning.
from langchain_classic.evaluation import load_evaluator
# laod_evaluator is a helper that gives you built-in evaluation tools like embedding distance, etc.
from dotenv import load_dotenv
#Reads your .env (enviornment variable) file so you don’t hardcode your secret keys in code
import openai
import os #To access environment variables like OPENAI_API_KEY.

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

def main(): # this is our main function like int main in c++
        #   Get embedding for a word.
    embedding_function = OpenAIEmbeddings()
        #   Here you create an embeddings generator.
        #    Now this object can take in text like "apple" and return a giant vector, e.g. [0.01, -0.12, ...].
    vector = embedding_function.embed_query("apple")
        #   embed_query("apple"): Sends the word “apple” to OpenAI’s embedding model.
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

        #   You’ve basically done:

        #   “Hey model, what is ‘apple’ in your math language?”

    # Compare vector of two words
    evaluator = load_evaluator("pairwise_embedding_distance") #You load an evaluator that calculates distance between embeddings.
    words = ("apple", "iphone") #small tuple/row of words
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
        #It internally embeds both strings and computes how far apart they are.
        #Smaller distance = more similar in meaning.
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
