#LangChain is a Python/JavaScript framework that helps you to build applications using LLMs
#Embeddings in OpenAI are numeric vector representations of text.
#They convert words, sentences, or documents into lists of numbers so that a computer can understand their meaning,
#similarity, and relationships.
#In simple terms:
#Embeddings = Text → Numbers that capture meaning.
from langchain_community.evaluation import load_evaluator
# laod_evaluator is a helper that gives you built-in evaluation tools like embedding distance, etc.
from rag_utils import get_embeddings, load_environment

load_environment()

def main(): # this is our main function like int main in c++
        #   Get embedding for a word.
    try:
        embedding_function, provider = get_embeddings("auto")
    except Exception:
        embedding_function, provider = get_embeddings("local")
        #   Here you create an embeddings generator.
        #    Now this object can take in text like "apple" and return a giant vector, e.g. [0.01, -0.12, ...].
    vector = embedding_function.embed_query("apple")
        #   embed_query("apple"): Sends the word “apple” to OpenAI’s embedding model.
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")
    print(f"Embedding provider: {provider}")

        #   You’ve basically done:

        #   “Hey model, what is ‘apple’ in your math language?”

    # Compare vector of two words
    evaluator = load_evaluator(
        "pairwise_embedding_distance",
        embeddings=embedding_function,
    ) #You load an evaluator that calculates distance between embeddings.
    words = ("apple", "iphone") #small tuple/row of words
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
        #It internally embeds both strings and computes how far apart they are.
        #Smaller distance = more similar in meaning.
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()
