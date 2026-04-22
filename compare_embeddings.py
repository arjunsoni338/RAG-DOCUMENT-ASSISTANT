from langchain_community.evaluation import load_evaluator
from rag_utils import get_embeddings, load_environment
load_environment()

def main():
    try:
        embedding_function, provider = get_embeddings("auto")
    except Exception:
        embedding_function, provider = get_embeddings("local")
        vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")
    print(f"Embedding provider: {provider}")
    evaluator = load_evaluator(
        "pairwise_embedding_distance",
        embeddings=embedding_function,
    )
    words = ("apple", "iphone") 
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")
    
if __name__ == "__main__":
    main()
