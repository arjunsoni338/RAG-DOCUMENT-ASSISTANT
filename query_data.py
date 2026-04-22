import argparse
import json
import math
import re
from collections import Counter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rag_utils import (
    LOCAL_STORE_PATH,
    get_chat_model,
    get_embeddings,
    load_embedding_provider,
    load_environment,
)

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "who", "what", "when", "where", "why", "how",
    "which", "this", "these", "those", "did", "do", "does", "had", "have",
    "i", "you", "they", "we", "she", "her", "his", "him", "them", "our",
    "their", "or", "if", "but", "about", "into", "than", "then",
}

def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b\w+\b", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }

def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [part.strip() for part in parts if part.strip()]

def search_local_documents(db: Chroma, query_text: str, k: int = 3) -> list[Document]:
    query_tokens = tokenize(query_text)
    raw_documents = db.get(include=["documents", "metadatas"])
    documents: list[Document] = []
    tokenized_documents: list[set[str]] = []
    for page_content, metadata in zip(
        raw_documents["documents"],
        raw_documents["metadatas"],
        strict=False,
    ):
        doc = Document(page_content=page_content, metadata=metadata or {})
        documents.append(doc)
        tokenized_documents.append(tokenize(page_content))
    document_frequency = Counter()
    for tokens in tokenized_documents:
        document_frequency.update(tokens)
    scored_documents: list[tuple[int, Document]] = []
    total_documents = max(len(documents), 1)
    for doc, doc_tokens in zip(documents, tokenized_documents, strict=False):
        overlap = query_tokens & doc_tokens
        score = 0.0
        for token in overlap:
            idf = math.log((1 + total_documents) / (1 + document_frequency[token])) + 1
            score += idf
        if query_text.lower() in doc.page_content.lower():
            score += 3.0
        if "alice" in query_tokens and "alice" in doc.page_content.lower():
            score += 0.5
        scored_documents.append((score, doc))
    scored_documents.sort(
        key=lambda item: (
            item[0],
            -len(item[1].page_content),
        ),
        reverse=True,
    )
    return [doc for score, doc in scored_documents[:k] if score > 0]

def load_local_documents() -> list[Document]:
    if not LOCAL_STORE_PATH.exists():
        return []
    data = json.loads(LOCAL_STORE_PATH.read_text(encoding="utf-8"))
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in data
    ]

def search_local_store(query_text: str, k: int = 3) -> list[Document]:
    documents = load_local_documents()
    if not documents:
        return []
    query_tokens = tokenize(query_text)
    document_frequency = Counter()
    tokenized_documents: list[set[str]] = []
    for doc in documents:
        tokens = tokenize(doc.page_content)
        tokenized_documents.append(tokens)
        document_frequency.update(tokens)
    total_documents = max(len(documents), 1)
    scored_documents: list[tuple[float, Document]] = []
    for doc, doc_tokens in zip(documents, tokenized_documents, strict=False):
        overlap = query_tokens & doc_tokens
        score = 0.0
        for token in overlap:
            idf = math.log((1 + total_documents) / (1 + document_frequency[token])) + 1
            score += idf
        if query_text.lower() in doc.page_content.lower():
            score += 3.0
        scored_documents.append((score, doc))
    scored_documents.sort(
        key=lambda item: (item[0], -len(item[1].page_content)),
        reverse=True,
    )
    return [doc for score, doc in scored_documents[:k] if score > 0]

def extract_subject_tokens(question: str) -> list[str]:
    words = re.findall(r"\b[A-Za-z][A-Za-z']*\b", question)
    lowered = [word.lower() for word in words]
    return [
        word
        for word, lowered_word in zip(words, lowered, strict=False)
        if lowered_word not in STOPWORDS
    ]

def is_speech_question(question: str) -> bool:
    lowered_question = question.lower()
    return lowered_question.startswith("what did") and any(
        verb in lowered_question
        for verb in (" say", " said", " ask", " asked", " reply", " replied", " cry", " cried", " call", " called")
    )

def rank_sentences(question: str, sentences: list[str]) -> list[tuple[float, str]]:
    question_tokens = tokenize(question)
    candidate_sentences: list[tuple[float, str]] = []
    subject_tokens = [token.lower() for token in extract_subject_tokens(question)]
    lowered_question = question.lower()
    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        lowered_sentence = sentence.lower()
        if not sentence_tokens:
            continue
        overlap = question_tokens & sentence_tokens
        score = float(len(overlap))
        if sentence.lower().startswith(("alice", "the", "she", "he")):
            score += 0.2
        if question.lower().startswith("who"):
            if any(token in lowered_sentence for token in subject_tokens):
                score += 1.5
            if re.search(r"\b(is|was)\b|’s|'s", lowered_sentence):
                score += 1.0
        if is_speech_question(question) and any(
            token in lowered_sentence for token in subject_tokens
        ):
            if any(
                verb in lowered_sentence
                for verb in ("say", "said", "asked", "cried", "replied", "called")
            ):
                score += 2.0
            if '"' in sentence or "“" in sentence:
                score += 0.5
        if question.lower().startswith("where") and any(
            word in lowered_sentence
            for word in ("in ", "at ", "under", "over", "near", "inside")
        ):
            score += 0.5
        if question.lower().startswith("when") and any(
            char.isdigit() for char in sentence
        ):
            score += 0.5
        if subject_tokens and all(token in lowered_sentence for token in subject_tokens):
            score += 0.5
        if lowered_question in lowered_sentence:
            score += 3.0
        if score > 0:
            candidate_sentences.append((score, sentence))
    candidate_sentences.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidate_sentences

def answer_from_context(
    question: str,
    documents: list[Document],
    corpus_documents: list[Document] | None = None,
) -> str:
    sentence_pool: list[str] = []
    for doc in documents:
        sentence_pool.extend(sentence_split(doc.page_content))
    ranked_sentences = rank_sentences(question, sentence_pool)
    if (
        question.lower().startswith("who")
        or is_speech_question(question)
    ) and corpus_documents is not None:
        full_corpus_sentences: list[str] = []
        for doc in corpus_documents:
            full_corpus_sentences.extend(sentence_split(doc.page_content))
        corpus_ranked = rank_sentences(question, full_corpus_sentences)
        if corpus_ranked:
            ranked_sentences = corpus_ranked
    if not ranked_sentences:
        return documents[0].page_content.strip()
    best_sentences: list[str] = []
    for _score, sentence in ranked_sentences:
        if sentence not in best_sentences:
            best_sentences.append(sentence)
        if question.lower().startswith("who") and len(best_sentences) == 1:
            break
        if is_speech_question(question) and len(best_sentences) == 1:
            break
        if len(best_sentences) == 2:
            break
    return " ".join(best_sentences)

def main():
    load_environment()

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved context before the answer.",
    )
    args = parser.parse_args()
    query_text = args.query_text

    provider = load_embedding_provider()
    if provider == "local":
        corpus_documents = load_local_documents()
        results = search_local_store(query_text, k=3)
    else:
        corpus_documents = None
        embedding_function, provider = get_embeddings(provider)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search(query_text, k=3)
    if len(results) == 0:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = get_chat_model()
    if model is None:
        response_text = answer_from_context(query_text, results, corpus_documents)
    else:
        try:
            response_text = model.invoke(prompt).content
        except Exception:
            response_text = answer_from_context(query_text, results, corpus_documents)
    sources = [doc.metadata.get("source", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    if args.show_context:
        formatted_response = (
            f"Retrieved context:\n{context_text}\n\n{formatted_response}"
        )
    print(formatted_response)

if __name__ == "__main__":
    main()