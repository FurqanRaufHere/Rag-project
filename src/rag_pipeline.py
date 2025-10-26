from retrieval import Retriever
from prompt import build_rag_messages
from groq_client import call_chat_completion
from typing import Dict, Any
import csv
from pathlib import Path

retriever = Retriever()

def run_rag(question: str, top_k: int = 4):
    retrieved = retriever.retrieve(question, top_k=top_k)
    messages = build_rag_messages(retrieved, question)
    answer = call_chat_completion(messages, model="llama-3.3-70b-versatile", max_tokens=512, temperature=0.0)
    return {"question": question, "retrieved": retrieved, "answer": answer}

def run_baseline(question: str):
    # baseline: call model with only the user question (and a short system message)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely and do not hallucinate."},
        {"role": "user", "content": question}
    ]
    answer = call_chat_completion(messages, model="llama-3.3-70b-versatile", max_tokens=512, temperature=0.0)
    return {"question": question, "answer": answer}

def save_comparison(results: list, out_csv: str = "data/comparison.csv"):
    keys = ["question", "rag_answer", "non_rag_answer", "rag_pages", "rag_scores"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question": r["question"],
                "rag_answer": r["rag"]["answer"],
                "non_rag_answer": r["baseline"]["answer"],
                "rag_pages": ";".join(str(x.get("page", "?")) for x in r["rag"]["retrieved"]),
                "rag_scores": ";".join(f"{x.get('score',0):.4f}" for x in r["rag"]["retrieved"])
            })
