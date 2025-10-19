# src/run_questions.py
from rag_pipeline import run_rag, run_baseline, save_comparison
from pathlib import Path

QUESTIONS = [
    "What is the first thing to understand about psychology of money?",
    "what is the psychology of money people don't understand?",
]

results = []
for q in QUESTIONS:
    print("Processing question:", q)
    rag = run_rag(q, top_k=4)
    baseline = run_baseline(q)
    results.append({"question": q, "rag": rag, "baseline": baseline})

# Save to CSV
save_comparison(results, out_csv="data/comparison.csv")
print("Saved comparison to data/comparison.csv")

# also print for quick inspection
for r in results:
    print("="*80)
    print("Q:", r["question"])
    print("RAG answer:\n", r["rag"]["answer"])
    print("Retrieved pages:", [c["page"] for c in r["rag"]["retrieved"]])
    print("BASELINE answer:\n", r["baseline"]["answer"])
