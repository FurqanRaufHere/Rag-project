# src/test_faiss_search.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load model, FAISS index, and chunks
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("data/faiss_index.idx")

with open("data/chunk_mapping.pkl", "rb") as f:
    chunks = pickle.load(f)

# Example query
query = "What is the main conclusion of this report?"
query_embedding = model.encode([query])
D, I = index.search(np.array(query_embedding).astype("float32"), 3)

print("Top relevant chunks:")
for idx in I[0]:
    print("—", chunks[idx][:300], "...")
