# src/build_faiss_cosine.py
import pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNKS_PKL = Path("data/chunks_small.pkl")  # change if your chunks file name is different
INDEX_OUT = Path("data/faiss_index_cosine.idx")
MAPPING_OUT = Path("data/chunk_mapping_cosine.pkl")

# Load chunks: expected a list of dicts with 'text' keys
with open(CHUNKS_PKL, "rb") as f:
    chunks = pickle.load(f)

texts = [c["text"] for c in chunks]
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get embeddings (float32)
emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

# Normalize embeddings (L2) for cosine similarity using inner product
faiss.normalize_L2(emb)

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors equals cosine similarity
index.add(emb)

faiss.write_index(index, str(INDEX_OUT))

# save mapping (chunks) as-is — index positions match chunks list order
with open(MAPPING_OUT, "wb") as f:
    pickle.dump(chunks, f)

print("Saved FAISS index to", INDEX_OUT, "and mapping to", MAPPING_OUT)
