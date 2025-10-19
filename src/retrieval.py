# src/retrieval.py
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"  # change if you used a different encoder
INDEX_PATH = Path("data/faiss_index_cosine.idx")
MAPPING_PATH = Path("data/chunk_mapping_cosine.pkl")

class Retriever:
    def __init__(self, index_path=INDEX_PATH, mapping_path=MAPPING_PATH):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(index_path))
        with open(mapping_path, "rb") as f:
            self.chunks = pickle.load(f)  # list of dicts
        self.dim = self.index.d

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        # normalize to unit length for cosine (we normalized index vectors)
        faiss.normalize_L2(vec)
        return vec

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        qv = self.embed_query(query)
        D, I = self.index.search(qv, top_k)
        idxs = I[0].tolist()
        scores = D[0].tolist()
        results = []
        for idx, score in zip(idxs, scores):
            # safety: if idx is -1 (no result) skip
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            results.append({"id": c["id"], "page": c.get("page"), "text": c["text"], "score": float(score)})
        return results
