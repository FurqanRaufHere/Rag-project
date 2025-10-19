# src/build_faiss_index.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load your chunks (use the file created in previous step)
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for all chunks
embeddings = model.encode(chunks, show_progress_bar=True)

# Convert embeddings to numpy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save both FAISS index and chunks
faiss.write_index(index, "data/faiss_index.idx")
with open("data/chunk_mapping.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ FAISS index built and saved with {len(chunks)} chunks.")
