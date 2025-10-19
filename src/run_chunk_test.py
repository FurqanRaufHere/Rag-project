# src/run_chunk_test.py
from pathlib import Path
import pickle
from extractors import extract_with_pdfplumber
from chunking import chunk_by_words

PDF_PATH = Path(__file__).parents[1] / "data" / "test_doc.pdf"

pages = extract_with_pdfplumber(str(PDF_PATH))
# try two chunk sizes
chunks_small = chunk_by_words(pages, chunk_size=200, overlap=50)
chunks_large = chunk_by_words(pages, chunk_size=500, overlap=100)

print("Small chunks:", len(chunks_small))
print("Large chunks:", len(chunks_large))

# Save sample chunks for inspection
with open(Path(__file__).parents[1] / "data" / "chunks_small.pkl", "wb") as f:
    pickle.dump(chunks_small, f)
with open(Path(__file__).parents[1] / "data" / "chunks_large.pkl", "wb") as f:
    pickle.dump(chunks_large, f)

# Print first 3 chunks to console for quick manual inspection
for c in chunks_small[:3]:
    print("="*30)
    print("Page:", c['page'])
    print(c['text'][:800], "...")
