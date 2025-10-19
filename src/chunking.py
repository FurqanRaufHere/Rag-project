# src/chunking.py
from typing import List, Dict
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def clean_text(text: str) -> str:
    # basic cleaning: normalize whitespace, remove strange repeated chars
    if not text:
        return ""
    text = text.replace("\r", " ")
    text = re.sub(r"\n{2,}", "\n\n", text)  # keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text

def chunk_by_words(pages: List[Dict], chunk_size: int = 200, overlap: int = 50) -> List[Dict]:
    """
    pages: list of {'page': int, 'text': str}
    returns list of {'id': int, 'page': int, 'text': str}
    chunk_size & overlap in words
    """
    chunks = []
    id_counter = 0
    for p in pages:
        txt = clean_text(p['text'])
        words = txt.split()
        if not words:
            continue
        i = 0
        while i < len(words):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"id": id_counter, "page": p['page'], "text": chunk_text})
            id_counter += 1
            i += chunk_size - overlap
    logging.info(f"Created {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
    return chunks
