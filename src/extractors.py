# src/extractors.py
import logging
from typing import List, Dict
import os

# PyPDF2
from PyPDF2 import PdfReader

# pdfplumber
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_with_pypdf2(path: str) -> List[Dict]:
    """
    Returns list of {'page': int, 'text': str}
    """
    texts = []
    logging.info(f"PyPDF2: opening {path}")
    try:
        reader = PdfReader(path)
    except Exception as e:
        logging.error(f"PyPDF2 failed to open PDF: {e}")
        return texts

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logging.warning(f"PyPDF2 extraction failed on page {i+1}: {e}")
            text = ""
        texts.append({"page": i + 1, "text": text})
    logging.info(f"PyPDF2: extracted {len(texts)} pages")
    return texts

def extract_with_pdfplumber(path: str) -> List[Dict]:
    texts = []
    logging.info(f"pdfplumber: opening {path}")
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logging.warning(f"pdfplumber failed on page {i+1}: {e}")
                    text = ""
                texts.append({"page": i + 1, "text": text})
    except Exception as e:
        logging.error(f"pdfplumber failed to open PDF: {e}")
    logging.info(f"pdfplumber: extracted {len(texts)} pages")
    return texts

def save_texts_to_files(texts: List[Dict], out_folder: str, prefix: str):
    os.makedirs(out_folder, exist_ok=True)
    for p in texts:
        fname = os.path.join(out_folder, f"{prefix}_page_{p['page']:03d}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(p['text'])
    logging.info(f"Saved {len(texts)} text files to {out_folder}")
