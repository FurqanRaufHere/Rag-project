# src/run_extract.py
from extractors import extract_with_pypdf2, extract_with_pdfplumber, save_texts_to_files
import os
from pathlib import Path

PDF_PATH = Path(__file__).parents[1] / "data" / "test_doc.pdf"
OUT_DIR = Path(__file__).parents[1] / "data" / "extracted"

if not PDF_PATH.exists():
    raise FileNotFoundError(f"Put your PDF at: {PDF_PATH}")

# PyPDF2
pypdf_texts = extract_with_pypdf2(str(PDF_PATH))
save_texts_to_files(pypdf_texts, str(OUT_DIR / "pypdf2"), "pypdf2")

# pdfplumber
pdfplumber_texts = extract_with_pdfplumber(str(PDF_PATH))
save_texts_to_files(pdfplumber_texts, str(OUT_DIR / "pdfplumber"), "pdfplumber")

# quick summary
print(f"PyPDF2 pages: {len(pypdf_texts)}")
print(f"pdfplumber pages: {len(pdfplumber_texts)}")
