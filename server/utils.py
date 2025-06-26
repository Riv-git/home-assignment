import hashlib, os, fitz  # PyMuPDF
from pathlib import Path
from typing import Dict

BASE_DIR = Path("storage")     # all PDFs & PNGs live here
BASE_DIR.mkdir(exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def pdf_to_png(pdf_path: Path, out_dir: Path) -> Dict[str, int]:
    """
    Render every page of *pdf_path* to PNG files inside *out_dir*.
    Returns {"pages": N}
    """
    doc = fitz.open(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for page_idx in range(doc.page_count):
        pix = doc.load_page(page_idx).get_pixmap(dpi=150)  # 150 DPI is fine for previews
        pix.save(out_dir / f"page_{page_idx+1:03}.png")

    return {"pages": doc.page_count}
