import json
import uuid
import re
from pathlib import Path

import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
UPLOAD_DIR = Path("uploads")
VECTOR_DIR = Path("vector_store")
VECTOR_DIR.mkdir(exist_ok=True)

INDEX_FILE = VECTOR_DIR / "index.faiss"
META_FILE = VECTOR_DIR / "meta.json"

# ================= LOAD MODEL =================
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)
DIM = model.get_sentence_embedding_dimension()

# ================= FAISS =================
index = faiss.IndexFlatIP(DIM)
meta = {}

# ================= TEXT CLEANER =================
def normalize_text(text: str) -> str:
    """
    Removes problematic Unicode ligatures and normalizes text
    """
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2013": "-",
        "\u2014": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def embed(text: str):
    vec = model.encode(text)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    return vec.astype("float32")

def extract_text(pdf_path: Path):
    reader = PdfReader(pdf_path)
    full = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                full.append(normalize_text(t))
        except:
            pass
    return "\n".join(full)

def split_chunks(text: str):
    chunks = re.split(r"\n{2,}", text)
    return [c.strip() for c in chunks if len(c.strip()) > 120]

# ================= INDEXING =================
print("🚀 Starting PDF indexing...")

for pdf in UPLOAD_DIR.glob("*.pdf"):
    print(f"📄 Indexing: {pdf.name}")

    text = extract_text(pdf)
    if not text.strip():
        print(f"⚠️ No text found in {pdf.name}, skipping")
        continue

    chunks = split_chunks(text)

    for chunk in chunks:
        vid = str(uuid.uuid4())
        index.add(embed(chunk).reshape(1, -1))
        meta[vid] = {
            "text": chunk,
            "source": pdf.name
        }

# ================= SAVE SAFELY =================
faiss.write_index(index, str(INDEX_FILE))

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("✅ All PDFs indexed successfully")
print(f"📊 Total vectors stored: {index.ntotal}")
