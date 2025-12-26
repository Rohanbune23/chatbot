import json
import uuid
from pathlib import Path

# IMPORTANT: use the FAISS backend file
from cb import (
    extract_pdf_paragraphs,
    add_paragraphs_to_faiss,
    uploaded_pdfs,
    UPLOADED_PDFS_FILE,
)

UPLOAD_DIR = Path("uploads")

pdf_files = list(UPLOAD_DIR.glob("*.pdf"))

added = False
for pdf in pdf_files:
    if pdf.name not in uploaded_pdfs:
        print("Adding:", pdf.name)
        paragraphs = extract_pdf_paragraphs(pdf)
        pdf_id = str(uuid.uuid4())
        uploaded_pdfs[pdf.name] = pdf_id
        add_paragraphs_to_faiss(paragraphs, pdf_id)
        added = True

if added:
    UPLOADED_PDFS_FILE.write_text(json.dumps(uploaded_pdfs))
    print("PDFs added successfully!")
else:
    print("No new PDFs found.")
