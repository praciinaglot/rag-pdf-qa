import sys
import pymupdf  # PyMuPDF, reads PDFs


def load_pdf(path):
    doc = pymupdf.open(path)
    return [(i, page.get_text()) for i, page in enumerate(doc, start=1)]


def chunk_text(text, size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_chunks(path):
    chunks = []
    for page_num, text in load_pdf(path):
        for piece in chunk_text(text):
            if piece.strip():
                chunks.append({"page": page_num, "text": piece})
    return chunks


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "doc.pdf"
    chunks = build_chunks(path)
    print(f"Total chunks: {len(chunks)}")
    mid = chunks[len(chunks) // 2]
    print(f"\nSample chunk (from page {mid['page']}):\n")
    print(mid["text"])
