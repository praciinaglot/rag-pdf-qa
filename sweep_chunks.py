import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ingest_old as base

evalset = json.load(open("evalset_clean.json"))
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
PREFIX = "Represent this sentence for searching relevant passages: "
pages = base.load_pdf("doc.pdf")
q_embs = embedder.encode([PREFIX + i["question"] for i in evalset], normalize_embeddings=True)


def run(size, overlap):
    chunks = []
    for page_num, text in pages:
        for piece in base.chunk_text(text, size, overlap):
            if piece.strip():
                chunks.append({"page": page_num, "text": piece})
    emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    hit1 = hit4 = 0
    for item, q in zip(evalset, q_embs):
        top = np.argsort(emb @ q)[::-1][:4]
        found = [chunks[i]["page"] for i in top]
        hit1 += found[0] == item["page"]
        hit4 += item["page"] in found
    return len(chunks), hit1, hit4


n = len(evalset)
print(f"\n{'size/overlap':<14}{'chunks':<10}{'Hit@1':<12}{'Hit@4'}")
for size, overlap in [(200, 25), (300, 40), (400, 50), (500, 60)]:
    c, h1, h4 = run(size, overlap)
    print(f"{str(size)+'/'+str(overlap):<14}{c:<10}{h1}/{n} = {h1/n:.0%}   {h4}/{n} = {h4/n:.0%}")
