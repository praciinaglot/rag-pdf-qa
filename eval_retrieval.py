import json
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import build_chunks

chunks = build_chunks("doc.pdf")
texts = [c["text"] for c in chunks]
evalset = json.load(open("evalset_clean.json"))

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
PREFIX = "Represent this sentence for searching relevant passages: "

hit1 = hit4 = hit4_near = 0
misses = []
for item in evalset:
    q_emb = embedder.encode(PREFIX + item["question"], normalize_embeddings=True)
    scores = embeddings @ q_emb
    top = np.argsort(scores)[::-1][:4]
    pages = [chunks[i]["page"] for i in top]
    right = item["page"]
    if pages[0] == right:
        hit1 += 1
    if right in pages:
        hit4 += 1
    if any(abs(p - right) <= 1 for p in pages):
        hit4_near += 1
    else:
        misses.append((right, pages, item["question"]))

n = len(evalset)
print(f"\nQuestions: {n}")
print(f"Hit@1 (right page is #1 result):   {hit1}/{n} = {hit1/n:.0%}")
print(f"Hit@4 (right page in top 4):       {hit4}/{n} = {hit4/n:.0%}")
print(f"Hit@4 within +-1 page:             {hit4_near}/{n} = {hit4_near/n:.0%}")
print("\nMISSES (correct page, pages found, question):")
for right, pages, q in misses:
    print(f"- page {right} | found {pages} | {q}")
