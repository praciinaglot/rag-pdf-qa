import json, importlib
import numpy as np
from sentence_transformers import SentenceTransformer

evalset = json.load(open("evalset_clean.json"))
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
PREFIX = "Represent this sentence for searching relevant passages: "


def run(module_name):
    mod = importlib.import_module(module_name)
    chunks = mod.build_chunks("doc.pdf")
    emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    hit1 = hit4 = 0
    for item in evalset:
        q = embedder.encode(PREFIX + item["question"], normalize_embeddings=True)
        top = np.argsort(emb @ q)[::-1][:4]
        pages = [chunks[i]["page"] for i in top]
        hit1 += pages[0] == item["page"]
        hit4 += item["page"] in pages
    return len(chunks), hit1, hit4


n = len(evalset)
print(f"\n{'chunker':<12}{'chunks':<10}{'Hit@1':<12}{'Hit@4'}")
for name in ["ingest_old", "ingest"]:
    c, h1, h4 = run(name)
    print(f"{name:<12}{c:<10}{h1}/{n} = {h1/n:.0%}   {h4}/{n} = {h4/n:.0%}")
