import json, time
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
import ingest

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
K = 6
client = genai.Client()
PREFIX = "Represent this sentence for searching relevant passages: "

TRAPS = [
    "Who won the cricket world cup?",
    "What is the capital of France?",
    "How do I configure a Raspberry Pi GPIO pin?",
    "What is the price of a Tesla Model 3?",
    "What is the recipe for butter chicken?",
]


def generate(prompt):
    for model in MODELS:
        for attempt in range(3):
            try:
                text = client.models.generate_content(model=model, contents=prompt).text
                time.sleep(4)
                return text.strip()
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    time.sleep(5 * (attempt + 1))
                    continue
                break
    return None


page_text = dict(ingest.load_pdf("doc.pdf"))
chunks = ingest.build_chunks("doc.pdf")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
emb = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)


def answer(question):
    q = embedder.encode(PREFIX + question, normalize_embeddings=True)
    top = np.argsort(emb @ q)[::-1][:K]
    context = "\n\n".join(f"[Page {chunks[i]['page']}]\n{chunks[i]['text']}" for i in top)
    prompt = f"""Answer the question using ONLY the context below.
Cite page numbers like (page 109).
If the answer is not in the context, say: I couldn't find this in the document.

Context:
{context}

Question: {question}"""
    return generate(prompt), [chunks[i]["page"] for i in top]


def judge(question, ref, ans):
    prompt = f"""You are grading an answer to a question about a textbook.
Question: {question}
Reference page from the textbook:
{ref}

Answer given: {ans}

Reply with exactly one word:
CORRECT if the answer is accurate and consistent with the reference page.
INCORRECT if it is wrong or contradicts the reference page.
NOT_FOUND if the answer says it could not find the information."""
    r = generate(prompt)
    return (r or "ERROR").upper().replace(".", "").split()[0]


evalset = json.load(open("evalset_clean.json"))
results = []
counts = {"CORRECT": 0, "INCORRECT": 0, "NOT_FOUND": 0, "ERROR": 0}
hit_and_correct = hit_total = 0

for n, item in enumerate(evalset, 1):
    ans, pages = answer(item["question"])
    if ans is None:
        verdict = "ERROR"
    else:
        verdict = judge(item["question"], page_text[item["page"]], ans)
    verdict = verdict if verdict in counts else "ERROR"
    counts[verdict] += 1
    hit = item["page"] in pages
    hit_total += hit
    hit_and_correct += hit and verdict == "CORRECT"
    results.append({"question": item["question"], "page": item["page"], "retrieved": pages, "answer": ans, "verdict": verdict})
    print(f"{n:>2}. {verdict:<10} page_hit={hit}  {item['question'][:70]}")

refused = 0
for q in TRAPS:
    ans, _ = answer(q)
    ok = ans is not None and "find this in the document" in ans.lower()
    refused += ok
    results.append({"question": q, "trap": True, "answer": ans, "refused": ok})
    print(f"TRAP refused={ok}  {q}")

n = len(evalset)
print("\n===== RESULTS =====")
print(f"Answer accuracy:            {counts['CORRECT']}/{n} = {counts['CORRECT']/n:.0%}")
print(f"  incorrect: {counts['INCORRECT']}, said not found: {counts['NOT_FOUND']}, errors: {counts['ERROR']}")
print(f"Accuracy when page retrieved: {hit_and_correct}/{hit_total}")
print(f"Trap questions refused:     {refused}/{len(TRAPS)}")
json.dump(results, open("eval_answers.json", "w"), indent=2)
