import json, random, time
from google import genai
from ingest import build_chunks

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
client = genai.Client()


def generate(prompt):
    for model in MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(model=model, contents=prompt).text
            except Exception as e:
                if "503" in str(e):
                    time.sleep(3 * (attempt + 1))
                    continue
                break
    return None


random.seed(42)
chunks = [c for c in build_chunks("doc.pdf") if len(c["text"]) > 600]
sample = random.sample(chunks, 25)

evalset = []
for c in sample:
    prompt = f"""Write ONE clear question that can be answered using only the passage below.
The question must make sense on its own (never say "the passage" or "the text").
Reply with only the question.

Passage:
{c['text']}"""
    q = generate(prompt)
    if q:
        evalset.append({"question": q.strip(), "page": c["page"]})
        print(len(evalset), "(page", c["page"], ")", q.strip())
    time.sleep(4)

json.dump(evalset, open("evalset.json", "w"), indent=2)
print("\nSaved", len(evalset), "questions to evalset.json")
