import os, tempfile, time
import numpy as np
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
import ingest

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
K = 6
PREFIX = "Represent this sentence for searching relevant passages: "

st.set_page_config(page_title="Ask your PDF")
st.title("Ask your PDF")
st.caption("Answers come only from the document, with page citations.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set.")
    st.stop()


@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_client():
    return genai.Client()


@st.cache_data(show_spinner="Reading and indexing the PDF...")
def index_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        path = f.name
    chunks = ingest.build_chunks(path)
    os.remove(path)
    emb = load_embedder().encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    return chunks, emb


def generate(prompt):
    client = get_client()
    for model in MODELS:
        for attempt in range(3):
            try:
                return client.models.generate_content(model=model, contents=prompt).text.strip()
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    time.sleep(3 * (attempt + 1))
                    continue
                break
    return "All models are busy right now. Please try again in a minute."


uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None:
    data = uploaded.getvalue()
elif os.path.exists("doc.pdf"):
    data = open("doc.pdf", "rb").read()
    st.info("Using the sample document. Upload your own PDF above to replace it.")
else:
    st.warning("Upload a PDF to get started.")
    st.stop()

chunks, emb = index_pdf(data)
st.write(f"Indexed {len(chunks)} chunks.")

question = st.chat_input("Ask a question about the document")
if question:
    with st.chat_message("user"):
        st.write(question)
    q = load_embedder().encode(PREFIX + question, normalize_embeddings=True)
    top = np.argsort(emb @ q)[::-1][:K]
    context = "\n\n".join(f"[Page {chunks[i]['page']}]\n{chunks[i]['text']}" for i in top)
    prompt = f"""Answer the question using ONLY the context below.
Cite page numbers like (page 109).
If the answer is not in the context, say: I couldn't find this in the document.

Context:
{context}

Question: {question}"""
    with st.spinner("Thinking..."):
        answer = generate(prompt)
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Sources"):
            for i in top:
                st.markdown(f"**Page {chunks[i]['page']}**")
                st.text(chunks[i]["text"])
