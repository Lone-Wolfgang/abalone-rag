import os
import requests
import pickle
import sentence_transformers
import faiss
import gradio as gr
from transformers import pipeline
import numpy as np
from sentence_transformers import CrossEncoder

# ------------------------------
# Configuration
# ------------------------------
INDEX_URL = "https://huggingface.co/LoneWolfgang/abalone-index/resolve/main/index.faiss"
DOCSTORE_URL = "https://huggingface.co/LoneWolfgang/abalone-index/resolve/main/docstore.pkl"
INDEX_DIR = "data/index"
SBERT = "all-MiniLM-L12-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ------------------------------
# Ensure data folder exists
# ------------------------------
os.makedirs(INDEX_DIR, exist_ok=True)

# ------------------------------
# Download helper
# ------------------------------
def download_file(url, dest_path):
    print(f"Downloading {url} ...")
    r = requests.get(url)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
        print(f"Saved to {dest_path}")

# Download index + docstore
download_file(INDEX_URL, os.path.join(INDEX_DIR, "index.faiss"))
download_file(DOCSTORE_URL, os.path.join(INDEX_DIR, "docstore.pkl"))

# ------------------------------
# Retriever
# ------------------------------
class Retriever:
    def __init__(self, index_dir, sbert, cross_encoder):
        index, segments = self._load_index(index_dir)
        self.index = index
        self.segments = segments
        
        # bi-encoder
        self.sbert = sentence_transformers.SentenceTransformer(sbert)

        # cross-encoder
        self.cross = CrossEncoder(cross_encoder)

    def _load_index(self, index_dir):
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "docstore.pkl"), "rb") as f:
            segments = pickle.load(f)
        return index, segments
    
    def preprocess_query(self, query):
        embedding = self.sbert.encode([query]).astype("float32")
        faiss.normalize_L2(embedding)
        return embedding

    def retrieve(self, query, k=50):
        # ---------- Step 1: SBERT Retrieval ----------
        # Embed the query using SBERT, and retrieve the top k candidates from the index.

        embedding = self.preprocess_query(query)
        D, I = self.index.search(embedding, k)

        candidates = []
        ce_pairs_segments = []

        for idx in I[0]:
            seg = self.segments[idx]
            candidates.append(seg)
            ce_pairs_segments.append([query, seg["text"]])

        # ---------- Step 2: Cross-Encoder Re-Rank ----------
        # Use the cross-encoder to re-rank the top k candidates.

        segment_scores = self.cross.predict(ce_pairs_segments)
        best_seg_idx = int(np.argmax(segment_scores))
        best_segment = candidates[best_seg_idx]

        # ---------- Step 3: Cross-Encoder Sentence Highlighting ----------
        # After selecting the best segment, rank the sentences for the final highlighting.
        
        sentences = best_segment["sentences"]
        ce_pairs_sentences = [[query, s] for s in sentences]
        sentence_scores = self.cross.predict(ce_pairs_sentences)

        best_sent_idx = int(np.argmax(sentence_scores))
        best_sentence = sentences[best_sent_idx].strip()

        highlighted_text = (
            best_segment["text"]
            .replace(best_sentence, f"**{best_sentence}**")
            .replace("\n", " ")
        )

        return {
            "text": highlighted_text,
            "url": best_segment.get("url"),
            "document_id": best_segment.get("document_id"),
            "segment_score": float(segment_scores[best_seg_idx]),
            "sentence_score": float(sentence_scores[best_sent_idx]),
        }

# ------------------------------
# Generators 
# ------------------------------
generators = {
    "TinyLlama": pipeline(
        "text-generation",
        model="LoneWolfgang/tinyllama-for-abalone-RAG",
        max_new_tokens=150,
        temperature=0.1,
    ),
    "FLAN-T5": pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200,
    )
}

retriever = Retriever(INDEX_DIR)

# ------------------------------
# Combined function: retrieve → generate
# ------------------------------
def answer_query(query, model_choice):
    doc = retriever.retrieve(query)

    url = doc["url"]
    context = doc["text"].replace("\n", " ")

    if model_choice == "No Generation":
        # Just return context, no model generation
        return (
            f"#### Response\n\n"
            f"{context}\n\n"
            f"---\n"
            f"[Source]({url})"
        )
    else:
        prompt = f"""
        You answer questions strictly using the provided context.

        Context: {context}

        Question: {query}
        """

        # Choose generator
        gen = generators[model_choice]

        if model_choice == "TinyLlama":
            out = gen(f"<|system|>{prompt}<|assistant|>")[0]["generated_text"]
            result = out.split("<|assistant|>")[-1].strip()
        else:
            # FLAN-T5 returns text in "generated_text"
            result = gen(prompt)[0]["generated_text"]

    return (
        f"#### Response\n\n"
        f"{result}\n\n"
        f"---\n"
        f"#### Context\n\n"
        f"{context}\n\n"
        f"---\n"
        f"[Source]({url})"
    )

# ------------------------------
# Gradio UI
# ------------------------------
demo = gr.Interface(
    fn=answer_query,
    inputs=[
        gr.Textbox(label="Enter your question"),
        gr.Radio(
            ["TinyLlama", "FLAN-T5", "No Generation"],
            label="Choose Model",
            value="No Generation"
        )
    ],
    outputs=gr.Markdown(label="Answer"),
    title="Abalone RAG Demo",
    description="""This RAG system uses [SBERT](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for initial retrieval and a [Cross Encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) for re-ranking and highlighting.

Sentence embeddings are computed and [indexed](https://huggingface.co/LoneWolfgang/abalone-index) using FAISS.

For generation, you can choose between:

- [FLAN-T5](https://huggingface.co/google/flan-t5-base) — Fast and reliable, the baseline experience.
- [Finetuned TinyLlama](https://huggingface.co/LoneWolfgang/tinyllama-for-abalone-RAG) — Slower, but more expressive.
- **No Generation** — Only retrieve and highlight relevant context without generating a response. Explore the retrieval quality.
"""
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
