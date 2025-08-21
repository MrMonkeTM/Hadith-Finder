import os, time, json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

INDEX_DIR = "data/faiss_index_e5"   # use a new folder to avoid clobbering the old
MODEL_NAME = "intfloat/e5-small-v2" # or "BAAI/bge-small-en-v1.5"  (higher accuracy, still fast)

def now(msg): print(time.strftime("[%H:%M:%S]"), msg)

def load_json(path: str):
    now(f"Loading JSON from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    now(f"Loaded {len(data)} records.")
    return data

def build_docs(rows):
    # e5 prefers "passage:" prefix for documents
    docs = []
    for h in rows:
        page = f"{h.get('reference','')}\n{h.get('title','')}\n{h.get('text','')}".strip()
        page = f"passage: {page}"
        docs.append(
            Document(
                page_content=page,
                metadata={
                    "book": h.get("book",""),
                    "reference": h.get("reference",""),
                    "title": h.get("title","")
                }
            )
        )
    return docs

def main():
    if os.path.exists(INDEX_DIR):
        now(f"Index already exists at {INDEX_DIR}. Skipping build.")
        return

    # 1) Fast, GPU-enabled embedder with big batches
    now(f"Initializing embedder: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},                      # change to "cpu" if no GPU
        encode_kwargs={"batch_size": 256, "normalize_embeddings": True}
    )

    # 2) Load JSON directly (faster than JSONLoader for large files)
    data = load_json("all_hadiths.json")

    # Sanity check small subset first if you like
    # data = data[:5000]

    # 3) Build documents (reference + title + text)
    now("Building Document objects ...")
    docs = build_docs(data)
    now(f"Docs ready: {len(docs)}")

    # 4) Create and persist FAISS index (this is the heavy step)
    now("Embedding & indexing (this can take a while the first time) ...")
    vs = FAISS.from_documents(docs, embeddings)  # LangChain handles embedding in batches
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    now(f"✅ Done. Saved FAISS index to {INDEX_DIR}")

if __name__ == "__main__":
    main()