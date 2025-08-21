from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = "data/faiss_index_e5"
MODEL_NAME = "intfloat/e5-small-v2"  # must match the index

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"},                 # "cpu" if no GPU
    encode_kwargs={"normalize_embeddings": True}
)

faiss_db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def search(q: str, k: int = 5, diversify=False):
    # e5 expects "query:" prefix
    q = f"query: {q}"
    if diversify:
        return faiss_db.max_marginal_relevance_search(q, k=k, fetch_k=max(20, k*3), lambda_mult=0.5)
    return faiss_db.similarity_search(q, k=k)
ask = True
while ask:
    query = input("Enter what hadith you want to look for: ")
    if query.lower() in ["exit", "quit", "q"]:
        ask = False
        print("Exiting search.")
        break
    # Example
    for r in search(query, k=5):
        
        print(r.page_content)

# # Reference lookup works better now because 'reference' is inside page_content
# for r in search("Sahih al-Bukhari 1", k=3):
#     print(f"[{r.metadata.get('book')}] {r.metadata.get('reference')} — {r.metadata.get('title')}")