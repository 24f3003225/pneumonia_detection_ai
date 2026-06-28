import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTOR_DB_DIR, EMBED_MODEL, TOP_K

_db = None

def load_db_once():
    global _db
    if _db is not None:
        return _db

    if not os.path.isdir(VECTOR_DB_DIR) or not os.listdir(VECTOR_DB_DIR):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    _db = FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    return _db

def build_query(age_group: str) -> str:
    ag = (age_group or "unknown").lower().strip()
    if ag == "child":
        return "WHO IMCI pneumonia classification fast breathing chest indrawing danger signs referral India"
    if ag == "adult":
        return "WHO adult pneumonia management oxygen therapy severity CAP India ICMR NCDC"
    return "pneumonia guideline symptoms danger signs oxygen saturation referral WHO India ICMR NCDC"

def retrieve_context(db, query: str, k: int = TOP_K) -> str:
    docs = db.similarity_search(query, k=k)
    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        txt = (d.page_content or "").strip().replace("\n", " ")
        if len(txt) > 1200:
            txt = txt[:1200] + "..."
        out.append(f"[{i}] (source={src}, page={page}) {txt}")
    return "\n".join(out)