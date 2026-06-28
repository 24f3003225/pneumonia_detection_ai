import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import PDF_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

def main():
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(f"PDF folder not found: {PDF_DIR}")

    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise RuntimeError(f"No PDFs found in: {PDF_DIR}")

    docs = []
    for fn in pdfs:
        path = os.path.join(PDF_DIR, fn)
        docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    db.save_local(VECTOR_DB_DIR)
    print(f"✅ RAG index saved to: {VECTOR_DB_DIR}")

if __name__ == "__main__":
    main()