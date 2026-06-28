import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload paths
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# Models (same locations your file expects)
MODELS_DIR = os.path.join(BASE_DIR, "models")
VALIDATOR_MODEL_PATH = os.path.join(MODELS_DIR, "xray_validator.h5")
PNEUMONIA_MODEL_PATH = os.path.join(MODELS_DIR, "densenet_best.h5")

# Preprocessing (matches final_predict.py)
IMG_SIZE = 224
XRAY_VALID_THRESHOLD = 0.5
PNEUMONIA_THRESHOLD = 0.5

# RAG
PDF_DIR = os.path.join(BASE_DIR, "data", "guidelines_pdfs")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "vector_db")
TOP_K = 8
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini
GEMINI_API_KEY = "AIzaSyDXeEQUzqHgvbY1zDUhniNO6FT5_QOyoM0"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")