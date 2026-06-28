import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uuid
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from config import UPLOAD_DIR, XRAY_VALID_THRESHOLD
from models.wrapper import predict_from_path
from rag.retriever import load_db_once, build_query, retrieve_context
from rag.prompt_template import build_prompt
from rag.llm_client import llm_generate

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB = load_db_once()

@app.route("/", methods=["GET"])
def home():
    return render_template("dashboard.html", result=None, error=None)

@app.route("/analyze", methods=["POST"])
def analyze():
    global DB

    if DB is None:
        return render_template("dashboard.html", result=None,
                               error="Vector DB missing. Run: python -m rag.build_index")

    f = request.files.get("image")
    if f is None or not f.filename:
        return render_template("dashboard.html", result=None, error="No image uploaded.")

    age_group = (request.form.get("age_group", "unknown") or "unknown").lower().strip()
    notes = (request.form.get("notes", "") or "").strip()

    ext = os.path.splitext(f.filename)[1].lower()
    uid = str(uuid.uuid4())[:10]
    filename = secure_filename(f"xray_{uid}{ext if ext else '.png'}")
    img_path = os.path.join(UPLOAD_DIR, filename)
    f.save(img_path)

    ok, xray_score, pneumo_prob, msg = predict_from_path(img_path)
    if not ok:
        return render_template("dashboard.html", result=None,
                               error=f"{msg} (xray_score={xray_score:.2f})")

    if xray_score < XRAY_VALID_THRESHOLD:
        return render_template("dashboard.html", result=None,
                               error=f"Not a valid chest X-ray (score={xray_score:.2f}).")

    # RAG + Gemini
    query = build_query(age_group)
    context = retrieve_context(DB, query, k=8)
    prompt = build_prompt(xray_score, pneumo_prob, age_group, notes, context)
    doctor_note = llm_generate(prompt)

    result = {
        "xray_valid_score": round(xray_score, 3),
        "pneumo_prob_percent": round(pneumo_prob * 100.0, 2),
        "age_group": age_group,
        "rag_query": query,
        "doctor_note": doctor_note,
    }
    return render_template("dashboard.html", result=result, error=None)

if __name__ == "__main__":
    app.run(debug=True)