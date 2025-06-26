# pdf_routes.py  ── revised
import os, io
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from utils import sha256_bytes, pdf_to_png, BASE_DIR
from models import db, PDFDocument
from .auth_routes import current_user

pdf_bp = Blueprint("pdf", __name__, url_prefix="/api")

ALLOWED = {".pdf"}

@pdf_bp.route("/pdf", methods=["POST"])
@current_user()                      # <-- injects the *user* object
def upload_pdf(user):
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400
    if Path(file.filename).suffix.lower() not in ALLOWED:
        return jsonify({"error": "must be a PDF"}), 400

    raw = file.read()
    sha = sha256_bytes(raw)

    # ----- cache hit? -----
    existing = PDFDocument.query.filter_by(user_id=user.id, sha256=sha).first()
    if existing:
        return jsonify({"message": "hi", "doc_id": existing.id, "cached": True})

    # ----- cache miss → persist -----
    filename = secure_filename(file.filename)             # keep original name
    user_dir = BASE_DIR / f"user_{user.id}"
    doc_dir  = user_dir / sha                             # 1 folder per hash
    pdf_path = doc_dir / filename
    png_dir  = doc_dir / "pages"

    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(raw)

    meta = pdf_to_png(pdf_path, png_dir)                  # render → PNGs

    doc = PDFDocument(
        user_id  = user.id,
        sha256   = sha,
        filename = filename,          # store real name in DB too
        pages    = meta["pages"],
        meta     = meta,
    )
    db.session.add(doc)
    db.session.commit()

    return jsonify({"message": "hi", "doc_id": doc.id, "cached": False})
