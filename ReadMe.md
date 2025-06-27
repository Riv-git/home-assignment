# Shape-Count Web App 🟣🟥⬛

[**▶︎ Watch the 2-min demo video**](https://www.loom.com/share/fea2a56cee3944f29c4b5ab65001c97b?sid=2d1b1bfb-2494-444c-bdcd-7a679302e42c)

A one-stop Flask application that takes your PDFs, renders each page to PNG and uses a compact CNN to count **circles, squares and rectangles**. Results are shown in an HTML dashboard and can be downloaded as CSV.

> **ML weights:** `checkpoints/best.pt`

---

## ✨ What you get

* **PDF → PNG** rendering via **PyMuPDF** (150 DPI).
* **Fast CPU inference** using a 4-layer CNN (≈ 92 KB).
* **Per-user caching** — identical PDFs are processed once.
* Minimal **SQLite** DB (`app.db`) to track users & documents.

---

## 🗂 Folder Guide

```
home-assignment/
├── checkpoints/        # best.pt (trained CNN)
├── server/
│   ├── app.py         # Flask factory
│   ├── models.py      # SQLAlchemy tables
│   ├── shape_counter.py # loads CNN + predict_counts()
│   ├── routes/        # web UI & PDF endpoints
│   └── templates/     # login.html, dashboard.html, count.html
├── shape_count_cnn.py # (optional) training script
└── storage/           # runtime cache (PDFs + PNGs)
```

---

## ⚡ Install & Run (3 commands)

```bash
# 1. deps
pip install flask flask-cors flask-login flask-wtf flask-sqlalchemy \
             python-dotenv PyJWT PyMuPDF torch torchvision pillow

# 2. secrets (env or .env)
export FLASK_SECRET="super-secret"
export JWT_SECRET="dev-secret"

# 3. launch
python -m server.app        # defaults to http://localhost:5000
```

That's it – open your browser, register, log in, upload a PDF and enjoy the counts.

---

## 🔍 How it works – under the hood

| Stage | What happens | Code pointer |
|-------|-------------|--------------|
| 1. Upload | User chooses a PDF in the dashboard, which submits a multipart form. | `routes/web_routes.py:dashboard()` |
| 2. De-dup | The raw file bytes are hashed (sha256). If the same user already uploaded that hash, we skip re-processing. | `utils.sha256_bytes()` |
| 3. Rendering | Fresh PDFs are stored under `storage/user_<id>/<hash>/`. PyMuPDF converts each page to `page_001.png`, `page_002.png`, … | `utils.pdf_to_png()` |
| 4. Counting | For every `page_*.png`, the helper `shape_counter.predict_counts()`: <br>• loads `checkpoints/best.pt` once (cached with `@lru_cache`). <br>• resizes the image to 64 × 64. <br>• runs the CNN → 3 numbers (circles, squares, rectangles). | `server/shape_counter.py` |
| 5. Display | The dashboard shows a table with counts, and each PNG is served on demand for previews. | `routes/web_routes.py` (`_build_mock_table`, `preview()`) |
| 6. Export | Click "Download CSV" to receive `yourfile_counts.csv` with per-page counts and image paths. | `routes/web_routes.py:download_csv()` |

---

## 🧠 Model in 30 seconds

| Component | Details |
|-----------|---------|
| Input | RGB 64 × 64 |
| Backbone | 4× Conv-ReLU-MaxPool blocks (16→32→64→128 filters) |
| Head | FC 128·4·4 → 256 → 3 |
| Loss | Mean-Squared-Error on raw counts |
| Metric | Mean Absolute Error (MAE) |

The training script (`shape_count_cnn.py`) splits 10% of the data for validation and saves the best checkpoint to `checkpoints/best.pt` whenever the val-loss improves.