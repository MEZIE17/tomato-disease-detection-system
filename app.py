"""
Tomato Leaf Disease Detection - Flask Web App
-----------------------------------------------
Serves a web UI where a user uploads a photo of a tomato leaf,
and a trained MobileNetV2 model predicts the disease class.

To go live, drop your trained files into models/:
    models/tomato_model.keras
    models/class_names.json
The app will detect them automatically on startup.
"""

import os
import json
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from services.predict import ModelService

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load disease reference info (descriptions, causes, treatment, prevention)
with open(os.path.join(BASE_DIR, "data", "disease_info.json")) as f:
    DISEASE_INFO = json.load(f)

# Model service handles loading the .keras model + class_names.json,
# and gracefully reports "not ready" if they haven't been added yet.
model_service = ModelService(
    model_path=os.path.join(BASE_DIR, "models", "tomato_model.keras"),
    class_names_path=os.path.join(BASE_DIR, "models", "class_names.json"),
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_class_name(raw_name: str) -> str:
    """Turn 'Tomato___Late_blight' into 'Late Blight'."""
    name = raw_name.split("___")[-1]
    return name.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_ready=model_service.is_ready())


@app.route("/predict", methods=["POST"])
def predict():
    if "leaf_image" not in request.files:
        flash("No file was selected.")
        return redirect(url_for("index"))

    file = request.files["leaf_image"]

    if file.filename == "":
        flash("No file was selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Please upload a PNG, JPG, or WEBP image.")
        return redirect(url_for("index"))

    # Save with a unique name to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    if not model_service.is_ready():
        flash("The model hasn't been added yet. See README.md to plug in your trained model.")
        return redirect(url_for("index"))

    try:
        raw_class, confidence, top3 = model_service.predict(filepath)
    except Exception as exc:  # pragma: no cover - safety net for bad images etc.
        flash(f"Couldn't process that image: {exc}")
        return redirect(url_for("index"))

    display_name = format_class_name(raw_class)
    info = DISEASE_INFO.get(raw_class, {})
    is_healthy = "healthy" in raw_class.lower()

    top3_formatted = [
        {"name": format_class_name(name), "confidence": round(conf * 100, 1)}
        for name, conf in top3
    ]

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
        disease_name=display_name,
        confidence=round(confidence * 100, 1),
        is_healthy=is_healthy,
        description=info.get("description", "No description available yet."),
        symptoms=info.get("symptoms", []),
        causes=info.get("causes", []),
        treatment=info.get("treatment", []),
        prevention=info.get("prevention", []),
        top3=top3_formatted,
        timestamp=datetime.now().strftime("%B %d, %Y — %H:%M"),
    )


@app.route("/about")
def about():
    return render_template("about.html", model_ready=model_service.is_ready())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
