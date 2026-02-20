import os
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np

import model


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_scent_engine():
    """
    Helper to build the ScentRecommender from data.json and va_descriptors.
    """
    data_path = BASE_DIR / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.json not found at {data_path}. "
            "Make sure your color->descriptor JSON is in the same folder as app.py."
        )
    color_map = model._build_color_map_from_data_json(str(data_path))
    return model.ScentRecommender(color_map, model.va_descriptors)


app = Flask(__name__)
app.secret_key = "change_this_in_production"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "photo" not in request.files:
            flash("No file part in request.")
            return redirect(request.url)

        file = request.files["photo"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Please upload an image file (png, jpg, jpeg, gif, webp).")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        # Run the CNN color extractor + perfume recommender
        try:
            # Step 1: image -> color percentages array
            arr = model.extract_color_percentages_array(str(save_path))  # shape (16,)
            color_percentages = {c: float(arr[i]) for i, c in enumerate(model.COLOR_LIST)}

            # Sort and take top 5 colors
            top5 = sorted(
                color_percentages.items(), key=lambda kv: kv[1], reverse=True
            )[:5]
            top5_dict = {c: p for c, p in top5}

            # Step 2: build scent engine and get recommendation based on top 5 colors
            engine = load_scent_engine()
            result = engine.get_scent(color_percentages=top5_dict)

            coordinates = result["coordinates"]
            recommendation = result["recommendation"]

            return render_template(
                "index.html",
                image_url=url_for("uploaded_file", filename=filename),
                top5_colors=top5,
                coordinates=coordinates,
                recommendation=recommendation,
            )
        except ImportError as e:
            flash(
                f"{e}. Make sure Pillow is installed: "
                '"python3 -m pip install pillow"'
            )
            return redirect(request.url)
        except Exception as e:  # pragma: no cover
            flash(f"Something went wrong while analyzing the image: {e}")
            return redirect(request.url)

    # GET request
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return app.send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == "__main__":
    # Run with: python app.py
    app.run(debug=True)

