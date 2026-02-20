# Chiyou – We Cure You

A small web app for **Chiyou** that reads the emotional color profile of an image and recommends a perfume inspired by Korean traditional medicine.

## Project layout

- `app.py` – Flask web server, file upload, routing, and integration of the model with the UI.
- `model.py` – CNN-style color extractor + valence/arousal based perfume recommender.
- `data.json` – color → descriptor data used to build emotional profiles.
- `colors_to_emotions.json` – optional helper mapping colors to emotions (source data).
- `templates/index.html` – Chiyou-branded upload and results page.
- `static/` – brand assets such as `chiyou_logo.png`.
- `uploads/` – runtime folder where uploaded images are stored (not needed in git).

## Setup

From this `chiyou` directory:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Make sure `data.json` is present in this folder (it should be committed to the repo).

## Run the app

From inside `chiyou/`:

```bash
python app.py
```

Then open `http://127.0.0.1:5000/` in your browser. Upload a photo to see the top colors, their emotional coordinates, and the Chiyou perfume recommendation.

## Create the git repository

From inside `chiyou/`:

```bash
git init
git add app.py model.py data.json colors_to_emotions.json templates static requirements.txt README.md
# Optionally ignore uploads and virtualenv
cat << 'GITEOF' > .gitignore
venv/
__pycache__/
uploads/
*.pyc
GITEOF

git add .gitignore
git commit -m "Initial commit: Chiyou color-to-scent recommender"
```

You can then add a remote (GitHub, GitLab, etc.) and push the repo as usual.
