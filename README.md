# Leaf & Ledger — Tomato Disease Detection Web App

A Flask web app that diagnoses tomato leaf diseases from a photo, using a
MobileNetV2 transfer-learning model trained in Google Colab.

## Project structure

```
tomato-disease-detection-system/
├── app.py                    # Flask app & routes
├── requirements.txt
├── data/
│   └── disease_info.json     # name, description, symptoms, causes, treatment, prevention
├── models/
│   ├── tomato_model.keras    # ← YOUR TRAINED MODEL GOES HERE
│   └── class_names.json      # ← YOUR CLASS NAMES GO HERE
├── services/
│   └── predict.py            # Loads model, runs predictions
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/
├── templates/
│   ├── base.html
│   ├── index.html             # Upload page
│   ├── result.html            # Diagnosis report page (now includes Symptoms)
│   └── about.html
└── training/
    ├── train_model.py         # Full MobileNetV2 training script (run in Colab)
    └── evaluate_model.py      # Accuracy / precision / recall / F1 / confusion matrix
```

## Step 1 — Add your trained model

Copy your downloaded `tomato_model.keras` and `class_names.json` into the
`models/` folder. No code changes needed — the app detects them
automatically.

> If you trained with a different image size than 224×224, update
> `image_size` in `services/predict.py`.

## Step 2 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

(If you only want to run the web app — not retrain or evaluate — you can
skip `matplotlib` and `scikit-learn`; they're only used by the scripts in
`training/`.)

## Step 3 — Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000**.

## Training script (training/train_model.py)

Run this in Colab if you want to retrain or reproduce your model. It:
- Loads your `dataset/train` and `dataset/val` folders
- Applies data augmentation (flip, rotation, zoom)
- Builds MobileNetV2 with a frozen base + custom classification head
- Trains with EarlyStopping and ModelCheckpoint
- **Fine-tunes**: unfreezes the top layers of MobileNetV2 for a second
  training phase at a lower learning rate, typically boosting accuracy
- Saves `tomato_model.keras`, `class_names.json`, and a
  `training_history.png` accuracy/loss chart — useful directly in your
  Chapter 4 (Results)

## Evaluation script (training/evaluate_model.py)

Run this after training to generate the metrics typically expected in a
Chapter 4 results section:
- Overall accuracy
- Per-class precision, recall, F1-score (`classification_report.txt`)
- Confusion matrix image (`confusion_matrix.png`)

```bash
cd training
python evaluate_model.py
```

## A note on Chapter One vs. implementation

If your Chapter One proposal describes a CNN + Vision Transformer (ViT)
hybrid architecture, be aware that **this implementation uses MobileNetV2
transfer learning only** (MobileNetV2 is itself a CNN, just not a hybrid
with a transformer). When writing Chapter Three (Methodology) and Chapter
Four (Results), describe what was actually built and evaluated —
MobileNetV2 — and you can mention a CNN-ViT hybrid as related work or a
suggested future improvement rather than something implemented here. This
keeps your report internally consistent and easy to defend.

## What this app does and doesn't do

**Does:** upload a photo → predict disease class → show confidence score →
display description, symptoms, causes, treatment, and prevention from
`disease_info.json`.

**Doesn't (by design, to keep scope manageable):** user accounts, a
database, prediction history, or an admin dashboard. All storage is JSON
files only, per the current project scope. These can be added later as
extensions if your final report calls for them.

## Troubleshooting

- **"No description available yet."** → the class name in
  `class_names.json` doesn't match a key in `disease_info.json`. Open both
  files and make sure the class names match exactly (case and underscores
  included).
- **Low confidence on every prediction** → check `IMAGE_SIZE` matches
  between training and `services/predict.py`, and consider more training
  epochs or a larger/cleaner dataset.
- **pip install errors on tensorflow/numpy** → don't pin exact versions;
  use the unpinned `requirements.txt` as provided, which lets pip pick
  versions compatible with your installed Python version.
