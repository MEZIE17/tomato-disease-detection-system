"""
evaluate_model.py
------------------
Loads a trained tomato disease model and evaluates it on a held-out
validation/test set, producing the metrics typically needed for a Chapter 4
results section:

    - Overall accuracy
    - Per-class precision, recall, F1-score
    - Confusion matrix (as a saved image)

Usage:
    python evaluate_model.py

Expects:
    output/tomato_model.keras (or update MODEL_PATH below)
    output/class_names.json
    dataset/val/  (same structure as training: one subfolder per class)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = "output/tomato_model.keras"
CLASS_NAMES_PATH = "output/class_names.json"
VAL_DIR = "dataset/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Loading model and class names...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    print("Loading validation dataset...")
    val_data = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Running predictions on validation set...")
    y_true = []
    y_pred = []

    for images, labels in val_data:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- Classification report: precision, recall, F1, accuracy --------
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\nClassification Report:\n")
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    # ---- Confusion matrix ------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Greens", colorbar=False)
    plt.title("Confusion Matrix — Tomato Leaf Disease Classification")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

    overall_accuracy = (y_true == y_pred).mean()
    print(f"\nOverall validation accuracy: {overall_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
