"""
train_model.py
---------------
Trains a MobileNetV2-based tomato leaf disease classifier using transfer
learning. Designed to be run in Google Colab (or locally with a GPU).

Expects a dataset already split into train/val folders, e.g. the
`tomatoleaf` Kaggle dataset:

    dataset/
      train/
        Tomato___Bacterial_spot/
        Tomato___Early_blight/
        ... (one folder per class)
      val/
        Tomato___Bacterial_spot/
        ...

Outputs:
    tomato_model.keras       — the trained model
    class_names.json         — class index -> label mapping
    training_history.png     — accuracy/loss curves (for your Chapter 4)
"""

import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------
# Configuration — adjust these paths/values as needed
# ---------------------------------------------------------------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT_LAYER = 100  # unfreeze from this layer onward during fine-tuning
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_datasets():
    train_data = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_data = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_data, val_data


def build_model(num_classes: int):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    conv_base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SIZE + (3,),
        pooling="avg",
    )
    conv_base.trainable = False  # frozen for the initial training phase

    model = models.Sequential(
        [
            layers.Input(shape=IMAGE_SIZE + (3,)),
            layers.Rescaling(1.0 / 255),
            data_augmentation,
            conv_base,
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, conv_base


def plot_history(history, fine_tune_history=None, save_path="output/training_history.png"):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    if fine_tune_history:
        acc += fine_tune_history.history["accuracy"]
        val_acc += fine_tune_history.history["val_accuracy"]
        loss += fine_tune_history.history["loss"]
        val_loss += fine_tune_history.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    print("Loading datasets...")
    train_data, val_data = load_datasets()
    class_names = train_data.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Improve pipeline performance
    autotune = tf.data.AUTOTUNE
    train_data = train_data.prefetch(buffer_size=autotune)
    val_data = val_data.prefetch(buffer_size=autotune)

    print("Building model...")
    model, conv_base = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "tomato_model_checkpoint.keras"),
        save_best_only=True,
        monitor="val_accuracy",
    )

    print("\n--- Phase 1: Training top layers (base frozen) ---")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=INITIAL_EPOCHS,
        callbacks=[early_stop, checkpoint],
    )

    # -----------------------------------------------------------------
    # Fine-tuning: unfreeze the top portion of MobileNetV2 and continue
    # training at a much lower learning rate. This squeezes out extra
    # accuracy once the new top layers have already learned something
    # reasonable.
    # -----------------------------------------------------------------
    print("\n--- Phase 2: Fine-tuning top layers of MobileNetV2 ---")
    conv_base.trainable = True
    for layer in conv_base.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    fine_tune_history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[early_stop, checkpoint],
    )

    plot_history(history, fine_tune_history, os.path.join(OUTPUT_DIR, "training_history.png"))

    final_loss, final_acc = model.evaluate(val_data)
    print(f"\nFinal validation accuracy: {final_acc * 100:.2f}%")
    print(f"Final validation loss: {final_loss:.4f}")

    model_path = os.path.join(OUTPUT_DIR, "tomato_model.keras")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Class names saved to {os.path.join(OUTPUT_DIR, 'class_names.json')}")
    print("\nCopy both files into models/ in the Flask app to activate diagnosis.")


if __name__ == "__main__":
    main()
