"""
ModelService
------------
Wraps loading + inference for the trained MobileNetV2 tomato disease model.

Designed to fail gracefully: if tomato_model.keras or class_names.json
aren't present yet, is_ready() returns False and the app shows a friendly
message instead of crashing. Once you drop in your trained files, it just
works — no code changes needed.
"""

import os
import json


class ModelService:
    def __init__(self, model_path: str, class_names_path: str, image_size=(224, 224)):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.image_size = image_size
        self._model = None
        self._class_names = None
        self._load_error = None
        self._try_load()

    def _try_load(self):
        if not (os.path.exists(self.model_path) and os.path.exists(self.class_names_path)):
            return  # Files not added yet — stay "not ready" silently

        try:
            # Imported here so the whole app doesn't hard-fail at startup
            # if TensorFlow isn't installed in a lightweight dev environment.
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path)
            with open(self.class_names_path) as f:
                self._class_names = json.load(f)
        except Exception as exc:
            self._load_error = str(exc)
            self._model = None

    def is_ready(self) -> bool:
        return self._model is not None and self._class_names is not None

    def predict(self, image_path: str):
        """
        Returns (predicted_class_name, confidence, top3_list)
        where top3_list is [(class_name, confidence), ...] sorted descending.
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded yet.")

        import numpy as np
        import tensorflow as tf

        img = tf.keras.utils.load_img(image_path, target_size=self.image_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self._model.predict(img_array, verbose=0)[0]

        top_indices = predictions.argsort()[-3:][::-1]
        top3 = [(self._class_names[i], float(predictions[i])) for i in top_indices]

        best_class, best_confidence = top3[0]
        return best_class, best_confidence, top3
