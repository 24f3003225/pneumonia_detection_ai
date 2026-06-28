import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import cv2
import tensorflow as tf

from config import IMG_SIZE, VALIDATOR_MODEL_PATH, PNEUMONIA_MODEL_PATH

from tensorflow.keras.layers import DepthwiseConv2D as _BaseDepthwiseConv2D

class PatchedDepthwiseConv2D(_BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

CUSTOM_OBJECTS = {"DepthwiseConv2D": PatchedDepthwiseConv2D}

_validator = None
_pneumo = None

def load_models_once():
    global _validator, _pneumo

    tf.config.run_functions_eagerly(True)

    if _validator is None:
        _validator = tf.keras.models.load_model(
            VALIDATOR_MODEL_PATH,
            compile=False,
            custom_objects=CUSTOM_OBJECTS
        )

    if _pneumo is None:
        _pneumo = tf.keras.models.load_model(
            PNEUMONIA_MODEL_PATH,
            compile=False,
            custom_objects=CUSTOM_OBJECTS
        )

    return _validator, _pneumo

def preprocess_path(image_path: str):

    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_from_path(image_path: str):
    """
    Returns:
      ok(bool), xray_score(float), pneumo_score(float), message(str)
    """
    validator, pneumo = load_models_once()

    x = preprocess_path(image_path)
    if x is None:
        return False, 0.0, 0.0, "Image not found or unreadable."

    xray_score = float(validator.predict(x, verbose=0)[0][0])
    if xray_score < 0.5:
        return False, xray_score, 0.0, "Not a Chest X-ray."

    pneumo_score = float(pneumo.predict(x, verbose=0)[0][0])
    return True, xray_score, pneumo_score, "OK"