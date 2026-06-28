import os
# ---- TF stability on servers/CPU ----
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from PIL import Image

tf.config.run_functions_eagerly(True)

from config import IMG_SIZE, PNEUMO_MODEL_PATH, VALIDATOR_MODEL_PATH

# -----------------------------
# PATCH: Ignore "groups" in DepthwiseConv2D during model load
# -----------------------------
from tensorflow.keras.layers import DepthwiseConv2D as _BaseDepthwiseConv2D

class PatchedDepthwiseConv2D(_BaseDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

CUSTOM_OBJECTS = {
    "DepthwiseConv2D": PatchedDepthwiseConv2D
}

_pneumo_model = None
_validator_model = None

def _load_models_once():
    global _pneumo_model, _validator_model

    if _validator_model is None:
        if not os.path.exists(VALIDATOR_MODEL_PATH):
            raise FileNotFoundError(f"Missing validator model: {VALIDATOR_MODEL_PATH}")
        _validator_model = tf.keras.models.load_model(
            VALIDATOR_MODEL_PATH,
            compile=False,
            custom_objects=CUSTOM_OBJECTS
        )
        try:
            _validator_model.compile(jit_compile=False)
        except Exception:
            pass

    if _pneumo_model is None:
        if not os.path.exists(PNEUMO_MODEL_PATH):
            raise FileNotFoundError(f"Missing pneumonia model: {PNEUMO_MODEL_PATH}")
        _pneumo_model = tf.keras.models.load_model(
            PNEUMO_MODEL_PATH,
            compile=False,
            custom_objects=CUSTOM_OBJECTS
        )
        try:
            _pneumo_model.compile(jit_compile=False)
        except Exception:
            pass

def preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    MATCHES your final_predict.py:
    - resize 224x224
    - scale 0..1
    """
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # [1,H,W,3]

def validate_is_chest_xray(pil_img: Image.Image) -> float:
    _load_models_once()
    x = preprocess(pil_img)
    pred = _validator_model.predict(x, verbose=0)
    return float(pred[0][0])

def predict_pneumonia_probability(pil_img: Image.Image) -> float:
    _load_models_once()
    x = preprocess(pil_img)
    pred = _pneumo_model.predict(x, verbose=0)
    return float(pred[0][0])