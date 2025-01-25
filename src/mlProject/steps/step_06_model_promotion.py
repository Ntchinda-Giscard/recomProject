from zenml import step, get_step_context
from zenml.client import Client
import bentoml
from bentoml._internal.models.model import Model
from bentoml._internal.tag import Tag
import tensorflow as tf
import numpy as np

@step(enable_cache=False)
def promote_to_bentoml(
    model: tf.keras.Model,
    X_user: np.ndarray, 
    X_movie: np.ndarray, 
    y_rating: np.ndarray) -> Tag:  # Explicit return type

    bentoml_model: Model = bentoml.tensorflow.save_model("RecommendNet", model)
    print(f"Tag oooo: {bentoml_model.tag}")
    return bentoml_model.tag  # Returns a string like "zenml_promoted_model:latest"