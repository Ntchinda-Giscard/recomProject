from zenml import step, get_step_context
from zenml.client import Client
import bentoml
from bentoml._internal.models.model import Model
import tensorflow as tf

@step(enable_cache=False)
def promote_to_bentoml(model: tf.keras.Model) -> None:  # Explicit return type
    model_version = get_step_context().model_version
    model = model_version.get_artifact("model").load()
    print(f"Artifact to promote {model_version}")
    # model = model_version.get_artifact("recommend_model").load()
    # bentoml_model: Model = bentoml.sklearn.save_model("RecommendNet", model)
    # return bentoml_model.tag  # Returns a string like "zenml_promoted_model:latest"