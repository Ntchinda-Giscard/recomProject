from zenml import step, get_step_context
from zenml.client import Client

import bentoml
from bentoml._internal.models.model import Model

@step(enable_cache=False)
def promote_to_bentoml() -> None:  # Explicit return type
    latest_model_version = client.get_latest_model_version(name="RecommendNet")

    print(f"Artifact {model_version}")
    # model = model_version.get_artifact("recommend_model").load()
    # bentoml_model: Model = bentoml.sklearn.save_model("RecommendNet", model)
    # return bentoml_model.tag  # Returns a string like "zenml_promoted_model:latest"