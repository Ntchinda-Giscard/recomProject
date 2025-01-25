from zenml import step, get_step_context
import bentoml
from bentoml._internal.models.model import Model

@step(enable_cache=False)
def promote_to_bentoml() -> str:  # Explicit return type
    model_version = get_step_context().model_version
    model = model_version.get_artifact("recommend_model").load()
    # bentoml_model: Model = bentoml.sklearn.save_model("RecommendNet", model)
    # return bentoml_model.tag  # Returns a string like "zenml_promoted_model:latest"