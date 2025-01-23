from zenml import step
from mlProject.components.model_evaluation import RecommendModeEvaluator
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig
from tensorflow.keras import Model
import numpy as np
from zenml import step, ArtifactConfig
from typing import Tuple, Annotated
import tensorflow as tf
from zenml.integrations.tensorflow.materializers.keras_materializer import KerasMaterializer


STAGE_NAME = "Model Evaluation"

class ModelEvaluationPiepline:

    def __init__(self) -> None:
        pass

    def main(self,
     X_val_user: np.ndarray, X_val_movie: np.ndarray, y_val_rating: np.ndarray, 
     model: Model) -> Model:
        print(f"Type of model inputed in evluation stage {type(model)}")
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_trainer_config()
        model_evaluation = RecommendModeEvaluator(config=model_evaluation_config)
        model_evaluation.log_into_mlflow(X_val_user, X_val_movie, y_val_rating, model)
        return model

@step(enable_cache=False, experiment_tracker= "dagshub_mlflow_tracker", output_materializers={
    "recommend_model": KerasMaterializer})
def model_evaluation(
    X_val_user: np.ndarray, X_val_movie: np.ndarray, y_val_rating: np.ndarray, 
    model: Annotated[tf.keras.Model, "recommend_model"]) -> Annotated[tf.keras.Model, "recommend_model"]:
    try:
        logger.info(f"\33[33m>>>>>5️⃣ {STAGE_NAME}🧪  step has started 🏁🏁<<<<<\33[0m")
        obj = ModelEvaluationPiepline()
        model = obj.main(X_val_user, X_val_movie, y_val_rating, model)
        logger.info(f"\33[33m>>>>> {STAGE_NAME} step has completed✅ x=========x\33[0m")
        return model
    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")
