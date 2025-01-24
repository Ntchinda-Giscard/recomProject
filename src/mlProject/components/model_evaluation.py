from pathlib import Path
import pandas as pd
import numpy as np
from mlProject.utils.common import save_json
from mlProject.components.model_trainer import RecommenderTrainer
from typing import Annotated, Tuple
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.base import ModelEvaluator
from tensorflow.keras import Model
import mlflow


class RecommendModeEvaluator(ModelEvaluator):
    def __init__(self, config: ModelTrainerConfig) -> None:

        self.config = config
    
    def eval_metrics(self, X_user_val: np.ndarray, X_movie_val: np.ndarray, y_rating_val: np.ndarray, model: Model) -> Tuple[
        Annotated[float, "MAE"], 
        Annotated[float, "MSE"]
    ]:

        metrics = model.evaluate([X_user_val, X_movie_val], y_rating_val)

        return metrics
    
    def log_into_mlflow(self, X_user_val: np.ndarray, X_movie_val: np.ndarray, y_rating_val: np.ndarray, model: Model):


        metrics = self.eval_metrics(X_user_val, X_movie_val, y_rating_val, model)
        mse = metrics[2]
        mae = metrics[1]
        hyperparameters = {
            "learning_rate": self.config.learning_rate,
            "validation_split": self.config.validation_split,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size
        }
        mlflow.log_params(hyperparameters)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
