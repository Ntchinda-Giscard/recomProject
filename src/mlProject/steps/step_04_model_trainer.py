from mlProject.components.model_trainer import RecommenderNet, RecommenderTrainer
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from zenml import step
from typing import Tuple, Annotated
import pandas as pd
import numpy as np
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.callbacks import History
from mlProject.utils.materializers import HistoryMaterializer, RecommenderNetMaterializer


STAGE_NAME = "Model trainer"


class ModelTrainerPipeline:

    def __init__(self) -> None:
        pass

    def main(self, training_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[
        Annotated[Model, "recomend_model"],
        Annotated[History, "model_history"]]:
        X_user, X_movie, y = training_data
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = RecommenderTrainer(config=model_trainer_config, user_shape=X_user.shape[1], movie_shape=X_movie.shape[1])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        model, history = model_trainer.train(
            X_user,
            X_movie,
            y,
            callbacks=callbacks
        )
    
        return model, history

@step(enable_cache=False, output_materializers={
    "recomend_model": RecommenderNetMaterializer,
    "model_history": HistoryMaterializer
})
def model_trainer(X_user: np.ndarray, X_movie: np.ndarray, y: np.ndarray) -> Tuple[
        Annotated[Model, "recomend_model"],
        Annotated[History, "model_history"]]:
    try:
        logger.info(f">>>>>\33[33m>>>>4️⃣ {STAGE_NAME}🤖  step has started 🏁🏁\33[0m<<<<<")
        obj = ModelTrainerPipeline()
        model, history = obj.main((X_user, X_movie, y))
        logger.info(f">>>>>\33[33m>>>> {STAGE_NAME}🤖 step has completed✅ \33[0mx=========x")
        return model, history
    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")

if __name__ == "__main__":
    num_samples = 10  # Number of samples
    user_vector_size = 21  # Length of each user vector
    movie_vector_size = 71  # Length of each movie vector

    # Generate random data
    X_user = np.random.rand(num_samples, user_vector_size)  # Shape (10, 21)
    X_movie = np.random.rand(num_samples, movie_vector_size)  # Shape (10, 71)
    y = np.random.rand(num_samples)  # Shape (10, 1)
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = ModelTrainerPipeline()
        obj.main((X_user, X_movie, y))
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
    except Exception as e:
        raise e