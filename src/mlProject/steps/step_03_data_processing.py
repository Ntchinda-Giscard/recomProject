from pathlib import Path
from mlProject.components.data_processing import DataPreprocessor
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from zenml import step
import pandas as pd
from typing import Tuple, Annotated
import numpy as np

STAGE_NAME = "Data Processing"

class DataProcessingPipeline:

    def __init__(self, validation: bool) -> None:
        self.validation = validation

    def main(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        try:
            config = ConfigurationManager()
            get_data_processing_config = config.get_data_processing_config()
            data_preprocessor = DataPreprocessor(config=get_data_processing_config)
            x_user, x_movie, y = data_preprocessor.process_data()
            X_train_user, X_train_movie, y_train_rating, X_val_user, X_val_movie, y_val_rating = data_preprocessor.train_validation_split(x_user, x_movie, y)

            return X_train_user, X_train_movie, y_train_rating, X_val_user, X_val_movie, y_val_rating
        except Exception as e:
            logger.exception(f"Oops😟! An error occured: {e} ")

@step(enable_cache=False)
def data_processing(validation_status: bool) -> Tuple[
        Annotated[np.ndarray, "X_train_user"],
        Annotated[np.ndarray, "X_train_movie"],
        Annotated[np.ndarray, "y_train_rating"], 
        Annotated[np.ndarray, "X_val_user"], 
        Annotated[np.ndarray, "X_val_movie"], 
        Annotated[np.ndarray, "y_val_rating"]]:
    try:
        logger.info(f"\33[33m>>>>3️⃣ {STAGE_NAME}🛠 step started 🏁🏁 <<<<< \33[0m")
        obj = DataProcessingPipeline(validation_status)
        X_train_user, X_train_movie, y_train_rating, X_val_user, X_val_movie, y_val_rating = obj.main()
        logger.info(f"\33[33m>>>> {STAGE_NAME}🛠 step completed ✅ x=========x\33[0m")

        return X_train_user, X_train_movie, y_train_rating, X_val_user, X_val_movie, y_val_rating

    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")
