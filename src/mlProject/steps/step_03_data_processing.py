from pathlib import Path
from mlProject.components.data_processing import DataPreprocessor
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from zenml import step
import pandas as pd
from typing import Tuple

STAGE_NAME = "Data Processing"

class DataProcessingPipeline:

    def __init__(self, validation: bool) -> None:
        self.validation = validation

    def main(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:


        try:
            config = ConfigurationManager()
            get_data_processing_config = config.get_data_processing_config()
            data_preprocessor = DataPreprocessor(config=get_data_processing_config)
            dataset = data_preprocessor.process_data()
            X_train, X_test, y_train, y_test = data_preprocessor.train_test_spliting(dataset)

        except Exception as e:
            logger.exception(f"Oops😟! An error occured: {e} ")
            raise e

@step(enable_cache=False)
def data_processing(validation_status: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        logger.info(f"\33[33m>>>>3️⃣ {STAGE_NAME} step started 🏁🏁 <<<<< \33[0m")
        obj = DataProcessingPipeline(validation_status)
        X_tarin, X_test, y_train, y_test = obj.main()
        logger.info(f"\33[33m>>>> {STAGE_NAME} step completed ✅ x=========x\33[0m")

        return X_tarin, X_test, y_train, y_test

    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")
