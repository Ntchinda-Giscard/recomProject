from pathlib import Path
from mlProject.components.data_transformation import DataTransformation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from zenml import step
import pandas as pd
from typing import Tuple

STAGE_NAME = "Data Preprocessing"

class DataTransformationPipeline:

    def __init__(self, validation: bool) -> None:
        self.validation = validation

    def main(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        

        try:
            config = ConfigurationManager()
            get_data_processing_config = config.get_data_processing_config()
            data_preprocessor = DataPreprocessor(config=get_data_processing_config)
            dataset = data_preprocessor.process_data()

        except Exception as e:
            logger.exception(f"Oops😟! An error occured: {e} ")
            raise e

@step(enable_cache=False)
def data_processing(data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info(f">>>> {STAGE_NAME} stage started...⏳ <<<<< ")
        obj = DataTransformationPipeline()
        train, test = obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed ✅")
        return train, test

    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")
        raise e



if __name__ =="__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed \n\nx=========x")
    
    except Exception as e:
        logger.exception(e)
        raise e