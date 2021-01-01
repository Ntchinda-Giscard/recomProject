from typing import Any
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger
from zenml import step
from pathlib import Path, PosixPath
import pandas as pd
from typing import Tuple

STAGE_NAME="Data validation"

class DataValidtionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> Tuple[bool, pd.DataFrame]:
        """
        This function is responsible for orchestrating the data validation pipeline.

        Parameters:
        self (DataValidtionPipeline): The instance of the DataValidtionPipeline class.

        Returns:
        Any: This function does not return any specific value. It is mainly used for orchestrating the data validation process.
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_configuration()
        data_validation = DataValidation(config=data_validation_config)
        status, data = data_validation.validate_all_columns()
        return status, data

@step(enable_cache=False)
def data_validation(data_path: PosixPath) -> pd.DataFrame:
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataValidtionPipeline()
        _, data_frame = obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed x=========x")
        return data_frame
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ =="__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataValidtionPipeline()
        obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed x=========x")
    
    except Exception as e:
        logger.exception(e)
        raise e