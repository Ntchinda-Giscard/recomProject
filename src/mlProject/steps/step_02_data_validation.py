from typing import Any
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import MoviesDataValidator
from mlProject import logger
from zenml import step
from pathlib import Path, PosixPath
import pandas as pd
from typing import Tuple

STAGE_NAME="Data validation"

class DataValidtionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> bool:
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
        status = MoviesDataValidator.validate_dataset()
        return status

@step(enable_cache=False)
def data_validation(data_path: PosixPath) -> bool:
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataValidtionPipeline()
        status = obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed x=========x")
        return status
    except Exception as e:
        logger.exception(e)
        raise e