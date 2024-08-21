from typing import Any
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger
from zenml import step

STAGE_NAME="Data validation"

class DataValidtionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> Any:
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
    data_validation.validate_all_columns()

@step
def data_validation():
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataValidtionPipeline()
        obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed x=========x")
    
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