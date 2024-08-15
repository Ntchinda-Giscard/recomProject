from typing import Any
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import DataValidation
from mlProject import logger


STAGE_NAME="Data validation"

class DataValidtionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> Any:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_configuration()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()


if __name__ =="__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataValidtionPipeline()
        obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed x=========x")
    
    except Exception as e:
        logger.exception(e)
        raise e