from typing import Any
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_validation import CSVDataValidator
from mlProject import logger
from zenml import step
from pathlib import Path, PosixPath
import pandas as pd
from typing import Tuple, Annotated

STAGE_NAME="Data validation"

class DataValidtionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> bool:
            """
            Performs data validation on a dataset using the specified configuration.

            Returns:
                bool: The status of the data validation process.
            """
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_configuration()
            data_validator = CSVDataValidator(config=data_validation_config)
            status = data_validator.validate_dataset()
            return status

@step(enable_cache=False)
def data_validation(data_path: PosixPath) -> Annotated[bool, "validation_state"]:
    try:
        logger.info(f"\33[33m>>>> 2️⃣ {STAGE_NAME} step started 🏁🏁 <<<<<\33[0m ")
        obj = DataValidtionPipeline()
        status = obj.main()
        logger.info(f"\33[33m>>>>> {STAGE_NAME} step has completed ✅x=========x\33[0m")
        return status
    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")