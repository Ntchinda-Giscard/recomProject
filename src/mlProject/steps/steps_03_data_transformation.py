from pathlib import Path
from mlProject.components.data_transformation import DataTransformation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from zenml import step
import pandas as pd
from typing import Tuple

STAGE_NAME = "Data transformation"

class DataTransformationPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to execute the data transformation pipeline.

        This function reads the validation status from a file, checks if the status is 'True',
        and if so, retrieves the data transformation configuration, creates a DataTransformation object,
        and performs the train-test split operation. If the status is not 'True', it raises an exception.

        Parameters:
        None

        Returns:
        None

        Raises:
        Exception: If the data schema is invalid.
        """

        try:

            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                train, test = data_transformation.train_test_spliting()

                return train, test
            
            else:
                raise Exception("Your data scheema is invalid")
        except Exception as e:
            raise e

@step(enable_cache=False)
def data_transformation(data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
        obj = DataTransformationPipeline()
        train, test = obj.main()
        logger.info(f">>>> {STAGE_NAME} stage completed \n\nx=========x")
        return train, test

    except Exception as e:
        logger.exception(e)
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