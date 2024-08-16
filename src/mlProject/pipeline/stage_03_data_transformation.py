from pathlib import Path
from mlProject.components.data_transformation import DataTransformation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger


STAGE_NAME = "Data transformation"

class DataTransformationPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> None:
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
            data_transformation.train_test_spliting()
        
        else:
            raise Exception("Your data scheema is invalid")
    except Exception as e:
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