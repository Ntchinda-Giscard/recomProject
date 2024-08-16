


from pathlib import Path

from mlProject.components.data_transformation import DataTransformation
from mlProject.config.configuration import ConfigurationManager


STAGE_NAME = "Data transformation"

class DataTransformationPipeline:

    def __init__(self) -> None:
        pass

    def main(self, ) -> None:

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