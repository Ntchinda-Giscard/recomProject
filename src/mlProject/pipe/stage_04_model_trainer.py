from mlProject.components.model_trainer import ModelTrainer
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from zenml import step
from typing import Tuple
import pandas as pd
from sklearn.linear_model import ElasticNet

STAGE_NAME = "Model trainer"


class ModelTrainerPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> ElasticNet:

        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()


@step(enable_cache=False)
def model_trainer(train: pd.DataFrame, test: pd.DataFrame) -> None:
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = ModelTrainerPipeline()
        model = obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
        # return model
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
    except Exception as e:
        raise e