from zenml import step
from mlProject.components.model_evaluation import ModelEvaluation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from sklearn.linear_model import ElasticNet


STAGE_NAME = "Model evaluation"

class ModelEvaluationPiepline:

    def __init__(self) -> None:
        pass

    def main(self) -> None:

        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()

@step
def model_evaluation(model: ElasticNet) -> None:
    try:
        logger.info(f" >>>>>>>> {STAGE_NAME} has started <<<<<<<<<< ")
        obj = ModelEvaluationPiepline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} has completed \n\n x==================x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    try:
        logger.info(f" >>>>>>>> {STAGE_NAME} has started <<<<<<<<<< ")
        obj = ModelEvaluationPiepline()
        obj.main()
        logger.info(f">>>>> {STAGE_NAME} has completed \n\n x==================x")
    except Exception as e:
        logger.exception(e)
        raise e