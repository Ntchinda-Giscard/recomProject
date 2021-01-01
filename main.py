from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataingestionPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidtionPipeline
from mlProject.pipeline.stage_03_data_transformation import DataTransformationPipeline
from mlProject.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from mlProject.pipeline.stage_05_model_evaluation import ModelEvaluationPiepline

STAGE_NAME = "Data ingestion"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
    obj = DataingestionPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
except Exception as e:
    logger.exception(e)
    # raise e
    pass

STAGE_NAME="Data validation"

try:
    logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
    obj = DataValidtionPipeline()
    obj.main()
    logger.info(f">>>> {STAGE_NAME} stage completed \n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data transformation"

try:
    logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>> {STAGE_NAME} stage completed \n\nx=========x")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Model trainer"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
    obj = ModelTrainerPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model evaluation"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
    obj = ModelEvaluationPiepline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
except Exception as e:
    logger.exception(e)
    raise e