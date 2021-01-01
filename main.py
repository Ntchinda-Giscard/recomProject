from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataingestionPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidtionPipeline

STAGE_NAME = "Data ingestion"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
    obj = DataingestionPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
except Exception as e:
    raise e

STAGE_NAME="Data validation"

try:
    logger.info(f">>>> {STAGE_NAME} stage started <<<<< ")
    obj = DataValidtionPipeline()
    obj.main()
    logger.info(f">>>> {STAGE_NAME} stage completed x=========x")

except Exception as e:
    logger.exception(e)
    raise e