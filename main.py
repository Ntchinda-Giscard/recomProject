from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataingestionPipeline


STAGE_NAME = "Data ingestion"

try:
    logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
    obj = DataingestionPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
except Exception as e:
    raise e