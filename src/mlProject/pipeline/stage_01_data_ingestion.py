from zenml import step
from mlProject.components.data_ingestion import Dataingestion
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger



STAGE_NAME = "Data ingestion stage"

class DataingestionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> None:

        config = ConfigurationManager()
        data_injestion_config = config.get_data_ingestion_config()
        data_ingestion = Dataingestion(config = data_injestion_config)
        data_ingestion.download_file()
        data_ingestion.extrat_zip_file()

@step
def data_ingestion() -> None:
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = DataingestionPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
    except Exception as e:
        raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = DataingestionPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
    except Exception as e:
        raise e