from zenml import step
from mlProject.components.data_ingestion import Dataingestion
from mlProject.config.configuration import ConfigurationManager
from mlProject.utils.materializers import PosixPathMaterializer
from mlProject import logger
from pathlib import Path, PosixPath




STAGE_NAME = "Data ingestion stage"

class DataingestionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> PosixPath:

        config = ConfigurationManager()
        data_injestion_config = config.get_data_ingestion_config()
        data_ingestion = Dataingestion(config = data_injestion_config)
        data_ingestion.download_file()
        path = data_ingestion.extrat_zip_file()
        return path

@step(output_materializers = PosixPathMaterializer)
def data_ingestion() -> PosixPath:
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = DataingestionPipeline()
        path = obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
        return PosixPath(path)
    except Exception as e:
        pass

if __name__ == "__main__":
    try:
        logger.info(f">>>>> Stage {STAGE_NAME} has started <<<<<")
        obj = DataingestionPipeline()
        obj.main()
        logger.info(f">>>>> Stage {STAGE_NAME} has completed \n\n x=========x")
    except Exception as e:
        raise e
        # pass