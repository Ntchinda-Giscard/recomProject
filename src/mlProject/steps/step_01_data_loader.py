from zenml import step
from mlProject.components.data_loader import MoviesDataLoader
from mlProject.config.configuration import ConfigurationManager
from mlProject.utils.materializers import PosixPathMaterializer
from mlProject import logger
from pathlib import Path, PosixPath




STAGE_NAME = "Data loader stage"

class DataingestionPipeline:

    def __init__(self) -> None:
        pass

    def main(self) -> PosixPath:

        config = ConfigurationManager()
        data_loader_config = config.get_data_ingestion_config()
        data_ingestion = MoviesDataLoader(config = data_loader_config)
        data_ingestion.download_data()
        path = data_ingestion.extract_zip_file()
        return path

@step(output_materializers = PosixPathMaterializer, enable_cache=False)
def data_loader() -> PosixPath:
    try:
        logger.info(f"\33[33m >>>>> 1️⃣ {STAGE_NAME} has started 🏁🏁 <<<<<\33[0m")
        obj = DataingestionPipeline()
        path = obj.main()
        logger.info(f"\33[33m>>>>> {STAGE_NAME} has completed ✅x=========x\33[0m")
        return PosixPath(path)
    except Exception as e:
        logger.exception(f"Oops😟! An error occured: {e} ")

