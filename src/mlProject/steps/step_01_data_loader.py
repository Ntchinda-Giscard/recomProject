from zenml import step
from mlProject.components.data_loader import MoviesDataLoader
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
        data_loader_config = config.get_data_ingestion_config()
        data_ingestion = MoviesDataLoader(config = data_loader_config)
        data_ingestion.download_data()
        path = data_ingestion.extract_zip_file()
        return path

# @step(output_materializers = PosixPathMaterializer, enable_cache=False)
# def data_loader() -> PosixPath:
#     try:
#         logger.info(f">>>>>1️⃣ Stage {STAGE_NAME} has started 🏁🏁<<<<<")
#         obj = DataingestionPipeline()
#         path = obj.main()
#         logger.info(f">>>>> Stage {STAGE_NAME} has completed ✅x=========x")
#         return PosixPath(path)
#     except Exception as e:
#         pass

try:
    logger.info(f">>>>>Stage 1️⃣ {STAGE_NAME} has started 🏁🏁<<<<<")
    obj = DataingestionPipeline()
    path = obj.main()
    print( f"output type {type(path)}"  )
    logger.info(f">>>>> Stage {STAGE_NAME} has completed ✅x=========x")
    # return PosixPath(path)
except Exception as e:
    logger.exception(f"Oops😟! An error occured: {e} ")
    raise e