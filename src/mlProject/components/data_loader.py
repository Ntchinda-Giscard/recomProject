#Components
import os
from pathlib import Path
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.entity.config_entity import DataIngestionConfig
from mlProject.utils.common import get_size
from mlProject.base import DataLoader


#Components
class MoviesDataLoader(DataLoader):
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_data(self) -> None:
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"File download! With the following information : \n{headers} ")
        else:
            logger.info(f"File already exists of size {get_size(Path(self.config.local_data_file))}")

    def extrat_zip_file(self) -> Path:
        unzip_path = self.config.unsizp_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        return unzip_path
