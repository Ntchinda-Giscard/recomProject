#Components
import os
from pathlib import Path
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.entity.config_entity import DataIngestionConfig
from mlProject.utils.common import get_size


class Dataingestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_file(self) -> None:

        if not os._exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename= self.config.local_data_file
            ) 
            logger.info(f"File download! With the following information : \n{headers} ")
        else:
            logger.info(f"File already exist of size {get_size(Path(self.config.local_data_file))}")
    
    def extrat_zip_file(self):
        unzip_path = self.config.unsizp_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
