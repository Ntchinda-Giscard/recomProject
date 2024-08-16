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
        """
        Downloads a file from a specified URL to a local directory.

        If the file already exists locally, it logs the file size and does not download it again.

        Parameters:
        - self: The instance of the class.
        - self.config.source_URL (str): The URL from which to download the file.
        - self.config.local_data_file (str): The local path where the file will be saved.

        Returns:
        - None: The function does not return any value.
        """

        if not os._exists(self.config.local_data_file):
            _, headers = request.urlretrieve(
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
