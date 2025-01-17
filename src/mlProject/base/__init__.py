from abc import ABC, abstractmethod
import pandas as pd
from mlProject.entity.config_entity import DataProcessingConfig


class DataLoader(ABC):

    @abstractmethod
    def download_data(self) -> None:
        pass

    @abstractmethod
    def extract_zip_file(self) -> None:
        pass


class DataValidator(ABC):

    @abstractmethod
    def validate(self) -> bool:
        pass

class FeatureExtractor(ABC):

    @abstractmethod
    def generate_features(self, config: DataProcessingConfig) -> pd.DataFrame:
        pass