from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):

    @abstractmethod
    def download_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def extrat_zip_file(self) -> None:
        pass