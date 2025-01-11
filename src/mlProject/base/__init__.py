from dataclasses import dataclass
from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class DataLoader(ABC):

    @abstractmethod
    def download_data(self) -> None:
        pass

    @abstractmethod
    def extrat_zip_file(self) -> None:
        pass


class DataValidator(ABC):

    @abstractmethod
    def validate(self) -> Any:
        pass

class DataProfiler(ABC):

    @abstractmethod
    def data_profiler(self) -> Any:
        pass