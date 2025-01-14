import os
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import DataTransformationConfig
import pandas as pd
from mlProject import logger
from typing import Tuple
import pandas as pd
from mlProject.base import FeatureExtractor
from mlProject.config.configuration import ConfigurationManager




class MovieFeatureExtractor(FeatureExtractor):
    def process_data(self) -> pd.DataFrame:
        pass


class UserFeatureExtractor(FeatureExtractor):
    def process_data(self) -> pd.DataFrame:
        pass


class DataProcessing:

    def __int__(
            self,
            config:ConfigurationManager,
            user_feature_extractor: UserFeatureExtractor,
            movie_feature_extractor: MovieFeatureExtractor
    ) -> None:
        self.config = config
        self.movie_feature_extractor = movie_feature_extractor
        self.user_feature_extractor = user_feature_extractor






class DataTransformation:

    def __init__(self, config: DataTransformationConfig) -> Tuple[pd.DataFrame, pd.DataFrame] :
        
        self.config = config

    def train_test_spliting(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        data = pd.read_csv(self.config.data_path)
        train,test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data splitted into test and training set")
        logger.info(train.shape)
        logger.info(test.shape)

        return train, test