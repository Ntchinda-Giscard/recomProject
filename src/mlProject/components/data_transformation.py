import os
from sklearn.model_selection import train_test_split
from mlProject.entity.config_entity import DataTransformationConfig
import pandas as pd
from mlProject import logger


class DataTransformation:

    def __init__(self, config: DataTransformationConfig) -> None:
        
        self.config = config

    def train_test_spliting(self):

        data = pd.read_csv(self.config.data_path)
        train,test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Data splitted into test and training set")
        logger.info(train.shape)
        logger.info(test.shape)