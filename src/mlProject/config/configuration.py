from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig, DataProcessingConfig,
                                            DataValidationConfig, ModelEvaluationConfig, ModelTrainerConfig
                                            )

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH,
            schema_filepath = SCHEMA_FILE_PATH
            ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unsizp_dir = config.unsizp_dir
        )

        return data_ingestion_config
    
    def get_data_validation_configuration(self) -> DataValidationConfig:

        config  = self.config.data_validations
        schema = self.schema


        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            all_schema= schema,
            unzip_file_dir = {'movies': config.movies, 'tags': config.tags, 'ratings': config.ratings},
            root_dir= config.root_dir,
            STATUS_FILE= config.STATUS_FILE
        )


        return data_validation_config

    def get_data_processing_config(self) -> DataProcessingConfig:

        config  = self.config.data_processing

        create_directories([config.root_dir])

        data_processing_config = DataProcessingConfig(
            movies= config.movies,
            tags= config.tags,
            ratings= config.ratings,
            root_dir= config.root_dir
        )

        return data_processing_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config  = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            learning_rate = config.learning_rate,
            validation_split = config.validation_split,
            batch_size = config.batch_size,
            epochs = config.epochs,
            model_name = config.model_name
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            test_data_path = config.test_data_path,
            model_path = config.model_path,
            all_params = params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri = "https://dagshub.com/ntchinda1998/recomProject.mlflow"
        )

        return model_evaluation_config