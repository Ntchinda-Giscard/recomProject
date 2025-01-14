from mlProject.entity.config_entity import DataValidationConfig
import pandera as pa
import pandas as pd
from pandera import Column, DataFrameSchema
from mlProject.base import DataValidator
from typing import List, Dict
from mlProject.base import DataValidator
from mlProject.constant import TYPE_MAPPING
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnQuantile
from evidently.tests.data_quality_tests import TestValueRange


class SchemaValidator(DataValidator):

    def __init__(self, columns: Dict,  dataset_name: str) -> None:
        self.columns = columns
        self. dataset_name =  dataset_name

    def get_schema(self) -> bool:      
        schema_dict = {col: Column(TYPE_MAPPING[col_type]) for col, col_type in self.columns.items()}
        return DataFrameSchema(schema_dict, strict=True)

    def validate(self, df: pd.DataFrame) -> bool:
        schema = self.get_schema()
        try:
            schema.validate(df)
            logger.info(f"Schema validation for {self.dataset_name}  passed")
            return True
        except pa.errors.SchemaError as e:
            logger.exception(f"Schema validation for {self.dataset_name}   failed: {e}")
            return False

class DataRangeValidator(DataValidator):
    def __init__(self, min_value: float, max_value: float):
        self.column_name = 'rating'
        self.min_value = min_value
        self.max_value = max_value


    def validate(self, df: pd.DataFrame) -> bool:
        test_suite = TestSuite(tests=[
            TestValueRange(
                column_name = self.column_name,
                left = self.min_value,
                right =self.max_value
            )
        ])
        test_suite.run(reference_data=df, current_data=df)
        results = test_suite.as_dict()
        test_statuses = [test['status'] for test in results['tests']]
        logger.info(f"Statuses of ran tests: {results} ")
        if all(status == 'SUCCESS' for status in test_statuses):
            logger.info(f"Data range validation for '{self.column_name}' passed.")
            return True
        else:
            logger.info(f"Data range validation for '{self.column_name}' failed.")
            return False

            import pandas as pd

class CSVDataValidator:
    def __init__(self, config: DataValidationConfig):
        
        self.schema = config.all_schema
        self.config = config
        

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def validate_dataset(self):

        try:
            for dataset_name in self.schema:
                df = self.load_dataset(self.config.unzip_file_dir[dataset_name])
                schema_validator = SchemaValidator(columns=self.schema[dataset_name].columns, dataset_name=dataset_name)
                if schema_validator.validate(df):
                    if dataset_name == 'ratings': 
                        range_validator = DataRangeValidator(self.schema[dataset_name].rating_range.min_value, self.schema[dataset_name].rating_range.max_value)
                        range_validator.validate(df)
            return True
        except Exception as e:
            logger.exception(e)
            raise e