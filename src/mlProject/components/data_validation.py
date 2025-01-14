from mlProject.entity.config_entity import DataValidationConfig
import pandera as pa
import pandas as pd
from pandera import Column, DataFrameSchema
from mlProject.base import DataValidator
from typing import List, Dict
from mlProject.base import DataValidator
from mlProject.constant import TYPE_MAPPING


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
    def __init__(self, column_name: str, min_value: float, max_value: float):
        self.column_name = column_name
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, df: pd.DataFrame) -> bool:
        test_suite = TestSuite(tests=[
            TestColumnValueRange(
                column_name=self.column_name,
                left=self.min_value,
                right=self.max_value
            )
        ])
        test_suite.run(current_data=df)
        results = test_suite.as_dict()
        test_status = results['tests'][0]['status']
        if test_status == 'SUCCESS':
            print(f"Data range validation for '{self.column_name}' passed.")
            return True
        else:
            print(f"Data range validation for '{self.column_name}' failed.")
            return False

class MoviesDataValidator:
    def __init__(self, config: DataValidationConfig):
        self.schema = config.all_schema
        

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def validate_dataset(self) -> bool:
        try:
            for dataset_name in self.schema:
                df = self.load_dataset(config.unzip_file_dir[dataset_name])
                schema_validator = SchemaValidator(columns=config[dataset_name].columns)
                if self.schema_validator(dataset_name, df):
                    if dataset_name == 'ratings': 
                        range_validator = DataRangeValidator(dataset_name, config[dataset_name].min_value, config[dataset_name].max_value)
                        range_validator.validate(df)
        except Exception as e:
            return True