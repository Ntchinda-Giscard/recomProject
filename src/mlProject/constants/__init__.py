from pathlib import Path
import pandera as pa

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")
TYPE_MAPPING = {
    'int': pa.Int,
    'float': pa.Float,
    'str': pa.String
}