import os
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def create_direcrories(paths: list, verbose=True) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file : {path_to_yaml} loaded successfuly")
        return ConfigBox(content)

    except BoxValueError:
        raise ValueError("Yaml file s empty")
    except Exception as e:
        raise e

@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
    logger.info(f"Json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:

    with open(path) as file:
        content = json.load(file)
    logger.info(f"Json file saved at: {path}")

    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:

    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")

    return data

@ensure_annotations
def get_size(path: Path) -> str:

    size_in_kb = round(os.path.getsize(path)/1024)

    return f"~{size_in_kb} KB"