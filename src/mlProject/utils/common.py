import os
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from mlProject import logger


def create_directories(paths: list, verbose=True) -> None:
    """
    Creates directories at the specified paths.

    This function iterates over a list of paths and creates each directory using the `os.makedirs()` function.
    If a directory already exists, it will not be recreated. If the `verbose` parameter is set to `True`,
    the function will log a message for each directory created using the `logger.info()` function.

    Parameters:
    paths (list): A list of strings representing the paths where directories need to be created.
    verbose (bool, optional): A flag indicating whether to log information about each created directory. Defaults to True.

    Returns:
    None
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    This function opens the specified YAML file, reads its content using the `yaml.safe_load()` function,
    and logs a success message using the `logger.info()` function. If the YAML file is empty,
    it raises a `ValueError` with the message "Yaml file is empty". If any other exception occurs,
    it is raised as is.

    Parameters:
    path_to_yaml (Path): The path to the YAML file to be read.

    Returns:
    ConfigBox: A ConfigBox object containing the content of the YAML file.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file : {path_to_yaml} loaded successfully")
        return ConfigBox(content)

    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e

# @ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """
    Saves a dictionary as a JSON file at the specified path.

    This function opens the specified file in write mode, writes the given dictionary as a JSON string
    to the file, and then closes the file. It also logs a message indicating the successful save operation.

    Parameters:
    path (Path): The path to the JSON file where the data will be saved.
    data (dict): The dictionary to be saved as a JSON file.

    Returns:
    None
    """
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
    logger.info(f"Json file saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file from the specified path and return its content as a ConfigBox object.

    This function opens the specified JSON file, reads its content using the `json.load()` function,
    and logs a message indicating the successful load operation.

    Parameters:
    path (Path): The path to the JSON file to be loaded.

    Returns:
    ConfigBox: A ConfigBox object containing the content of the JSON file.
    """
    with open(path) as file:
        content = json.load(file)
    logger.info(f"Json file loaded from: {path}")

    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """
    Save a given data object to a binary file using joblib.

    This function takes a data object and a file path as input. It uses the joblib library to serialize
    the data object and save it to the specified file path. After saving the data, it logs a message
    indicating the successful save operation.

    Parameters:
    data (Any): The data object to be saved. This can be of any type that is serializable by joblib.
    path (Path): The file path where the data will be saved.

    Returns:
    None
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file from the specified path and return its content.

    This function uses the joblib library to deserialize a binary file and load its content.
    The binary file should have been saved using the `save_bin()` function.

    Parameters:
    path (Path): The path to the binary file to be loaded.

    Returns:
    Any: The deserialized content of the binary file. The type of the returned object depends on the data saved in the file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")

    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Calculate and return the size of a file in kilobytes.

    Parameters:
    path (Path): The path to the file for which the size needs to be calculated.

    Returns:
    str: The size of the file in kilobytes, formatted as a string with a tilde prefix.
    """
    size_in_kb = round(os.path.getsize(path)/1024)

    return f"~{size_in_kb} KB"