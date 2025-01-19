import os
from pathlib import PosixPath
from typing import Type
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from tensorflow.keras.callbacks import History
import json
import tensorflow as tf

class PosixPathMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (PosixPath,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[PosixPath]) -> PosixPath:
        """Read from artifact store."""
        with self.artifact_store.open(os.path.join(self.uri, 'path.txt'), 'r') as f:
            path_str = f.read()
        return PosixPath(path_str)

    def save(self, path: PosixPath) -> None:
        """Write to artifact store."""
        with self.artifact_store.open(os.path.join(self.uri, 'path.txt'), 'w') as f:
            f.write(str(path))


class RecommenderNetMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (tf.keras.Model,)
    ASSOCIATED_ARTIFACT_TYPES = ("Model",)

    def handle_input(self, data_type):
        model_path = os.path.join(self.artifact.uri, "recommend_model.h5")
        return tf.keras.models.load_model(model_path)

    def handle_return(self, model):
        model_path = os.path.join(self.artifact.uri, "recommend_model.h5")
        model.save(model_path)


class HistoryMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (History,)
    ASSOCIATED_ARTIFACT_TYPES = (ArtifactType.DATA,)

    def handle_input(self, data_type):
        history_path = os.path.join(self.artifact.uri, "history.json")
        with open(history_path, "r") as f:
            history_dict = json.load(f)
        history = History()
        history.history = history_dict
        return history

    def handle_return(self, history: History):
        history_path = os.path.join(self.artifact.uri, "history.json")
        with open(history_path, "w") as f:
            json.dump(history.history, f)
