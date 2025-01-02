import os
from pathlib import PosixPath
from typing import Type
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer

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
