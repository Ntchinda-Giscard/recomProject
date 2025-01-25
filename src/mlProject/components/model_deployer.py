import bentoml
from bentoml.validators import DType, Shape
import numpy as np
from typing import Annotated
import tensorflow as tf

SERVICE_NAME = "Recommnder-Service"
@bentoml.service(
    name=SERVICE_NAME
)
class RecommenderService:
    def __init__(self):
        # load model
        self.model = bentoml.tensorflow.load_model(MODEL_NAME)
        self.model.eval()

    @bentoml.api()
    async def predict_rating(
        self, 
        X_user: Annotated[np.ndarray, DType("float32"), Shape((24,))],
        X_movie: Annotated[np.ndarray, DType("float32"), Shape((71,))]
    ) -> np.ndarray:
        inp = np.expand_dims(inp, (0, 1))
        output_tensor = await self.model(torch.tensor(inp))
        return to_numpy(output_tensor)
