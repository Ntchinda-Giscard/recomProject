import bentoml
from bentoml.validators import DType, Shape
import numpy as np
from typing import Annotated
import tensorflow as tf

SERVICE_NAME = "Recommnder-Service"
MODEL_NAME = "RecommendNet"
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
        output_tensor = await self.model.predict([X_user, X_movie])
        return output_tensor
