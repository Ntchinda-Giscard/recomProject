from pathlib import Path
import joblib


class PredictionPipeline:

    def __init__(self) -> None:
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
    
    def predict(self, data):

        prediciotns = self.model.predict(data)

        return prediciotns