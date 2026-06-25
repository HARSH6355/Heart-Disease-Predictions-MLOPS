import sys
from dataclasses import asdict, dataclass

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class HeartDiseaseInput:
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


class PredictPipeline:
    def __init__(
        self,
        model_path: str = "artifacts/model.pkl",
        preprocessor_path: str = "artifacts/preprocessor.pkl",
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path

    def predict(self, features: pd.DataFrame) -> dict[str, float | int]:
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            transformed_features = preprocessor.transform(features)
            prediction = int(model.predict(transformed_features)[0])

            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(transformed_features)[0][prediction])

            return {
                "prediction": prediction,
                "confidence": confidence if confidence is not None else 0.0,
            }
        except Exception as error:
            raise CustomException(error, sys)
