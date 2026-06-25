import os
import sys
from typing import Any

import dill
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate

from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: Any) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Saved object to %s", file_path)
    except Exception as error:
        raise CustomException(error, sys)


def load_object(file_path: str) -> Any:
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as error:
        raise CustomException(error, sys)


def evaluate_binary_classifier(model: Any, x_test: Any, y_test: Any) -> dict[str, float]:
    predictions = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
    else:
        probabilities = predictions

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def cross_validate_binary_classifier(model: Any, x_train: Any, y_train: Any, cv: int = 5) -> dict[str, float]:
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)

    return {
        metric.replace("test_", "cv_"): float(np.mean(values))
        for metric, values in cv_results.items()
        if metric.startswith("test_")
    }
