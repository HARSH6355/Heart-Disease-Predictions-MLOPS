import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "data", "train.csv")
    test_data_path: str = os.path.join("artifacts", "data", "test.csv")
    dataset_id: int = 45
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig | None = None):
        self.config = config or DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str, str]:
        logging.info("Starting data ingestion")

        try:
            data_frame = self._fetch_heart_disease_data()
            data_frame = self._prepare_target(data_frame)

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            data_frame.to_csv(self.config.raw_data_path, index=False)
            logging.info("Saved raw data to %s", self.config.raw_data_path)

            train_set, test_set = train_test_split(
                data_frame,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=data_frame["target"],
            )

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Saved train data to %s", self.config.train_data_path)
            logging.info("Saved test data to %s", self.config.test_data_path)
            logging.info("Completed data ingestion")

            return (
                self.config.train_data_path,
                self.config.test_data_path,
                self.config.raw_data_path,
            )

        except Exception as error:
            raise CustomException(error, sys)

    def _fetch_heart_disease_data(self) -> pd.DataFrame:
        heart_disease = fetch_ucirepo(id=self.config.dataset_id)
        features = heart_disease.data.features
        targets = heart_disease.data.targets

        data_frame = pd.concat([features, targets], axis=1)
        logging.info("Fetched dataset with shape %s", data_frame.shape)
        return data_frame

    @staticmethod
    def _prepare_target(data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame.copy()

        if "target" in data_frame.columns:
            return data_frame

        if "num" not in data_frame.columns:
            raise ValueError("Expected target column 'num' was not found in dataset")

        data_frame["target"] = data_frame["num"].apply(lambda value: 0 if value == 0 else 1)
        data_frame.drop(columns=["num"], inplace=True)
        return data_frame


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path, raw_path = ingestion.initiate_data_ingestion()
    print(f"Raw data saved at: {raw_path}")
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")
