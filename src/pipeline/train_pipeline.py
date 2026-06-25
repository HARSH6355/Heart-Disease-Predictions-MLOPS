import sys

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.components.model_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run_pipeline(self) -> dict[str, float]:
        try:
            logging.info("Training pipeline started")

            train_path, test_path, _ = DataIngestion().initiate_data_ingestion()

            (
                train_arr,
                test_arr,
                train_target,
                test_target,
                _,
            ) = DataTransformation().initiate_data_transformation(train_path, test_path)

            _, metrics = ModelTrainer().initiate_model_trainer(
                train_arr,
                test_arr,
                train_target,
                test_target,
            )

            logging.info("Training pipeline completed")
            return metrics
        except Exception as error:
            raise CustomException(error, sys)


if __name__ == "__main__":
    results = TrainPipeline().run_pipeline()
    print("Training complete")
    for metric, value in results.items():
        print(f"{metric}: {value}")
