import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


TARGET_COLUMN = "target"
CATEGORICAL_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self, config: DataTransformationConfig | None = None):
        self.config = config or DataTransformationConfig()

    def get_data_transformer_object(self, train_df: pd.DataFrame) -> ColumnTransformer:
        try:
            feature_columns = [column for column in train_df.columns if column != TARGET_COLUMN]
            categorical_columns = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
            numerical_columns = [column for column in feature_columns if column not in categorical_columns]

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns),
                ]
            )

            logging.info("Prepared preprocessing pipelines")
            return preprocessor
        except Exception as error:
            raise CustomException(error, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
                raise ValueError(f"Both train and test data must contain '{TARGET_COLUMN}'")

            preprocessing_obj = self.get_data_transformer_object(train_df)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info("Saved preprocessing object")

            return (
                input_feature_train_arr,
                input_feature_test_arr,
                target_feature_train_df,
                target_feature_test_df,
                self.config.preprocessor_obj_file_path,
            )
        except Exception as error:
            raise CustomException(error, sys)
