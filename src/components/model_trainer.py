import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import cross_validate_binary_classifier, evaluate_binary_classifier, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    metrics_plot_path: str = os.path.join("artifacts", "plots", "model_comparison.png")
    experiment_name: str = "heart-disease-classification"
    random_state: int = 42


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig | None = None):
        self.config = config or ModelTrainerConfig()

    def initiate_model_trainer(self, x_train, x_test, y_train, y_test) -> tuple[str, dict[str, float]]:
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.config.random_state),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=self.config.random_state,
                ),
            }

            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment(self.config.experiment_name)

            results: dict[str, dict[str, float]] = {}
            fitted_models = {}

            for model_name, model in models.items():
                logging.info("Training model: %s", model_name)
                cv_metrics = cross_validate_binary_classifier(model, x_train, y_train)
                model.fit(x_train, y_train)
                test_metrics = evaluate_binary_classifier(model, x_test, y_test)

                all_metrics = {**cv_metrics, **test_metrics}
                results[model_name] = all_metrics
                fitted_models[model_name] = model

                with mlflow.start_run(run_name=model_name):
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(all_metrics)
                    mlflow.sklearn.log_model(model, artifact_path="model")

            best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
            best_model = fitted_models[best_model_name]
            best_metrics = results[best_model_name]

            self._save_model_comparison_plot(results)

            with mlflow.start_run(run_name="best-model-summary"):
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metrics(best_metrics)
                mlflow.log_artifact(self.config.metrics_plot_path)

            save_object(self.config.trained_model_file_path, best_model)
            logging.info("Best model: %s with metrics: %s", best_model_name, best_metrics)

            return self.config.trained_model_file_path, {"best_model": best_model_name, **best_metrics}

        except Exception as error:
            raise CustomException(error, sys)

    def _save_model_comparison_plot(self, results: dict[str, dict[str, float]]) -> None:
        os.makedirs(os.path.dirname(self.config.metrics_plot_path), exist_ok=True)

        model_names = list(results.keys())
        roc_auc_scores = [results[name]["roc_auc"] for name in model_names]
        recall_scores = [results[name]["recall"] for name in model_names]

        x_positions = range(len(model_names))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar([x - width / 2 for x in x_positions], roc_auc_scores, width=width, label="ROC-AUC")
        plt.bar([x + width / 2 for x in x_positions], recall_scores, width=width, label="Recall")
        plt.xticks(list(x_positions), model_names)
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Heart Disease Model Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.metrics_plot_path)
        plt.close()
