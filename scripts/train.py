from src.pipeline.train_pipeline import TrainPipeline


if __name__ == "__main__":
    metrics = TrainPipeline().run_pipeline()
    for name, value in metrics.items():
        print(f"{name}: {value}")
