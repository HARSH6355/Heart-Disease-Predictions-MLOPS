import json

from src.pipeline.predict_pipeline import HeartDiseaseInput, PredictPipeline


if __name__ == "__main__":
    with open("sample_request.json", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    input_data = HeartDiseaseInput(**payload)
    result = PredictPipeline().predict(input_data.to_dataframe())
    print(result)
