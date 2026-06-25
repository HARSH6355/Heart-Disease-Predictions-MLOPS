from src.pipeline.predict_pipeline import HeartDiseaseInput


def test_heart_disease_input_to_dataframe_has_expected_columns():
    payload = HeartDiseaseInput(
        age=54,
        sex=1,
        cp=4,
        trestbps=130,
        chol=250,
        fbs=0,
        restecg=1,
        thalach=150,
        exang=0,
        oldpeak=1.0,
        slope=2,
        ca=0,
        thal=3,
    )

    df = payload.to_dataframe()

    assert df.shape == (1, 13)
    assert list(df.columns) == [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]
