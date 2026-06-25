import pandas as pd

from src.components.data_ingestion import DataIngestion


def test_prepare_target_converts_num_to_binary_target():
    df = pd.DataFrame(
        {
            "age": [50, 60, 70],
            "num": [0, 1, 4],
        }
    )

    prepared = DataIngestion._prepare_target(df)

    assert "num" not in prepared.columns
    assert prepared["target"].tolist() == [0, 1, 1]
