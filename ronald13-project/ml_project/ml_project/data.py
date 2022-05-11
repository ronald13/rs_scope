from pathlib import Path
from typing import Tuple
import click
import pandas as pd

def get_dataset(
        csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    target_train = dataset["Cover_Type"]
    features_train = dataset.drop("Cover_Type", axis=1)
    return features_train, target_train