import os
import warnings

import pandas as pd

from dataset.splitting_dataset import get_time_series_split_dataframe
from training.constants import ALL_MODELS
from training.train_and_evaluate_models import (
    expanding_window_train_and_evaluate_models,
)


warnings.simplefilter("ignore")

set_name = "expanded"  # SET_NAME
PATH_TO_DATASET = os.path.join("preprocessed_bestiaries", f"bestiaries_{set_name}.csv")
MIN_MONSTERS_NUMBER = 200
START_MONSTERS_NUMBER = 213

if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    bestiaries = bestiaries.drop(columns=["name"])

    ts_dataframes = get_time_series_split_dataframe(
        bestiaries, MIN_MONSTERS_NUMBER, START_MONSTERS_NUMBER
    )
    print(len(ts_dataframes), len(ts_dataframes[0]))

    results_test, results_train = expanding_window_train_and_evaluate_models(
        ALL_MODELS, dataframes=ts_dataframes, set_name=set_name
    )
