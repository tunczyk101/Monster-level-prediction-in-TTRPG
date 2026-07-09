import os
import warnings

import pandas as pd

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import get_time_series_split_dataframe
from training.constants import ALL_MODELS, SET_NAME
from training.train_and_evaluate_models import (
    expanding_window_train_and_evaluate_models,
)


warnings.simplefilter("ignore")

PATH_TO_DATASET = os.path.join("preprocessed_bestiaries", f"bestiaries_{SET_NAME}.csv")
MIN_MONSTERS_NUMBER = 200
START_MONSTERS_NUMBER = 213

if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    bestiaries = min_max_scale_data(bestiaries)
    bestiaries = bestiaries.drop(columns=["name"])

    ts_dataframes = get_time_series_split_dataframe(
        bestiaries, MIN_MONSTERS_NUMBER, START_MONSTERS_NUMBER
    )
    print(len(ts_dataframes), len(ts_dataframes[0]))

    results_test, results_train = expanding_window_train_and_evaluate_models(
        ALL_MODELS,
        dataframes=ts_dataframes,
        thresholds=[[0.05 * i for i in range(1, 20)], [0.05 * i for i in range(5, 16)]],
    )
