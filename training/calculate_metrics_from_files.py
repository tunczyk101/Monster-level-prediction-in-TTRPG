import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd

from dataset.creating_dataset import min_max_scale_data
from dataset.splitting_dataset import split_dataframe
from training.constants import (
    ALL_MODELS,
    PATH_TO_DATASET,
    TEST_RESULT_FILE,
    TRAIN_RESULT_FILE,
)
from training.train_and_evaluate_models import calculate_results_from_files


warnings.simplefilter("ignore")


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)
    bestiaries = min_max_scale_data(bestiaries)

    X_train, X_test, y_train, y_test = split_dataframe(bestiaries)

    results_test, results_train = calculate_results_from_files(
        ALL_MODELS,
        y_train,
        y_test,
        thresholds=[[0.05 * i for i in range(1, 20)], [0.05 * i for i in range(5, 16)]],
        save_files=(TRAIN_RESULT_FILE, TEST_RESULT_FILE),
    )
