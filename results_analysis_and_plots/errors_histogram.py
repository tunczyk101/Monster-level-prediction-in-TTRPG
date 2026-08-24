from collections import Counter
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from dataset.splitting_dataset import split_dataframe
from results_analysis_and_plots.constants import OTHER_PLOTS_FOLDER
from training.constants import (
    CHOSEN_MODEL,
    MODELS_RESULTS_DIR,
    PATH_TO_DATASET,
    SET_NAME,
)


def load_models_results(model_name: str) -> np.ndarray:
    y_pred_test = pd.read_csv(
        os.path.join(MODELS_RESULTS_DIR, f"{SET_NAME}_{model_name}_test.csv"),
        index_col=False,
        header=None,
        names=["predictions"],
    )["predictions"].to_numpy()

    return y_pred_test


def plot_error_histogram(
    y_true: np.ndarray, model_name: str, scale: str = "linear"
) -> None:
    y_pred = load_models_results(model_name)

    errors = y_pred - y_true

    error_counts = Counter(errors.round(0).astype(int))
    error_keys = set(error_counts.keys())

    all_errors = np.arange(min(error_keys), max(error_keys) + 1)
    counts = [error_counts.get(err, 0) for err in all_errors]

    plt.figure()
    plt.bar(all_errors, counts, color="skyblue")
    plt.xlabel("Prediction error")
    plt.xticks(all_errors)
    plt.ylabel("Frequency")
    plt.yscale(scale)
    plt.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, f"error_histogram_{model_name}_{scale}.svg"),
    )
    plt.savefig(
        os.path.join(OTHER_PLOTS_FOLDER, f"error_histogram_{model_name}_{scale}.pdf"),
    )


if __name__ == "__main__":
    bestiaries = pd.read_csv(PATH_TO_DATASET, index_col=0)

    _, _, _, y_test = split_dataframe(bestiaries)

    y_test += 1

    for scale in ["linear", "log"]:
        plot_error_histogram(y_test, CHOSEN_MODEL, scale=scale)
